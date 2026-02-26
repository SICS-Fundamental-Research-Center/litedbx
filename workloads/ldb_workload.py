import json
import pandas as pd
import logging
from pathlib import Path
from typing import Tuple
from collections import defaultdict
from data_structure import LdbData, SemCQ, PopulationSpec, PopulationSpecs, FeatureRefinementResponse
from llm import LdbLLMClient, PROMPTS
from common import (
    pred_and_eval, 
    select_coreset, 
    compute_feature_importance,
    encode_features,
    train_classifier,
    evaluate_classifier,
)

logger = logging.getLogger(__name__)

class LdbWorkload:
    def __init__(self, data_dir: str, scenario: str, queries: dict[str, SemCQ], config: dict) -> None:
        
        self.llm_client = LdbLLMClient()

        self.data_dir = data_dir
        self.scenario = scenario
        self.ldb_data = LdbData(data_dir=data_dir)
        self.queries = queries
        self.config = config
        self.random_seed = config["random_seed"]
        self.b_lab = config["b_lab"]
        self.b_se = config["b_se"]
        self.b_rew = config["b_rew"]
        self.b_fs = config["b_fs"]

        self.sigma_satisfied_data = {}
        self.labeled_data = {}
        self.unlabeled_data = {}
        self.coresets = {}

        self.feature_spaces = {}
        
        self.base_schema = self.ldb_data.df.columns.tolist()
        self.candidate_external_features = []

        self.CKPT_path = Path(__file__).parent.parent / ".ckpt" / self.scenario
        self.CKPT_path.mkdir(parents=True, exist_ok=True)

    def apply_sigma(self, debug=False):
        for q_name, sem_cq in self.queries.items():

            data = self.ldb_data.sigma_retrieve(sem_cq.Sigma, reset_index=True)
            labels = self._build_ground_truth_labels(data, q_name, sem_cq.selected, debug=debug)

            self.sigma_satisfied_data[q_name] = {
                "data": data,
                "labels": labels,
            }

    
    async def init_coresets(self, enable_refinement: bool = False, debug: bool = False,
                            max_refine_iter: int = 3, f1_threshold: float = 0.01,
                            n_bad_cases: int = 5):

        for q_name, sem_cq in self.queries.items():
            # Acquire labeled set.
            self._acquire_human_label(q_name=q_name, debug=debug)

            # Acquire initial feature space.
            self.feature_spaces[q_name] = self._init_feature_space(q_name, sem_cq)

            # Populate feature values for the acquired feature space.
            result_df = await self._materialize_features(
                q_name=q_name,
                tag="labeled_full",
                data=self.labeled_data[q_name]["data"],
                feature_specs=self.feature_spaces[q_name],
            )

            # Store materialized data in coreset
            self.coresets[q_name] = {
                "data": LdbData(df=result_df, config=self.ldb_data.config),
                "labels": self.labeled_data[q_name]["labels"],
            }

            # Refine the feature space if enabled.
            if enable_refinement:
                await self._refine_feature_space(
                    q_name=q_name,
                    sem_cq=sem_cq,
                    max_iter=max_refine_iter,
                    f1_threshold=f1_threshold,
                    n_bad_cases=n_bad_cases,
                )    

    async def populate_unlabeled_data(self):
        for q_name, spec in self.feature_spaces.items():
            self.unlabeled_data[q_name]["data"].df = await self._materialize_features(
                    q_name=q_name,
                    tag="unlabeled_full",
                    data=self.unlabeled_data[q_name]["data"],
                    feature_specs=spec,
                    is_remote=False
                )

    
    def expand_coreset(self, debug: bool = False):
        for q_name, coreset in self.coresets.items():
            labeled_X = coreset["data"].exclude_fk_and_id()
            labeled_Y = coreset["labels"]
            unlabeled_X = self.unlabeled_data[q_name]["data"].exclude_fk_and_id()

            selected_X_idx, selected_Y = select_coreset(
                labeled_X=labeled_X,
                labeled_Y=labeled_Y,
                unlabeled_X=unlabeled_X,
                k_neighbors=self.config.get("k_neighbors", 5),
            )

            # Update the coreset with the selected samples.
            selected_X = \
                self.unlabeled_data[q_name]["data"].df.iloc[selected_X_idx].reset_index(drop=True)
            self.coresets[q_name]["data"].df = \
                pd.concat([self.coresets[q_name]["data"].df, selected_X], ignore_index=True)
            self.coresets[q_name]["labels"] = pd.concat([labeled_Y, selected_Y], ignore_index=True)

            logger.info(f"Expanded coreset for query {q_name}: added {len(selected_X)} samples. New coreset size: {len(self.coresets[q_name]['data'].df)}")

            if debug:
                ground_truth_Y = self.unlabeled_data[q_name]["labels"].iloc[selected_X_idx].reset_index(drop=True)
                eval_results = evaluate_classifier(selected_Y, ground_truth_Y)
                logger.info((
                    f"Debug evaluation of expanded coreset for query {q_name}: "
                    f"TP={eval_results['TP']}, FP={eval_results['FP']}, FN={eval_results['FN']}, "
                    f"Precision={eval_results['precision']:.4f}, Recall={eval_results['recall']:.4f}, F1={eval_results['f1']:.4f}."
                ))

    
    def generate_candidate_external_features(self):

        importance_sum = defaultdict(float)

        for coreset in self.coresets.values():
            X = encode_features(coreset["data"].exclude_fk_and_id())
            Y = coreset["labels"].astype(int)
            for feat, imp in compute_feature_importance(X, Y).itertuples(index=False):
                importance_sum[feat] += imp

        # Collect external features (excluding base schema)
        base_set = set(self.base_schema)
        external_feats = list({
            spec.target_col
            for space in self.feature_spaces.values()
            for spec in space
            if spec.target_col not in base_set
        })

        # Sort by importance and select top-k
        top_feats = sorted(external_feats, key=lambda f: importance_sum.get(f, 0), reverse=True)[:self.b_se]

        # Combine with base schema
        self.candidate_external_features = top_feats

        logger.info(f"Selected {len(top_feats)} external features: {top_feats}")

        return self.candidate_external_features


    def select_schema_and_rewrite_query(self, debug: bool = False):

        best_static_error = float('inf')
        best_schema = None
        best_rules = None        

        for i in range(len(self.candidate_external_features)):
            active_external_features = self.candidate_external_features[:i+1]

            for q_name, _ in self.queries.items():

                # TODO: Propagation labels
                self._label_propagation(q_name, active_external_features, debug=debug)
                
                # TODO: Estimate objective error score.

                # TODO: Estimate subjective error score.

                # Update the best schema and best rules.

                pass


    async def _refine_feature_space(self, q_name: str, sem_cq: SemCQ,
                                    max_iter: int = 3, f1_threshold: float = 0.01,
                                    n_bad_cases: int = 5):
        # Record the previous-round F1-score.
        prev_f1 = pred_and_eval(
            self.labeled_data[q_name]["data"].exclude_fk_and_id(), 
            self.labeled_data[q_name]["labels"])['f1']
        f1_score_trace = [prev_f1]

        for iteration in range(max_iter):
            logger.info(f"Refinement iteration {iteration + 1}/{max_iter} for {q_name}")

            # Evaluate current feature space
            df_for_clf = self.coresets[q_name]["data"].exclude_fk_and_id()
            labels = self.coresets[q_name]["labels"]
            feedback = pred_and_eval(df_for_clf, labels)

            # Check stopping criteria
            f1_improvement = feedback['f1'] - prev_f1
            prev_f1 = feedback['f1']
            f1_score_trace.append(prev_f1)
            logger.info(f"Iteration {iteration + 1}: F1 improvement = {f1_improvement:.4f}")
            if iteration > 0 and f1_improvement < f1_threshold:
                logger.info(f"F1 improvement ({f1_improvement:.4f}) below threshold ({f1_threshold}). Stopping refinement.")
                break

            # Prepare prompt for LLM
            # TODO: [WARN] Using this simplified impl, we should turn off the refinement if there are multiple semantic predicates in the CQ.
            sem_pred = sem_cq.Ps[0]  # Assume single semantic predicate for now
            prompt, bad_cases = self._prepare_refinement(
                sem_pred=sem_pred,
                current_features=self.feature_spaces[q_name],
                f1_improvement=f1_improvement,
                feedback=feedback,
                n_bad_cases=n_bad_cases,
                original_df=self.labeled_data[q_name]["data"].df,
            )

            # Call LLM for refinement suggestions
            llm_response = self.llm_client.invoke(
                modality=sem_pred.modality,
                is_remote=True,
                prompt=prompt,
                data_items=bad_cases,
                response_model=FeatureRefinementResponse,
            )

            # Apply refinements
            features_to_add = llm_response.to_add  # type: ignore
            features_to_remove = set(llm_response.to_remove)  # type: ignore
            if not features_to_add and not features_to_remove:
                logger.info(f"LLM suggested no changes. Stopping refinement.")
                break

            # Remove low-importance features
            if features_to_remove:
                self.feature_spaces[q_name] = [
                    spec for spec in self.feature_spaces[q_name]
                    if spec.target_col not in features_to_remove
                ]
                self.coresets[q_name]["data"].df.drop(columns=features_to_remove, inplace=True)
                logger.info(f"Removed {len(features_to_remove)} features: {features_to_remove}")

            # Add new features
            if features_to_add:
                # Check the feature budget before adding
                available_budget = self.b_fs - len(self.feature_spaces[q_name])
                if len(features_to_add) > available_budget:
                    features_to_add = features_to_add[:available_budget]
                    logger.info(f"Feature budget exceeded. Only adding {available_budget} features.")
                self.feature_spaces[q_name].extend(features_to_add)
                logger.info(f"Adding {len(features_to_add)} new features")

                # Materialize only the new features and merge (directly cover the previous version)
                new_feature_specs = features_to_add
                self.coresets[q_name]["data"].df = await self._materialize_features(
                    q_name=q_name,
                    tag="labeled_full",
                    data=self.coresets[q_name]["data"],
                    feature_specs=new_feature_specs,
                    reuse=False
                )
            else:
                logger.info(f"No new features to add in this iteration.")

            # Cache refined feature space (directly cover the previous refined feature space)
            ckpt_path = self.CKPT_path / f"{q_name}_feature_space.json"
            ckpt_data = {
                q_name: [spec.model_dump() for spec in self.feature_spaces[q_name]]
            }
            with open(ckpt_path, 'w') as f:
                json.dump(ckpt_data, f, indent=2)
                logger.info(f"Cached refined feature space to: {ckpt_path}")

        logger.info(f"Feature space refinement completed for {q_name}")
        logger.info(f"F1 score trace over iterations: {f1_score_trace}")


    def _prepare_refinement(self, sem_pred, current_features: list[PopulationSpec],
                            f1_improvement: float,
                            feedback: dict, n_bad_cases: int, 
                            original_df: pd.DataFrame) -> Tuple[str, list[str]]:
        """Build the refinement prompt with feedback from classifier evaluation."""

        # Format current features
        current_features_str = "\n".join([
            f"  - {spec.target_col} ({spec.feature_type}): {spec.prompt[:100]}..."
            for spec in current_features
        ])

        # Format feature importance
        feature_importance_str = "\n".join([
            f"  - {feat}: {imp:.4f}"
            for feat, imp in list(feedback['feature_importance'].items())[:10]  # Top 10
        ])

        # Format bad cases (most uncertain ones) with original source data
        bad_cases_df = feedback['bad_cases'].head(n_bad_cases)
        bad_cases = []
        for _, row in bad_cases_df.iterrows():
            original_idx = row['_original_index']
            bad_cases.append(original_df.loc[original_idx, sem_pred.field])

        return PROMPTS["REFINE_FEATURE_SPACE_PROMPT"].format(
            TASK_DESC=sem_pred.prompt,
            SOURCE_FIELD=sem_pred.field,
            MODALITY=sem_pred.modality,
            CURRENT_FEATURES=current_features_str,
            FEATURE_IMPORTANCE=feature_importance_str,
            F1_SCORE=feedback['f1'],
            F1_IMPROVEMENT=f1_improvement,
            FEATURE_BUDGET=self.b_fs,
        ), bad_cases


    def _build_ground_truth_labels(
            self, data: LdbData, q_name: str, 
            selected_columns: list[str], debug=False) -> pd.Series:
        ground_truth_df = pd.read_csv(f"{self.data_dir}/ground_truth/{q_name}.csv")
        ground_truth_df = ground_truth_df[selected_columns]
        ground_truth_set = set(tuple(row) for row in ground_truth_df.values)

        labels = data.df[selected_columns].apply(
            lambda row: tuple(row) in ground_truth_set,
            axis=1
        ).reset_index(drop=True)

        if debug:
            true_rows = data.df[labels][selected_columns]
            true_rows_set = set(tuple(row) for row in true_rows.values)
            assert true_rows_set == ground_truth_set, \
                f"[DebugErr] Fail to build ground truth labels for query {q_name}."
            logger.debug(f"Ground truth of {self.scenario}.{q_name}: {labels.sum()} positives / {len(labels)} samples. Oracle selectivity: {labels.sum() / len(labels):.4f}")

        return labels


    def _acquire_human_label(self, q_name: str, debug: bool = False):

        data = self.sigma_satisfied_data[q_name]["data"]
        labels = self.sigma_satisfied_data[q_name]["labels"]

        labeled_indices = data.df.sample(n=self.b_lab, random_state=self.random_seed).index
        remaining_indices = data.df.index.difference(labeled_indices)

        self.labeled_data[q_name] = {
            "data": LdbData(
                df=data.df.loc[labeled_indices].reset_index(drop=True), 
                config=data.config),
            "labels": labels.loc[labeled_indices].reset_index(drop=True),
        }
        self.unlabeled_data[q_name] = {
            "data": LdbData(
                df=data.df.loc[remaining_indices].reset_index(drop=True),
                config=data.config),
            "labels": labels.loc[remaining_indices].reset_index(drop=True),
        }

        if debug:
            logger.debug(f"Acquired labels for {self.scenario}.{q_name}: {self.labeled_data[q_name]['labels'].sum()} positive labels / {len(self.labeled_data[q_name]['data'].df)} labeled samples. Selectivity: {self.labeled_data[q_name]['labels'].sum() / len(self.labeled_data[q_name]['data'].df):.4f}")


    def _init_feature_space(self, q_name: str, sem_cq: SemCQ) -> list[PopulationSpec]:
        ckpt_path = self.CKPT_path / f"{q_name}_feature_space.json"

        # Load the cached feature space if it exists
        if ckpt_path.exists():
            logger.info(f"Loading cached initial feature space for query {q_name}...")
            with open(ckpt_path, 'r') as f:
                cached_data = json.load(f)
            return [PopulationSpec(**spec) for spec in cached_data[q_name]]

        # Generate initial feature space using LLM.
        feature_space = []
        for sem_pred in sem_cq.Ps:
            field = sem_pred.field
            sampled_data = \
                self.labeled_data[q_name]["data"].df[field].sample(n=10, random_state=self.random_seed).tolist()
            prompt = PROMPTS["GEN_FEAT_CANDIDATE_PROMPT"].format(
                MODALITY=sem_pred.modality,
                DESC=sem_pred.prompt,
                SAMPLE_DATA="\n".join(sampled_data),
                SOURCE_COL=field,
                FEATURE_BUDGET=self.b_fs,
            )

            llm_response = self.llm_client.invoke(
                modality=sem_pred.modality,
                is_remote=True,
                prompt=prompt,
                data_items=sampled_data,
                response_model=PopulationSpecs,
            )

            feature_space.extend(llm_response.value)  # type: ignore

        # Cache the generated feature space for future use.
        ckpt_data = {
            q_name: [spec.model_dump() for spec in feature_space]
        }
        with open(ckpt_path, 'w') as f:
            json.dump(ckpt_data, f, indent=2)
            logger.info(f"Cached initial feature space for query {q_name} to: {ckpt_path}")

        return feature_space


    async def _materialize_features(self, q_name: str, tag: str, data: LdbData, 
                                    feature_specs: list[PopulationSpec],
                                    is_remote: bool=True, 
                                    reuse: bool=True) -> pd.DataFrame:
        ckpt_path = self.CKPT_path / f"{q_name}_{tag}.csv"

        if ckpt_path.exists() and reuse:
            logger.info(f"Loading cached materialized features for query {q_name} with tag {tag} from: {ckpt_path}")
            return pd.read_csv(ckpt_path)

        df_cp = data.df.copy()
        for spec in feature_specs:
            df_cp[spec.target_col] = await data._sem_map(
                spec=spec,
                llm_client=self.llm_client,
                is_remote=is_remote
            )

        df_cp.to_csv(ckpt_path, index=False)
        logger.info(f"Cached materialized features for query {q_name} with tag {tag} to: {ckpt_path}")
        return df_cp


    def _label_propagation(self, q_name: str, active_external_features: list[str], debug: bool = False):
        train_X = self.coresets[q_name]["data"].select_active_features(active_external_features)
        train_Y = self.coresets[q_name]["labels"].astype(int)
        test_X = self.unlabeled_data[q_name]["data"].select_active_features(active_external_features)
        test_Y = self.unlabeled_data[q_name]["labels"].astype(int)

        # Encode features and train classifier
        train_X_proc = encode_features(train_X)
        test_X_proc = encode_features(test_X)
        clf = train_classifier(train_X_proc, train_Y)

        # Predict labels for unlabeled data
        pred_Y = pd.Series(clf.predict(test_X_proc), index=test_Y.index)

        # TODO: perform query translation.

        if debug:
            # Appendix with ground truth labels.
            visible_labels = self.labeled_data[q_name]["labels"].astype(int)
            pred_Y_complete = pd.concat([visible_labels, pred_Y], ignore_index=True)
            test_Y_complete = pd.concat([visible_labels, test_Y], ignore_index=True)

            # Evaluate the label propagation results with ground truth.
            eval_results = evaluate_classifier(pred_Y_complete, test_Y_complete)
            logger.info((
                f"Debug evaluation of label propagation for query {q_name} with {len(active_external_features)} external features {active_external_features}: "
                f"TP={eval_results['TP']}, FP={eval_results['FP']}, FN={eval_results['FN']}, "
                f"Precision={eval_results['precision']:.4f}, Recall={eval_results['recall']:.4f}, F1={eval_results['f1']:.4f}."
            ))

    def _objective_error_estimation(self):
        pass
    
    def _subjective_error_estimation(self):
        pass

