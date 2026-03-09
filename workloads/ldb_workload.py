import json
import pandas as pd
import logging
from pathlib import Path
from typing import Tuple, Any, cast
from collections import defaultdict
from data_structure import LdbData, SemCQ, PopulationSpec, FeatureRefinementResponse
from llm import LdbLLMClient, PROMPTS
from common import (
    select_coreset,
    compute_feature_importance,
    encode_features,
    evaluate_classifier,
)
from workloads.workload_utils import (
    build_contrastive_batch,
    build_feature_generation_prompt,
    build_ground_truth_labels,
    compute_objective_error,
    compute_subjective_error,
    report_usage_statistics,
    report_evaluation_trace,
    perform_label_propagation,
    pred_and_eval,
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

        self.usage_statistics = {
            item: {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "prompt_cost": 0.0,
                "completion_cost": 0.0,
                "total_cost": 0.0,
            } for item in [
                "feature_space_init",
                "feature_space_refine",
                "materialize_labeled_full",
                "materialize_unlabeled_full",
            ]
        }

    def apply_sigma(self, debug=False):
        for q_name, sem_cq in self.queries.items():

            data = self.ldb_data.sigma_retrieve(sem_cq.Sigma, reset_index=True)
            labels = build_ground_truth_labels(data.df, q_name, sem_cq.selected, self.data_dir, debug=debug)

            self.sigma_satisfied_data[q_name] = {
                "data": data,
                "labels": labels,
            }

    
    async def init_coresets(self, debug: bool = False):

        for q_name, sem_cq in self.queries.items():
            # Acquire labeled set.
            self._acquire_human_label(q_name=q_name, debug=debug)

            # Acquire initial feature space.
            self.feature_spaces[q_name] = await self._init_feature_space(q_name, sem_cq)

            # Store materialized data in coreset
            self.coresets[q_name] = {
                "data": LdbData(df=self.labeled_data[q_name]["data"].df, config=self.ldb_data.config),
                "labels": self.labeled_data[q_name]["labels"],
            }


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
                mode="emperical",
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


    def select_schema_and_rewrite_query(self, debug: bool = False) -> Tuple[dict, dict]:

        best_static_error = float('inf')
        best_statistics = {}   
        execution_trace = {}

        for i in range(len(self.candidate_external_features)):

            active_external_features = self.candidate_external_features[:i+1]
            accumulated_error = 0

            stats = [
                "rules", "features", "pred_eval", "trans_eval", "L_rew",
                "penalty_rew", "L_LOO", "penalty_LOO", "L_obj", "L_subj", "L_static"
            ]

            execution_results: dict[str, Any] = {stat: {} for stat in stats}
            execution_results["L_avg"] = float('inf')

            for q_name, _ in self.queries.items():
                # Propagation labels
                train_X = self.coresets[q_name]["data"].select_active_features(active_external_features)
                train_Y = self.coresets[q_name]["labels"].astype(int)
                test_X = self.unlabeled_data[q_name]["data"].select_active_features(active_external_features)
                test_Y = self.unlabeled_data[q_name]["labels"].astype(int)
                visible_labels = self.labeled_data[q_name]["labels"].astype(int)

                rules, pred_Y, trans_Y, pred_eval_results, trans_eval_results = \
                    perform_label_propagation(
                        train_X, train_Y, test_X, test_Y, visible_labels,
                        self.b_rew, debug
                    )

                # Estimate objective error score.
                L_rew, penalty_rew = compute_objective_error(
                    pred_Y=pred_Y, trans_Y=trans_Y, b_rew=self.b_rew,
                    schema_arity=len(active_external_features) + len(self.base_schema),
                    query_size=len(self.queries),
                    selected_data_size=len(pred_Y) + self.b_lab,
                    delta=self.config["delta"])
                L_obj = L_rew + penalty_rew

                # Estimate subjective error score.
                L_LOO, penalty_LOO = compute_subjective_error(
                    X=self.labeled_data[q_name]["data"].select_active_features(active_external_features),
                    Y=self.labeled_data[q_name]["labels"].astype(int),
                    query_size=len(self.queries),
                    data_size=len(self.labeled_data[q_name]["labels"]),
                    delta=self.config["delta"],
                    loo_step=self.config["loo_step"])
                L_subj = L_LOO + penalty_LOO

                L_static = L_obj + L_subj
                accumulated_error += L_static

                # Store metrics organized by type
                execution_results[q_name] = {
                    "rules": rules,
                    "features": active_external_features,
                    "pred_eval": pred_eval_results,
                    "trans_eval": trans_eval_results,
                    "L_rew": L_rew,
                    "penalty_rew": penalty_rew,
                    "L_LOO": L_LOO,
                    "penalty_LOO": penalty_LOO,
                    "L_obj": L_obj,
                    "L_subj": L_subj,
                    "L_static": L_static,
                }
                execution_results["rules"][q_name] = rules
                execution_results["features"][q_name] = active_external_features
                execution_results["pred_eval"][q_name] = pred_eval_results
                execution_results["trans_eval"][q_name] = trans_eval_results
                execution_results["L_rew"][q_name] = L_rew
                execution_results["penalty_rew"][q_name] = penalty_rew
                execution_results["L_LOO"][q_name] = L_LOO
                execution_results["penalty_LOO"][q_name] = penalty_LOO
                execution_results["L_obj"][q_name] = L_obj
                execution_results["L_subj"][q_name] = L_subj
                execution_results["L_static"][q_name] = L_static

                if debug:
                    logger.info(
                        f"Estimated for {q_name} with {i+1} external features: "
                        f"L_obj = {L_obj:.4f} (L_rew={L_rew:.4f}, penalty={penalty_rew:.4f}), "
                        f"L_subj = {L_subj:.4f} (L_LOO={L_LOO:.4f}, penalty={penalty_LOO:.4f})"
                    )

            average_error = accumulated_error / len(self.queries)
            execution_results["L_avg"] = average_error
            execution_trace[i] = execution_results

            if average_error < best_static_error:
                best_static_error = average_error
                best_statistics = execution_results

        return best_statistics, execution_trace


    def _acquire_human_label(self, q_name: str, debug: bool = False):

        data = self.sigma_satisfied_data[q_name]["data"]
        labels = self.sigma_satisfied_data[q_name]["labels"]

        labeled_indices = data.df.sample(n=self.b_lab, random_state=self.random_seed).index

        # Fallback: ensure minority class has at least minority_threshold samples
        num_pos_sampled = labels.loc[labeled_indices].sum()
        num_neg_sampled = len(labeled_indices) - num_pos_sampled
        minority_threshold = max(1, int(0.1 * self.b_lab))
        if min(num_pos_sampled, num_neg_sampled) < minority_threshold:
            # Need to resample with minority constraint
            pos_indices = labels[labels == True].index
            neg_indices = labels[labels == False].index

            # Allocate budget: minority class gets minority_threshold, majority gets the rest
            if num_pos_sampled < num_neg_sampled:
                pos_to_sample = min(minority_threshold, len(pos_indices))
                neg_to_sample = min(self.b_lab - pos_to_sample, len(neg_indices))
            else:
                neg_to_sample = min(minority_threshold, len(neg_indices))
                pos_to_sample = min(self.b_lab - neg_to_sample, len(pos_indices))

            # Check if we have enough total samples
            if pos_to_sample + neg_to_sample < self.b_lab:
                logger.warning(
                    f"Query {q_name}: Not enough samples to fill budget. "
                    f"Sampling {pos_to_sample} pos + {neg_to_sample} neg = {pos_to_sample + neg_to_sample} < {self.b_lab}"
                )

            # Resample with the constraint
            labeled_pos = pd.Series(pos_indices).sample(n=pos_to_sample, random_state=self.random_seed)
            labeled_neg = pd.Series(neg_indices).sample(n=neg_to_sample, random_state=self.random_seed)
            labeled_indices = pd.concat([labeled_pos, labeled_neg])

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
            logger.info((f"Acquired labels for {self.scenario}.{q_name}: "
                         f"{self.labeled_data[q_name]['labels'].sum()} positive labels / "
                         f"{len(self.labeled_data[q_name]['data'].df)} labeled samples. Selectivity: "
                         f"{self.labeled_data[q_name]['labels'].sum() / len(self.labeled_data[q_name]['data'].df):.4f}"))


    async def _init_feature_space(self, q_name: str, sem_cq: SemCQ) -> list[PopulationSpec]:
        ckpt_path = self.CKPT_path / f"{q_name}_feature_space.json"
        ckpt_usage_path = self.CKPT_path / f"{q_name}_usage_feature_space.json"
        ckpt_data_path = self.CKPT_path / f"{q_name}_labeled_full.csv"

        # Load the cached feature space and cached usage if it exists
        if ckpt_path.exists():
            assert ckpt_usage_path.exists(), \
                f"[Error] Checkpoint for feature space exists but usage statistics is missing for query {q_name}."
            assert ckpt_data_path.exists(), \
                f"[Error] Checkpoint for feature space exists but materialized labeled data is missing for query {q_name}."
            logger.info(f"Loading cached initial feature space for query {q_name}...")
            with open(ckpt_path, 'r') as f:
                cached_data = json.load(f)
            logger.info(f"Loading cached usage statistics for initial feature space for query {q_name}...")
            with open(ckpt_usage_path, 'r') as f:
                cached_usage = json.load(f)
                self._update_statistics("feature_space_init", cached_usage)
            with open(ckpt_data_path, 'r') as f:
                cached_df = pd.read_csv(f)
                self.labeled_data[q_name]["data"] = LdbData(df=cached_df, config=self.ldb_data.config)
            return [PopulationSpec(**spec) for spec in cached_data[q_name]]

        # Initialize feature space
        feature_space = []

        # Process each semantic predicate
        for sem_pred in sem_cq.Ps:
            field = sem_pred.field

            # Divide pos / neg labeled data into batches
            labels = self.labeled_data[q_name]["labels"]
            pos_indices = labels[labels == True].index.tolist()
            neg_indices = labels[labels == False].index.tolist()

            # Calculate batch size N = min(5, num_pos, num_neg)
            N = min(5, len(pos_indices), len(neg_indices))
            assert N > 0, f"No positive or negative samples for query {q_name}, field {field}"

            # Create batches
            num_pos_batches = (len(pos_indices) + N - 1) // N  # Ceiling division
            num_neg_batches = (len(neg_indices) + N - 1) // N
            num_batches = max(num_pos_batches, num_neg_batches)

            logger.info(f"Initializing feature space for {q_name}.{field}: "
                       f"{len(pos_indices)} pos, {len(neg_indices)} neg, "
                       f"batch_size={N}, num_batches={num_batches}")

            prev_f1 = None
            previous_feedback = None
            iteration = 0

            # Iterate through batches (max 10 iterations or until all batches used)
            record = set()
            max_iter_num = 1
            max_iterations = min(max_iter_num, num_batches)
            for batch_idx in range(max_iterations):
                iteration = batch_idx
                pos_idx = iteration % num_batches
                neg_idx = iteration % num_batches
                if (pos_idx, neg_idx) in record:
                    logger.info(f"Batch combination already processed. Ending iterations.")
                    break
                record.add((pos_idx, neg_idx))

                # Get pos/neg batch for this iteration
                pos_start = pos_idx * N
                pos_end = min(pos_start + N, len(pos_indices))
                neg_start = neg_idx * N
                neg_end = min(neg_start + N, len(neg_indices))

                # Get current batch indices and data
                current_pos_indices = pos_indices[pos_start:pos_end] 
                current_neg_indices = neg_indices[neg_start:neg_end]
                pos_batch_data = self.labeled_data[q_name]["data"].df.loc[current_pos_indices, field].tolist() 
                neg_batch_data = self.labeled_data[q_name]["data"].df.loc[current_neg_indices, field].tolist()

                # Build data items and metadata
                data_items, metadata = build_contrastive_batch(
                    sem_pred, pos_batch_data, neg_batch_data,
                    current_pos_indices, current_neg_indices, previous_feedback,
                    self.labeled_data[q_name]["data"].df
                )

                # Build prompt
                prompt = build_feature_generation_prompt(
                    sem_pred, feature_space, previous_feedback, iteration,
                    self.b_fs, PROMPTS["GEN_FEAT_CANDIDATE_PROMPT"]
                )

                # Call LLM
                llm_response = cast(FeatureRefinementResponse, self.llm_client.invoke(
                    modality=sem_pred.modality,
                    is_remote=True,
                    prompt=prompt,
                    data_items=data_items,
                    data_items_metadata=metadata,
                    response_model=FeatureRefinementResponse,
                ))

                # Apply feature changes
                features_to_remove = [f for f in llm_response.to_remove if f not in self.base_schema]
                if features_to_remove:
                    feature_space[:] = [spec for spec in feature_space if spec.target_col not in features_to_remove]
                    self.labeled_data[q_name]["data"].df.drop(columns=features_to_remove, inplace=True)
                
                if llm_response.to_add:
                    feature_space.extend(llm_response.to_add)
                    self.labeled_data[q_name]["data"].df = await self._materialize_features(
                        q_name=q_name,
                        tag=f"labeled_full",
                        data=self.labeled_data[q_name]["data"],
                        feature_specs=llm_response.to_add,
                        reuse=False
                    )

                # Evaluate and enforce budget
                feedback = pred_and_eval(
                    self.labeled_data[q_name]["data"].exclude_fk_and_id(),
                    self.labeled_data[q_name]["labels"]
                )
                if len(feature_space) > self.b_fs:
                    feature_can_be_removed = [k for k, v in feedback["feature_importance"].items() if k not in self.base_schema]
                    feature_to_be_removed = feature_can_be_removed[-(len(feature_space) - self.b_fs):]
                    feature_space[:] = [spec for spec in feature_space if spec.target_col not in feature_to_be_removed]
                    self.labeled_data[q_name]["data"].df.drop(columns=feature_to_be_removed, inplace=True)
                    logger.info(f"Removed {len(feature_to_be_removed)} features to enforce budget.")

                logger.info(f"Iteration {iteration}: F1={feedback['f1']:.4f}, "
                           f"Added={len(llm_response.to_add)}, Removed={len(llm_response.to_remove)}")

                # Check stopping criteria: F1 drops > 0.05
                if prev_f1 is not None:
                    f1_drop = prev_f1 - feedback['f1']
                    if f1_drop > 0.05:
                        logger.info(f"F1 dropped by {f1_drop:.4f} > 0.05. Stopping iteration.")
                        break

                prev_f1 = feedback['f1']
                previous_feedback = feedback

        # Record the usage statistics for this phase.
        self._update_statistics("feature_space_init", self.llm_client.get_usage_statistics())
        self.llm_client.reset_usage_statistics()

        # Cache the generated feature space and the usage statistics for future use.
        ckpt_data = {
            q_name: [spec.model_dump() for spec in feature_space]
        }
        with open(ckpt_path, 'w') as f:
            json.dump(ckpt_data, f, indent=2)
            logger.info(f"Cached initial feature space for query {q_name} to: {ckpt_path}")
        with open(ckpt_usage_path, 'w') as f:
            json.dump(self.usage_statistics["feature_space_init"], f, indent=2)
            logger.info(f"Cached usage statistics for initial feature space for query {q_name} to: {ckpt_usage_path}")

        logger.info(f"Initialized feature space for {q_name}: {len(feature_space)} features")

        return feature_space


    async def _materialize_features(self, q_name: str, tag: str, data: LdbData, 
                                    feature_specs: list[PopulationSpec],
                                    is_remote: bool=True, 
                                    reuse: bool=True) -> pd.DataFrame:
        ckpt_path = self.CKPT_path / f"{q_name}_{tag}.csv"
        ckpt_usage_path = self.CKPT_path / f"{q_name}_usage_{tag}.json"

        if ckpt_path.exists() and reuse:
            assert ckpt_usage_path.exists(), \
                f"[Error] Checkpoint for materialized features exists but usage statistics is missing for query {q_name} with tag {tag}."
            materialized_df = pd.read_csv(ckpt_path)
            logger.info(f"Loading cached materialized features for query {q_name} with tag {tag} from: {ckpt_path}")
            with open(ckpt_usage_path, "r") as f:
                cached_usage = json.load(f)
                self._update_statistics(f"materialize_{tag}", cached_usage)
            logger.info(f"Loaded cached usage statistics for materializing features for query {q_name} with tag {tag} from: {ckpt_usage_path}")
            return materialized_df

        df_cp = data.df.copy()
        for spec in feature_specs:
            df_cp[spec.target_col] = await data._sem_map(
                spec=spec,
                llm_client=self.llm_client,
                is_remote=is_remote
            )
        # Record the usage statistics.
        self._update_statistics(f"materialize_{tag}", self.llm_client.get_usage_statistics())
        self.llm_client.reset_usage_statistics()

        # Store the usage statistics to the cache.
        with open(ckpt_usage_path, 'w') as f:
            json.dump(self.usage_statistics[f"materialize_{tag}"], f, indent=2)
            logger.info(f"Cached usage statistics for materializing features for query {q_name} with tag {tag} to: {ckpt_usage_path}")

        df_cp.to_csv(ckpt_path, index=False)
        logger.info(f"Cached materialized features for query {q_name} with tag {tag} to: {ckpt_path}")

        return df_cp


    def _update_statistics(self, key, value):
        assert key in self.usage_statistics, f"Invalid statistics key: {key}"
        for k, v in value.items():
            self.usage_statistics[key][k] += v


    def _report_usage_statistics(self):
        report_usage_statistics(self.usage_statistics)
        

    def _report_evaluation_trace(self, execution_trace: dict):
        report_evaluation_trace(execution_trace)

