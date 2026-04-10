import json
import pandas as pd
import logging
from pathlib import Path
from typing import Tuple, Any, cast
from collections import defaultdict
import pandas.api.types as ptypes
from time import time
from data_structure import LdbData, SemCQ, PopulationSpec, Predicate, LdbDataManager
from data_structure.llm_resp_templates import PredicateResponses, PredicateResponse, RelevantFieldsResponse
from llm import LdbLLMClient, PROMPTS
from common import (
    select_coreset,
    compute_feature_importance,
    encode_features,
    evaluate_classifier,
    clf_to_rules,
    apply_rules,
)
from workloads.workload_utils import (
    compute_objective_error,
    compute_subjective_error,
    report_usage_statistics,
    report_evaluation_trace,
    perform_label_propagation,
    compute_inc_error_certificate,
)
from workloads.feature_utils import (
    initialize_feature_space,
    enforce_feature_budget,
)

logger = logging.getLogger(__name__)

class LdbWorkload:
    def __init__(self, data_dir: str, scenario: str, queries: dict[str, SemCQ], config: dict) -> None:
        
        self.llm_client = LdbLLMClient()
        self.dynamic_setting = config["dynamic_setting"]

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
        self.eta = config["eta"]
        self.delta = config["delta"]
        self.enable_hitl = config["enable_hitl"]
        self.enable_conf_pred = config["enable_conf_pred"]
        self.enable_conf_struct = config["enable_conf_struct"]
        self.enable_enrich = config["enable_enrich"]
        self.enable_rewrite = config["enable_rewrite"]

        self.data_manager = LdbDataManager(
            data_dir=self.data_dir,
            scenario=self.scenario,
            queries=self.queries,
            llm_client=self.llm_client,
            dynamic_steps=self.dynamic_setting
        )

        self.data_stream = []
        self.sigma_satisfied_data = [{} for _ in range(len(self.dynamic_setting))]
        self.unlabeled_data = [{} for _ in range(len(self.dynamic_setting))]
        self.labeled_data = {}
        self.coresets = {}
        self.rules = {}

        self.feature_spaces = {}
        
        self.base_schema = self.ldb_data.df.columns.tolist()
        self.candidate_external_features = []

        self.CKPT_path = Path(__file__).parent.parent / ".data_ckpt" / self.scenario \
            / "_".join(str(step) for step in self.dynamic_setting)
        for q_name in self.queries.keys():
            (self.CKPT_path / q_name).mkdir(parents=True, exist_ok=True)

        self.usage_statistics = [{
            item: {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "prompt_cost": 0.0,
                "completion_cost": 0.0,
                "total_cost": 0.0,
            } for item in [
                "sigma_augmentation",
                "feature_space_init",
                "feature_space_refine",
                "materialize_labeled_full",
                "materialize_unlabeled_full",
                "materialize_unlabeled_inc",
            ]
        } for _ in range(len(self.dynamic_setting))]


    def inject_exp_setting(self, exp_group: str, exp_patch: dict):
        assert exp_group != "", "Exp group cannot be empty when injecting exp setting."
        assert exp_patch is not None, "Exp patch cannot be None when injecting exp setting."
        for k, v in exp_patch.items():
            assert k in self.config, f"Invalid config key in exp patch: {k}"
            self.config[k] = v

            logger.info(f"Exp patch applied: {k}={v}")

        # Flush the config setting.
        self.random_seed = self.config["random_seed"]
        self.b_lab = self.config["b_lab"]
        self.b_se = self.config["b_se"]
        self.b_rew = self.config["b_rew"]
        self.b_fs = self.config["b_fs"]
        self.eta = self.config["eta"]
        self.delta = self.config["delta"]
        self.enable_hitl = self.config["enable_hitl"]
        self.enable_conf_pred = self.config["enable_conf_pred"]
        self.enable_conf_struct = self.config["enable_conf_struct"]
        self.enable_enrich = self.config["enable_enrich"]
        self.enable_rewrite = self.config["enable_rewrite"]

        exp_term = "_".join([str(v[0])+"="+str(v[1]) for v in list(zip(exp_patch.keys(), exp_patch.values()))])
        self.CKPT_path = Path(__file__).parent.parent / ".data_ckpt" / exp_group / self.scenario / exp_term \
            / "_".join(str(step) for step in self.dynamic_setting)
        for q_name in self.queries.keys():
            (self.CKPT_path / q_name).mkdir(parents=True, exist_ok=True)

        # Update the ckpt path of data manager.
        self.data_manager.CKPT_path = self.CKPT_path


    def refine_sigma_satisfied_data(self):
        for q_name, sem_cq in self.queries.items():
            logger.info(f"Augmenting Sigma for query {q_name}...")

            df_clean = self.data_manager.sigma_satisfied_data[0][q_name]["ldb_data"].exclude_fk_and_id()
            schema_info = self._collect_table_schema(df_clean)
            query_desc = self._build_query_description(sem_cq)


            cache_path = self.CKPT_path / q_name / "prefilter_ucq.json"
            if cache_path.exists():
                with open(cache_path, 'r') as f:
                    cached_results = json.load(f)
            else:
                # ========== Step 1: Identify Query-Relevant Fields ==========
                logger.info(f"Step 1: Identifying query-relevant fields for query {q_name}...")
                identify_fields_prompt = PROMPTS["IDENTIFY_RELEVANT_FIELDS_PROMPT"].format(
                    query_desc=query_desc,
                    schema_info=schema_info
                )

                fields_response = cast(RelevantFieldsResponse, self.llm_client.invoke(
                    is_remote=True,
                    modality="Text",
                    prompt=identify_fields_prompt,
                    response_model=RelevantFieldsResponse,
                ))
                self._update_statistics("sigma_augmentation", self.llm_client.get_usage_statistics())
                self.llm_client.reset_usage_statistics()

                relevant_fields_by_category = fields_response.value

                # Enforce the validity of the response
                for category, fields in relevant_fields_by_category.items():
                    relevant_fields_by_category[category] = [f for f in fields if f in df_clean.columns]

                # Log categorized fields
                total_fields = sum(len(fields) for fields in relevant_fields_by_category.values())
                logger.info(f"Identified {total_fields} relevant fields across {len(relevant_fields_by_category)} category/categories:")
                for category, fields in relevant_fields_by_category.items():
                    if fields:
                        logger.info(f"  - {category}: {fields}")

                # ========== Step 2: Generate UCQ with High Confidence ==========
                logger.info(f"Step 2: Generating UCQ for query {q_name}...")

                generate_ucq_prompt = PROMPTS["GENERATE_UCQ_PROMPT"].format(
                    query_desc=query_desc,
                    relevant_fields=self._format_relevant_fields(relevant_fields_by_category),
                    schema_info=schema_info
                )

                ucq_response = cast(PredicateResponses, self.llm_client.invoke(
                    is_remote=True,
                    modality="Text",
                    prompt=generate_ucq_prompt,
                    response_model=PredicateResponses,
                ))
                self._update_statistics("sigma_augmentation", self.llm_client.get_usage_statistics())
                self.llm_client.reset_usage_statistics()

                # Save to cache for reproducibility
                cached_results = {
                    "field_resp": relevant_fields_by_category,
                    "ucq_resp": ucq_response.model_dump(),
                }
                with open(cache_path, 'w') as f:
                    json.dump(cached_results, f, indent=2)

            new_ucq = self._expand_ucq([
                [PredicateResponse(**pred) for pred in group] \
                    for group in cached_results["ucq_resp"]["value"]])
            if not new_ucq:
                logger.info(f"No UCQ predicates suggested for query {q_name}")


            # Apply the new UCQ to narrow down the data
            self.data_manager.refine_sigma_satisfied_data(
                q_name=q_name, ucq=new_ucq
            )


    async def construct_feature_space(self, debug: bool = False):

        await self.data_manager.acquire_annotation_and_init_coreset(
            b_lab=self.b_lab,
            seed=self.random_seed,
        )

        for q_name, sem_cq in self.queries.items():
            # Acquire initial feature space.
            ckpt_path = self.CKPT_path / q_name / "feature_space.json"
            ckpt_usage_path = self.CKPT_path / q_name / "usage_feature_space.json"

            if ckpt_path.exists() and ckpt_usage_path.exists():
                with open(ckpt_path, 'r') as f:
                    feature_space = json.load(f)
                with open(ckpt_usage_path, 'r') as f:
                    usage_statistics = json.load(f)
                logger.info(f"Loaded feature space from checkpoint for query {q_name}.")
                self.data_manager.enriched_features[q_name] = [PopulationSpec(**spec) for spec in feature_space]
                self._update_statistics("feature_space_init", usage_statistics)
                continue

            feature_space, usage_statistics = await initialize_feature_space(
                feature_budget=self.b_se * 2,
                data_manager=self.data_manager,
                q_name=q_name,
                sem_cq=sem_cq,
                llm_client=self.llm_client,
            )
            self.data_manager.enriched_features[q_name] = feature_space
            self._update_statistics("feature_space_init", usage_statistics)
            self.llm_client.reset_usage_statistics()
            with open(ckpt_usage_path, 'w') as f:
                json.dump(usage_statistics, f, indent=2)        

        # Enforce the feature budget for batched query.
        if len(self.queries) > 1:
            new_fs, stat = await enforce_feature_budget(
                pop_specs=self.data_manager.enriched_features,
                sem_preds=self.queries,
                feature_budget=self.b_se,
                llm_client=self.llm_client,
            )
            self.data_manager.enriched_features = new_fs
            self._update_statistics("feature_space_init", stat)

        for q_name, _ in self.queries.items():
            feature_space = self.data_manager.enriched_features[q_name]
            with open(ckpt_path, 'w') as f:
                json.dump([spec.model_dump() for spec in feature_space], f, indent=2)
            logger.info(f"Saved feature space and usage statistics to checkpoint for query {q_name}.")


    async def sync_with_enriched_features(self, tag: str = ""):
        for q_name in self.queries.keys():
            stat_coreset = await self.data_manager\
                .sync_coreset_features(q_name, tag=tag, enable_cache=True)
            stat_sigma = await self.data_manager.\
                sync_sigma_satisfied_data_features(q_name, stream_idx=0, tag=tag, enable_cache=True)

            self._update_statistics("materialize_labeled_full", stat_coreset)
            self._update_statistics("materialize_unlabeled_full", stat_sigma)
            self.llm_client.reset_usage_statistics()

    
    def expand_coresets(self, inc_round: int = 0, debug: bool = False):
        for q_name in self.data_manager.coresets.keys():
            labeled_X = self.data_manager.coresets[q_name]["ldb_data"].exclude_fk_and_id()
            labeled_Y = self.data_manager.coresets[q_name]["labels"]
            unlabeled_X = self.data_manager.sigma_satisfied_data[inc_round][q_name]["ldb_data"].exclude_fk_and_id()

            mode = "empirical" if inc_round == 0 else "inc"
            selected_X_idx, selected_Y, new_lb, new_ub = select_coreset(
                labeled_X=labeled_X,
                labeled_Y=labeled_Y,
                unlabeled_X=unlabeled_X,
                k_neighbors=self.config.get("k_neighbors", 5),
                mode=mode,
                lb=self.data_manager.coresets[q_name]["lb"],
                ub=self.data_manager.coresets[q_name]["ub"],
                enable_conf_pred=self.enable_conf_pred,
                enable_conf_struct=self.enable_conf_struct,
            )
            self.data_manager.coresets[q_name]["lb"] = new_lb
            self.data_manager.coresets[q_name]["ub"] = new_ub

            # Update the coreset with the selected samples.
            selected_X = \
                self.data_manager.sigma_satisfied_data[inc_round][q_name]["ldb_data"]\
                    .df.iloc[selected_X_idx].reset_index(drop=True)
            self.data_manager.coresets[q_name]["ldb_data"].df = \
                pd.concat([self.data_manager.coresets[q_name]["ldb_data"].df, selected_X], ignore_index=True)
            self.data_manager.coresets[q_name]["labels"] = pd.concat([labeled_Y, selected_Y], ignore_index=True)

            
            logger.info((
                f"Inc-Round {inc_round}: Expanded coreset for query {q_name}: "
                f"added {len(selected_X)} samples. "
                f"New coreset size: {len(self.data_manager.coresets[q_name]['ldb_data'].df)}"))

            if debug:
                ground_truth_Y = \
                    self.data_manager.sigma_satisfied_data[inc_round][q_name]["labels"]\
                        .df.iloc[selected_X_idx].reset_index(drop=True)
                eval_results = evaluate_classifier(Y_pred=selected_Y, Y_true=ground_truth_Y)
                logger.info((
                    f"Debug evaluation of expanded coreset for query {q_name}: "
                    f"TP={eval_results['TP']}, FP={eval_results['FP']}, FN={eval_results['FN']}, "
                    f"Precision={eval_results['precision']:.4f}, "
                    f"Recall={eval_results['recall']:.4f}, F1={eval_results['f1']:.4f}."
                ))    


    async def rank_and_trim_feature_space(self, reuse: bool = False):

        ckpt_path = self.CKPT_path / "ranked_feature_space.json"

        top_feats = []
        if ckpt_path.exists() and reuse:
            with open(ckpt_path, 'r') as f:
                ranked_features = json.load(f)
            top_feats = ranked_features[:self.b_se]
            logger.info(f"Loaded ranked feature space from checkpoint.")
        else:
            importance_sum = defaultdict(float)
            for coreset in self.data_manager.coresets.values():
                X = encode_features(coreset["ldb_data"].exclude_fk_and_id())
                Y = coreset["labels"].astype(int)
                for feat, imp in compute_feature_importance(X, Y).itertuples(index=False):
                    importance_sum[feat] += imp
            # Collect external features (excluding base schema)
            external_feats = list({
                spec.target_col
                for space in self.data_manager.enriched_features.values()
                for spec in space
            })
            # Sort by importance and select top-k
            ranked_feats = sorted(external_feats, key=lambda f: importance_sum.get(f, 0), reverse=True)
            top_feats = ranked_feats[:self.b_se]
            logger.info(f"Selected {len(top_feats)} external features: {top_feats}")
            with open(ckpt_path, 'w') as f:
                json.dump(ranked_feats, f, indent=2)

        # Trim the feature space to only include the candidate external features.
        self.data_manager.trimmed_feature_names = top_feats
        for q_name in self.data_manager.enriched_features.keys():
            self.data_manager.enriched_features[q_name] = [
                spec for spec in self.data_manager.enriched_features[q_name] if spec.target_col in top_feats
            ]

        # sync the trimmed feature space.
        for q_name in self.data_manager.coresets.keys():
            _ = await self.data_manager.\
                sync_coreset_features(q_name, tag="trimmed", enable_cache=True)

        for q_name in self.data_manager.sigma_satisfied_data[0].keys():
            _ = await self.data_manager.\
                sync_sigma_satisfied_data_features(q_name, stream_idx=0, tag="trimmed", enable_cache=True)

        # Patch: If the generated trimmed feature space is inconsistent with the cached features, 
        # flush the current trimmed feature space.
        for q_name in self.data_manager.coresets.keys():
            ldb_data = self.data_manager.coresets[q_name]["ldb_data"]
            features = ldb_data.df.columns.tolist()
            original_schema = ldb_data.base_features + ldb_data.id_features + ldb_data.foreign_keys
            external_features = [f for f in features if f not in original_schema]
            self.data_manager.trimmed_feature_names = external_features
            self.data_manager.enriched_features[q_name] = [
                spec for spec in self.data_manager.enriched_features[q_name] \
                    if spec.target_col in external_features
            ]
    

    def rewrite_and_execute_query(self, debug: bool = False) -> Tuple[dict, dict]:

        best_static_error = float('inf')
        best_statistics = {}   
        execution_trace = {}
        rules_trace = {}

        assert len(self.data_manager.trimmed_feature_names) > 0, "No available features to be selected."

        for i in range(len(self.data_manager.trimmed_feature_names)):

            accumulated_error = 0

            stats = [
                "rules", "features", "pred_eval", "trans_eval", "L_rew",
                "penalty_rew", "L_LOO", "penalty_LOO", "L_obj", "L_subj", "L_static"
            ]

            execution_results: dict[str, Any] = {stat: {} for stat in stats}
            execution_results["L_avg"] = float('inf')

            for q_name, _ in self.queries.items():
                # Propagate labels
                active_external_features = [
                    spec.target_col for spec in self.data_manager.enriched_features[q_name] 
                    if spec.target_col in self.data_manager.trimmed_feature_names[:i+1]
                ]
                train_X = self.data_manager.coresets[q_name]["ldb_data"].select_active_features(active_external_features)
                train_Y = self.data_manager.coresets[q_name]["labels"].astype(int)
                test_X = self.data_manager.sigma_satisfied_data[0][q_name]["ldb_data"].select_active_features(active_external_features)
                test_Y = self.data_manager.sigma_satisfied_data[0][q_name]["labels"].astype(int)

                clf, pred_Y_li = perform_label_propagation(train_X, train_Y, [test_X], [test_Y])
                self.data_manager.sigma_satisfied_data[0][q_name]["propagated_labels"] = pred_Y_li[0]

                # Translated the query.
                if q_name not in rules_trace.keys():
                    rules_trace[q_name] = []
                rules = clf_to_rules(
                    clf, feature_names=train_X.columns.tolist(), 
                    disjunction_budget=self.b_rew, 
                    X_train=encode_features(train_X).to_numpy(), 
                    y_train=train_Y.to_numpy(), debug=debug)
                rules_trace[q_name].append(rules)

                # Apply the rules.
                trans_Y = apply_rules(rules, encode_features(test_X))

                # Evaluate the predications and translations.
                pred_eval_results = self.data_manager.eval_query_quality(
                    q_name=q_name, 
                    selected_cols=self.queries[q_name].selected,
                    stream_idx=0,
                    pred_labels=pred_Y_li,
                )
                trans_eval_results = self.data_manager.eval_query_quality(
                    q_name=q_name, 
                    selected_cols=self.queries[q_name].selected,
                    stream_idx=0,
                    pred_labels=[trans_Y]
                )

                # Estimate objective error score.
                observed_size = self.data_manager.coresets[q_name]["observed_size"]
                L_rew, penalty_rew = compute_objective_error(
                    pred_Y=pred_Y_li[0], trans_Y=trans_Y, b_rew=self.b_rew,
                    schema_arity=len(train_X.columns.tolist()),
                    query_size=len(self.queries),
                    selected_data_size=len(pred_Y_li[0]) + self.b_lab,  # TODO: the b_lab may be adjusted.
                    delta=self.delta)
                L_obj = L_rew + penalty_rew

                # Estimate subjective error score.
                L_LOO, penalty_LOO = compute_subjective_error(
                    X=self.data_manager.sigma_satisfied_data[0][q_name]["ldb_data"]\
                        .select_active_features(active_external_features).iloc[:observed_size],
                    Y=self.data_manager.sigma_satisfied_data[0][q_name]["labels"].astype(int)[:observed_size],
                    query_size=len(self.queries),
                    data_size=observed_size,
                    delta=self.delta,
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

        # TODO: temporarily set the "longest" rule as the best rule.
        for q_name, rules in rules_trace.items():
            trimmed_feature_names = self.data_manager.trimmed_feature_names
            active_feature_names = [
                spec.target_col for spec in self.data_manager.enriched_features[q_name] 
                if spec.target_col in trimmed_feature_names
            ]
            self.rules[q_name] = {
                "active_external_features": active_feature_names,
                "ucq": rules[-1]
            }

        return best_statistics, execution_trace


    def rewrite_and_execute_query_noEnr(self, debug: bool = False):
        rules_trace = {}

        for q_name, _ in self.queries.items():
            # Propagate labels
            active_external_features = []
            train_X = self.data_manager.coresets[q_name]["ldb_data"].select_active_features(active_external_features)
            train_Y = self.data_manager.coresets[q_name]["labels"].astype(int)
            test_X = self.data_manager.sigma_satisfied_data[0][q_name]["ldb_data"].select_active_features(active_external_features)
            test_Y = self.data_manager.sigma_satisfied_data[0][q_name]["labels"].astype(int)

            clf, pred_Y_li = perform_label_propagation(train_X, train_Y, [test_X], [test_Y])
            self.data_manager.sigma_satisfied_data[0][q_name]["propagated_labels"] = pred_Y_li[0]

            # Translated the query.
            if q_name not in rules_trace.keys():
                rules_trace[q_name] = []
            rules = clf_to_rules(
                clf, feature_names=train_X.columns.tolist(), 
                disjunction_budget=self.b_rew, 
                X_train=encode_features(train_X).to_numpy(), 
                y_train=train_Y.to_numpy(), debug=debug)

            # Apply the rules.
            trans_Y = apply_rules(rules, encode_features(test_X))

            # Evaluate translations.
            trans_eval_results = self.data_manager.eval_query_quality(
                q_name=q_name, 
                selected_cols=self.queries[q_name].selected,
                stream_idx=0,
                pred_labels=[trans_Y]
            )

            print("=" * 30)
            print(trans_eval_results)
            print("=" * 30)

           

    async def rewrite_and_execute_query_noRew(self, debug: bool = False):

        from data_structure.llm_resp_templates import BooleanFeatureResponse

        assert len(self.data_manager.trimmed_feature_names) > 0, "No available features to be selected."

        for q_name, _ in self.queries.items():
            # Propagate labels
            active_external_features = [
                spec.target_col for spec in self.data_manager.enriched_features[q_name] 
                if spec.target_col in self.data_manager.trimmed_feature_names
            ]
            test_X = self.data_manager.sigma_satisfied_data[0][q_name]["ldb_data"].select_active_features(active_external_features)
            test_Y = self.data_manager.sigma_satisfied_data[0][q_name]["labels"].astype(int)

            # Construct the parallel invoking task.
            sem_query = self.queries[q_name]

            prompt = "Determine if the provided data item satisfies the following conditions: " + \
                " AND ".join([pred.succ_cond for pred in sem_query.Ps])

            data_items = [
                [", ".join([f"{col}: {val}" for col, val in row.items()])]
                for _, row in test_X.iterrows()
            ]

            resp = await self.llm_client.invoke_parallel(
                is_remote=False,
                modality="Text",
                prompt=prompt,
                data_items=data_items,
                response_model=BooleanFeatureResponse,
            )
            pred_Y = pd.Series([r.value for r in resp])

            pred_eval_results = self.data_manager.eval_query_quality(
                q_name=q_name,
                selected_cols=sem_query.selected,
                stream_idx=0,
                pred_labels=[pred_Y]
            )
            print("=" * 30)
            print(pred_eval_results)
            print("=" * 30)



    async def incremental_processing(self, debug: bool = False) -> Tuple[bool, list]:

        eval_results = []
        rerun = False

        for inc_round in range(1, len(self.dynamic_setting)):
        
            start_time = time()
            eval_results_single_step = {q_name: {
                "error_certificate": 0.0,
                "pred_eval": {},
                "trans_eval": {},
            } for q_name in self.queries.keys()}

            for q_name, _ in self.queries.items():
                if self.data_manager.sigma_satisfied_data[inc_round][q_name]["ldb_data"].df.empty:
                    logger.info(f"Inc-Round {inc_round}: No sigma-satisfied data available.")
                    break

                active_external_features = self.rules[q_name]["active_external_features"]

                # Materialize the unlabeled data.
                await self.data_manager.\
                    sync_sigma_satisfied_data_features(q_name=q_name, tag="init", stream_idx=inc_round, enable_cache=True)

                # Expand the coreset.
                self.expand_coresets(inc_round=inc_round, debug=debug)

                # Propagate labels
                train_X = self.data_manager.coresets[q_name]["ldb_data"].select_active_features(active_external_features)
                train_Y = self.data_manager.coresets[q_name]["labels"].astype(int)
                test_X_li = [
                    self.data_manager.sigma_satisfied_data[i][q_name]["ldb_data"].select_active_features(active_external_features)
                    if not self.data_manager.sigma_satisfied_data[i][q_name]["ldb_data"].df.empty
                    else pd.DataFrame(columns=train_X.columns)
                    for i in range(inc_round + 1)
                ]

                test_Y_li = [
                    self.data_manager.sigma_satisfied_data[i][q_name]["labels"].astype(int)
                    if not self.data_manager.sigma_satisfied_data[i][q_name]["ldb_data"].df.empty
                    else pd.Series(dtype=int)
                    for i in range(inc_round + 1)
                ]
                _, pred_Y_li = perform_label_propagation(train_X, train_Y, test_X_li, test_Y_li)

                # Compute the error certificate.
                self.data_manager.sigma_satisfied_data[inc_round][q_name]["propagated_labels"] = pred_Y_li[-1]
                observed_data_size = self.data_manager.coresets[q_name]["observed_size"]
                err_certificate = compute_inc_error_certificate(
                    label_Y=self.data_manager.coresets[q_name]["labels"][:observed_data_size].astype(int),
                    prev_prop_Y_li=[
                        self.data_manager.sigma_satisfied_data[i][q_name]["propagated_labels"]
                        if not self.data_manager.sigma_satisfied_data[i][q_name]["ldb_data"].df.empty
                        else pd.Series(dtype=int)
                        for i in range(inc_round + 1)
                    ],
                    prop_Y_li=pred_Y_li,
                )
                logger.info(f"Inc-Round {inc_round}, Query {q_name}: Est. error certificate: {err_certificate:.4f}")

                # Apply the rules.
                trans_Y_li = []
                if err_certificate < self.eta:
                    trans_Y_li = [
                            apply_rules(
                                self.rules[q_name]["ucq"],
                                encode_features(test_X_li[i]),
                            ) for i in range(inc_round + 1)
                        ]
                else:
                    rerun = True
                    return rerun, []

                # Evaluate the predications and translations.
                eval_results_single_step[q_name]["error_certificate"] = err_certificate
                eval_results_single_step[q_name]["pred_eval"] = \
                    self.data_manager.eval_query_quality(
                        q_name=q_name, 
                        selected_cols=self.queries[q_name].selected,
                        stream_idx=inc_round,
                        pred_labels=pred_Y_li,
                    )
                eval_results_single_step[q_name]["trans_eval"] = \
                    self.data_manager.eval_query_quality(
                        q_name=q_name, 
                        selected_cols=self.queries[q_name].selected,
                        stream_idx=inc_round,
                        pred_labels=trans_Y_li,
                    )
            
            end_time = time()
            eval_results.append({
                "inc_round": inc_round,
                "inc_ratio": self.dynamic_setting[inc_round],
                "eval_results": eval_results_single_step,
                "eval_time": end_time - start_time
            })
        return False, eval_results


    def _update_statistics(self, key, value):
        assert key in self.usage_statistics[0], f"Invalid statistics key: {key}"
        for k, v in value.items():
            self.usage_statistics[0][key][k] += v


    def _report_usage_statistics(self):
        report_usage_statistics(self.usage_statistics[0])
        

    def _report_evaluation_trace(self, execution_trace: dict):
        if execution_trace == {}:
            return
        report_evaluation_trace(execution_trace)


    def _collect_table_schema(self, df: pd.DataFrame) -> str:
        """Collect and format table schema information for LLM prompt."""
        schema_lines = []

        for col in df.columns:
            # Numerical columns
            if ptypes.is_numeric_dtype(df[col]):
                col_min = df[col].min()
                col_max = df[col].max()
                schema_lines.append(f"- {col} (numerical): range [{col_min}, {col_max}]")

            # Categorical columns (object, category, bool)
            else:
                # value_counts() returns results in descending order by default
                value_counts = df[col].value_counts()
                top_n = min(20, len(value_counts))
                top_values = value_counts.head(top_n).index.tolist()

                # Format the values
                if len(value_counts) > 20:
                    values_str = ", ".join(str(v) for v in top_values) + f", ... ({len(value_counts)} total unique values)"
                else:
                    values_str = ", ".join(str(v) for v in top_values)

                schema_lines.append(f"- {col} (categorical): [{values_str}]")

        return "\n".join(schema_lines)


    def _build_query_description(self, sem_cq: SemCQ) -> str:
        """Build a query description from semantic predicates."""
        if not sem_cq.Ps:
            return "No semantic predicates provided."

        descriptions = []
        for sem_pred in sem_cq.Ps:
            desc = f"Field: {sem_pred.field}\n"
            desc += f"Success condition: {sem_pred.succ_cond}\n"
            desc += f"Prompt: {sem_pred.prompt}"
            descriptions.append(desc)

        return "\n\n".join(descriptions)


    def _expand_ucq(self, ucq_responses: list[list[PredicateResponse]]) -> list[list[Predicate]]:
        from itertools import product

        expanded_ucq = []

        for group in ucq_responses:
            if not group:
                continue

            # Separate predicates by operator and field/value count
            eq_multi_preds = []  # == with multiple fields or values
            other_preds = []     # All other predicates

            for pred_resp in group:
                field_list = pred_resp.field
                value_list = pred_resp.value

                if pred_resp.op == '==' and (len(field_list) > 1 or len(value_list) > 1):
                    # For == with multiple fields or values: create alternatives
                    # Each field-value combination becomes a separate alternative group
                    alternatives = []
                    for field in field_list:
                        for value in value_list:
                            alternatives.append(
                                Predicate(field=field, op=pred_resp.op, value=value)
                            )
                    eq_multi_preds.append(alternatives)

                elif pred_resp.op == '!=' and (len(field_list) > 1 or len(value_list) > 1):
                    # For != with multiple fields or values: expand to conjunctive predicates
                    # All field-value combinations must be true (AND logic within group)
                    for field in field_list:
                        for value in value_list:
                            other_preds.append(
                                Predicate(field=field, op=pred_resp.op, value=value)
                            )

                else:
                    # Single field and single value (or comparison operators with single value)
                    # Extract the single field and single value
                    single_field = field_list[0]
                    single_value = value_list[0]
                    other_preds.append(
                        Predicate(field=single_field, op=pred_resp.op, value=single_value)
                    )

            # Combine eq_multi_preds (alternatives) with other_preds using Cartesian product
            # eq_multi_preds contains lists of alternatives, and we pick one from each list
            # Then combine with all other_preds using AND logic
            if eq_multi_preds:
                # Generate all combinations of alternatives
                for combo in product(*eq_multi_preds):
                    new_group = list(combo) + other_preds
                    expanded_ucq.append(new_group)
            else:
                # No eq_multi_preds, just use other_preds
                expanded_ucq.append(other_preds)

        return expanded_ucq


    def _format_relevant_fields(self, fields_by_semantic_group: dict[str, list[str]]) -> str:
        if not fields_by_semantic_group:
            return "No relevant fields identified"

        lines = []
        for group_name, fields in sorted(fields_by_semantic_group.items()):
            if fields:
                fields_str = ", ".join(fields)
                lines.append(f"- **{group_name}**: {fields_str}")

        return "\n".join(lines) if lines else "No relevant fields identified"



    def _report_dynamic_results(self, eval_results: list):
        inc_rounds = [res["inc_round"] for res in eval_results]
        inc_ratios = [res["inc_ratio"] for res in eval_results]
        eval_resuls_single_step = [res["eval_results"] for res in eval_results]
        eval_time = [res["eval_time"] for res in eval_results]
        
        queries = eval_resuls_single_step[0].keys()

        try:
            for q in queries:
                eval_q_single_step = [res[q] for res in eval_resuls_single_step]

                trans_f1s = [res["trans_eval"]["f1"] if res else None for res in eval_q_single_step]
                pred_f1s = [res["pred_eval"]["f1"] if res else None for res in eval_q_single_step]
                error_certificates = [res["error_certificate"] if res else None for res in eval_q_single_step]

                results = pd.DataFrame({
                    "inc_round": inc_rounds,
                    "inc_ratio": inc_ratios,
                    "trans_f1": trans_f1s,
                    "pred_f1": pred_f1s,
                    "error_certificate": error_certificates,
                    "eval_time": eval_time,
                })

                print("=" * 20 + f" Incremental Evaluation Results for Query {q}" + "=" * 20)
                print(results)
        except Exception as e:
            print(eval_results)
