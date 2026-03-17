import json
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Tuple, Any, cast, Optional
from collections import defaultdict
import pandas.api.types as ptypes
from data_structure import LdbData, SemCQ, PopulationSpec, FeatureRefinementResponse, Predicate
from data_structure.llm_resp_templates import PredicateResponses, PredicateResponse, RelevantFieldsResponse
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
        self.dynamic_setting = config["dynamic_setting"]

        self.data_dir = data_dir
        self.scenario = scenario if self.dynamic_setting == [1.0] else f"dyn_{scenario}"
        self.ldb_data = LdbData(data_dir=data_dir)
        self.queries = queries
        self.config = config
        self.random_seed = config["random_seed"]
        self.b_lab = config["b_lab"]
        self.b_se = config["b_se"]
        self.b_rew = config["b_rew"]
        self.b_fs = config["b_fs"]

        self.sigma_satisfied_data = []
        self.labeled_data = [{}]
        self.unlabeled_data = [{}]
        self.coresets = [{}]

        self.feature_spaces = {}
        
        self.base_schema = self.ldb_data.df.columns.tolist()
        self.candidate_external_features = []

        self.CKPT_path = Path(__file__).parent.parent / ".ckpt" / self.scenario
        self.CKPT_path.mkdir(parents=True, exist_ok=True)

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
            ]
        } for _ in range(len(self.dynamic_setting))]

    def apply_sigma(self, debug=False):
        for q_name, sem_cq in self.queries.items():

            data = self.ldb_data.sigma_retrieve(sem_cq.Sigma, reset_index=True)
            labels = build_ground_truth_labels(data.df, q_name, sem_cq.selected, self.data_dir, debug=debug)

            # Get indices for positive and negative samples
            rem_pos_idx = labels[labels == True].index
            rem_neg_idx = labels[labels == False].index

            base_ratio = 0
            for r in self.dynamic_setting:
                delta_ratio = (r - base_ratio) / (1 - base_ratio)

                sel_pos_idx = rem_pos_idx.to_series().sample(frac=delta_ratio, random_state=self.random_seed).index
                sel_neg_idx = rem_neg_idx.to_series().sample(frac=delta_ratio, random_state=self.random_seed).index
                sel_idx = sel_pos_idx.union(sel_neg_idx)
                shuffled_idx = sel_idx.to_series().sample(frac=1, random_state=self.random_seed).index

                self.sigma_satisfied_data.append({
                    q_name: {
                        "data": LdbData(
                            df=data.df.loc[shuffled_idx].reset_index(drop=True),
                            config=data.config
                        ),
                        "labels": labels.loc[shuffled_idx].reset_index(drop=True),
                    }
                })

                rem_pos_idx = rem_pos_idx.difference(sel_pos_idx)
                rem_neg_idx = rem_neg_idx.difference(sel_neg_idx)
                base_ratio = r


    def augment_sigma_and_apply(self):
        for q_name, sem_cq in self.queries.items():
            logger.info(f"Augmenting Sigma for query {q_name}...")

            # Get the data without FK and ID columns
            data = self.sigma_satisfied_data[0][q_name]["data"]
            df_clean = data.exclude_fk_and_id()

            # Collect table information
            schema_info = self._collect_table_schema(df_clean)

            # Build query description from semantic predicates
            query_desc = self._build_query_description(sem_cq)


            cache_path = self.CKPT_path / f"{q_name}_prefilter_ucq.json"
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
            filtered_data = self.sigma_satisfied_data[0][q_name]["data"].sigma_retrieve_ucq(new_ucq, reset_index=True)

            # Rebuild labels for the filtered data
            labels = build_ground_truth_labels(
                filtered_data.df, q_name, sem_cq.selected, self.data_dir, debug=False
            )

            # Update the sigma_satisfied_data
            self.sigma_satisfied_data[0][q_name]["data"] = filtered_data
            self.sigma_satisfied_data[0][q_name]["labels"] = labels

            logger.info(
                f"Successfully applied UCQ for query {q_name}. "
                f"Data size: {len(data.df)} -> {len(filtered_data.df)} rows"
            )

            if self.b_lab >= len(filtered_data.df):
                self.b_lab = len(filtered_data.df) // 2
                logger.info(f"Adjusted b_lab to {self.b_lab} due to small data size after UCQ application for query {q_name}.")

    
    async def init_coresets(self, debug: bool = False):

        for q_name, sem_cq in self.queries.items():
            # Acquire labeled set.
            self._acquire_human_label(q_name=q_name, debug=debug)

            # Acquire initial feature space.
            self.feature_spaces[q_name] = await self._init_feature_space(q_name, sem_cq)

            # Store materialized data in coreset
            self.coresets[0][q_name] = {
                "data": LdbData(df=self.labeled_data[0][q_name]["data"].df, config=self.ldb_data.config),
                "labels": self.labeled_data[0][q_name]["labels"],
            }


    async def populate_unlabeled_data(self):
        for q_name, spec in self.feature_spaces.items():
            self.unlabeled_data[0][q_name]["data"].df = await self._materialize_features(
                    q_name=q_name,
                    tag="unlabeled_full",
                    data=self.unlabeled_data[0][q_name]["data"],
                    feature_specs=spec,
                    is_remote=False
                )

    
    def expand_coreset(self, debug: bool = False):
        for q_name, coreset in self.coresets[0].items():
            labeled_X = coreset["data"].exclude_fk_and_id()
            labeled_Y = coreset["labels"]
            unlabeled_X = self.unlabeled_data[0][q_name]["data"].exclude_fk_and_id()

            selected_X_idx, selected_Y = select_coreset(
                labeled_X=labeled_X,
                labeled_Y=labeled_Y,
                unlabeled_X=unlabeled_X,
                k_neighbors=self.config.get("k_neighbors", 5),
                mode="emperical",
            )

            # Update the coreset with the selected samples.
            selected_X = \
                self.unlabeled_data[0][q_name]["data"].df.iloc[selected_X_idx].reset_index(drop=True)
            self.coresets[0][q_name]["data"].df = \
                pd.concat([self.coresets[0][q_name]["data"].df, selected_X], ignore_index=True)
            self.coresets[0][q_name]["labels"] = pd.concat([labeled_Y, selected_Y], ignore_index=True)

            logger.info(f"Expanded coreset for query {q_name}: added {len(selected_X)} samples. New coreset size: {len(self.coresets[0][q_name]['data'].df)}")

            if debug:
                ground_truth_Y = self.unlabeled_data[0][q_name]["labels"].iloc[selected_X_idx].reset_index(drop=True)
                eval_results = evaluate_classifier(selected_Y, ground_truth_Y)
                logger.info((
                    f"Debug evaluation of expanded coreset for query {q_name}: "
                    f"TP={eval_results['TP']}, FP={eval_results['FP']}, FN={eval_results['FN']}, "
                    f"Precision={eval_results['precision']:.4f}, Recall={eval_results['recall']:.4f}, F1={eval_results['f1']:.4f}."
                ))

    
    def generate_candidate_external_features(self):

        importance_sum = defaultdict(float)

        for coreset in self.coresets[0].values():
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
                train_X = self.coresets[0][q_name]["data"].select_active_features(active_external_features)
                train_Y = self.coresets[0][q_name]["labels"].astype(int)
                test_X = self.unlabeled_data[0][q_name]["data"].select_active_features(active_external_features)
                test_Y = self.unlabeled_data[0][q_name]["labels"].astype(int)
                visible_labels = self.labeled_data[0][q_name]["labels"].astype(int)

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
                    X=self.labeled_data[0][q_name]["data"].select_active_features(active_external_features),
                    Y=self.labeled_data[0][q_name]["labels"].astype(int),
                    query_size=len(self.queries),
                    data_size=len(self.labeled_data[0][q_name]["labels"]),
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

        data = self.sigma_satisfied_data[0][q_name]["data"]
        labels = self.sigma_satisfied_data[0][q_name]["labels"]

        labeled_indices = data.df.sample(n=self.b_lab, random_state=self.random_seed).index

        # Fallback: ensure minority class has at least minority_threshold samples
        num_pos_sampled = labels.loc[labeled_indices].sum()
        num_neg_sampled = len(labeled_indices) - num_pos_sampled
        minority_threshold = max(1, int(0.05 * self.b_lab))
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

        self.labeled_data[0][q_name] = {
            "data": LdbData(
                df=data.df.loc[labeled_indices].reset_index(drop=True),
                config=data.config),
            "labels": labels.loc[labeled_indices].reset_index(drop=True),
        }
        self.unlabeled_data[0][q_name] = {
            "data": LdbData(
                df=data.df.loc[remaining_indices].reset_index(drop=True),
                config=data.config),
            "labels": labels.loc[remaining_indices].reset_index(drop=True),
        }

        if debug:
            logger.info((f"Acquired labels for {self.scenario}.{q_name}: "
                         f"{self.labeled_data[0][q_name]['labels'].sum()} positive labels / "
                         f"{len(self.labeled_data[0][q_name]['data'].df)} labeled samples. Selectivity: "
                         f"{self.labeled_data[0][q_name]['labels'].sum() / len(self.labeled_data[0][q_name]['data'].df):.4f}"))


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
                self.labeled_data[0][q_name]["data"] = LdbData(df=cached_df, config=self.ldb_data.config)
            return [PopulationSpec(**spec) for spec in cached_data[q_name]]

        # Initialize feature space
        feature_space = []

        # Process each semantic predicate
        for sem_pred in sem_cq.Ps:
            field = sem_pred.field

            # Divide pos / neg labeled data into batches
            labels = self.labeled_data[0][q_name]["labels"]
            pos_indices = labels[labels == True].index.tolist()
            neg_indices = labels[labels == False].index.tolist()

            # Calculate batch size N = min(5, num_pos, num_neg)
            N = min(5, len(pos_indices), len(neg_indices))

            # Handle case when N=0 (all samples are positive or negative)
            if N == 0:
                logger.warning(f"All samples are {'positive' if len(pos_indices) > 0 else 'negative'} "
                             f"for query {q_name}, field {field}. Using fallback mode with single-class samples.")
                # Use all available samples with their labels
                all_indices = pos_indices + neg_indices
                all_labels = [True] * len(pos_indices) + [False] * len(neg_indices)
                all_data = self.labeled_data[0][q_name]["data"].df.loc[all_indices, field].tolist()

                # Build data items and metadata for single-class case
                data_items = all_data
                metadata = [
                    {"label": label, "sample_id": int(idx)}
                    for label, idx in zip(all_labels, all_indices)
                ]

                # Build prompt
                prompt = build_feature_generation_prompt(
                    sem_pred, feature_space, None, 0,
                    self.b_fs, PROMPTS["GEN_FEAT_CANDIDATE_PROMPT"],
                    data_df=self.labeled_data[0][q_name]["data"].df
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
                    self.labeled_data[0][q_name]["data"].df.drop(columns=features_to_remove, inplace=True)

                if llm_response.to_add:
                    feature_space.extend(llm_response.to_add)
                    self.labeled_data[0][q_name]["data"].df = await self._materialize_features(
                        q_name=q_name,
                        tag=f"labeled_full",
                        data=self.labeled_data[0][q_name]["data"],
                        feature_specs=llm_response.to_add,
                        reuse=False,
                        is_remote=False,
                    )

                logger.info(f"Fallback mode: Added={len(llm_response.to_add)}, Removed={len(llm_response.to_remove)}")
                continue  # Skip to next semantic predicate

            # Create batches (normal case with both pos and neg samples)
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
                pos_batch_data = self.labeled_data[0][q_name]["data"].df.loc[current_pos_indices, field].tolist() 
                neg_batch_data = self.labeled_data[0][q_name]["data"].df.loc[current_neg_indices, field].tolist()

                # Build data items and metadata
                data_items, metadata = build_contrastive_batch(
                    sem_pred, pos_batch_data, neg_batch_data,
                    current_pos_indices, current_neg_indices, previous_feedback,
                    self.labeled_data[0][q_name]["data"].df
                )

                # Build prompt
                prompt = build_feature_generation_prompt(
                    sem_pred, feature_space, previous_feedback, iteration,
                    self.b_fs, PROMPTS["GEN_FEAT_CANDIDATE_PROMPT"],
                    data_df=self.labeled_data[0][q_name]["data"].df
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
                    self.labeled_data[0][q_name]["data"].df.drop(columns=features_to_remove, inplace=True)
                
                if llm_response.to_add:
                    feature_space.extend(llm_response.to_add)
                    self.labeled_data[0][q_name]["data"].df = await self._materialize_features(
                        q_name=q_name,
                        tag=f"labeled_full",
                        data=self.labeled_data[0][q_name]["data"],
                        feature_specs=llm_response.to_add,
                        reuse=False,
                        is_remote=False,
                    )

                # Evaluate and enforce budget
                feedback = pred_and_eval(
                    self.labeled_data[0][q_name]["data"].exclude_fk_and_id(),
                    self.labeled_data[0][q_name]["labels"]
                )
                if len(feature_space) > self.b_fs:
                    feature_can_be_removed = [k for k, v in feedback["feature_importance"].items() if k not in self.base_schema]
                    feature_to_be_removed = feature_can_be_removed[-(len(feature_space) - self.b_fs):]
                    feature_space[:] = [spec for spec in feature_space if spec.target_col not in feature_to_be_removed]
                    self.labeled_data[0][q_name]["data"].df.drop(columns=feature_to_be_removed, inplace=True)
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
            json.dump(self.usage_statistics[0]["feature_space_init"], f, indent=2)
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
            json.dump(self.usage_statistics[0][f"materialize_{tag}"], f, indent=2)
            logger.info(f"Cached usage statistics for materializing features for query {q_name} with tag {tag} to: {ckpt_usage_path}")

        df_cp.to_csv(ckpt_path, index=False)
        logger.info(f"Cached materialized features for query {q_name} with tag {tag} to: {ckpt_path}")

        return df_cp


    def _update_statistics(self, key, value):
        assert key in self.usage_statistics[0], f"Invalid statistics key: {key}"
        for k, v in value.items():
            self.usage_statistics[0][key][k] += v


    def _report_usage_statistics(self):
        report_usage_statistics(self.usage_statistics[0])
        

    def _report_evaluation_trace(self, execution_trace: dict):
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

