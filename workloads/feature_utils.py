import pandas as pd
import logging
from typing import Tuple, cast
from data_structure import SemCQ, PopulationSpec, FeatureRefinementResponse, LdbDataManager
from llm import LdbLLMClient, PROMPTS
from workloads.workload_utils import pred_and_eval


logger = logging.getLogger(__name__)


class ContrastiveSampleIterator:
    
    def __init__(
            self, 
            pos_samples_idx: list, neg_samples_idx: list, 
            batch_size: int, max_iters: int = 1):
        self.pos_samples_idx = pos_samples_idx
        self.neg_samples_idx = neg_samples_idx
        self.batch_size = batch_size
        self.max_iters = max_iters
        self.num_pos_batches = (len(pos_samples_idx) + batch_size - 1) // batch_size
        self.num_neg_batches = (len(neg_samples_idx) + batch_size - 1) // batch_size

        self.iter_num = 0
        self.pos_ptr = 0
        self.neg_ptr = 0


    def reset(self):
        self.iter_num = 0
        self.pos_ptr = 0
        self.neg_ptr = 0

    
    def has_next_batch(self) -> bool:
        if self._has_next_pos_batch() and self._has_next_neg_batch():
            return True
        
        if self.iter_num == 0 and (
            self._has_next_pos_batch() or self._has_next_neg_batch()
        ):
            return True

        return False

    
    def next_batch(self) -> Tuple[list, list]:
        assert self.has_next_batch(), "No more batches available."

        pos_batch_data, neg_batch_data = [], []

        if self._has_next_pos_batch():
            pos_start = self.pos_ptr
            pos_end = min(self.pos_ptr + self.batch_size, len(self.pos_samples_idx))
            pos_batch_data = self.pos_samples_idx[pos_start:pos_end]
            self.pos_ptr = pos_end

        if self._has_next_neg_batch():
            neg_start = self.neg_ptr
            neg_end = min(self.neg_ptr + self.batch_size, len(self.neg_samples_idx))
            neg_batch_data = self.neg_samples_idx[neg_start:neg_end]
            self.neg_ptr = neg_end

        self.iter_num += 1

        return pos_batch_data, neg_batch_data


    def build_contrastive_batch(
        self,
        sem_pred,
        pos_batch_data: list,
        neg_batch_data: list,
        pos_batch_indices: list,
        neg_batch_indices: list,
        previous_feedback: dict | None,
        labeled_data_df: pd.DataFrame,
    ) -> Tuple[list, list]:
        data_items = pos_batch_data + neg_batch_data
        metadata = [
            {"label": True, "sample_id": int(i)} for i in pos_batch_indices
        ] + [
            {"label": False, "sample_id": int(i)} for i in neg_batch_indices
        ]
    
        # Add bad cases
        if previous_feedback and not previous_feedback['bad_cases'].empty:
            for _, row in previous_feedback['bad_cases'].head(5).iterrows():
                field_content = labeled_data_df.loc[
                    int(row['_original_index']), sem_pred.field
                ]
                data_items.append(field_content)
                metadata.append({
                    "label": bool(row['_true_label']),
                    "sample_id": int(row['_original_index']),
                    "is_bad_case": True,
                    "misclassified_as": bool(row['_predicted_label'])
                })
    
        return data_items, metadata


    def _has_next_pos_batch(self) -> bool:
        return self.iter_num < self.max_iters and \
            self.iter_num < self.num_pos_batches

    def _has_next_neg_batch(self) -> bool:
        return self.iter_num < self.max_iters and \
            self.iter_num < self.num_neg_batches


async def initialize_feature_space(
    b_fs: int,
    data_manager: LdbDataManager, 
    q_name: str,
    sem_cq: SemCQ,
    llm_client: LdbLLMClient,
) -> Tuple[list[PopulationSpec], dict]:


    usage_statistics = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "prompt_cost": 0.0,
        "completion_cost": 0.0,
        "total_cost": 0.0,
    }

    labels = data_manager.coresets[q_name]["labels"]
    pos_samples_idx = data_manager.coresets[q_name]["ldb_data"].df[labels == True].index.tolist()
    neg_samples_idx = data_manager.coresets[q_name]["ldb_data"].df[labels == False].index.tolist()

    sample_iterator = ContrastiveSampleIterator(
        pos_samples_idx=pos_samples_idx, neg_samples_idx=neg_samples_idx, batch_size=5, max_iters=1
    )

    prev_f1 = None
    previous_feedback = None
    feature_space = []

    base_schema = data_manager.coresets[q_name]["ldb_data"].df.columns.tolist()

    while sample_iterator.has_next_batch():
        pos_idx, neg_idx = sample_iterator.next_batch()

        for sem_pred in sem_cq.Ps:
            field = sem_pred.field
            pos_samples = data_manager.coresets[q_name]["ldb_data"].df.loc[pos_idx, field].tolist()
            neg_samples = data_manager.coresets[q_name]["ldb_data"].df.loc[neg_idx, field].tolist()

            data_items, metadata = sample_iterator.build_contrastive_batch(
                sem_pred=sem_pred,
                pos_batch_data=pos_samples,
                neg_batch_data=neg_samples,
                pos_batch_indices=pos_idx,
                neg_batch_indices=neg_idx,
                previous_feedback=previous_feedback,
                labeled_data_df=data_manager.coresets[q_name]["ldb_data"].df,
            )

            prompt = _build_feature_generation_prompt(
                sem_pred=sem_pred,
                feature_space=feature_space,
                previous_feedback=previous_feedback,
                iteration=sample_iterator.iter_num - 1,
                feature_budget=b_fs,
                prompt_template=PROMPTS["GEN_FEAT_CANDIDATE_PROMPT"],
                data_df=data_manager.coresets[q_name]["ldb_data"].df,
            )

            llm_response = cast(FeatureRefinementResponse, llm_client.invoke(
                modality=sem_pred.modality,
                is_remote=True,
                prompt=prompt,
                data_items=data_items,
                data_items_metadata=metadata,
                response_model=FeatureRefinementResponse,
            ))

            # Update the feature space.
            features_to_remove = [f for f in llm_response.to_remove if f not in base_schema]
            features_to_add = [spec for spec in llm_response.to_add if \
                               spec.target_col not in base_schema and \
                                spec.target_col not in [s.target_col for s in feature_space]]
            feature_space = [spec for spec in feature_space if spec.target_col not in features_to_remove]
            feature_space.extend(features_to_add)

            data_manager.enriched_features[q_name] = feature_space
            stat = await data_manager.sync_coreset_features(q_name=q_name, enable_cache=False)
            for k, v in stat.items():
                usage_statistics[k] += v

            # Evaluate and enforce budget
            feedback = pred_and_eval(
                data_manager.coresets[q_name]["ldb_data"].exclude_fk_and_id(),
                data_manager.coresets[q_name]["labels"]
            )

            if len(feature_space) > b_fs:
                feature_can_be_removed = [
                    k for k, _ in feedback["feature_importance"].items() if k not in base_schema]
                feature_to_be_removed = feature_can_be_removed[-(len(feature_space) - b_fs):]
                feature_space[:] = [
                    spec for spec in feature_space if spec.target_col not in feature_to_be_removed]

                data_manager.enriched_features[q_name] = feature_space                    
                stat = await data_manager.sync_coreset_features(q_name=q_name, enable_cache=False)
                for k, v in stat.items():
                    usage_statistics[k] += v

                logger.info(f"Removed {len(feature_to_be_removed)} features to enforce budget.")

            logger.info(f"Iteration {sample_iterator.iter_num - 1}: F1={feedback['f1']:.4f}, "
                       f"Added={len(llm_response.to_add)}, Removed={len(llm_response.to_remove)}")

            # Check stopping criteria: F1 drops > 0.05
            if prev_f1 is not None:
                f1_drop = prev_f1 - feedback['f1']
                if f1_drop > 0.05:
                    logger.info(f"F1 dropped by {f1_drop:.4f} > 0.05. Stopping iteration.")
                    break

            prev_f1 = feedback['f1']
            previous_feedback = feedback

    return feature_space, usage_statistics


def _build_feature_generation_prompt(
    sem_pred,
    feature_space: list[PopulationSpec],
    previous_feedback: dict | None,
    iteration: int,
    feature_budget: int,
    prompt_template: str,
    data_df: pd.DataFrame | None = None,
    num_samples: int = 3,
) -> str:

    is_first = iteration == 0 or len(feature_space) == 0

    # Build schema and sample data section
    schema_sample_section = ""
    if data_df is not None and not data_df.empty:
        # List all columns
        columns_str = ", ".join(data_df.columns.tolist())
        schema_section = f"=== Current Data Schema ===\nColumns: {columns_str}\n"

        # Show sample data
        sample_rows = data_df.head(num_samples)
        sample_data_str = sample_rows.to_string(index=True)
        sample_section = f"\n=== Sample Data (first {num_samples} rows) ===\n{sample_data_str}\n"

        schema_sample_section = schema_section + sample_section

    if is_first:
        return prompt_template.format(
            MODALITY=sem_pred.modality,
            DESC=sem_pred.prompt,
            SOURCE_COL=sem_pred.field,
            FEATURE_BUDGET=feature_budget,
            SCHEMA_SAMPLE_SECTION=schema_sample_section,
            PREVIOUS_FEATURES_SECTION="",
            PERFORMANCE_FEEDBACK_SECTION="",
            INSTRUCTIONS_SECTION="Propose an initial set of discriminative features that can effectively distinguish positive from negative samples.",
            CONSTRAINTS_ADDITIONAL="",
        )

    # Build current features section
    current_features_str = "\n".join([
        f"  - {spec.target_col} ({spec.feature_type}): {spec.prompt[:80]}..."
        for spec in feature_space
    ])
    previous_features_section = f"\n\n=== Current Feature Space ===\n{current_features_str}\n"

    # Build feedback section
    if previous_feedback:
        importance_str = "\n".join([
            f"  - {feat}: {imp:.4f}"
            for feat, imp in list(previous_feedback['feature_importance'].items())[:10]
        ])
        performance_feedback_section = f"""
\n=== Performance Feedback ===
\n1. Feature Importance (sorted by importance):\n{importance_str}
\n2. F1 Score:\n   - With current features: {previous_feedback['f1']:.4f}\n
"""
        constraints = """- Make sure "to_remove" contains EXACTLY the feature names from the current feature space (check the "Current Feature Space" section above).
- Avoid proposing features that are already in the current feature space."""
    else:
        performance_feedback_section = ""
        constraints = "- Avoid proposing features that are already in the current feature space."

    return prompt_template.format(
        MODALITY=sem_pred.modality,
        DESC=sem_pred.prompt,
        SOURCE_COL=sem_pred.field,
        FEATURE_BUDGET=feature_budget,
        SCHEMA_SAMPLE_SECTION=schema_sample_section,
        PREVIOUS_FEATURES_SECTION=previous_features_section,
        PERFORMANCE_FEEDBACK_SECTION=performance_feedback_section,
        INSTRUCTIONS_SECTION="Propose features that will improve classification performance based on the feedback above.",
        CONSTRAINTS_ADDITIONAL=constraints,
    )
            




