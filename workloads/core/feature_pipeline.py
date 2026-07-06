# pylint: disable=missing-class-docstring,logging-fstring-interpolation
# pylint: disable=duplicate-code,missing-function-docstring
# pylint: disable=unspecified-encoding,unused-argument,invalid-name
# pylint: disable=too-many-locals,too-many-instance-attributes
# pylint: disable=too-many-arguments,too-many-positional-arguments
"""Feature-space and coreset pipeline helpers."""

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.tree import DecisionTreeClassifier

from data_structure import (
    FeatureRefinementResponse,
    LdbDataManager,
    PopulationSpec,
    SemCQ,
)
from llm import PROMPTS, LdbLLMClient
from workloads.utils import (
    compute_feature_importance,
    encode_features,
)

logger = logging.getLogger(__name__)


class FeaturePipeline:
    """Methods for feature construction, materialization, and trimming."""

    def __init__(
        self,
        data_manager: LdbDataManager,
        queries: dict[str, SemCQ],
        ckpt_path: Path,
        llm_client: LdbLLMClient,
        usage_statistics: list[dict[str, Any]],
        random_seed: int,
        b_lab: int,
        b_se: int,
    ) -> None:
        self.data_manager = data_manager
        self.queries = queries
        self.CKPT_path = ckpt_path
        self.llm_client = llm_client
        self.usage_statistics = usage_statistics
        self.random_seed = random_seed
        self.b_lab = b_lab
        self.b_se = b_se

    def update_statistics(self, key: str, value: dict[str, Any]) -> None:
        assert key in self.usage_statistics[0], f"Invalid statistics key: {key}"
        for stat_key, stat_value in value.items():
            self.usage_statistics[0][key][stat_key] += stat_value

    async def construct_feature_space(self, debug: bool = False) -> None:
        await self.data_manager.acquire_annotation_and_init_coreset(
            b_lab=self.b_lab,
            seed=self.random_seed,
        )

        for q_name, sem_cq in self.queries.items():
            ckpt_path = self.CKPT_path / q_name / "feature_space.json"
            ckpt_usage_path = (
                self.CKPT_path / q_name / "usage_feature_space.json"
            )

            if ckpt_path.exists() and ckpt_usage_path.exists():
                with open(ckpt_path) as f:
                    feature_space = json.load(f)
                with open(ckpt_usage_path) as f:
                    usage_statistics = json.load(f)
                logger.info(
                    "Loaded feature space from checkpoint for query %s.",
                    q_name,
                )
                self.data_manager.enriched_features[q_name] = [
                    PopulationSpec(**spec) for spec in feature_space
                ]
                self.update_statistics("feature_space_init", usage_statistics)
                continue

            (
                feature_space,
                usage_statistics,
            ) = await self._initialize_feature_space(
                q_name=q_name,
                sem_cq=sem_cq,
                feature_budget=self.b_se * 2,
            )
            self.data_manager.enriched_features[q_name] = feature_space
            self.update_statistics("feature_space_init", usage_statistics)
            self.llm_client.reset_usage_statistics()
            with open(ckpt_usage_path, "w") as f:
                json.dump(usage_statistics, f, indent=2)

        if len(self.queries) > 1:
            new_fs, stat = await self._enforce_feature_budget(
                pop_specs=self.data_manager.enriched_features,
                feature_budget=self.b_se,
            )
            self.data_manager.enriched_features = new_fs
            if stat:
                self.update_statistics("feature_space_init", stat)

        for q_name in self.queries:
            ckpt_path = self.CKPT_path / q_name / "feature_space.json"
            feature_space = self.data_manager.enriched_features[q_name]
            with open(ckpt_path, "w") as f:
                json.dump(
                    [spec.model_dump() for spec in feature_space],
                    f,
                    indent=2,
                )
            logger.info(
                "Saved feature space and usage statistics to checkpoint "
                "for query %s.",
                q_name,
            )

    async def sync_with_enriched_features(self, tag: str = "") -> None:
        for q_name in self.queries:
            stat_coreset = await self.data_manager.sync_coreset_features(
                q_name, tag=tag, enable_cache=True
            )
            stat_sigma = (
                await self.data_manager.sync_sigma_satisfied_data_features(
                    q_name, stream_idx=0, tag=tag, enable_cache=True
                )
            )

            self.update_statistics("materialize_labeled_full", stat_coreset)
            self.update_statistics("materialize_unlabeled_full", stat_sigma)
            self.llm_client.reset_usage_statistics()

    async def rank_and_trim_feature_space(self, reuse: bool = False) -> None:
        ckpt_path = self.CKPT_path / "ranked_feature_space.json"

        if ckpt_path.exists() and reuse:
            with open(ckpt_path) as f:
                ranked_features = json.load(f)
            top_feats = ranked_features[: self.b_se]
            logger.info("Loaded ranked feature space from checkpoint.")
        else:
            importance_sum = defaultdict(float)
            for coreset in self.data_manager.coresets.values():
                X = encode_features(coreset["ldb_data"].exclude_fk_and_id())
                Y = coreset["labels"].astype(int)
                for feat, imp in compute_feature_importance(X, Y).itertuples(
                    index=False
                ):
                    importance_sum[feat] += imp

            enriched_features = self.data_manager.enriched_features
            external_feats = list(
                {
                    spec.target_col
                    for space in enriched_features.values()
                    for spec in space
                }
            )
            ranked_feats = sorted(
                external_feats,
                key=lambda feat: importance_sum.get(feat, 0),
                reverse=True,
            )
            top_feats = ranked_feats[: self.b_se]
            logger.info(
                "Selected %s external features: %s",
                len(top_feats),
                top_feats,
            )
            with open(ckpt_path, "w") as f:
                json.dump(ranked_feats, f, indent=2)

        self.data_manager.trimmed_feature_names = top_feats
        for q_name in self.data_manager.enriched_features:
            self.data_manager.enriched_features[q_name] = [
                spec
                for spec in self.data_manager.enriched_features[q_name]
                if spec.target_col in top_feats
            ]

        for q_name in self.data_manager.coresets:
            await self.data_manager.sync_coreset_features(
                q_name, tag="trimmed", enable_cache=True
            )

        for q_name in self.data_manager.sigma_satisfied_data[0]:
            await self.data_manager.sync_sigma_satisfied_data_features(
                q_name, stream_idx=0, tag="trimmed", enable_cache=True
            )

        for q_name in self.data_manager.coresets:
            ldb_data = self.data_manager.coresets[q_name]["ldb_data"]
            features = ldb_data.df.columns.tolist()
            original_schema = (
                ldb_data.base_features
                + ldb_data.id_features
                + ldb_data.foreign_keys
            )
            external_features = [
                feat for feat in features if feat not in original_schema
            ]
            self.data_manager.trimmed_feature_names = external_features
            self.data_manager.enriched_features[q_name] = [
                spec
                for spec in self.data_manager.enriched_features[q_name]
                if spec.target_col in external_features
            ]

    async def _initialize_feature_space(
        self,
        q_name: str,
        sem_cq: SemCQ,
        feature_budget: int,
    ) -> tuple[list[PopulationSpec], dict]:
        usage_statistics = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "prompt_cost": 0.0,
            "completion_cost": 0.0,
            "total_cost": 0.0,
        }

        labels = self.data_manager.coresets[q_name]["labels"]
        coreset_data = self.data_manager.coresets[q_name]["ldb_data"]
        pos_samples_idx = coreset_data.df[labels.astype(bool)].index.tolist()
        neg_samples_idx = coreset_data.df[~labels.astype(bool)].index.tolist()

        sample_iterator = ContrastiveSampleIterator(
            pos_samples_idx=pos_samples_idx,
            neg_samples_idx=neg_samples_idx,
            batch_size=5,
            max_iters=1,
        )

        prev_f1 = None
        previous_feedback = None
        feature_space = []
        base_schema = coreset_data.df.columns.tolist()

        while sample_iterator.has_next_batch():
            pos_idx, neg_idx = sample_iterator.next_batch()
            total_added = 0
            total_removed = 0

            for sem_pred in sem_cq.Ps:
                pos_samples = coreset_data.df.loc[
                    pos_idx, sem_pred.field
                ].tolist()
                neg_samples = coreset_data.df.loc[
                    neg_idx, sem_pred.field
                ].tolist()

                data_items, metadata = sample_iterator.build_contrastive_batch(
                    sem_pred=sem_pred,
                    pos_batch_data=pos_samples,
                    neg_batch_data=neg_samples,
                    pos_batch_indices=pos_idx,
                    neg_batch_indices=neg_idx,
                    previous_feedback=previous_feedback,
                    labeled_data_df=coreset_data.df,
                )

                prompt = self._build_feature_generation_prompt(
                    sem_pred=sem_pred,
                    feature_space=feature_space,
                    previous_feedback=previous_feedback,
                    iteration=sample_iterator.iter_num - 1,
                    feature_budget=feature_budget,
                    prompt_template=PROMPTS["GEN_FEAT_CANDIDATE_PROMPT"],
                    data_df=coreset_data.df,
                )

                llm_response = cast(
                    FeatureRefinementResponse,
                    self.llm_client.invoke(
                        modality=sem_pred.modality,
                        is_remote=True,
                        prompt=prompt,
                        data_items=data_items,
                        data_items_metadata=metadata,
                        response_model=FeatureRefinementResponse,
                    ),
                )

                features_to_remove = [
                    f for f in llm_response.to_remove if f not in base_schema
                ]
                existing_targets = [s.target_col for s in feature_space]
                features_to_add = [
                    spec
                    for spec in llm_response.to_add
                    if spec.target_col not in base_schema
                    and spec.target_col not in existing_targets
                ]
                feature_space = [
                    spec
                    for spec in feature_space
                    if spec.target_col not in features_to_remove
                ]
                feature_space.extend(features_to_add)

                total_added += len(llm_response.to_add)
                total_removed += len(llm_response.to_remove)

            self.data_manager.enriched_features[q_name] = feature_space
            stat = await self.data_manager.sync_coreset_features(
                q_name=q_name, enable_cache=False
            )
            for k, v in stat.items():
                usage_statistics[k] += v

            feedback = self._pred_and_eval(
                coreset_data.exclude_fk_and_id(),
                self.data_manager.coresets[q_name]["labels"],
            )

            if len(feature_space) > feature_budget:
                removable_features = [
                    k
                    for k, _ in feedback["feature_importance"].items()
                    if k not in base_schema
                ]
                removed_features = removable_features[
                    -(len(feature_space) - feature_budget) :
                ]
                feature_space[:] = [
                    spec
                    for spec in feature_space
                    if spec.target_col not in removed_features
                ]

                self.data_manager.enriched_features[q_name] = feature_space
                stat = await self.data_manager.sync_coreset_features(
                    q_name=q_name, enable_cache=False
                )
                for k, v in stat.items():
                    usage_statistics[k] += v

                logger.info(
                    "Removed %s features to enforce budget.",
                    len(removed_features),
                )

            logger.info(
                "Iteration %s: F1=%.4f, Added=%s, Removed=%s",
                sample_iterator.iter_num - 1,
                feedback["f1"],
                total_added,
                total_removed,
            )

            if prev_f1 is not None:
                f1_drop = prev_f1 - feedback["f1"]
                if f1_drop > 0.05:
                    logger.info(
                        "F1 dropped by %.4f > 0.05. Stopping iteration.",
                        f1_drop,
                    )
                    break

            prev_f1 = feedback["f1"]
            previous_feedback = feedback

        return feature_space, usage_statistics

    async def _enforce_feature_budget(
        self,
        pop_specs: dict[str, list[PopulationSpec]],
        feature_budget: int,
    ) -> tuple[dict[str, list[PopulationSpec]], dict[str, Any]]:
        flattened_specs = []
        for _, specs in pop_specs.items():
            flattened_specs.extend(specs)

        if len(flattened_specs) <= feature_budget:
            return pop_specs, {}

        joint_succ_conditions = (
            "The joint success conditions for this query is:\n"
        )
        for q_name, sem_cq in self.queries.items():
            joint_succ_conditions += f"\nQuery: {q_name}\n"
            for sem_pred in sem_cq.Ps:
                joint_succ_conditions += f"  - {sem_pred.succ_cond}\n"

        features_str = "\n".join(
            [
                (
                    f"  - {spec.target_col} ({spec.feature_type}): "
                    f"{spec.prompt[:100]}..."
                )
                for spec in flattened_specs
            ]
        )

        prompt = f"""You are given a set of features and need to select at most
{feature_budget} features that can best meet the joint success conditions.

=== Joint Success Conditions ===
{joint_succ_conditions}

=== Current Features ({len(flattened_specs)} total) ===
{features_str}

Your task is to identify which features can be eliminated while still meeting
the requirements. Consider:
1. Remove features that are duplicated or redundant
2. Remove features that are not important for meeting the success conditions
3. Keep the most discriminative and relevant features

Response format:
- Keep `to_add` as an empty list (do not add any new features)
- Put the feature names that can be eliminated in `to_remove`
  (target_col values only)

Please ensure you select exactly
{len(flattened_specs) - feature_budget} features to remove."""

        resp = cast(
            FeatureRefinementResponse,
            self.llm_client.invoke(
                is_remote=True,
                modality="Text",
                prompt=prompt,
                response_model=FeatureRefinementResponse,
            ),
        )
        usage_stat = self.llm_client.get_usage_statistics()
        self.llm_client.reset_usage_statistics()

        result = {}
        for q_name, specs in pop_specs.items():
            kept_specs = [
                spec for spec in specs if spec.target_col not in resp.to_remove
            ]
            result[q_name] = kept_specs

        return result, usage_stat

    @staticmethod
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

        schema_sample_section = ""
        if data_df is not None and not data_df.empty:
            columns_str = ", ".join(data_df.columns.tolist())
            schema_section = (
                f"=== Current Data Schema ===\nColumns: {columns_str}\n"
            )
            sample_rows = data_df.head(num_samples)
            sample_data_str = sample_rows.to_string(index=True)
            sample_section = (
                f"\n=== Sample Data (first {num_samples} rows) ===\n"
                f"{sample_data_str}\n"
            )
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
                INSTRUCTIONS_SECTION=(
                    "Propose an initial set of discriminative features "
                    "that can effectively distinguish positive from "
                    "negative samples."
                ),
                CONSTRAINTS_ADDITIONAL="",
            )

        current_features_str = "\n".join(
            [
                (
                    f"  - {spec.target_col} ({spec.feature_type}): "
                    f"{spec.prompt[:80]}..."
                )
                for spec in feature_space
            ]
        )
        previous_features_section = (
            f"\n\n=== Current Feature Space ===\n{current_features_str}\n"
        )

        if previous_feedback:
            importance_str = "\n".join(
                [
                    f"  - {feat}: {imp:.4f}"
                    for feat, imp in list(
                        previous_feedback["feature_importance"].items()
                    )[:10]
                ]
            )
            performance_feedback_section = f"""
\n=== Performance Feedback ===
\n1. Feature Importance (sorted by importance):\n{importance_str}
\n2. F1 Score:\n   - With current features: {previous_feedback["f1"]:.4f}\n
"""
            constraints = (
                '- Make sure "to_remove" contains EXACTLY the feature names '
                'from the current feature space (check the "Current Feature '
                'Space" section above).\n'
                "- Avoid proposing features that are already in the current "
                "feature space."
            )
        else:
            performance_feedback_section = ""
            constraints = (
                "- Avoid proposing features that are already in the current "
                "feature space."
            )

        return prompt_template.format(
            MODALITY=sem_pred.modality,
            DESC=sem_pred.prompt,
            SOURCE_COL=sem_pred.field,
            FEATURE_BUDGET=feature_budget,
            SCHEMA_SAMPLE_SECTION=schema_sample_section,
            PREVIOUS_FEATURES_SECTION=previous_features_section,
            PERFORMANCE_FEEDBACK_SECTION=performance_feedback_section,
            INSTRUCTIONS_SECTION=(
                "Propose features that will improve classification "
                "performance based on the feedback above."
            ),
            CONSTRAINTS_ADDITIONAL=constraints,
        )

    @staticmethod
    def _pred_and_eval(df: pd.DataFrame, labels: pd.Series) -> dict:
        logger.info(
            "Training enriched classifier with %s features.", len(df.columns)
        )
        df_proc = encode_features(df)

        clf = DecisionTreeClassifier(
            max_depth=3,
            min_samples_leaf=3,
            min_samples_split=3,
            class_weight="balanced",
            random_state=42,
        )
        clf.fit(df_proc, labels)

        preds = clf.predict(df_proc)
        f1 = f1_score(labels, preds, zero_division=0)
        feature_importance = dict(
            zip(df.columns, clf.feature_importances_, strict=True)
        )
        feature_importance = dict(
            sorted(
                feature_importance.items(),
                key=lambda item: item[1],
                reverse=True,
            )
        )

        misclassified_mask = labels != preds
        proba = clf.predict_proba(df_proc)
        if len(np.unique(labels)) == 1:
            pred_proba = np.ones(len(df_proc)) * (1 if clf.classes_[0] else 0)
        else:
            pred_proba = np.asarray(proba)[:, 1]
        bad_cases = df[misclassified_mask].copy()

        bad_cases["_original_index"] = bad_cases.index
        bad_cases["_true_label"] = labels[misclassified_mask].values
        bad_cases["_predicted_label"] = preds[misclassified_mask]
        bad_cases["_pred_proba"] = pred_proba[misclassified_mask]
        bad_cases["_uncertainty"] = abs(bad_cases["_pred_proba"] - 0.5)
        bad_cases = bad_cases.sort_values("_uncertainty", ascending=True)
        bad_cases = bad_cases.drop(columns=["_uncertainty"])

        logger.info("Enriched F1: %.4f", f1)
        logger.info(
            "Bad cases: %s / %s samples misclassified",
            misclassified_mask.sum(),
            len(labels),
        )

        return {
            "f1": f1,
            "feature_importance": feature_importance,
            "bad_cases": bad_cases,
        }


class ContrastiveSampleIterator:
    def __init__(
        self,
        pos_samples_idx: list,
        neg_samples_idx: list,
        batch_size: int,
        max_iters: int = 1,
    ):
        self.pos_samples_idx = pos_samples_idx
        self.neg_samples_idx = neg_samples_idx
        self.batch_size = batch_size
        self.max_iters = max_iters
        self.num_pos_batches = (
            len(pos_samples_idx) + batch_size - 1
        ) // batch_size
        self.num_neg_batches = (
            len(neg_samples_idx) + batch_size - 1
        ) // batch_size

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

    def next_batch(self) -> tuple[list, list]:
        assert self.has_next_batch(), "No more batches available."

        pos_batch_data, neg_batch_data = [], []

        if self._has_next_pos_batch():
            pos_start = self.pos_ptr
            pos_end = min(
                self.pos_ptr + self.batch_size, len(self.pos_samples_idx)
            )
            pos_batch_data = self.pos_samples_idx[pos_start:pos_end]
            self.pos_ptr = pos_end

        if self._has_next_neg_batch():
            neg_start = self.neg_ptr
            neg_end = min(
                self.neg_ptr + self.batch_size, len(self.neg_samples_idx)
            )
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
    ) -> tuple[list, list]:
        data_items = pos_batch_data + neg_batch_data
        metadata = [
            {"label": True, "sample_id": int(i)} for i in pos_batch_indices
        ] + [{"label": False, "sample_id": int(i)} for i in neg_batch_indices]

        # Add bad cases
        if previous_feedback and not previous_feedback["bad_cases"].empty:
            for _, row in previous_feedback["bad_cases"].head(5).iterrows():
                field_content = labeled_data_df.loc[
                    int(row["_original_index"]), sem_pred.field
                ]
                data_items.append(field_content)
                metadata.append(
                    {
                        "label": bool(row["_true_label"]),
                        "sample_id": int(row["_original_index"]),
                        "is_bad_case": True,
                        "misclassified_as": bool(row["_predicted_label"]),
                    }
                )

        return data_items, metadata

    def _has_next_pos_batch(self) -> bool:
        return (
            self.iter_num < self.max_iters
            and self.iter_num < self.num_pos_batches
        )

    def _has_next_neg_batch(self) -> bool:
        return (
            self.iter_num < self.max_iters
            and self.iter_num < self.num_neg_batches
        )
