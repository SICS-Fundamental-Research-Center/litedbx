# pylint: disable=duplicate-code
"""Coreset container and annotation operations."""

import hashlib
import json
import logging
from pathlib import Path
from typing import TypedDict

import numpy as np
import pandas as pd

from llm import LdbLLMClient

from .annotation_sampling import (
    AnnotationSelection,
    automatic_annotation_strategy,
)
from .annotation_sampling import (
    coerce_bool_labels as _coerce_bool_labels,
)
from .annotation_sampling import (
    select_annotation_sample as _select_annotation_sample,
)
from .ldb_data import LdbData
from .llm_resp_templates import PopulationSpec
from .sem_query import SemCQ
from .sigma_satisfied_data import SigmaSatisfiedData

logger = logging.getLogger(__name__)


class CoresetRecord(TypedDict):
    """
    Record for one query coreset.

    `lb` and `ub` indicate the confidence bounds for coreset maintenance.
    """

    ldb_data: LdbData
    labels: pd.Series
    observed_size: int
    lb: float
    ub: float
    estimated_selectivity: float | None


class CoresetStore(dict[str, CoresetRecord]):
    """
    Coresets keyed by query name.

    Basic schema:
    {
        "Q1": CoresetRecord,
        ...<queries>
    }
    """

    async def acquire_annotation_and_init(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        queries: dict[str, SemCQ],
        sigma_satisfied_data: SigmaSatisfiedData,
        complete_config: dict,
        llm_client: LdbLLMClient,
        ckpt_root: Path,
        b_lab: int,
        feature_spaces: dict[str, list[PopulationSpec]],
        pseudo_ckpt_root: Path | None = None,
        seed: int = 42,
        use_hitl: bool = True,
    ) -> None:
        """Acquire query annotations and initialize each query coreset."""
        for q_name in queries:
            await self.acquire_query_annotation_and_init(
                q_name=q_name,
                b_lab=b_lab,
                feature_space=feature_spaces[q_name],
                sigma_satisfied_data=sigma_satisfied_data,
                complete_config=complete_config,
                queries=queries,
                llm_client=llm_client,
                ckpt_root=ckpt_root,
                pseudo_ckpt_root=pseudo_ckpt_root,
                stream_idx=0,
                seed=seed,
                use_hitl=use_hitl,
            )

    async def acquire_query_annotation_and_init(  # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
        self,
        q_name: str,
        b_lab: int,
        feature_space: list[PopulationSpec],
        sigma_satisfied_data: SigmaSatisfiedData,
        complete_config: dict,
        queries: dict[str, SemCQ],
        llm_client: LdbLLMClient,
        ckpt_root: Path,
        pseudo_ckpt_root: Path | None = None,
        stream_idx: int = 0,
        seed: int = 42,
        use_hitl: bool = True,
    ) -> None:
        """Acquire labels for one query and move samples into its coreset."""
        record = sigma_satisfied_data[stream_idx][q_name]
        data = record["ldb_data"].df.copy()
        labels = record["labels"]
        if labels is None:
            raise ValueError(
                f"Labels for query '{q_name}' in stream-{stream_idx} "
                "have not been initialized."
            )

        if use_hitl:
            acquired_labels = labels
        else:
            acquired_labels = await acquire_pseudo_labels_by_llm(
                sigma_satisfied_data=sigma_satisfied_data,
                queries=queries,
                llm_client=llm_client,
                ckpt_root=pseudo_ckpt_root or ckpt_root,
                q_name=q_name,
                stream_idx=stream_idx,
            )

        labeling_budget = _clamp_labeling_budget(
            b_lab, len(data), q_name, stream_idx
        )
        semantic_modalities = {"Text", "Image", "VectorText", "VectorImage"}
        has_semantic_proxy = bool(queries[q_name].Ps) and all(
            predicate.modality in semantic_modalities
            for predicate in queries[q_name].Ps
        )
        annotation_strategy = automatic_annotation_strategy(
            has_semantic_proxy=has_semantic_proxy,
        )
        pseudo_labels = None
        if annotation_strategy != "random":
            pseudo_labels = (
                acquired_labels
                if not use_hitl
                else await acquire_pseudo_labels_by_llm(
                    sigma_satisfied_data=sigma_satisfied_data,
                    queries=queries,
                    llm_client=llm_client,
                    ckpt_root=pseudo_ckpt_root or ckpt_root,
                    q_name=q_name,
                    stream_idx=stream_idx,
                )
            )
        if pseudo_labels is not None:
            _attach_single_predicate_proxy_feature(
                data=data,
                query=queries[q_name],
                feature_space=feature_space,
                proxy_labels=pseudo_labels,
            )
        annotation_selection = _select_annotation_sample(
            data=data,
            pseudo_labels=pseudo_labels,
            labeling_budget=labeling_budget,
            strategy=annotation_strategy,
            seed=seed,
        )
        labeled_indices = annotation_selection.indices

        remaining_indices = data.index.difference(labeled_indices)
        estimated_selectivity = _estimate_selectivity(
            acquired_labels=acquired_labels,
            labeled_indices=labeled_indices,
            pseudo_labels=pseudo_labels,
            sampling_design=annotation_selection,
        )
        self.upsert(
            q_name=q_name,
            data=data,
            acquired_labels=acquired_labels,
            labeled_indices=labeled_indices,
            complete_config=complete_config,
            estimated_selectivity=estimated_selectivity,
        )
        update_sigma_after_labeling(
            sigma_satisfied_data=sigma_satisfied_data,
            q_name=q_name,
            stream_idx=stream_idx,
            data=data,
            labels=labels,
            labeled_indices=labeled_indices,
            remaining_indices=remaining_indices,
        )
        logger.info(
            "Initialized coreset for query '%s' in stream-%s with %s "
            "labeled samples (%s pos / %s neg) "
            "Remaining Sigma-satisfied data has %s samples.",
            q_name,
            stream_idx,
            len(labeled_indices),
            acquired_labels.loc[labeled_indices].sum(),
            len(labeled_indices) - acquired_labels.loc[labeled_indices].sum(),
            len(remaining_indices),
        )

    async def sync_features(  # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
        self,
        q_name: str,
        enriched_features: dict[str, list[PopulationSpec]],
        llm_client: LdbLLMClient,
        ckpt_root: Path,
        tag: str = "",
        enable_cache: bool = True,
        is_remote: bool = False,
    ) -> dict:
        """Synchronize enriched features for one query coreset."""
        ckpt_path = ckpt_root / q_name / f"coreset_{tag}.csv"
        if q_name not in enriched_features:
            raise ValueError(
                f"Enriched features for query '{q_name}' not found"
            )
        ldb_data = self[q_name]["ldb_data"]
        expected_columns = ldb_data.expected_enriched_columns(
            enriched_features[q_name]
        )
        context_path = ckpt_path.with_suffix(".context.json")
        context_key = ldb_data.feature_materialization_context_key(
            enriched_features[q_name], llm_client, is_remote
        )
        cached_context_key = None
        if context_path.exists():
            with context_path.open(encoding="utf-8") as context_file:
                cached_context_key = json.load(context_file).get("key")
        if (
            enable_cache
            and ckpt_path.exists()
            and cached_context_key == context_key
        ):
            logger.debug(
                "Loading enriched coreset for query '%s' from cache.", q_name
            )
            cached_df = pd.read_csv(ckpt_path)
            label_count = len(self[q_name]["labels"])
            cache_schema = set(cached_df.columns)
            cache_compatible = ldb_data.reuse_cached_features(cached_df)
            if cache_compatible and cache_schema == expected_columns:
                return llm_client.get_usage_statistics()
            if cache_compatible:
                logger.info(
                    "Reusing compatible coreset feature cache for query %s; "
                    "missing columns=%s, obsolete columns=%s.",
                    q_name,
                    sorted(expected_columns - cache_schema),
                    sorted(cache_schema - expected_columns),
                )
            else:
                logger.warning(
                    "Ignoring incompatible coreset cache for query %s: "
                    "rows=%s (expected %s).",
                    q_name,
                    len(cached_df),
                    label_count,
                )

        await ldb_data.sync_with_enriched_features(
            enriched_features=enriched_features[q_name],
            llm_client=llm_client,
            is_remote=is_remote,
        )

        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        ldb_data.df.to_csv(ckpt_path, index=False)
        with context_path.open("w", encoding="utf-8") as context_file:
            json.dump({"key": context_key}, context_file, indent=2)

        llm_usage_statistics = llm_client.get_usage_statistics()
        llm_client.reset_usage_statistics()
        return llm_usage_statistics

    def upsert(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        q_name: str,
        data: pd.DataFrame,
        acquired_labels: pd.Series,
        labeled_indices: pd.Index,
        complete_config: dict,
        estimated_selectivity: float | None = None,
    ) -> None:
        """Create or extend a query coreset."""
        if q_name not in self:
            self[q_name] = {
                "ldb_data": LdbData(
                    df=data.loc[labeled_indices].reset_index(drop=True).copy(),
                    config=complete_config,
                ),
                "labels": acquired_labels.loc[labeled_indices]
                .reset_index(drop=True)
                .copy(),
                "observed_size": len(labeled_indices),
                "lb": float("inf"),
                "ub": float("-inf"),
                "estimated_selectivity": estimated_selectivity,
            }
            return

        logger.debug("Extending existing coreset for query '%s'.", q_name)
        self[q_name]["ldb_data"].df = pd.concat(
            [
                self[q_name]["ldb_data"].df,
                data.loc[labeled_indices].reset_index(drop=True).copy(),
            ],
            ignore_index=True,
        )
        self[q_name]["labels"] = pd.concat(
            [
                self[q_name]["labels"],
                acquired_labels.loc[labeled_indices]
                .reset_index(drop=True)
                .copy(),
            ],
            ignore_index=True,
        )
        self[q_name]["observed_size"] += len(labeled_indices)
        if estimated_selectivity is not None:
            self[q_name]["estimated_selectivity"] = estimated_selectivity


async def acquire_pseudo_labels_by_llm(  # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    sigma_satisfied_data: SigmaSatisfiedData,
    queries: dict[str, SemCQ],
    llm_client: LdbLLMClient,
    ckpt_root: Path,
    q_name: str,
    stream_idx: int = 0,
    enable_cache: bool = True,
) -> pd.Series:
    """Acquire pseudo-labels for one query using the configured LLM."""
    data = sigma_satisfied_data[stream_idx][q_name]["ldb_data"]
    semcq = queries[q_name]
    cache_key = _pseudo_label_cache_key(data.df, semcq, llm_client)
    ckpt_path = (
        ckpt_root / q_name / f"pseudo_labels_{stream_idx}_{cache_key}.csv"
    )
    if enable_cache and ckpt_path.exists():
        logger.debug("Loading pseudo labels generated by LLM from cache.")
        return pd.read_csv(ckpt_path, index_col=0).iloc[:, 0]
    spec_labels = []
    for idx, sem_pred in enumerate(semcq.Ps):
        spec = PopulationSpec(
            source_col=sem_pred.field,
            source_modality=sem_pred.modality,  # type: ignore
            target_col=f"llm_label_{idx}",
            prompt=sem_pred.prompt,
            feature_type="bool",
        )
        spec_label = await data.sem_map(
            spec=spec, llm_client=llm_client, is_remote=False
        )
        spec_labels.append(spec_label)

    if not spec_labels:
        raise ValueError("Empty SemPredicates")

    result = spec_labels[0]
    for label in spec_labels[1:]:
        result = result & label

    if enable_cache:
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(ckpt_path, index=True, header=True)

    return result


def update_sigma_after_labeling(  # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    sigma_satisfied_data: SigmaSatisfiedData,
    q_name: str,
    stream_idx: int,
    data: pd.DataFrame,
    labels: pd.Series,
    labeled_indices: pd.Index,
    remaining_indices: pd.Index,
) -> None:
    """Remove labeled data from Sigma data and update audit buckets."""
    record = sigma_satisfied_data[stream_idx][q_name]
    record["ldb_data"].df = (
        data.loc[remaining_indices].reset_index(drop=True).copy()
    )
    record["labels"] = (
        labels.loc[remaining_indices].reset_index(drop=True).copy()
    )

    pos_labeled_indices = labeled_indices[labels.loc[labeled_indices]]
    neg_labeled_indices = labeled_indices[~labels.loc[labeled_indices]]
    record["selected_data"] = pd.concat(
        [
            record["selected_data"],
            data.loc[pos_labeled_indices].reset_index(drop=True).copy(),
        ],
        ignore_index=True,
    )
    record["selected_labels"] = pd.concat(
        [
            record["selected_labels"],
            labels.loc[pos_labeled_indices].reset_index(drop=True).copy(),
        ],
        ignore_index=True,
    )
    record["discarded_data"] = pd.concat(
        [
            record["discarded_data"],
            data.loc[neg_labeled_indices].reset_index(drop=True).copy(),
        ],
        ignore_index=True,
    )
    record["discarded_labels"] = pd.concat(
        [
            record["discarded_labels"],
            labels.loc[neg_labeled_indices].reset_index(drop=True).copy(),
        ],
        ignore_index=True,
    )


def _clamp_labeling_budget(
    labeling_budget: int, data_size: int, q_name: str, stream_idx: int
) -> int:
    """Clamp a requested labeling budget to fit available data."""
    if labeling_budget < 0:
        raise ValueError("Labeling budget cannot be negative.")
    if labeling_budget <= data_size:
        return labeling_budget

    logger.warning(
        "Requested labeled budget %s exceeds the data scale %s for query '%s' "
        "in stream-%s.",
        labeling_budget,
        data_size,
        q_name,
        stream_idx,
    )
    adjusted_budget = data_size
    logger.debug(
        "Adjusted labeled budget to %s for query '%s' in stream-%s.",
        adjusted_budget,
        q_name,
        stream_idx,
    )
    return adjusted_budget


def _post_stratified_selectivity(
    labels: pd.Series,
    local_labels: pd.Series,
    labeled_indices: pd.Index,
) -> float | None:
    """Estimate prevalence after sampling within local-model strata."""
    estimate = 0.0
    for local_value in (False, True):
        stratum_indices = local_labels[local_labels == local_value].index
        if len(stratum_indices) == 0:
            continue
        sampled_indices = labeled_indices.intersection(stratum_indices)
        if len(sampled_indices) == 0:
            return None
        stratum_weight = len(stratum_indices) / len(local_labels)
        estimate += stratum_weight * float(labels.loc[sampled_indices].mean())
    return estimate


def _design_based_selectivity(
    labels: pd.Series,
    selection: AnnotationSelection,
) -> float | None:
    """Estimate prevalence for a uniform annotation sample."""
    if selection.mode == "uniform":
        return float(labels.loc[selection.indices].mean())
    return None


def _estimate_selectivity(
    acquired_labels: pd.Series,
    labeled_indices: pd.Index,
    pseudo_labels: pd.Series | None,
    sampling_design: AnnotationSelection | None = None,
) -> float | None:
    """Estimate prevalence from released annotations and sampling design."""
    annotated_labels = _coerce_bool_labels(
        acquired_labels.loc[labeled_indices], "acquired labels"
    )
    if pseudo_labels is None:
        return float(annotated_labels.mean())

    local_labels = _coerce_bool_labels(
        pseudo_labels.reindex(acquired_labels.index),
        "local pseudo labels",
    )
    design_estimate = (
        _design_based_selectivity(acquired_labels, sampling_design)
        if sampling_design is not None
        else None
    )
    if design_estimate is not None:
        return design_estimate
    return _post_stratified_selectivity(
        labels=annotated_labels,
        local_labels=local_labels,
        labeled_indices=labeled_indices,
    )


def _attach_single_predicate_proxy_feature(
    data: pd.DataFrame,
    query: SemCQ,
    feature_space: list[PopulationSpec],
    proxy_labels: pd.Series,
) -> None:
    """Reuse a text acquisition proxy as its exact semantic feature."""
    if len(query.Ps) != 1:
        return
    predicate = query.Ps[0]
    normalized_prompt = " ".join(predicate.prompt.split()).casefold()
    matches = [
        spec
        for spec in feature_space
        if spec.source_col == predicate.field
        and spec.feature_type == "bool"
        and " ".join(spec.prompt.split()).casefold() == normalized_prompt
    ]
    if len(matches) == 1:
        data[matches[0].target_col] = proxy_labels.reindex(data.index).astype(
            bool
        )


def _pseudo_label_cache_key(
    data: pd.DataFrame, semcq: SemCQ, llm_client: LdbLLMClient
) -> str:
    """Build a stable cache key independent of budget and expansion policy."""
    source_cols = list(dict.fromkeys(pred.field for pred in semcq.Ps))
    row_hash = np.asarray(
        pd.util.hash_pandas_object(data[source_cols], index=True),
        dtype=np.uint64,
    ).tobytes()
    payload = {
        "cache_schema": 6,
        "local_models": llm_client.config.get("LOCAL_MODELS", {}),
        "inference": {
            key: llm_client.config.get(key)
            for key in (
                "max_tokens",
                "top_p",
                "temperature",
                "random_seed",
            )
        },
        "predicates": [
            {
                "field": pred.field,
                "modality": pred.modality,
                "prompt": pred.prompt,
            }
            for pred in semcq.Ps
        ],
        "rows": len(data),
        "row_hash": hashlib.sha1(row_hash).hexdigest(),
    }
    serialized = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha1(serialized.encode("utf-8")).hexdigest()[:16]
