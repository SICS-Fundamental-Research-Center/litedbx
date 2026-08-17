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
    annotation_sample_weights,
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
    annotation_weights: pd.Series


class CoresetStore(dict[str, CoresetRecord]):
    """
    Coresets keyed by query name.

    Basic schema:
    {
        "Q1": CoresetRecord,
        ...<queries>
    }
    """

    async def acquire_annotation(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        queries: dict[str, SemCQ],
        sigma_satisfied_data: SigmaSatisfiedData,
        llm_client: LdbLLMClient,
        ckpt_root: Path,
        b_lab: int,
        pseudo_ckpt_root: Path | None = None,
        seed: int = 42,
        use_hitl: bool = True,
        enable_cache: bool = True,
    ) -> dict[str, AnnotationSelection]:
        """Acquire query annotations and initialize each query coreset."""
        selection_dict = {}
        for q_name in queries:
            selection_dict[q_name] = await self.acquire_query_annotation(
                q_name=q_name,
                b_lab=b_lab,
                sigma_satisfied_data=sigma_satisfied_data,
                queries=queries,
                llm_client=llm_client,
                ckpt_root=ckpt_root,
                pseudo_ckpt_root=pseudo_ckpt_root,
                stream_idx=0,
                seed=seed,
                use_hitl=use_hitl,
                enable_cache=enable_cache,
            )
        return selection_dict

    async def init_coreset(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        queries: dict[str, SemCQ],
        sigma_satisfied_data: SigmaSatisfiedData,
        annotation_selections: dict[str, AnnotationSelection],
        complete_config: dict,
        llm_client: LdbLLMClient,
        ckpt_root: Path,
        pseudo_ckpt_root: Path | None = None,
        use_hitl: bool = True,
        enable_cache: bool = True,
    ) -> None:
        for q_name in queries:
            await self.init_query_coreset(
                q_name=q_name,
                sigma_satisfied_data=sigma_satisfied_data,
                complete_config=complete_config,
                annotation_selection=annotation_selections[q_name],
                queries=queries,
                llm_client=llm_client,
                ckpt_root=ckpt_root,
                pseudo_ckpt_root=pseudo_ckpt_root,
                stream_idx=0,
                use_hitl=use_hitl,
                enable_cache=enable_cache,
            )

    async def acquire_query_annotation(  # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals,unused-argument
        self,
        q_name: str,
        b_lab: int,
        sigma_satisfied_data: SigmaSatisfiedData,
        queries: dict[str, SemCQ],
        llm_client: LdbLLMClient,
        ckpt_root: Path,
        pseudo_ckpt_root: Path | None = None,
        stream_idx: int = 0,
        seed: int = 42,
        use_hitl: bool = True,
        enable_cache: bool = True,
    ) -> AnnotationSelection:
        """Acquire labels for one query."""
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
                enable_cache=enable_cache,
            )

        labeling_budget = _clamp_labeling_budget(
            b_lab, len(data), q_name, stream_idx
        )
        diversity_columns = list(
            dict.fromkeys(
                predicate.field
                for predicate in queries[q_name].Ps
                if predicate.modality in {"Text", "VectorText"}
            )
        )
        semantic_modalities = {"Text", "Image", "VectorText", "VectorImage"}
        has_semantic_proxy = bool(queries[q_name].Ps) and all(
            predicate.modality in semantic_modalities
            for predicate in queries[q_name].Ps
        )
        annotation_strategy = automatic_annotation_strategy(
            diversity_columns=diversity_columns,
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
                    enable_cache=enable_cache,
                )
            )
        annotation_selection = _select_annotation_sample(
            data=data,
            pseudo_labels=pseudo_labels,
            labeling_budget=labeling_budget,
            strategy=annotation_strategy,
            seed=seed,
            diversity_columns=diversity_columns,
        )
        retry_seed = 0
        while (
            labels.loc[annotation_selection.indices].nunique() < 2
            and retry_seed <= 42
        ):
            logger.warning(
                "Query '%s' in stream-%s has only one class after labeling "
                "Retrying with a new seed %s.",
                q_name,
                stream_idx,
                retry_seed,
            )
            annotation_selection = _select_annotation_sample(
                data=data,
                pseudo_labels=pseudo_labels,
                labeling_budget=labeling_budget,
                strategy=annotation_strategy,
                seed=retry_seed,
                diversity_columns=diversity_columns,
            )
            retry_seed += 1
        return annotation_selection

    async def init_query_coreset(  # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals,unused-argument
        self,
        q_name: str,
        sigma_satisfied_data: SigmaSatisfiedData,
        complete_config: dict,
        annotation_selection: AnnotationSelection,
        queries: dict[str, SemCQ],
        llm_client: LdbLLMClient,
        ckpt_root: Path,
        pseudo_ckpt_root: Path | None = None,
        stream_idx: int = 0,
        use_hitl: bool = True,
        enable_cache: bool = True,
    ) -> None:
        """Move samples into its coreset based on the annotation selection."""
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
                enable_cache=enable_cache,
            )

        labeled_indices = annotation_selection.indices
        annotation_weights = annotation_sample_weights(
            population_size=len(data), selection=annotation_selection
        )

        remaining_indices = data.index.difference(labeled_indices)
        estimated_selectivity = _estimate_selectivity(
            acquired_labels=acquired_labels,
            selection=annotation_selection,
        )
        self.upsert(
            q_name=q_name,
            data=data,
            acquired_labels=acquired_labels,
            labeled_indices=labeled_indices,
            complete_config=complete_config,
            estimated_selectivity=estimated_selectivity,
            annotation_weights=annotation_weights,
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
        num_pos = acquired_labels.loc[labeled_indices].sum()
        num_neg = len(labeled_indices) - num_pos
        if num_pos * num_neg == 0:
            raise ValueError(
                f"Query '{q_name}' in stream-{stream_idx} has only one class "
                f"after labeling: {num_pos} pos / {num_neg} neg. "
                "Please ensure that the labeling budget is sufficient to "
                "capture both classes."
            )
        logger.info(
            "Initialized coreset for query '%s' in stream-%s with %s "
            "labeled samples (%s pos / %s neg) "
            "Remaining Sigma-satisfied data has %s samples.",
            q_name,
            stream_idx,
            len(labeled_indices),
            num_pos,
            num_neg,
            len(remaining_indices),
        )
        while True:
            pass

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
        if enable_cache and context_path.exists():
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

        if enable_cache:
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
        annotation_weights: pd.Series | None = None,
    ) -> None:
        """Create or extend a query coreset."""
        if annotation_weights is None:
            annotation_weights = pd.Series(1.0, index=labeled_indices)
        local_weights = annotation_weights.reindex(labeled_indices)
        if local_weights.isna().any():
            raise ValueError(
                "Annotation weights must cover every released row."
            )
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
                "annotation_weights": local_weights.reset_index(
                    drop=True
                ).copy(),
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
        self[q_name]["annotation_weights"] = pd.concat(
            [
                self[q_name]["annotation_weights"],
                local_weights.reset_index(drop=True).copy(),
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
            prompt=_annotation_proxy_prompt(sem_pred.prompt),
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


def _estimate_selectivity(
    acquired_labels: pd.Series,
    selection: AnnotationSelection,
) -> float | None:
    """Estimate prevalence from released annotations and sampling weights."""

    BETA_alpha, BETA_beta = 1, 1

    if len(selection.strata) != 2:
        logger.warning(
            "Selectivity estimation requires exactly two strata: "
            f"positive and negative. Found {len(selection.strata)}."
        )
        return None

    pos_stratum, neg_stratum = selection.strata[0], selection.strata[1]
    est_selectivity = pos_stratum.population_size / (
        pos_stratum.population_size + neg_stratum.population_size
    )

    pseudo_pos_indices = pos_stratum.anchor_indices.append(
        pos_stratum.random_indices
    )
    pseudo_neg_indices = neg_stratum.anchor_indices.append(
        neg_stratum.random_indices
    )

    acquired_pos_labels = _coerce_bool_labels(
        acquired_labels.loc[pseudo_pos_indices], "acquired pos labels"
    )
    acquired_neg_labels = _coerce_bool_labels(
        acquired_labels.loc[pseudo_neg_indices], "acquired neg labels"
    )

    pos_true_posterier = (BETA_alpha + acquired_pos_labels.sum()) / (
        BETA_alpha + BETA_beta + len(acquired_pos_labels)
    )
    neg_true_posterier = (BETA_alpha + acquired_neg_labels.sum()) / (
        BETA_alpha + BETA_beta + len(acquired_neg_labels)
    )

    rectified_selectivity = (
        est_selectivity * pos_true_posterier
        + (1 - est_selectivity) * neg_true_posterier
    )

    return rectified_selectivity


def _annotation_proxy_prompt(task_prompt: str) -> str:
    """Apply a universal conservative standard to annotation proxies."""
    return (
        "Evaluate one input using the semantic task below. Return true only "
        "when the input directly provides sufficient evidence for the "
        "requested condition. A merely related or nonspecific observation "
        "is not sufficient. Do not introduce criteria not stated by the "
        "task.\n\nSemantic task:\n" + task_prompt
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
        "cache_schema": 7,
        "proxy_prompt_schema": 1,
        "remote_models": llm_client.config.get("REMOTE_MODELS", {}),
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
