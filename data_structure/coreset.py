"""Coreset container and annotation operations."""

import logging
from pathlib import Path
from typing import TypedDict

import pandas as pd

from llm import LdbLLMClient

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
        seed: int = 42,
        use_hitl: bool = True,
    ) -> None:
        """Acquire query annotations and initialize each query coreset."""
        for q_name in queries:
            await self.acquire_query_annotation_and_init(
                q_name=q_name,
                b_lab=b_lab,
                sigma_satisfied_data=sigma_satisfied_data,
                complete_config=complete_config,
                queries=queries,
                llm_client=llm_client,
                ckpt_root=ckpt_root,
                stream_idx=0,
                seed=seed,
                use_hitl=use_hitl,
            )

    async def acquire_query_annotation_and_init(  # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
        self,
        q_name: str,
        b_lab: int,
        sigma_satisfied_data: SigmaSatisfiedData,
        complete_config: dict,
        queries: dict[str, SemCQ],
        llm_client: LdbLLMClient,
        ckpt_root: Path,
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
                ckpt_root=ckpt_root,
                q_name=q_name,
                stream_idx=stream_idx,
            )

        labeling_budget = _clamp_labeling_budget(
            b_lab, len(data), q_name, stream_idx
        )
        labeled_indices = data.sample(
            n=labeling_budget, random_state=seed
        ).index
        labeled_indices = _include_missing_minority_classes(
            acquired_labels=acquired_labels,
            labeled_indices=labeled_indices,
            b_lab=b_lab,
            labeling_budget=labeling_budget,
            q_name=q_name,
            stream_idx=stream_idx,
        )

        remaining_indices = data.index.difference(labeled_indices)
        self.upsert(
            q_name=q_name,
            data=data,
            acquired_labels=acquired_labels,
            labeled_indices=labeled_indices,
            complete_config=complete_config,
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
            "labeled samples. Remaining Sigma-satisfied data has %s samples.",
            q_name,
            stream_idx,
            len(labeled_indices),
            len(remaining_indices),
        )

    async def sync_features(  # pylint: disable=too-many-arguments,too-many-positional-arguments
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
        if enable_cache and ckpt_path.exists():
            logger.debug(
                "Loading enriched coreset for query '%s' from cache.", q_name
            )
            cached_df = pd.read_csv(ckpt_path)
            label_count = len(self[q_name]["labels"])
            if len(cached_df) == label_count:
                self[q_name]["ldb_data"].df = cached_df
                return llm_client.get_usage_statistics()

            logger.warning(
                "Ignoring cached enriched coreset for query '%s' because "
                "row count %s does not match label count %s.",
                q_name,
                len(cached_df),
                label_count,
            )

        if q_name not in enriched_features:
            raise ValueError(
                f"Enriched features for query '{q_name}' not found"
            )

        await self[q_name]["ldb_data"].sync_with_enriched_features(
            enriched_features=enriched_features[q_name],
            llm_client=llm_client,
            is_remote=is_remote,
        )

        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        self[q_name]["ldb_data"].df.to_csv(ckpt_path, index=False)

        llm_usage_statistics = llm_client.get_usage_statistics()
        llm_client.reset_usage_statistics()
        return llm_usage_statistics

    def upsert(
        self,
        q_name: str,
        data: pd.DataFrame,
        acquired_labels: pd.Series,
        labeled_indices: pd.Index,
        complete_config: dict,
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
    ckpt_path = ckpt_root / q_name / f"pseudo_labels_{stream_idx}.csv"
    if enable_cache and ckpt_path.exists():
        logger.debug("Loading pseudo labels generated by LLM from cache.")
        return pd.read_csv(ckpt_path, index_col=0).iloc[:, 0]

    data = sigma_satisfied_data[stream_idx][q_name]["ldb_data"]
    semcq = queries[q_name]

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
    if labeling_budget < data_size:
        return labeling_budget

    logger.warning(
        "Requested labeled budget %s exceeds the data scale %s for query '%s' "
        "in stream-%s.",
        labeling_budget,
        data_size,
        q_name,
        stream_idx,
    )
    adjusted_budget = data_size // 2
    logger.debug(
        "Adjusted labeled budget to %s for query '%s' in stream-%s.",
        adjusted_budget,
        q_name,
        stream_idx,
    )
    return adjusted_budget


def _include_missing_minority_classes(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    acquired_labels: pd.Series,
    labeled_indices: pd.Index,
    b_lab: int,
    labeling_budget: int,
    q_name: str,
    stream_idx: int,
) -> pd.Index:
    """Add minority examples when the sampled labels have one class."""
    del b_lab, labeling_budget
    num_pos_sampled = acquired_labels.loc[labeled_indices].sum()
    num_neg_sampled = len(labeled_indices) - num_pos_sampled
    minority_bias = max(min(num_pos_sampled - 1, num_neg_sampled - 1, 2), 0)

    if num_pos_sampled == 0:
        pos_indices = acquired_labels[acquired_labels].index
        pos_to_add = min(minority_bias, len(pos_indices))
        labeled_indices = labeled_indices.union(pos_indices[:pos_to_add])
        logger.info(
            "Minority class (positive) is not sampled for query '%s' "
            "in stream-%s. Added %s positive samples. Current labeled "
            "set has %s pos samples out of %s samples.",
            q_name,
            stream_idx,
            pos_to_add,
            acquired_labels.loc[labeled_indices].sum(),
            len(labeled_indices),
        )
    if num_neg_sampled == 0:
        neg_indices = acquired_labels[~acquired_labels].index
        neg_to_add = min(minority_bias, len(neg_indices))
        labeled_indices = labeled_indices.union(neg_indices[:neg_to_add])
        logger.info(
            "Minority class (negative) is not sampled for query '%s' "
            "in stream-%s. Added %s negative samples. Current labeled "
            "set has %s neg samples out of %s samples.",
            q_name,
            stream_idx,
            neg_to_add,
            len(labeled_indices) - acquired_labels.loc[labeled_indices].sum(),
            len(labeled_indices),
        )
    return labeled_indices
