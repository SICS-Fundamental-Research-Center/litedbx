"""Sigma-satisfied data container and operations."""

import logging
from pathlib import Path
from typing import TypedDict

import pandas as pd

from llm import LdbLLMClient

from .ldb_data import LdbData
from .llm_resp_templates import PopulationSpec
from .sem_query import Predicate, SemCQ

logger = logging.getLogger(__name__)


class SigmaRecord(TypedDict):
    """Record for one query's Sigma-satisfied data in one stream."""

    ldb_data: LdbData
    labels: pd.Series | None
    propagated_labels: pd.Series | None
    selected_data: pd.DataFrame
    selected_labels: pd.Series
    discarded_data: pd.DataFrame
    discarded_labels: pd.Series


class SigmaSatisfiedData(list[dict[str, SigmaRecord]]):
    """
    Sigma-filtered data across streams and queries.

    Basic schema:
    [
        {
            "Q1": SigmaRecord,
            ...<queries>
        },
        ...<streams>
    ]
    """

    def initialize(
        self,
        data_stream: list[LdbData],
        queries: dict[str, SemCQ],
        complete_config: dict,
        data_dir: str,
    ) -> None:
        """Retrieve Sigma-satisfied data and build ground-truth labels."""
        logger.info("Start Sigma-satisfied data retrieval.")

        if len(self) > 0:
            raise RuntimeError(
                "Sigma-satisfied data has already been initialized."
            )
        self[:] = [{} for _ in range(len(data_stream))]

        for stream_idx, stream_data in enumerate(data_stream):
            for q_name, sem_cq in queries.items():
                self.apply_query_sigma_and_build_ground_truth(
                    stream_idx=stream_idx,
                    q_name=q_name,
                    ucq=[sem_cq.Sigma],
                    stream_data=stream_data,
                    selected_cols=sem_cq.selected,
                    complete_config=complete_config,
                    data_dir=data_dir,
                )

        logger.info("Sigma-satisfied data retrieval completed.")

    def refine(
        self,
        q_name: str,
        ucq: list[list[Predicate]],
        queries: dict[str, SemCQ],
        complete_config: dict,
        data_dir: str,
    ) -> None:
        """Refine Sigma-satisfied data for one query across streams."""
        for stream_idx in range(len(self)):
            self.apply_query_sigma_and_build_ground_truth(
                stream_idx=stream_idx,
                q_name=q_name,
                ucq=ucq,
                stream_data=None,
                selected_cols=queries[q_name].selected,
                complete_config=complete_config,
                data_dir=data_dir,
            )

    def apply_query_sigma_and_build_ground_truth(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        stream_idx: int,
        q_name: str,
        ucq: list[list[Predicate]],
        stream_data: LdbData | None,
        selected_cols: list[str],
        complete_config: dict,
        data_dir: str,
    ) -> None:
        """Apply Sigma retrieval and build query ground truth."""
        self._validate_stream_idx(stream_idx)
        if q_name not in self[stream_idx]:
            if stream_data is None:
                raise ValueError(
                    "Stream data is required when initializing Sigma data"
                )
            self._init_query_record(
                stream_idx=stream_idx,
                q_name=q_name,
                ucq=ucq,
                stream_data=stream_data,
                complete_config=complete_config,
            )
        else:
            self._refine_query_record(
                stream_idx=stream_idx,
                q_name=q_name,
                ucq=ucq,
                complete_config=complete_config,
            )

        self._build_ground_truth_if_needed(
            stream_idx=stream_idx,
            q_name=q_name,
            selected_cols=selected_cols,
            data_dir=data_dir,
        )

    async def sync_features(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        q_name: str,
        stream_idx: int,
        enriched_features: dict[str, list[PopulationSpec]],
        llm_client: LdbLLMClient,
        ckpt_root: Path,
        tag: str = "",
        enable_cache: bool = True,
        is_remote: bool = False,
    ) -> dict:
        """Synchronize enriched features for one Sigma-satisfied dataset."""
        filename = f"stream_{stream_idx}_sigma_satisfied_data_{tag}.csv"
        ckpt_path = ckpt_root / q_name / filename
        if enable_cache and ckpt_path.exists():
            logger.debug(
                "Loading enriched Sigma-satisfied data for query '%s' "
                "in stream-%s from cache.",
                q_name,
                stream_idx,
            )
            cached_df = pd.read_csv(ckpt_path)
            labels = self[stream_idx][q_name]["labels"]
            label_count = (
                len(labels)
                if labels is not None
                else len(self[stream_idx][q_name]["ldb_data"].df)
            )
            if len(cached_df) == label_count:
                self[stream_idx][q_name]["ldb_data"].df = cached_df
                return llm_client.get_usage_statistics()

            logger.warning(
                "Ignoring cached enriched Sigma-satisfied data for query "
                "'%s' in stream-%s because row count %s does not match "
                "label count %s.",
                q_name,
                stream_idx,
                len(cached_df),
                label_count,
            )

        if q_name not in enriched_features:
            raise ValueError(
                f"Enriched features for query '{q_name}' not found"
            )

        await self[stream_idx][q_name]["ldb_data"].sync_with_enriched_features(
            enriched_features=enriched_features[q_name],
            llm_client=llm_client,
            is_remote=is_remote,
        )

        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        self[stream_idx][q_name]["ldb_data"].df.to_csv(ckpt_path, index=False)

        llm_usage_statistics = llm_client.get_usage_statistics()
        llm_client.reset_usage_statistics()
        return llm_usage_statistics

    def eval_query_quality(
        self,
        q_name: str,
        selected_cols: list[str],
        stream_idx: int,
        pred_labels: list[pd.Series],
    ) -> dict:
        """Evaluate predicted labels against query ground truth."""
        ground_truth, retrieved_data = self._collect_quality_sets(
            q_name=q_name,
            selected_cols=selected_cols,
            stream_idx=stream_idx,
            pred_labels=pred_labels,
        )
        tp = len(ground_truth.intersection(retrieved_data))
        fp = len(retrieved_data - ground_truth)
        fn = len(ground_truth - retrieved_data)
        return _quality_metrics(q_name, stream_idx, tp, fp, fn)

    def compute_stream_stat(
        self, q_name: str
    ) -> dict[str, list[float] | list[int] | float | int]:
        """Compute per-stream and overall statistics for one query."""
        stream_selectivities = []
        stream_sizes = []
        total_positive = 0
        total_size = 0

        for stream_idx, stream_records in enumerate(self):
            if q_name not in stream_records:
                raise ValueError(
                    f"Query {q_name!r} is missing in stream-{stream_idx}."
                )

            labels = stream_records[q_name]["labels"]
            if labels is None:
                labels = pd.Series(dtype=bool)
                logger.warning(
                    "Ground-truth labels for query '%s' in stream-%s are "
                    "not assigned. Assuming empty labels.",
                    q_name,
                    stream_idx,
                )

            stream_size = len(labels)
            positive_count = int(labels.sum())
            stream_selectivity = (
                positive_count / stream_size if stream_size > 0 else 0.0
            )
            stream_selectivities.append(float(stream_selectivity))
            stream_sizes.append(stream_size)
            total_positive += positive_count
            total_size += stream_size

        overall_selectivity = (
            total_positive / total_size if total_size > 0 else 0.0
        )
        return {
            "stream_selectivities": stream_selectivities,
            "overall_selectivity": float(overall_selectivity),
            "stream_sizes": stream_sizes,
            "total_size": total_size,
        }

    @staticmethod
    def new_record(ldb_data: LdbData) -> SigmaRecord:
        """Create the standard Sigma-satisfied data record."""
        return {
            "ldb_data": ldb_data,
            "labels": None,
            "propagated_labels": None,
            "selected_data": pd.DataFrame(),
            "discarded_data": pd.DataFrame(),
            "selected_labels": pd.Series(dtype=bool),
            "discarded_labels": pd.Series(dtype=bool),
        }

    def record(self, stream_idx: int, q_name: str) -> SigmaRecord:
        """Return the Sigma-satisfied record for a stream and query."""
        return self[stream_idx][q_name]

    def _validate_stream_idx(self, stream_idx: int) -> None:
        """Validate that the stream slot has been initialized."""
        if stream_idx < 0 or stream_idx >= len(self):
            raise IndexError(
                f"Stream index {stream_idx} is outside initialized "
                f"Sigma-satisfied data streams: {len(self)}."
            )

    def _collect_quality_sets(
        self,
        q_name: str,
        selected_cols: list[str],
        stream_idx: int,
        pred_labels: list[pd.Series],
    ) -> tuple[set[tuple], set[tuple]]:
        """Collect ground-truth and retrieved row keys for quality metrics."""
        if len(pred_labels) != stream_idx + 1:
            raise ValueError(
                f"The number of predicted label series ({len(pred_labels)}) "
                f"does not match the number of streams ({stream_idx + 1})."
            )
        ground_truth, retrieved_data = set(), set()
        for sid in range(stream_idx + 1):
            stream_ground_truth, stream_retrieved = self._stream_quality_sets(
                q_name=q_name,
                selected_cols=selected_cols,
                stream_idx=sid,
                pred_labels=pred_labels[sid],
            )
            ground_truth.update(stream_ground_truth)
            retrieved_data.update(stream_retrieved)
        return ground_truth, retrieved_data

    def _stream_quality_sets(
        self,
        q_name: str,
        selected_cols: list[str],
        stream_idx: int,
        pred_labels: pd.Series,
    ) -> tuple[set[tuple], set[tuple]]:
        """Collect quality row keys for one stream."""
        ss_data = self[stream_idx][q_name]
        ss_df = ss_data["ldb_data"].df

        if len(pred_labels) != len(ss_df):
            raise ValueError(
                f"The number of predicted labels ({len(pred_labels)}) "
                f"does not match the number of Sigma-satisfied samples "
                f"({len(ss_df)}) for query '{q_name}' in stream-{stream_idx}."
            )

        if ss_data["labels"] is None:
            raise ValueError(
                f"Ground-truth labels for query '{q_name}' in stream-"
                f"{stream_idx} have not been initialized."
            )

        retrieved_labels = pred_labels.astype(bool).to_numpy()
        ground_truth_labels = ss_data["labels"].astype(bool).to_numpy()
        ground_truth = _row_keys(ss_df.iloc[ground_truth_labels], selected_cols)
        retrieved_data = _row_keys(ss_df.iloc[retrieved_labels], selected_cols)

        selected_keys = _selected_quality_keys(ss_data, selected_cols)
        ground_truth.update(selected_keys)
        retrieved_data.update(selected_keys)
        ground_truth.update(
            _discarded_ground_truth_keys(
                ss_data=ss_data,
                selected_cols=selected_cols,
                q_name=q_name,
                stream_idx=stream_idx,
            )
        )
        return ground_truth, retrieved_data

    def _init_query_record(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        stream_idx: int,
        q_name: str,
        ucq: list[list[Predicate]],
        stream_data: LdbData,
        complete_config: dict,
    ) -> None:
        """Initialize one query Sigma record for a stream."""
        prev_data_scale = len(stream_data.df)
        selected_indices = stream_data.sigma_retrieve_ucq(ucq)
        selected_data = (
            stream_data.df.loc[selected_indices].reset_index(drop=True).copy()
        )
        self[stream_idx][q_name] = self.new_record(
            LdbData(df=selected_data, config=complete_config)
        )
        post_data_scale = len(self[stream_idx][q_name]["ldb_data"].df)
        logger.info(
            "Applied Sigma retrieval for query '%s' in stream-%s: "
            "%s -> %s rows.",
            q_name,
            stream_idx,
            prev_data_scale,
            post_data_scale,
        )

    def _refine_query_record(
        self,
        stream_idx: int,
        q_name: str,
        ucq: list[list[Predicate]],
        complete_config: dict,
    ) -> None:
        """Refine one existing query Sigma record for a stream."""
        record = self.record(stream_idx, q_name)
        if record["ldb_data"] is None or record["labels"] is None:
            raise ValueError(
                "Sigma-satisfied data and labels must be initialized "
                "before refining Sigma retrieval"
            )
        prev_data_scale = len(record["ldb_data"].df)
        selected_indices = record["ldb_data"].sigma_retrieve_ucq(ucq)
        selected_data = (
            record["ldb_data"]
            .df.loc[selected_indices]
            .reset_index(drop=True)
            .copy()
        )
        discarded_data = (
            record["ldb_data"]
            .df.drop(index=selected_indices)
            .reset_index(drop=True)
            .copy()
        )
        selected_labels = (
            record["labels"].loc[selected_indices].reset_index(drop=True).copy()
        )
        discarded_labels = (
            record["labels"]
            .drop(index=selected_indices)
            .reset_index(drop=True)
            .copy()
        )

        record["ldb_data"] = LdbData(df=selected_data, config=complete_config)
        record["labels"] = selected_labels
        record["discarded_data"] = pd.concat(
            [record["discarded_data"], discarded_data], ignore_index=True
        )
        record["discarded_labels"] = pd.concat(
            [record["discarded_labels"], discarded_labels], ignore_index=True
        )

        introduced_fn = sum(record["discarded_labels"])
        post_data_scale = len(record["ldb_data"].df)
        logger.info(
            "Refined Sigma-satisfied data for query '%s' in stream-%s: "
            "%s -> %s rows.",
            q_name,
            stream_idx,
            prev_data_scale,
            post_data_scale,
        )
        if introduced_fn > 0:
            logger.warning(
                "Refining Sigma-satisfied data for query '%s' in stream-%s "
                "introduced %s false negatives.",
                q_name,
                stream_idx,
                introduced_fn,
            )

    def _build_ground_truth_if_needed(
        self,
        stream_idx: int,
        q_name: str,
        selected_cols: list[str],
        data_dir: str,
    ) -> None:
        """Build ground-truth labels for one query record when absent."""
        record = self[stream_idx][q_name]
        if record["labels"] is not None:
            logger.debug(
                "Ground truth for query '%s' in stream-%s already exists. "
                "Skip building ground truth.",
                q_name,
                stream_idx,
            )
            return

        ground_truth_df = pd.read_csv(
            Path(data_dir) / "ground_truth" / f"{q_name}.csv"
        )[selected_cols]
        ground_truth_set = set(tuple(row) for row in ground_truth_df.values)

        labels = (
            record["ldb_data"]
            .df[selected_cols]
            .apply(lambda row: tuple(row) in ground_truth_set, axis=1)
            .reset_index(drop=True)
        )
        record["labels"] = labels

        logger.debug(
            "Ground truth of %s-stream-%s has %s pos samples out of "
            "%s samples, oracle selectivity = %.4f.",
            q_name,
            stream_idx,
            labels.sum(),
            len(labels),
            labels.mean(),
        )

        if len(ground_truth_set) != labels.sum():
            logger.warning(
                "Ground truth for query '%s' in stream-%s has %s positive "
                "samples, but Sigma-satisfied data has %s positive samples.",
                q_name,
                stream_idx,
                len(ground_truth_set),
                labels.sum(),
            )


def _selected_quality_keys(
    ss_data: SigmaRecord, selected_cols: list[str]
) -> set[tuple]:
    """Return selected keys after validating selected labels."""
    selected_df = ss_data["selected_data"]
    selected_labels = ss_data["selected_labels"]
    if sum(selected_labels) != len(selected_df):
        raise ValueError(
            "All selected data should be labeled as positive samples"
        )
    return _row_keys(selected_df, selected_cols)


def _discarded_ground_truth_keys(
    ss_data: SigmaRecord,
    selected_cols: list[str],
    q_name: str,
    stream_idx: int,
) -> set[tuple]:
    """Return positive discarded keys that still belong to ground truth."""
    discarded_labels = ss_data["discarded_labels"]
    if sum(discarded_labels) <= 0:
        return set()

    logger.debug(
        "Found %s positive samples in the discarded data "
        "for query '%s' in stream-%s.",
        sum(discarded_labels),
        q_name,
        stream_idx,
    )
    discarded_df = ss_data["discarded_data"]
    return _row_keys(discarded_df[discarded_labels], selected_cols)


def _row_keys(df: pd.DataFrame, selected_cols: list[str]) -> set[tuple]:
    """Convert selected dataframe columns into comparable row keys."""
    return set(df.apply(lambda row: tuple(row[selected_cols]), axis=1))


def _quality_metrics(
    q_name: str, stream_idx: int, tp: int, fp: int, fn: int
) -> dict:
    """Build precision, recall, and F1 metrics from counts."""
    if fp == 0 and fn == 0 and tp == 0:
        logger.debug(
            "Both prediction and ground truth are empty for query '%s' "
            "in stream-%s.",
            q_name,
            stream_idx,
        )
        return {
            "f1": 1.0,
            "precision": 1.0,
            "recall": 1.0,
            "TP": tp,
            "FP": fp,
            "FN": fn,
        }

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return {
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "TP": tp,
        "FP": fp,
        "FN": fn,
    }
