"""Manage LiteDBX datasets, streams, labels, and coresets."""

import logging
import random
from pathlib import Path

import pandas as pd

from llm import LdbLLMClient

from .ldb_data import LdbData
from .llm_resp_templates import PopulationSpec
from .sem_query import Predicate, SemCQ

logger = logging.getLogger(__name__)
random.seed(42)


class LdbDataManager:  # pylint: disable=too-many-instance-attributes
    """Manage data streams, Sigma-satisfied data, and coresets."""

    def __init__(
        self,
        data_dir: str,
        scenario: str,
        queries: dict[str, SemCQ],
        llm_client: LdbLLMClient,
        dynamic_steps: list[float],
    ):

        self.data_dir = data_dir
        self.scenario = scenario
        self.complete_dataset = LdbData(data_dir=data_dir)
        self.queries = queries
        self.llm_client = llm_client
        self.dynamic_steps = dynamic_steps
        self.CKPT_path = (  # pylint: disable=invalid-name
            Path(__file__).parent.parent
            / ".data_ckpt"
            / scenario
            / "_".join(str(step) for step in dynamic_steps)
        )
        self.CKPT_path.mkdir(parents=True, exist_ok=True)

        self.enriched_features: dict[str, list[PopulationSpec]] = {}
        self.trimmed_feature_names: list[str] = []

        """
        schema of data_stream:
        [
            LdbData, ...<streams>
        ]
        """
        self.data_stream = []

        """
        schema of sigma_satisfied_data:
        [
            {
                "Q1": {
                    "ldb_data": LdbData,
                    "labels": pd.Series,
                    "propagated_labels": pd.Series,
                    "selected_data": pd.DataFrame,
                    "selected_labels": pd.Series,
                    "discarded_data": pd.DataFrame,
                    "discarded_labels": pd.Series,
                }, ...<queries>
            }, ...<streams>
        ]
        """
        self.sigma_satisfied_data = []

        """
        schema of coreset:
        {
            "Q1": {
                "ldb_data": LdbData,
                "labels": pd.Series,
                "observed_size": int,
                "lb": int,
                "ub": int,
            }, ...<queries> 
        }
        `lb` and `ub` indicate the confidence bounds for coreset maintenance.
        """
        self.coresets = {}

    def init_data_stream(self) -> None:
        """
        Build the data stream based on the dynamic setting.
        """
        logger.info("Start data stream construction.")

        total_rows = len(self.complete_dataset.df)
        indices = list(range(total_rows))

        random.shuffle(indices)

        steps = [0] + self.dynamic_steps
        data_ladder = [int(total_rows * step) for step in steps]

        for i in range(1, len(data_ladder)):
            selected_indices = indices[data_ladder[i - 1] : data_ladder[i]]
            df = (
                self.complete_dataset.df.iloc[selected_indices]
                .copy()
                .reset_index(drop=True)
            )

            self.data_stream.append(
                LdbData(df=df, config=self.complete_dataset.config)
            )

        logger.info("[S] Data stream construction completed.")

    def init_sigma_satisfied_data(self) -> None:
        """
        Retrieve the Sigma-satisfied data for each stream and
        construct the ground truth accordingly.
        """
        logger.info("Start Sigma-satisfied data retrieval.")
        for stream_idx in range(len(self.data_stream)):
            for q_name, sem_cq in self.queries.items():
                self._apply_query_sigma_and_build_ground_truth(
                    stream_idx=stream_idx, q_name=q_name, ucq=[sem_cq.Sigma]
                )
        logger.info("[S] Sigma-satisfied data retrieval completed.")

    def refine_sigma_satisfied_data(
        self, q_name: str, ucq: list[list[Predicate]]
    ) -> None:
        """
        [Optional]
        Refine (narrow down) the Sigma-satisfied data.
        """
        for stream_idx in range(len(self.data_stream)):
            self._apply_query_sigma_and_build_ground_truth(
                stream_idx=stream_idx, q_name=q_name, ucq=ucq
            )

    async def acquire_annotation_and_init_coreset(
        self, b_lab: int, seed: int = 42, use_hitl: bool = True
    ) -> None:
        """Acquire query annotations and initialize each query coreset."""
        for q_name in self.queries:
            await self._acquire_query_annotation_and_init_coreset(
                b_lab=b_lab,
                q_name=q_name,
                stream_idx=0,
                seed=seed,
                use_hitl=use_hitl,
            )

    async def sync_coreset_features(
        self,
        q_name: str,
        tag: str = "",
        enable_cache: bool = True,
        is_remote: bool = False,
    ) -> dict:
        """Sync enriched coreset features and return LLM usage stats."""
        ckpt_path = self.CKPT_path / q_name / f"coreset_{tag}.csv"
        if enable_cache and ckpt_path.exists():
            logger.info(
                "Loading enriched coreset for query '%s' from cache.",
                q_name,
            )
            self.coresets[q_name]["ldb_data"].df = pd.read_csv(ckpt_path)
            return self.llm_client.get_usage_statistics()

        assert q_name in self.enriched_features, (
            f"Enriched features for query '{q_name}' not found. "
        )
        await self.coresets[q_name]["ldb_data"].sync_with_enriched_features(
            enriched_features=self.enriched_features[q_name],
            llm_client=self.llm_client,
            is_remote=is_remote,
        )

        # Flush the cache.
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        self.coresets[q_name]["ldb_data"].df.to_csv(ckpt_path, index=False)

        llm_usage_statistics = self.llm_client.get_usage_statistics()
        self.llm_client.reset_usage_statistics()
        return llm_usage_statistics

    async def sync_sigma_satisfied_data_features(
        self,
        q_name: str,
        tag: str = "",
        stream_idx: int = 0,
        enable_cache: bool = True,
        is_remote: bool = False,
    ) -> dict:
        """Sync enriched Sigma-satisfied data and return LLM usage stats."""
        ckpt_path = (
            self.CKPT_path
            / q_name
            / (f"stream_{stream_idx}_sigma_satisfied_data_{tag}.csv")
        )
        if enable_cache and ckpt_path.exists():
            logger.info(
                "Loading enriched Sigma-satisfied data for query '%s' "
                "in stream-%s from cache.",
                q_name,
                stream_idx,
            )
            self.sigma_satisfied_data[stream_idx][q_name][
                "ldb_data"
            ].df = pd.read_csv(ckpt_path)
            return self.llm_client.get_usage_statistics()

        assert q_name in self.enriched_features, (
            f"Enriched features for query '{q_name}' not found. "
        )
        await self.sigma_satisfied_data[stream_idx][q_name][
            "ldb_data"
        ].sync_with_enriched_features(
            enriched_features=self.enriched_features[q_name],
            llm_client=self.llm_client,
            is_remote=is_remote,
        )

        # Flush the cache.
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        self.sigma_satisfied_data[stream_idx][q_name]["ldb_data"].df.to_csv(
            ckpt_path, index=False
        )

        llm_usage_statistics = self.llm_client.get_usage_statistics()
        self.llm_client.reset_usage_statistics()
        return llm_usage_statistics

    def eval_query_quality(  # pylint: disable=too-many-locals
        self,
        q_name: str,
        selected_cols: list[str],
        stream_idx: int,
        pred_labels: list[pd.Series],
    ) -> dict:
        """Evaluate predicted labels against query ground truth."""
        ground_truth, retrieved_data = set(), set()
        for sid in range(stream_idx + 1):
            # 1. Process the sigma-satisfied data.
            ss_data = self.sigma_satisfied_data[sid][q_name]
            ss_df = ss_data["ldb_data"].df
            ground_truth_labels = ss_data["labels"]
            retrieved_labels = pred_labels[sid].astype(bool)
            ground_truth.update(
                ss_df[ground_truth_labels].apply(
                    lambda row: tuple(row[selected_cols]), axis=1
                )
            )
            retrieved_data.update(
                ss_df[retrieved_labels].apply(
                    lambda row: tuple(row[selected_cols]), axis=1
                )
            )

            # 2. Involve the selected items with human annotation.
            selected_df = ss_data["selected_data"]
            selected_labels = ss_data["selected_labels"]
            assert sum(selected_labels) == len(selected_df), (
                "All selected data should be labeled as positive samples."
            )
            ground_truth.update(
                selected_df.apply(lambda row: tuple(row[selected_cols]), axis=1)
            )
            retrieved_data.update(
                selected_df.apply(lambda row: tuple(row[selected_cols]), axis=1)
            )

            # 3. Involve the discarded items.
            discarded_df = ss_data["discarded_data"]
            discarded_labels = ss_data["discarded_labels"]
            if sum(discarded_labels) > 0:
                logger.warning(
                    "Found %s positive samples in the discarded data "
                    "for query '%s' in stream-%s.",
                    sum(discarded_labels),
                    q_name,
                    sid,
                )
                ground_truth.update(
                    discarded_df[discarded_labels].apply(
                        lambda row: tuple(row[selected_cols]), axis=1
                    )
                )

        # Compute the evaluation metrics.
        tp = len(ground_truth.intersection(retrieved_data))
        fp = len(retrieved_data - ground_truth)
        fn = len(ground_truth - retrieved_data)

        # Avoid the case when no positive ground-truth sample exists,
        # which leads to TP=FP=FN=0.
        if fp == 0 and fn == 0 and tp == 0:
            logger.info(
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

    def _apply_query_sigma_and_build_ground_truth(  # pylint: disable=too-many-locals
        self, stream_idx: int, q_name: str, ucq: list[list[Predicate]]
    ) -> None:
        """
        Apply the Sigma retrieval and build the ground truth accordingly
        for a specific query and stream.
        """

        if len(self.sigma_satisfied_data) == 0:
            self.sigma_satisfied_data = [
                {} for _ in range(len(self.data_stream))
            ]

        # ==============================================
        # (1) Apply sigma retrieval with the refined UCQ.
        # ==============================================
        if q_name not in self.sigma_satisfied_data[stream_idx]:
            # Sigma retrieval
            assert self.data_stream[stream_idx] is not None, (
                "Data stream should be initialized before applying "
                "Sigma retrieval."
            )
            prev_data_scale = len(self.data_stream[stream_idx].df)
            selected_indices = self.data_stream[stream_idx].sigma_retrieve_ucq(
                ucq
            )
            selected_data = (
                self.data_stream[stream_idx]
                .df.loc[selected_indices]
                .reset_index(drop=True)
                .copy()
            )
            self.sigma_satisfied_data[stream_idx][q_name] = {
                "ldb_data": LdbData(
                    df=selected_data, config=self.complete_dataset.config
                ),
                "labels": None,
                "propagated_labels": None,
                "selected_data": pd.DataFrame(),
                "discarded_data": pd.DataFrame(),
                "selected_labels": pd.Series(),
                "discarded_labels": pd.Series(),
            }
            post_data_scale = len(
                self.sigma_satisfied_data[stream_idx][q_name]["ldb_data"].df
            )
            logger.info(
                "Applied Sigma retrieval for query '%s' in stream-%s: "
                "%s -> %s rows.",
                q_name,
                stream_idx,
                prev_data_scale,
                post_data_scale,
            )
        else:
            # Refined sigma retrieval
            assert (
                self.sigma_satisfied_data[stream_idx][q_name]["ldb_data"]
                is not None
                and self.sigma_satisfied_data[stream_idx][q_name]["labels"]
                is not None
            ), (
                "Sigma-satisfied data and labels should be initialized "
                "before refining Sigma retrieval."
            )
            prev_data_scale = len(
                self.sigma_satisfied_data[stream_idx][q_name]["ldb_data"].df
            )

            selected_indices = self.sigma_satisfied_data[stream_idx][q_name][
                "ldb_data"
            ].sigma_retrieve_ucq(ucq)
            selected_data = (
                self.sigma_satisfied_data[stream_idx][q_name]["ldb_data"]
                .df.loc[selected_indices]
                .reset_index(drop=True)
                .copy()
            )
            discarded_data = (
                self.sigma_satisfied_data[stream_idx][q_name]["ldb_data"]
                .df.drop(index=selected_indices)
                .reset_index(drop=True)
                .copy()
            )
            selected_labels = (
                self.sigma_satisfied_data[stream_idx][q_name]["labels"]
                .loc[selected_indices]
                .reset_index(drop=True)
                .copy()
            )
            discarded_labels = (
                self.sigma_satisfied_data[stream_idx][q_name]["labels"]
                .drop(index=selected_indices)
                .reset_index(drop=True)
                .copy()
            )

            self.sigma_satisfied_data[stream_idx][q_name]["ldb_data"] = LdbData(
                df=selected_data, config=self.complete_dataset.config
            )
            self.sigma_satisfied_data[stream_idx][q_name]["labels"] = (
                selected_labels
            )
            self.sigma_satisfied_data[stream_idx][q_name]["discarded_data"] = (
                pd.concat(
                    [
                        self.sigma_satisfied_data[stream_idx][q_name][
                            "discarded_data"
                        ],
                        discarded_data,
                    ],
                    ignore_index=True,
                )
            )
            self.sigma_satisfied_data[stream_idx][q_name][
                "discarded_labels"
            ] = pd.concat(
                [
                    self.sigma_satisfied_data[stream_idx][q_name][
                        "discarded_labels"
                    ],
                    discarded_labels,
                ],
                ignore_index=True,
            )

            introduced_fn = sum(
                self.sigma_satisfied_data[stream_idx][q_name][
                    "discarded_labels"
                ]
            )

            post_data_scale = len(
                self.sigma_satisfied_data[stream_idx][q_name]["ldb_data"].df
            )
            logger.info(
                "Refined Sigma-satisfied data for query '%s' in stream-%s: "
                "%s -> %s rows. Est. # FNs: %s.",
                q_name,
                stream_idx,
                prev_data_scale,
                post_data_scale,
                introduced_fn,
            )

        # ==============================================
        # (2) Build ground truth.
        # ==============================================

        if self.sigma_satisfied_data[stream_idx][q_name]["labels"] is not None:
            logger.info(
                "Ground truth for query '%s' in stream-%s already exists. "
                "Skip building ground truth.",
                q_name,
                stream_idx,
            )
            return

        # Hacked ground_truth.
        # hacked_query_name = "Q3a"
        hacked_query_name = q_name

        selected_cols = self.queries[q_name].selected
        ground_truth_df = pd.read_csv(
            f"{self.data_dir}/ground_truth/{hacked_query_name}.csv"
        )[selected_cols]
        ground_truth_set = set(tuple(row) for row in ground_truth_df.values)

        labels = (
            self.sigma_satisfied_data[stream_idx][q_name]["ldb_data"]
            .df[selected_cols]
            .apply(lambda row: tuple(row) in ground_truth_set, axis=1)
            .reset_index(drop=True)
        )

        self.sigma_satisfied_data[stream_idx][q_name]["labels"] = labels

        # Report the oracle selectivity.
        logger.info(
            "Ground truth of %s-stream-%s has %s pos samples out of "
            "%s samples, oracle selectivity = %.4f.",
            q_name,
            stream_idx,
            labels.sum(),
            len(labels),
            labels.mean(),
        )

    async def _acquire_query_annotation_and_init_coreset(  # pylint: disable=too-many-locals
        self,
        q_name: str,
        b_lab: int,
        stream_idx: int = 0,
        seed: int = 42,
        use_hitl: bool = True,
    ) -> None:

        data = self.sigma_satisfied_data[stream_idx][q_name][
            "ldb_data"
        ].df.copy()
        labels = self.sigma_satisfied_data[stream_idx][q_name]["labels"]
        if use_hitl:
            acquired_labels = self.sigma_satisfied_data[stream_idx][q_name][
                "labels"
            ]
        else:
            acquired_labels = await self._acquire_pseudo_labels_by_llm(q_name)

        labeling_budget = b_lab
        if labeling_budget >= len(data):
            logger.info(
                "[W] Requested labeled budget %s exceeds the data scale "
                "%s for query '%s' in stream-%s.",
                labeling_budget,
                len(data),
                q_name,
                stream_idx,
            )
            labeling_budget = len(data) // 2
            logger.info(
                "Adjusted labeled budget to %s for query '%s' in stream-%s.",
                labeling_budget,
                q_name,
                stream_idx,
            )

        labeled_indices = data.sample(
            n=labeling_budget, random_state=seed
        ).index

        # Check whether both classes are sampled.
        num_pos_sampled = acquired_labels.loc[labeled_indices].sum()
        num_neg_sampled = len(labeled_indices) - num_pos_sampled
        minority_bias = min(
            2, b_lab - labeling_budget
        )  # Add bias to avoid single-class situation.
        if num_pos_sampled == 0 and minority_bias > 0:
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
        if num_neg_sampled == 0 and minority_bias > 0:
            neg_indices = acquired_labels[~acquired_labels].index
            neg_to_add = min(minority_bias, len(neg_indices))
            labeled_indices = labeled_indices.union(neg_indices[:neg_to_add])
            logger.info(
                "Minority class (negative) is not sampled for query '%s' "
                "in stream-%s. Added %s negative samples. Current labeled "
                "set has %s pos samples out of %s samples.",
                q_name,
                stream_idx,
                neg_to_add,
                acquired_labels.loc[labeled_indices].sum(),
                len(labeled_indices),
            )

        remaining_indices = data.index.difference(labeled_indices)

        # Move the labeled data from `sigma_satisfied_data` to the `coreset`.
        if q_name not in self.coresets:
            self.coresets[q_name] = {
                "ldb_data": LdbData(
                    df=data.loc[labeled_indices].reset_index(drop=True).copy(),
                    config=self.complete_dataset.config,
                ),
                "labels": acquired_labels.loc[labeled_indices]
                .reset_index(drop=True)
                .copy(),
                "observed_size": len(labeled_indices),
                "lb": float("inf"),
                "ub": float("-inf"),
            }
        else:
            logger.info("[W] Coreset for query '%s' already exists.", q_name)
            self.coresets[q_name]["ldb_data"].df = pd.concat(
                [
                    self.coresets[q_name]["ldb_data"].df,
                    data.loc[labeled_indices].reset_index(drop=True).copy(),
                ],
                ignore_index=True,
            )
            self.coresets[q_name]["labels"] = pd.concat(
                [
                    self.coresets[q_name]["labels"],
                    acquired_labels.loc[labeled_indices]
                    .reset_index(drop=True)
                    .copy(),
                ],
                ignore_index=True,
            )
            self.coresets[q_name]["observed_size"] += len(labeled_indices)

        # Update the `sigma_satisfied_data`.
        self.sigma_satisfied_data[stream_idx][q_name]["ldb_data"].df = (
            data.loc[remaining_indices].reset_index(drop=True).copy()
        )
        self.sigma_satisfied_data[stream_idx][q_name]["labels"] = (
            labels.loc[remaining_indices].reset_index(drop=True).copy()
        )

        # Update the selected and discarded data for analysis.
        pos_labeled_indices = labeled_indices[labels.loc[labeled_indices]]
        neg_labeled_indices = labeled_indices[~labels.loc[labeled_indices]]
        self.sigma_satisfied_data[stream_idx][q_name]["selected_data"] = (
            pd.concat(
                [
                    self.sigma_satisfied_data[stream_idx][q_name][
                        "selected_data"
                    ],
                    data.loc[pos_labeled_indices].reset_index(drop=True).copy(),
                ],
                ignore_index=True,
            )
        )
        self.sigma_satisfied_data[stream_idx][q_name]["selected_labels"] = (
            pd.concat(
                [
                    self.sigma_satisfied_data[stream_idx][q_name][
                        "selected_labels"
                    ],
                    labels.loc[pos_labeled_indices]
                    .reset_index(drop=True)
                    .copy(),
                ],
                ignore_index=True,
            )
        )
        self.sigma_satisfied_data[stream_idx][q_name]["discarded_data"] = (
            pd.concat(
                [
                    self.sigma_satisfied_data[stream_idx][q_name][
                        "discarded_data"
                    ],
                    data.loc[neg_labeled_indices].reset_index(drop=True).copy(),
                ],
                ignore_index=True,
            )
        )
        self.sigma_satisfied_data[stream_idx][q_name]["discarded_labels"] = (
            pd.concat(
                [
                    self.sigma_satisfied_data[stream_idx][q_name][
                        "discarded_labels"
                    ],
                    labels.loc[neg_labeled_indices]
                    .reset_index(drop=True)
                    .copy(),
                ],
                ignore_index=True,
            )
        )
        logger.info(
            "Initialized coreset for query '%s' in stream-%s with %s "
            "labeled samples. Remaining Sigma-satisfied data has %s samples.",
            q_name,
            stream_idx,
            len(labeled_indices),
            len(remaining_indices),
        )

    async def _acquire_pseudo_labels_by_llm(
        self, q_name: str, stream_idx: int = 0, enable_cache: bool = True
    ) -> pd.Series:

        ckpt_path = self.CKPT_path / q_name / f"pseudo_labels_{stream_idx}.csv"
        if enable_cache and ckpt_path.exists():
            logger.info("Loading pseudo labels generated by LLM.")
            return pd.read_csv(ckpt_path, index_col=0).iloc[:, 0]

        data = self.sigma_satisfied_data[stream_idx][q_name]["ldb_data"]
        semcq = self.queries[q_name]

        spec_labels = []
        for idx, sem_pred in enumerate(semcq.Ps):
            spec = PopulationSpec(
                source_col=sem_pred.field,
                source_modality=sem_pred.modality,  # type: ignore
                target_col=f"llm_label_{idx}",
                prompt=sem_pred.prompt,
                feature_type="bool",
            )

            spec_label = await data._sem_map(  # pylint: disable=protected-access
                spec=spec, llm_client=self.llm_client, is_remote=False
            )
            spec_labels.append(spec_label)

        assert len(spec_labels) > 0, "Empty SemPredicates."

        result = spec_labels[0]
        for label in spec_labels[1:]:
            result = result & label

        # Save to cache
        if enable_cache:
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            result.to_csv(ckpt_path, index=True, header=True)

        return result
