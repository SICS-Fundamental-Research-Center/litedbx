import pandas as pd
import logging
import random
from pathlib import Path
from llm import LdbLLMClient
from .ldb_data import LdbData
from .sem_query import SemCQ, Predicate
from .llm_resp_templates import PopulationSpec

logger = logging.getLogger(__name__)
random.seed(42)


class LdbDataManager:
    def __init__(
            self, 
            data_dir: str,
            scenario: str,
            queries: dict[str, SemCQ],
            llm_client: LdbLLMClient,
            dynamic_steps: list[float]):

        self.data_dir = data_dir
        self.scenario = scenario
        self.complete_dataset = LdbData(data_dir=data_dir)
        self.queries = queries
        self.llm_client = llm_client
        self.dynamic_steps = dynamic_steps
        self.CKPT_path = Path(__file__).parent.parent / ".data_ckpt" / scenario \
            / "_".join(str(step) for step in dynamic_steps)
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
                    "deduplicated_num_pos": int,
                    "num_fn": int,
                    "num_tp": int,
                }, ...<queries>
            }, ...<streams>
        ]
        The sigma retrieval may introduce FNs, hence we track it with `num_fn`.
        We can also require the human-annotated labels, hence we track it with `num_tp`.
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
            selected_indices = indices[data_ladder[i-1]:data_ladder[i]]
            df = self.complete_dataset.df.iloc[selected_indices].copy().reset_index(drop=True)

            self.data_stream.append(LdbData(df=df, config=self.complete_dataset.config))

        logger.info("[S] Data stream construction completed.")

    
    def init_sigma_satisfied_data(self) -> None:
        """
        Retrieve the Sigma-satisfied data for each stream and 
        construct the ground truth accordingly.
        """
        logger.info("Start Sigma-satisfied data retrieval.")
        for stream_idx in range(len(self.data_stream)):
            for q_name, sem_cq in self.queries.items():
                _ = self._apply_query_sigma_and_build_ground_truth(
                    stream_idx=stream_idx,
                    q_name=q_name,
                    ucq=[sem_cq.Sigma]
                )
        logger.info("[S] Sigma-satisfied data retrieval completed.")


    def refine_sigma_satisfied_data(
            self,
            q_name: str,
            ucq: list[list[Predicate]]) -> None:
        """
        [Optional]
        Refine (narrow down) the Sigma-satisfied data.
        """
        for stream_idx in range(len(self.data_stream)):
            self._apply_query_sigma_and_build_ground_truth(
                stream_idx=stream_idx,
                q_name=q_name,
                ucq=ucq
            )


    def acquire_annotation_and_init_coreset(
            self, b_lab: int, seed: int = 42) -> None:
        for q_name in self.queries.keys():
            self._acquire_query_annotation_and_init_coreset(
                b_lab=b_lab, 
                q_name=q_name, 
                stream_idx=0, 
                seed=seed
            )


    async def sync_coreset_features(
            self,
            q_name: str,
            tag: str = "",
            enable_cache: bool = True,
            is_remote: bool = False) -> dict:

        ckpt_path = self.CKPT_path / q_name / f"coreset_{tag}.csv"
        if enable_cache and ckpt_path.exists():
            logger.info(f"Loading enriched coreset for query '{q_name}' from cache.")
            self.coresets[q_name]["ldb_data"].df = pd.read_csv(ckpt_path)
            return self.llm_client.get_usage_statistics()

        assert q_name in self.enriched_features.keys(), (
            f"Enriched features for query '{q_name}' not found. "
        )
        await self.coresets[q_name]["ldb_data"].sync_with_enriched_features(
            enriched_features=self.enriched_features[q_name],
            llm_client=self.llm_client,
            is_remote=is_remote
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
            is_remote: bool = False) -> dict:

        ckpt_path = self.CKPT_path / q_name / (
            f"stream_{stream_idx}_sigma_satisfied_data_{tag}.csv"
        )
        if enable_cache and ckpt_path.exists():
            logger.info((
                f"Loading enriched Sigma-satisfied data for "
                f"query '{q_name}' in stream-{stream_idx} from cache."
            ))
            self.sigma_satisfied_data[stream_idx][q_name]['ldb_data'].df = pd.read_csv(ckpt_path)
            return self.llm_client.get_usage_statistics() 

        assert q_name in self.enriched_features.keys(), (
            f"Enriched features for query '{q_name}' not found. "
        )
        await self.sigma_satisfied_data[stream_idx][q_name]['ldb_data']\
            .sync_with_enriched_features(
                enriched_features=self.enriched_features[q_name],
                llm_client=self.llm_client,
                is_remote=is_remote
            )

        # Flush the cache.
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        self.sigma_satisfied_data[stream_idx][q_name]['ldb_data'].df.to_csv(ckpt_path, index=False)
        
        llm_usage_statistics = self.llm_client.get_usage_statistics() 
        self.llm_client.reset_usage_statistics()
        return llm_usage_statistics


    def _apply_query_sigma_and_build_ground_truth(
            self, 
            stream_idx: int,
            q_name: str,
            ucq: list[list[Predicate]]) -> None:
        """
        Apply the Sigma retrieval and build the ground truth accordingly
        for a specific query and stream.
        """

        if len(self.sigma_satisfied_data) == 0:
            self.sigma_satisfied_data = [{} for _ in range(len(self.data_stream))]

        # ==============================================
        # (1) Apply sigma retrieval with the refined UCQ.
        # ==============================================
        if q_name not in self.sigma_satisfied_data[stream_idx].keys():
            # Sigma retrieval
            assert self.data_stream[stream_idx] is not None, (
                "Data stream should be initialized before applying Sigma retrieval."
            )
            prev_data_scale = len(self.data_stream[stream_idx].df)
            self.sigma_satisfied_data[stream_idx][q_name] = {
                "ldb_data": self.data_stream[stream_idx].sigma_retrieve_ucq(ucq, reset_index=True),
                "labels": None,
                "propagated_labels": None,
                "num_fn": 0,
                "num_tp": 0,
                "deduplicated_num_pos": 0,
            }
            logger.info((
                f"Applied Sigma retrieval for query '{q_name}' in stream-{stream_idx}: "
                f"{prev_data_scale} -> {len(self.sigma_satisfied_data[stream_idx][q_name]['ldb_data'].df)} rows."
            ))
        else:
            # Refined sigma retrieval
            assert self.sigma_satisfied_data[stream_idx][q_name]['ldb_data'] is not None, (
                "Sigma-satisfied data should be initialized before refining Sigma retrieval."
            )
            prev_data_scale = len(self.sigma_satisfied_data[stream_idx][q_name]['ldb_data'].df)

            self.sigma_satisfied_data[stream_idx][q_name]['ldb_data'] = \
                self.sigma_satisfied_data[stream_idx][q_name]['ldb_data'].sigma_retrieve_ucq(ucq, reset_index=True)

            post_data_scale = len(self.sigma_satisfied_data[stream_idx][q_name]['ldb_data'].df)
            logger.info((
                f"Refined Sigma-satisfied data for query '{q_name}' in stream-{stream_idx}: "
                f"{prev_data_scale} -> {post_data_scale} rows."
            ))

        # ==============================================
        # (2) Build ground truth.
        # ==============================================

        # Handle the case when all samples are eliminated by the augmented Sigma retrieval.
        if len(self.sigma_satisfied_data[stream_idx][q_name]['ldb_data'].df) == 0:
            assert self.sigma_satisfied_data[stream_idx][q_name]['labels'] is not None
            prev_num_true = self.sigma_satisfied_data[stream_idx][q_name]['labels'].sum()
            self.sigma_satisfied_data[stream_idx][q_name]['num_fn'] += prev_num_true
            self.sigma_satisfied_data[stream_idx][q_name]['labels'] = pd.Series([])
            logger.info((
                f"Refined Sigma retrieval eliminates all samples, resulting in {prev_num_true} FNs for query "
                f"'{q_name}' in stream-{stream_idx}."))
            return

        selected_cols = self.queries[q_name].selected
        ground_truth_df = pd.read_csv(
            f"{self.data_dir}/ground_truth/{q_name}.csv"
        )[selected_cols]
        ground_truth_set = set(tuple(row) for row in ground_truth_df.values)
                
        labels = self.sigma_satisfied_data[stream_idx][q_name]['ldb_data'].df[selected_cols].apply(
            lambda row: tuple(row) in ground_truth_set,
            axis=1
        ).reset_index(drop=True)
        positive_samples = self.sigma_satisfied_data[stream_idx][q_name]['ldb_data'].df[labels][selected_cols]
        deduplicated_num_pos = len(set(tuple(row) for row in positive_samples.values))

        if self.sigma_satisfied_data[stream_idx][q_name]['labels'] is None:
            self.sigma_satisfied_data[stream_idx][q_name]["labels"] = labels
            self.sigma_satisfied_data[stream_idx][q_name]["deduplicated_num_pos"] = deduplicated_num_pos
            self.sigma_satisfied_data[stream_idx][q_name]["num_fn"] = 0
            logger.info((
                f"[W] Duplication in {q_name}-stream-{stream_idx} ground truth introduces: "
                f"{len(positive_samples) - deduplicated_num_pos} additional TPs. "
                f"The deduplicated number of pos sample is {deduplicated_num_pos}."
            ))
        else:
            # Integrity check.
            prev_num_true = self.sigma_satisfied_data[stream_idx][q_name]['labels'].sum()
            curr_num_true = labels.sum()
            curr_true_rows = self.sigma_satisfied_data[stream_idx][q_name]['ldb_data'].df[labels][selected_cols]
            curr_true_rows_set = set(tuple(row) for row in curr_true_rows.values)
            assert curr_true_rows_set.issubset(ground_truth_set), (
                f"Fail to build ground truth for query '{q_name}' in stream-{stream_idx}."
            )

            num_new_fn = self.sigma_satisfied_data[stream_idx][q_name]['num_fn'] + \
                (prev_num_true - curr_num_true)
            self.sigma_satisfied_data[stream_idx][q_name]["num_fn"] = num_new_fn 

            logger.info((
                f"[W] Refined Sigma retrieval for {q_name}-stream-{stream_idx} results in "
                f"{num_new_fn} FNs."))
            
            self.sigma_satisfied_data[stream_idx][q_name]["labels"] = labels
            self.sigma_satisfied_data[stream_idx][q_name]["deduplicated_num_pos"] = deduplicated_num_pos

        # Report the oracle selectivity.
        logger.info((
            f"Ground truth of {q_name}-stream-{stream_idx} has {labels.sum()} pos samples "
            f"out of {len(labels)} samples, "
            f"oracle selectivity = {labels.mean():.4f}."
        ))


    def _acquire_query_annotation_and_init_coreset(
            self, q_name: str, b_lab: int, stream_idx: int = 0, seed: int = 42) -> None:

        data = self.sigma_satisfied_data[stream_idx][q_name]["ldb_data"]
        labels = self.sigma_satisfied_data[stream_idx][q_name]["labels"]

        labeling_budget = b_lab
        if labeling_budget >= len(data.df):
            logger.info((
                f"[W] Requested labeled budget {labeling_budget} exceeds the data scale "
                f"{len(data.df)} for query '{q_name}' in stream-{stream_idx}. "
            ))
            labeling_budget = len(data.df) // 2
            logger.info((
                f"Adjusted labeled budget to {labeling_budget} for query '{q_name}' in stream-{stream_idx}."
            ))

        labeled_indices = data.df.sample(n=labeling_budget, random_state=seed).index

        # Check whether the sampled data contains both positive and negative samples.
        num_pos_sampled = labels.loc[labeled_indices].sum()
        num_neg_sampled = len(labeled_indices) - num_pos_sampled
        minority_bias = min(2, b_lab - labeling_budget)  # Add bias to avoid single-class situation.
        if num_pos_sampled == 0 and minority_bias > 0:
            pos_indices = labels[labels == True].index
            pos_to_add = min(minority_bias, len(pos_indices))
            labeled_indices = labeled_indices.union(pos_indices[:pos_to_add])
            logger.info((
                f"Minority class (positive) is not sampled for query '{q_name}' in stream-{stream_idx}. "
                f"Added {pos_to_add} positive samples to the labeled set. "
                f"Current labeled set has {labels.loc[labeled_indices].sum()} pos samples out of {len(labeled_indices)} samples."
            ))
        if num_neg_sampled == 0 and minority_bias > 0:
            neg_indices = labels[labels == False].index
            neg_to_add = min(minority_bias, len(neg_indices))
            labeled_indices = labeled_indices.union(neg_indices[:neg_to_add])
            logger.info((
                f"Minority class (negative) is not sampled for query '{q_name}' in stream-{stream_idx}. "
                f"Added {neg_to_add} negative samples to the labeled set. "
                f"Current labeled set has {labels.loc[labeled_indices].sum()} neg samples out of {len(labeled_indices)} samples."
            ))

        remaining_indices = data.df.index.difference(labeled_indices)

        # Move the labeled data from `sigma_satisfied_data` to the `coreset`.
        if q_name not in self.coresets.keys():
            self.coresets[q_name] = {
                "ldb_data": LdbData(
                    df=data.df.loc[labeled_indices].reset_index(drop=True), 
                    config=data.config),
                "labels": labels.loc[labeled_indices].reset_index(drop=True),
                "observed_size": len(labeled_indices),
                "lb": float('inf'),
                "ub": float('-inf'),
            }
        else:
            logger.info((
                f"[W] Coreset for query '{q_name}' already exists. "
            ))
            self.coresets[q_name]["ldb_data"].df = pd.concat([
                self.coresets[q_name]["ldb_data"].df,
                data.df.loc[labeled_indices].reset_index(drop=True)
            ], ignore_index=True)
            self.coresets[q_name]["labels"] = pd.concat([
                self.coresets[q_name]["labels"],
                labels.loc[labeled_indices].reset_index(drop=True)
            ], ignore_index=True)
            self.coresets[q_name]["observed_size"] += len(labeled_indices)

        # Update the `sigma_satisfied_data`.
        num_tp = labels.loc[labeled_indices].sum()
        self.sigma_satisfied_data[stream_idx][q_name]["num_tp"] = num_tp
        self.sigma_satisfied_data[stream_idx][q_name]["ldb_data"].df = \
            data.df.loc[remaining_indices].reset_index(drop=True)
        self.sigma_satisfied_data[stream_idx][q_name]["labels"] = \
            labels.loc[remaining_indices].reset_index(drop=True)

        logger.info((
            f"Acquired labels for {q_name}: "
            f"{num_tp} pos samples / {len(labeled_indices)} total samples."
        ))
    