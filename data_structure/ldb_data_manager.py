import pandas as pd
import logging
import random
from pathlib import Path
from llm import LdbLLMClient
from .ldb_data import LdbData
from .sem_query import SemCQ, Predicate
from .llm_resp_templates import PopulationSpec

logger = logging.getLogger(__name__)


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

        self.enriched_features: list[PopulationSpec] = []

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
                    "ground_truth": pd.Series,
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
            indices = indices[data_ladder[i-1]:data_ladder[i]]
            df = self.complete_dataset.df.iloc[indices].copy().reset_index(drop=True)

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
            ucq: list[list[Predicate]],
            stream_idx: int = 0) -> None:
        """
        [Optional]
        Refine (narrow down) the Sigma-satisfied data.
        """
        self._apply_query_sigma_and_build_ground_truth(
            stream_idx=stream_idx,
            q_name=q_name,
            ucq=ucq
        )


    def acquire_annotation_and_init_coreset(
            self, b_lab: int, seed: int = 42) -> None:
        for idx in range(len(self.data_stream)):
            for q_name in self.queries.keys():
                self._acquire_query_annotation_and_init_coreset(
                    b_lab=b_lab, 
                    q_name=q_name, 
                    stream_idx=idx, 
                    seed=seed
                )


    def sync_coreset_features(
            self,
            enriched_features: list[PopulationSpec],
            enable_cache: bool = True,
            is_remote: bool = False) -> dict:

        for q_name, coreset in self.coresets.items():
            ckpt_path = self.CKPT_path / f"{q_name}_coreset.csv"
            if enable_cache and ckpt_path.exists():
                logger.info(f"Loading enriched coreset for query '{q_name}' from cache.")
                coreset["ldb_data"].df = pd.read_csv(ckpt_path)
                continue

            coreset["ldb_data"].sync_with_enriched_features(
                enriched_features=enriched_features,
                llm_client=self.llm_client,
                is_remote=is_remote
            )

        llm_usage_statistics = self.llm_client.get_usage_statistics() 
        self.llm_client.reset_usage_statistics()
        return llm_usage_statistics


    def sync_sigma_satisfied_data_features(
            self,
            enriched_features: list[PopulationSpec],
            enable_cache: bool = True,
            is_remote: bool = False) -> dict:

        for stream_idx in range(len(self.data_stream)):
            for q_name in self.queries.keys():
                ckpt_path = self.CKPT_path / (
                    f"{q_name}_stream_{stream_idx}_sigma_satisfied_data.csv"
                )
                if enable_cache and ckpt_path.exists():
                    logger.info((
                        f"Loading enriched Sigma-satisfied data for "
                        f"query '{q_name}' in stream-{stream_idx} from cache."
                    ))
                    self.sigma_satisfied_data[stream_idx][q_name]['ldb_data'].df = pd.read_csv(ckpt_path)
                    continue

                self.sigma_satisfied_data[stream_idx][q_name]['ldb_data']\
                    .sync_with_enriched_features(
                        enriched_features=enriched_features,
                        llm_client=self.llm_client,
                        is_remote=is_remote
                    )
        
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
            self.sigma_satisfied_data[stream_idx][q_name] = {
                "ldb_data": self.data_stream[stream_idx].sigma_retrieve_ucq(ucq, reset_index=True),
                "ground_truth": None,
                "num_fn": 0,
                "num_tp": 0,
            }
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
        selected_cols = self.queries[q_name].selected
        ground_truth_df = pd.read_csv(
            f"{self.data_dir}/ground_truth/{q_name}.csv"
        )[selected_cols]
        ground_truth_set = set(tuple(row) for row in ground_truth_df.values)
                
        labels = self.sigma_satisfied_data[stream_idx][q_name]['ldb_data'].df[selected_cols].apply(
            lambda row: tuple(row) in ground_truth_set,
            axis=1
        ).reset_index(drop=True)
        self.sigma_satisfied_data[stream_idx][q_name]["ground_truth"] = labels

        # Integrity check.
        true_rows = self.sigma_satisfied_data[stream_idx][q_name]['ldb_data'].df[labels][selected_cols]
        true_rows_set = set(tuple(row) for row in true_rows.values)
        assert true_rows_set.issubset(ground_truth_set), (
            f"Fail to build ground truth for query '{q_name}' in stream-{stream_idx}."
        )
        if len(true_rows_set) < len(ground_truth_set):
            num_fn = len(ground_truth_set) - len(true_rows_set)
            self.sigma_satisfied_data[stream_idx][q_name]["num_fn"] = num_fn
            logger.info((
                f"[W] Sigma-retrieving introduces {num_fn} FNs for {q_name} in stream-{stream_idx} "
            ))

        # Report the oracle selectivity.
        logger.info((
            f"Ground truth of {q_name}-stream-{stream_idx} has {labels.sum()} pos samples "
            f"out of {len(labels)} samples, "
            f"oracle selectivity = {labels.mean():.4f}."
        ))


    def _acquire_query_annotation_and_init_coreset(
            self, q_name: str, b_lab: int, stream_idx: int = 0, seed: int = 42) -> None:

        data = self.sigma_satisfied_data[stream_idx][q_name]["data"]
        labels = self.sigma_satisfied_data[stream_idx][q_name]["labels"]

        if b_lab >= len(data.df):
            logger.info((
                f"[W] Requested labeled budget {b_lab} exceeds the data scale "
                f"{len(data.df)} for query '{q_name}' in stream-{stream_idx}. "
            ))
            b_lab = len(data.df) // 2
            logger.info((
                f"Adjusted labeled budget to {b_lab} for query '{q_name}' in stream-{stream_idx}."
            ))

        labeled_indices = data.df.sample(n=b_lab, random_state=seed).index

        """ [Backup]
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
        """

        remaining_indices = data.df.index.difference(labeled_indices)

        # Move the labeled data from `sigma_satisfied_data` to the `coreset`.
        if q_name not in self.coresets.keys():
            self.coresets[q_name] = {
                "ldb_data": data.df.loc[labeled_indices].reset_index(drop=True),
                "labels": labels.loc[labeled_indices].reset_index(drop=True),
                "lb": float('inf'),
                "ub": float('-inf'),
            }
        else:
            logger.info((
                f"[W] Coreset for query '{q_name}' already exists. "
            ))
            self.coresets[q_name]["ldb_data"] = pd.concat([
                self.coresets[q_name]["ldb_data"],
                data.df.loc[labeled_indices].reset_index(drop=True)
            ], ignore_index=True)
            self.coresets[q_name]["labels"] = pd.concat([
                self.coresets[q_name]["labels"],
                labels.loc[labeled_indices].reset_index(drop=True)
            ], ignore_index=True)

        # Update the `sigma_satisfied_data`.
        num_tp = labels.loc[labeled_indices].sum()
        self.sigma_satisfied_data[stream_idx][q_name]["num_tp"] += num_tp
        self.sigma_satisfied_data[stream_idx][q_name]["ground_truth"] = \
            labels.loc[remaining_indices].reset_index(drop=True)

        logger.info((
            f"Acquired labels for {q_name}: "
            f"{num_tp} pos samples / {len(labeled_indices)} total samples."
        ))
    