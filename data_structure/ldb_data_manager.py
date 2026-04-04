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

    
    def eval_query_quality(self, q_name: str, selected_cols: list[str], 
                           stream_idx: int, pred_labels: list[pd.Series]) -> dict:
        ground_truth, retrieved_data = set(), set()
        for sid in range(stream_idx + 1):
            # 1. Process the sigma-satisfied data.
            ss_data = self.sigma_satisfied_data[sid][q_name]
            ss_df = ss_data["ldb_data"].df
            ground_truth_labels = ss_data["labels"]
            retrieved_labels = pred_labels[sid].astype(bool)
            ground_truth.update(ss_df[ground_truth_labels]\
                                .apply(lambda row: tuple(row[selected_cols]), axis=1))
            retrieved_data.update(ss_df[retrieved_labels]\
                                  .apply(lambda row: tuple(row[selected_cols]), axis=1))
            
            # 2. Involve the selected items with human annotation.
            selected_df = ss_data["selected_data"]
            selected_labels = ss_data["selected_labels"]
            assert sum(selected_labels) == len(selected_df), (
                "All selected data should be labeled as positive samples."
            )
            ground_truth.update(selected_df.apply(lambda row: tuple(row[selected_cols]), axis=1))
            retrieved_data.update(selected_df.apply(lambda row: tuple(row[selected_cols]), axis=1))

            # 3. Involve the discarded items.
            discarded_df = ss_data["discarded_data"]
            discarded_labels = ss_data["discarded_labels"]
            if sum(discarded_labels) > 0:
                logger.warning((
                    f"Found {sum(discarded_labels)} positive samples in the discarded data "
                    f"for query '{q_name}' in stream-{sid}."
                ))
                ground_truth.update(discarded_df[discarded_labels]\
                                        .apply(lambda row: tuple(row[selected_cols]), axis=1))

        # Compute the evaluation metrics.
        TP = len(ground_truth.intersection(retrieved_data))
        FP = len(retrieved_data - ground_truth)
        FN = len(ground_truth - retrieved_data)

        # Avoid the case when there is no positive sample in the ground truth, which leads to TP=FP=FN=0.
        if FP == 0 and FN == 0 and TP == 0:
            logger.info((
                f"Both prediction and ground truth are empty for query '{q_name}' in stream-{stream_idx}."
            ))
            return {
                'f1': 1.0,
                'precision': 1.0,
                'recall': 1.0,
                'TP': TP,
                'FP': FP,
                'FN': FN,
            }

        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        return {
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'TP': TP,
            'FP': FP,
            'FN': FN,
        }


        

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
            selected_indices = self.data_stream[stream_idx].sigma_retrieve_ucq(ucq)
            selected_data = self.data_stream[stream_idx].df\
                .loc[selected_indices].reset_index(drop=True).copy()
            self.sigma_satisfied_data[stream_idx][q_name] = {
                "ldb_data": LdbData(df=selected_data, config=self.complete_dataset.config),
                "labels": None,
                "propagated_labels": None,
                "selected_data": pd.DataFrame(),
                "discarded_data": pd.DataFrame(),
                "selected_labels": pd.Series(),
                "discarded_labels": pd.Series(),
            }
            logger.info((
                f"Applied Sigma retrieval for query '{q_name}' in stream-{stream_idx}: "
                f"{prev_data_scale} -> {len(self.sigma_satisfied_data[stream_idx][q_name]['ldb_data'].df)} rows."
            ))
        else:
            # Refined sigma retrieval
            assert self.sigma_satisfied_data[stream_idx][q_name]['ldb_data'] is not None and \
                self.sigma_satisfied_data[stream_idx][q_name]['labels'] is not None, (
                "Sigma-satisfied data and labels should be initialized before refining Sigma retrieval."
            )
            prev_data_scale = len(self.sigma_satisfied_data[stream_idx][q_name]['ldb_data'].df)

            selected_indices = self.sigma_satisfied_data[stream_idx][q_name]['ldb_data']\
                .sigma_retrieve_ucq(ucq)
            selected_data = self.sigma_satisfied_data[stream_idx][q_name]['ldb_data'].df\
                .loc[selected_indices].reset_index(drop=True).copy()
            discarded_data = self.sigma_satisfied_data[stream_idx][q_name]['ldb_data'].df\
                .drop(index=selected_indices).reset_index(drop=True).copy()
            selected_labels = self.sigma_satisfied_data[stream_idx][q_name]['labels']\
                .loc[selected_indices].reset_index(drop=True).copy()
            discarded_labels = self.sigma_satisfied_data[stream_idx][q_name]['labels']\
                .drop(index=selected_indices).reset_index(drop=True).copy()
            
            self.sigma_satisfied_data[stream_idx][q_name]['ldb_data'] = \
                LdbData(df=selected_data, config=self.complete_dataset.config)
            self.sigma_satisfied_data[stream_idx][q_name]['labels'] = selected_labels
            self.sigma_satisfied_data[stream_idx][q_name]['discarded_data'] = \
                pd.concat([
                    self.sigma_satisfied_data[stream_idx][q_name]['discarded_data'], 
                    discarded_data
                ], ignore_index=True)
            self.sigma_satisfied_data[stream_idx][q_name]['discarded_labels'] = \
                pd.concat([
                    self.sigma_satisfied_data[stream_idx][q_name]['discarded_labels'], 
                    discarded_labels
                ], ignore_index=True)

            introduced_fn = sum(self.sigma_satisfied_data[stream_idx][q_name]['discarded_labels'])

            post_data_scale = len(self.sigma_satisfied_data[stream_idx][q_name]['ldb_data'].df)
            logger.info((
                f"Refined Sigma-satisfied data for query '{q_name}' in stream-{stream_idx}: "
                f"{prev_data_scale} -> {post_data_scale} rows. "
                f"Est. # FNs: {introduced_fn}."
            ))

        # ==============================================
        # (2) Build ground truth.
        # ==============================================

        if self.sigma_satisfied_data[stream_idx][q_name]['labels'] is not None:
            logger.info((
                f"Ground truth for query '{q_name}' in stream-{stream_idx} already exists. "
                f"Skip building ground truth."
            ))
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

        self.sigma_satisfied_data[stream_idx][q_name]["labels"] = labels

        # Report the oracle selectivity.
        logger.info((
            f"Ground truth of {q_name}-stream-{stream_idx} has {labels.sum()} pos samples "
            f"out of {len(labels)} samples, "
            f"oracle selectivity = {labels.mean():.4f}."
        ))


    def _acquire_query_annotation_and_init_coreset(
            self, q_name: str, b_lab: int, stream_idx: int = 0, seed: int = 42) -> None:

        data = self.sigma_satisfied_data[stream_idx][q_name]["ldb_data"].df.copy()
        labels = self.sigma_satisfied_data[stream_idx][q_name]["labels"]

        labeling_budget = b_lab
        if labeling_budget >= len(data):
            logger.info((
                f"[W] Requested labeled budget {labeling_budget} exceeds the data scale "
                f"{len(data)} for query '{q_name}' in stream-{stream_idx}. "
            ))
            labeling_budget = len(data) // 2
            logger.info((
                f"Adjusted labeled budget to {labeling_budget} for query '{q_name}' in stream-{stream_idx}."
            ))

        labeled_indices = data.sample(n=labeling_budget, random_state=seed).index

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

        remaining_indices = data.index.difference(labeled_indices)

        # Move the labeled data from `sigma_satisfied_data` to the `coreset`.
        if q_name not in self.coresets.keys():
            self.coresets[q_name] = {
                "ldb_data": LdbData(
                    df=data.loc[labeled_indices].reset_index(drop=True).copy(), 
                    config=self.complete_dataset.config),
                "labels": labels.loc[labeled_indices].reset_index(drop=True).copy(),
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
                data.loc[labeled_indices].reset_index(drop=True).copy()
            ], ignore_index=True)
            self.coresets[q_name]["labels"] = pd.concat([
                self.coresets[q_name]["labels"],
                labels.loc[labeled_indices].reset_index(drop=True).copy()
            ], ignore_index=True)
            self.coresets[q_name]["observed_size"] += len(labeled_indices)

        # Update the `sigma_satisfied_data`.
        self.sigma_satisfied_data[stream_idx][q_name]["ldb_data"].df = \
            data.loc[remaining_indices].reset_index(drop=True).copy()
        self.sigma_satisfied_data[stream_idx][q_name]["labels"] = \
            labels.loc[remaining_indices].reset_index(drop=True).copy()

        # Update the selected and discarded data for analysis.
        pos_labeled_indices = labeled_indices[labels.loc[labeled_indices] == True]
        neg_labeled_indices = labeled_indices[labels.loc[labeled_indices] == False]
        self.sigma_satisfied_data[stream_idx][q_name]["selected_data"] = \
            pd.concat([
                self.sigma_satisfied_data[stream_idx][q_name]["selected_data"],
                data.loc[pos_labeled_indices].reset_index(drop=True).copy()
            ], ignore_index=True)
        self.sigma_satisfied_data[stream_idx][q_name]["selected_labels"] = \
            pd.concat([
                self.sigma_satisfied_data[stream_idx][q_name]["selected_labels"],
                labels.loc[pos_labeled_indices].reset_index(drop=True).copy()
            ], ignore_index=True)
        self.sigma_satisfied_data[stream_idx][q_name]["discarded_data"] = \
            pd.concat([
                self.sigma_satisfied_data[stream_idx][q_name]["discarded_data"],
                data.loc[neg_labeled_indices].reset_index(drop=True).copy()
            ], ignore_index=True)
        self.sigma_satisfied_data[stream_idx][q_name]["discarded_labels"] = \
            pd.concat([
                self.sigma_satisfied_data[stream_idx][q_name]["discarded_labels"],
                labels.loc[neg_labeled_indices].reset_index(drop=True).copy()
            ], ignore_index=True)
        logger.info((
            f"Initialized coreset for query '{q_name}' in stream-{stream_idx} with "
            f"{len(labeled_indices)} labeled samples. "
            f"Remaining Sigma-satisfied data has {len(remaining_indices)} samples."
        ))
