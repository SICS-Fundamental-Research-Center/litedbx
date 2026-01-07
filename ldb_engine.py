import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Union, Literal, Dict
from llm_client import LiteLLMWrapper, BooleanFeatureResponse
import asyncio


@dataclass
class CQ:
    static_rules: List[
        Tuple[
            str,
            Literal["Eq", "Gt", "Lt", "Ge", "Le", "In"],
            Union[str, int, float, List[Union[str, int, float]]],
        ]
    ]
    sem_rules: List[Tuple[str, str]]


@dataclass
class UCQ:
    select_cols: List[str]
    rules: List[CQ]


@dataclass
class EvalResult:
    tp: int
    fp: int
    fn: int
    precision: float
    recall: float
    f1: float


class LDBEngine:
    def __init__(self,
                 dataset_name: str, 
                 workloads: Dict[str, UCQ]):
        
        print("Initializing LDBEngine...")
        self.WD = Path(__file__).parent
        self.dataset_path = self.WD / "data" / dataset_name
        self.ground_truth_paths = self.dataset_path / "ground_truth"
        self.database = \
            pd.read_csv(self.dataset_path / "data_full.csv").reset_index(drop=True)
        print(f"Loaded database {dataset_name} with {len(self.database)} rows.")

        self.workloads = workloads
        self.ground_truth = self._init_ground_truth(list(workloads.keys()))
        print("Loaded ground truths.")
        self.llm_client = LiteLLMWrapper()
        print("Initialized LDBEngine.")


    def apply(self, queries: List[str]) -> None:

        # Step 1: Prefilter easy samples using static and semantic rules.
        retrieved_df, remaining_df = [], []
        for query_name in queries:

            workload = self.workloads[query_name]

            # Step 1: Filter the data based on static rules.
            static_view = self._prefilter_by_static_rules(self.database, workload)
            print(f"After static prefiltering: {len(static_view)}/{len(self.database)} rows remain.")

            # Step 2: Filter the data based on semantic rules.
            early_positive_view, sem_view = \
                self._prefilter_by_semantic_rules(static_view, workload, drop_neg=True)
            early_positive_view = early_positive_view[workload.select_cols]

            retrieved_df.append(early_positive_view.reset_index(drop=True))
            remaining_df.append(sem_view.reset_index(drop=True))
            print(f"After semantic prefiltering: {len(sem_view)}/{len(static_view)} rows remain.")


        # Finally, evaluate against ground truth.
        for query_name, retrieved_df in zip(queries, retrieved_df):
            retrieved_df.to_csv(self.dataset_path / f"retrieved_{query_name}.csv", index=False)
            assert self.ground_truth[query_name] is not None, f"Ground truth for {query_name} not found."
            retrieved_set = set(
                tuple(row)
                for row in retrieved_df[workload.select_cols].itertuples(index=False, name=None)
            )
            gt_set = self.ground_truth[query_name]
            eval_result = self._evaluate(retrieved_set, gt_set)


    def _init_ground_truth(self, query_names: List[str]) -> Dict[str, set]:
        assert self.workloads is not None, \
            "Workloads must be provided to initialize ground truths."

        ground_truths = {}
        for query_name in query_names:
            gt_df = pd.read_csv(self.ground_truth_paths / f"{query_name}.csv").reset_index(drop=True)
            gt_df = gt_df[self.workloads[query_name].select_cols]
            gt_set = set(
                tuple(row)
                for row in gt_df.itertuples(index=False, name=None)
            )
            ground_truths[query_name] = gt_set
        return ground_truths

    def _prefilter_by_static_rules(self, df: pd.DataFrame, query: UCQ) -> pd.DataFrame:
        result = pd.DataFrame()

        # Make sure all selected columns cannot have null values.
        involved_cols = set()
        for cq in query.rules:
            for col, _, _ in cq.static_rules:
                involved_cols.add(col)
            for col, _ in cq.sem_rules:
                involved_cols.add(col)
        df = df.dropna(subset=list(involved_cols))

        # Apply static rules.
        for cq in query.rules:
            df_cp = df.copy()
            for col, op, val in cq.static_rules:
                if op == "Eq":
                    df_cp = df_cp[df_cp[col] == val]
                elif op == "Gt":
                    df_cp = df_cp[df_cp[col] > val]
                elif op == "Lt":
                    df_cp = df_cp[df_cp[col] < val]
                elif op == "Ge":
                    df_cp = df_cp[df_cp[col] >= val]
                elif op == "Le":
                    df_cp = df_cp[df_cp[col] <= val]
                elif op == "In":
                    df_cp = df_cp[df[col].isin(val)]  # type: ignore
                else:
                    raise ValueError(f"Unsupported operation: {op}")
            result = pd.concat([result, df_cp], ignore_index=True)
        return result.drop_duplicates()

    def _prefilter_by_semantic_rules(self, 
                                     df: pd.DataFrame, 
                                     query: UCQ,
                                     drop_neg: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
        
        fired_df = df.copy()
        sem_flags = []
        for cq in query.rules:
            # Reset the sem_flag column.
            fired_df["sem_flag"] = 0

            # Color using one collection of conjunctive rules.
            colored_df = self._sem_coloring(fired_df, cq.sem_rules)

            # Collect the sem_flags.
            sem_flags.append(colored_df["sem_flag"].tolist())

        # Aggregate the sem_flags to sem_flag.
        aggregated_flags = []
        for i in range(len(sem_flags[0])):
            values_at_index = [flags[i] for flags in sem_flags]
            if 1 in values_at_index:
                aggregated_flags.append(1)
            elif all(v == 0 for v in values_at_index):
                aggregated_flags.append(0)
            else:
                aggregated_flags.append(-1)

        fired_df["sem_flag"] = aggregated_flags
        early_positive_df = fired_df[fired_df["sem_flag"] == 1]
        result_df = fired_df[fired_df["sem_flag"] == 0] if drop_neg \
            else fired_df[fired_df["sem_flag"] == 0 | (fired_df["sem_flag"] == -1)]

        return early_positive_df.drop(columns=["sem_flag"]), result_df.drop(columns=["sem_flag"])

    def _sem_coloring(
            self,
            df: pd.DataFrame,
            sem_rules: List[Tuple[str, str]],) -> pd.DataFrame:

        """
        Filter high-confidence negative samples or
        low-confidence positive samples based on, e.g., proxy models.
        """

        df_cp = df.copy()
        if "sem_flag" not in df_cp.columns:
            df_cp["sem_flag"] = 0

        for col_name, semantic_desc in sem_rules:

            modality = self._detect_modality(df_cp[col_name].iloc[0])
            consensus_results = asyncio.run(
                self.llm_client.invoke_parallel_consensus(
                    modality=modality,
                    prompt=semantic_desc,
                    data_items=df_cp[col_name].astype(str).tolist(),
                    response_model=BooleanFeatureResponse,
                )
            )
            for result in consensus_results:
                pos, is_match = result
                col_idx = df_cp.columns.get_loc("sem_flag")  # type: ignore
                if df_cp.iat[pos, col_idx] == -1:  # type: ignore
                    continue
                df_cp.iat[pos, col_idx] = 1 if is_match else -1  # type: ignore

        return df_cp
        
    def _detect_modality(self, data_item: str) -> str:
        if any(
            data_item.endswith(extension) for extension in [".png", ".jpg", ".jpeg"]
        ):
            return "IMAGE"
        else:
            return "TEXT"

    def _evaluate(self, retrieved_set: set, gt_set: set) -> EvalResult:
        tp = len(retrieved_set & gt_set)
        fp = len(retrieved_set - gt_set)
        fn = len(gt_set - retrieved_set)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        print(f"TP: {tp}, FP: {fp}, FN: {fn}")
        print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
        return EvalResult(tp=tp, fp=fp, fn=fn, precision=precision, recall=recall, f1=f1)


if __name__ == "__main__":
    q1 = UCQ(
        select_cols=["patient_id"],
        rules=[
            CQ(
                static_rules=[],
                sem_rules=[("symptoms", (
                    "You are a medical expert." 
                    "Please determine if the following symptoms indicate an allergy."
                    "Please JUST answer \"True\" if they do, and \"False\" otherwise."
                    "Do NOT provide any explanations."))],
            ),
        ],
    )

    ldb = LDBEngine(
        dataset_name="medical",
        workloads={"Q1": q1}
    )

    ldb.apply(["Q1"])

