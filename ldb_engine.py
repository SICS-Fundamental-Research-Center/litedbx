import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Union, Literal, Dict, Type
from llm_client import LiteLLMWrapper, BooleanFeatureResponse
import asyncio
from pydantic import BaseModel
from time import time


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
    rewritten_rules: List[
        Tuple[
            str,
            Literal["Eq", "Gt", "Lt", "Ge", "Le", "In"],
            Union[str, int, float, bool, List[Union[str, int, float, bool]]],
        ]
    ]


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


class PopulationSpec(BaseModel):
    source_col: str
    target_col: str
    prompt: str
    var_type: Literal["bool", "float", "int"]


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


    async def apply(self, queries: List[str]) -> None:

        # Step 1: Prefilter easy samples using static and semantic rules.
        retrieved_dfs, remaining_dfs = {}, {}
        for query_name in queries:

            workload = self.workloads[query_name]

            # Step 1.1: Filter the data based on static rules.
            static_view = self._prefilter_by_static_rules(self.database, workload)
            print(f"After static prefiltering: {len(static_view)}/{len(self.database)} rows remain.")

            # Step 1.2: Filter the data based on semantic rules.
            early_positive_view, sem_view = \
                await self._prefilter_by_semantic_rules(static_view, workload,
                                                        early_positive=False, 
                                                        drop_neg=True)
            early_positive_view = early_positive_view[workload.select_cols]

            retrieved_dfs[query_name] = early_positive_view.reset_index(drop=True)
            print(f"After semantic prefiltering: {len(early_positive_view)}/{len(static_view)} rows retrieved.")
            remaining_dfs[query_name] = sem_view.reset_index(drop=True)
            print(f"After semantic prefiltering: {len(sem_view)}/{len(static_view)} rows remain.")

        # Step 2: Generate LLM-based pseudo-labels for the remaining samples.
        for query_name, remaining_df in remaining_dfs.items():
            workload = self.workloads[query_name]
            for cq in workload.rules:
                for col_name, semantic_desc in cq.sem_rules:
                    new_col_name = f"{col_name}_sem_mapped"
                    remaining_df = await self._sem_mapping(
                        remaining_df,
                        col_name,
                        new_col_name,
                        semantic_desc,
                        BooleanFeatureResponse
                    )
                cq.rewritten_rules.append(
                    (new_col_name, "Eq", True)
                )
        
        # Step 3: Generate feature space for each queries.
        #   TODO: Generate feature space JOINTLY for all queries.

            
        # Finally, apply the rewritten rules and evaluate against ground truth.
        for query_name in queries:
            retrieved_df = retrieved_dfs[query_name]
            remaining_df = remaining_dfs[query_name]
            workload = self.workloads[query_name]
            filtered_df = self._filter_by_rewritten_rules(remaining_df, workload)
            print(f"After filtering by rewritten rules: {len(filtered_df)}/{len(remaining_df)} rows retrieved.")
            final_df = pd.concat([retrieved_df, filtered_df], ignore_index=True).drop_duplicates()
            print(f"After applying rewritten rules: {len(final_df)}/{len(self.database)} rows retrieved.")

            final_df.to_csv(self.dataset_path / f"retrieved_{query_name}.csv", index=False)
            assert self.ground_truth[query_name] is not None, f"Ground truth for {query_name} not found."

            result_set = set(
                tuple(row)
                for row in final_df[workload.select_cols].itertuples(index=False, name=None)
            )
            gt_set = self.ground_truth[query_name]
            eval_result = self._evaluate(result_set, gt_set)


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

    async def _prefilter_by_semantic_rules(self, 
                                     df: pd.DataFrame, 
                                     query: UCQ,
                                     early_positive: bool = True,
                                     drop_neg: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
        
        fired_df = df.copy()
        sem_flags = []
        for cq in query.rules:
            # Reset the sem_flag column.
            fired_df["sem_flag"] = 0

            # Color using one collection of conjunctive rules.
            colored_df = await self._sem_coloring(fired_df, cq.sem_rules)

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

        early_positive_df, result_df = pd.DataFrame(), pd.DataFrame()

        if early_positive:
            early_positive_df = fired_df[fired_df["sem_flag"] == 1].drop(columns=["sem_flag"])
        else:
            early_positive_df = pd.DataFrame(columns=fired_df.columns).drop(columns=["sem_flag"])

        if drop_neg and early_positive:
            early_positive_df = fired_df[fired_df["sem_flag"] == 1].drop(columns=["sem_flag"])
            result_df = fired_df[fired_df["sem_flag"] == 0].drop(columns=["sem_flag"])
        elif drop_neg and not early_positive:
            early_positive_df = pd.DataFrame(columns=fired_df.columns).drop(columns=["sem_flag"])
            result_df = fired_df[(fired_df["sem_flag"] == 0) | (fired_df["sem_flag"] == 1)].drop(columns=["sem_flag"])
        elif early_positive and not drop_neg:
            early_positive_df = fired_df[fired_df["sem_flag"] == 1].drop(columns=["sem_flag"])
            result_df = fired_df[fired_df["sem_flag"] == 0 | (fired_df["sem_flag"] == -1)].drop(columns=["sem_flag"])
        else:
            early_positive_df = pd.DataFrame(columns=fired_df.columns).drop(columns=["sem_flag"])
            result_df = fired_df.drop(columns=["sem_flag"])

        return early_positive_df, result_df

    def _filter_by_rewritten_rules(self, df: pd.DataFrame, query: UCQ) -> pd.DataFrame:
        result = pd.DataFrame()

        for cq in query.rules:
            df_cp = df.copy()
            for col, op, val in cq.rewritten_rules:
                print(f"Applying rewritten rule: {col} {op} {val}")
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

    async def _sem_coloring(
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
            consensus_results = await self.llm_client.invoke_parallel_consensus(
                modality=modality,
                prompt=semantic_desc,
                data_items=df_cp[col_name].astype(str).tolist(),
                response_model=BooleanFeatureResponse,
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

    async def _sem_mapping(
            self,
            df: pd.DataFrame,
            col_name: str,
            new_col_name: str,
            prompt: str,
            response_model: Type[BaseModel]) -> pd.DataFrame:
        
        data_items = df[col_name].astype(str).tolist()
        modality = self._detect_modality(data_items[0] if data_items else "")
        llm_labels = await self.llm_client.invoke_parallel(
            modality=modality,
            is_remote=True,
            prompt=prompt,
            data_items=data_items,
                response_model=response_model,
        )
        df[new_col_name] = llm_labels
        print(f"Semantic mapping column '{new_col_name}' added.")

        return df


    async def _generate_feature_space(
            self,
            workloads: Dict[str, UCQ],
            data_views: Dict[str, pd.DataFrame],) -> Dict[str, PopulationSpec]:
        
        population_specs = {}
        for query_name, workload in workloads.items():
            pass

        return population_specs


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
                rewritten_rules=[]
            ),
        ],
    )

    ldb = LDBEngine(
        dataset_name="medical",
        workloads={"Q1": q1}
    )

    start = time()

    asyncio.run(ldb.apply(["Q1"]))

    end = time()
    print(f"Total execution time: {end - start} seconds")
