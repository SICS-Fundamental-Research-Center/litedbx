import asyncio
import json
from pydantic import BaseModel
from time import time
import pandas as pd
import random
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple, Union, Literal, Dict, Type
from llm_client import LiteLLMWrapper, BooleanFeatureResponse, IntFeatureResponse, FloatFeatureResponse
from prompts import PROMPTS
from ldb_classifier import RuleClassifier


@dataclass
class CQ:
    static_rules: List[
        Tuple[
            str,
            Literal["Eq", "Gt", "Lt", "Ge", "Le", "In"],
            Union[str, int, float, List[Union[str, int, float]]],
        ]
    ] = field(default_factory=list)
    sem_rules: List[Tuple[str, str]] = field(default_factory=list)
    backup_rules: List[
        Tuple[
            str,
            Literal["Eq", "Gt", "Lt", "Ge", "Le", "In"],
            Union[str, int, float, bool, List[Union[str, int, float, bool]]],
        ]
    ] = field(default_factory=list)
    rewritten_pos_rules: List[
        Tuple[
            str,
            Literal["Eq", "Gt", "Lt", "Ge", "Le", "In"],
            Union[str, int, float, bool, List[Union[str, int, float, bool]]],
        ]
    ] = field(default_factory=list)
    rewritten_neg_rules: List[
        Tuple[
            str,
            Literal["Eq", "Gt", "Lt", "Ge", "Le", "In"],
            Union[str, int, float, bool, List[Union[str, int, float, bool]]],
        ]
    ] = field(default_factory=list)


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
    feature_type: Literal["bool", "float", "int"]

class PopulationSpecs(BaseModel):
    value: List[PopulationSpec]


class LDBEngine:
    def __init__(self,
                 dataset_name: str, 
                 workloads: Dict[str, UCQ],
                 feature_enrich_budget: int = 3,
                 query_rewrite_budget: int = 3,
                 external_keys: List[str] = []) -> None:
        
        self.feature_enrich_budget = feature_enrich_budget
        self.query_rewrite_budget = query_rewrite_budget
        self.external_keys = external_keys
        
        print("Initializing LDBEngine...")
        self.WD = Path(__file__).parent
        self.dataset_path = self.WD / "data" / dataset_name
        self.ckpt_home = self.dataset_path / ".ckpt"
        self.ckpt_home.mkdir(parents=True, exist_ok=True)
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
            static_view = self._prefilter_by_static_rules(self.database, query_name, enable_cache=True)
            retrieved_dfs[query_name] = pd.DataFrame(columns=static_view.columns)
            remaining_dfs[query_name] = static_view.reset_index(drop=True)
            print(f"After static prefiltering: {len(static_view)}/{len(self.database)} rows remain.")

            # Step 1.2: Filter the data based on semantic rules.
            early_positive_view, sem_view = \
                await self._prefilter_by_semantic_rules(static_view, query_name,
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
                cq.backup_rules.append(
                    (new_col_name, "Eq", True)
                )
        
        # Step 3: Generate feature space for each queries.
        #   TODO: Generate feature space JOINTLY for all queries.
        population_specs = await self._generate_feature_space(
            self.workloads,
            remaining_dfs,
            PopulationSpecs
        )

        # Step 4: Iteratively enrich the database and learn UCQs.
        num_selected_features = 0
        while num_selected_features < self.feature_enrich_budget:

            for query_name, ucq in self.workloads.items():
                pop_specs = population_specs[query_name]
                remaining_df = remaining_dfs[query_name]
                
                # Step 4.1 Feature pre-selection.
                # TODO: For testing, we only accept the first 3 features.
                pre_selected_specs = pop_specs.value[:self.feature_enrich_budget]

                # Step 4.2 Feature population.
                remaining_df = await self._sem_multi_mapping(
                    remaining_df,
                    PopulationSpecs(value=pre_selected_specs),
                    enable_cache=True
                )

                # Step 4.3 Feature selection.
                # TODO: For testing, we directly accept all selected features.
                num_selected_features += self.feature_enrich_budget

                # Step 4.3 UCQ learning.
                ground_truth_set = self.ground_truth[query_name]
                remaining_df["label"] = remaining_df.apply(
                    lambda row: tuple(row[ucq.select_cols]) in ground_truth_set,
                    axis=1
                )
                exclude_cols = self.external_keys + ucq.select_cols
                for cq in ucq.rules:
                    for col, _, _ in cq.backup_rules:
                        exclude_cols.append(col)

                # Generate classification rules and get pos/neg rules
                pos_rules, neg_rules = self._generate_classification_rules(
                    query_name, remaining_df.drop(columns=exclude_cols))

                # Populate the rules into the CQ objects
                for cq in ucq.rules:
                    cq.rewritten_pos_rules = pos_rules
                    cq.rewritten_neg_rules = neg_rules

                # Update the remaining_dfs with new features.
                remaining_dfs[query_name] = remaining_df.drop(columns=["label"])

            
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
            eval_result = self._evaluate_set(result_set, gt_set)


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

    def _prefilter_by_static_rules(
            self, 
            df: pd.DataFrame, 
            query_name: str,
            enable_cache: bool = True) -> pd.DataFrame:
        
        # Check whether the base result already exists.
        if enable_cache and (self.ckpt_home / f"{query_name}_base.csv").exists():
            print(f"Loading cached base result for query {query_name}...")
            return pd.read_csv(self.ckpt_home / f"{query_name}_base.csv").reset_index(drop=True) 

        query = self.workloads[query_name]
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
        result = result.drop_duplicates()
        
        if enable_cache:
            result.to_csv(self.ckpt_home / f"{query_name}_base.csv", index=False)
            print(f"Stored base result of {query_name} to checkpoint.")

        return result

    async def _prefilter_by_semantic_rules(
            self, 
            df: pd.DataFrame, 
            query_name: str,
            early_positive: bool = True,
            drop_neg: bool = True,
            enable_cache: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if enable_cache and \
            (self.ckpt_home / f"{query_name}_sem_early_positive.csv").exists() and \
                (self.ckpt_home / f"{query_name}_sem_prefilter_result.csv").exists():
            print(f"Loading cached semantic prefilter results for query {query_name}...")
            early_positive_df = pd.read_csv(self.ckpt_home / f"{query_name}_sem_early_positive.csv").reset_index(drop=True)
            result_df = pd.read_csv(self.ckpt_home / f"{query_name}_sem_prefilter_result.csv").reset_index(drop=True)
            return early_positive_df, result_df
        
        query = self.workloads[query_name]
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

        if enable_cache:
            early_positive_df.to_csv(self.ckpt_home / f"{query_name}_sem_early_positive.csv", index=False)
            result_df.to_csv(self.ckpt_home / f"{query_name}_sem_prefilter_result.csv", index=False)
            print(f"Stored semantic prefilter results of {query_name} to checkpoint.")

        return early_positive_df, result_df

    def _filter_by_rewritten_rules(self, df: pd.DataFrame, query: UCQ) -> pd.DataFrame:

        # Step 1: Use LLM-guided backup rules to recall relatively high-confidence samples.
        backup_result = pd.DataFrame()
        for cq in query.rules:
            df_cp = df.copy()
            for col, op, val in cq.backup_rules:
                print(f"Applying backup rule: {col} {op} {val}")
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
                    df_cp = df_cp[df_cp[col].isin(val)]  # type: ignore
                else:
                    raise ValueError(f"Unsupported operation: {op}")
            backup_result = pd.concat([backup_result, df_cp], ignore_index=True)
        backup_result = backup_result.drop_duplicates()

        # Step 2: Use rewritten neg rules to reduce false positives.
        # For each CQ, apply neg rules as POSITIVE conjunction to get samples to EXCLUDE,
        # then remove them from backup_result (De Morgan's law)
        for cq in query.rules:
            if not cq.rewritten_neg_rules:
                continue

            # Apply all neg rules as positive conjunction to get the set to exclude
            df_exclude = backup_result.copy()
            for col, op, val in cq.rewritten_neg_rules:
                print(f"Applying rewritten neg rule (to get exclusion set): {col} {op} {val}")
                if op == "Eq":
                    df_exclude = df_exclude[df_exclude[col] == val]
                elif op == "Gt":
                    df_exclude = df_exclude[df_exclude[col] > val]
                elif op == "Lt":
                    df_exclude = df_exclude[df_exclude[col] < val]
                elif op == "Ge":
                    df_exclude = df_exclude[df_exclude[col] >= val]
                elif op == "Le":
                    df_exclude = df_exclude[df_exclude[col] <= val]
                elif op == "In":
                    df_exclude = df_exclude[df_exclude[col].isin(val)]  # type: ignore
                else:
                    raise ValueError(f"Unsupported operation: {op}")

            # Remove the excluded set from backup_result
            exclude_indices = set(df_exclude.index)
            backup_result = backup_result[~backup_result.index.isin(exclude_indices)]

        # # Step 3: Use rewritten pos rules to reduce false negatives.
        # # Apply pos rules as positive conjunction to get samples to INCLUDE
        # pos_result = pd.DataFrame()
        # for cq in query.rules:
        #     if not cq.rewritten_pos_rules:
        #         continue

        #     df_pos = df.copy()
        #     for col, op, val in cq.rewritten_pos_rules:
        #         print(f"Applying rewritten pos rule: {col} {op} {val}")
        #         if op == "Eq":
        #             df_pos = df_pos[df_pos[col] == val]
        #         elif op == "Gt":
        #             df_pos = df_pos[df_pos[col] > val]
        #         elif op == "Lt":
        #             df_pos = df_pos[df_pos[col] < val]
        #         elif op == "Ge":
        #             df_pos = df_pos[df_pos[col] >= val]
        #         elif op == "Le":
        #             df_pos = df_pos[df_pos[col] <= val]
        #         elif op == "In":
        #             df_pos = df_pos[df_pos[col].isin(val)]  # type: ignore
        #         else:
        #             raise ValueError(f"Unsupported operation: {op}")

        #     pos_result = pd.concat([pos_result, df_pos], ignore_index=True)

        # Combine backup_result (after negation) with pos_result and deduplicate
        # final_result = pd.concat([backup_result, pos_result], ignore_index=True).drop_duplicates()
        final_result = backup_result.drop_duplicates()

        return final_result

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

    def _evaluate_set(self, retrieved_set: set, gt_set: set) -> EvalResult:
        tp = len(retrieved_set & gt_set)
        fp = len(retrieved_set - gt_set)
        fn = len(gt_set - retrieved_set)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        print(f"TP: {tp}, FP: {fp}, FN: {fn}")
        print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
        return EvalResult(tp=tp, fp=fp, fn=fn, precision=precision, recall=recall, f1=f1)

    def _evaluate_list(self, pred: List[int], label: List[int]) -> EvalResult:
        assert len(pred) == len(label), "Prediction and label lists must have the same length."

        tp = sum(1 for p, l in zip(pred, label) if p == 1 and l == 1)
        fp = sum(1 for p, l in zip(pred, label) if p == 1 and l == 0)
        fn = sum(1 for p, l in zip(pred, label) if p == 0 and l == 1)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        print(f"TP: {tp}, FP: {fp}, FN: {fn}")
        print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
        return EvalResult(tp=tp, fp=fp, fn=fn, precision=precision, recall=recall, f1=f1)

    async def _sem_multi_mapping(
            self,
            df: pd.DataFrame,
            mapping_specs: PopulationSpecs,
            enable_cache: bool = True) -> pd.DataFrame:

        mapping_signature = "_".join(
            [spec.target_col for spec in mapping_specs.value]
        )
        cache_path = self.ckpt_home / f"sem_multi_mapping_{mapping_signature}.csv"

        if enable_cache and cache_path.exists():
            print(f"Loading cached semantic multi-mapping for features: {mapping_signature}...")
            cached_df = pd.read_csv(cache_path).reset_index(drop=True)
            assert len(cached_df) == len(df), (
                f"Cached semantic multi-mapping length does not match current dataframe length. "
                f"Expected {len(df)}, got {len(cached_df)}."
            )
            return cached_df

        for spec in mapping_specs.value:
            df = await self._sem_mapping(
                df,
                spec.source_col,
                spec.target_col,
                spec.prompt,
                BooleanFeatureResponse if spec.feature_type == "bool" else
                IntFeatureResponse if spec.feature_type == "int" else
                FloatFeatureResponse,
                enable_cache=False
            )

        if enable_cache:
            df.to_csv(cache_path, index=False)
            print(f"Stored semantic multi-mapping for features: {mapping_signature} to checkpoint.")

        return df
    
    async def _sem_mapping(
            self,
            df: pd.DataFrame,
            col_name: str,
            new_col_name: str,
            prompt: str,
            response_model: Type[BaseModel],
            enable_cache: bool = True) -> pd.DataFrame:
        
        if enable_cache and (self.ckpt_home / f"sem_mapping_{new_col_name}.csv").exists():
            print(f"Loading cached semantic mapping for column '{new_col_name}'...")
            cached_df = pd.read_csv(self.ckpt_home / f"sem_mapping_{new_col_name}.csv").reset_index(drop=True)
            assert len(cached_df) == len(df), \
                "Cached semantic mapping length does not match current dataframe length."
            df[new_col_name] = cached_df[new_col_name]
            return df
        
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

        if enable_cache:
            df.to_csv(self.ckpt_home / f"sem_mapping_{new_col_name}.csv", index=False)
            print(f"Stored semantic mapping for column '{new_col_name}' to checkpoint.")

        return df

    async def _generate_feature_space(
            self,
            workloads: Dict[str, UCQ],
            data_views: Dict[str, pd.DataFrame],
            response_model: Type[PopulationSpecs],
            enable_cache: bool = True) -> Dict[str, PopulationSpecs]:

        feature_space_signature = "_".join(workloads.keys())
        cache_path = self.ckpt_home / f"feature_space_{feature_space_signature}.json"

        if enable_cache and cache_path.exists():
            print("Loading cached feature space...")
            with open(cache_path, 'r') as f:
                cached_data = json.load(f)
            # Deserialize back to Dict[str, List[PopulationSpec]]
            population_specs = {
                query_name: PopulationSpecs(value=[PopulationSpec(**spec) for spec in specs])
                for query_name, specs in cached_data.items()
            }
            return population_specs
        
        population_specs = {}
        for query_name, workload in workloads.items():
            data_view = data_views[query_name]
            for rule in workload.rules:
                for col_name, semantic_desc in rule.sem_rules:
                    # Determine the modality of the source column.
                    data_modality = self._detect_modality(
                        data_view.iloc[0][col_name]
                    ) if len(data_view) > 0 else "TEXT"

                    # Sample data from the source column.
                    sample_data = data_view[col_name].astype(str).dropna().\
                        sample(n=min(10, len(data_view)), random_state=42).tolist()

                    prompt = PROMPTS["GEN_FEAT_CANDIDATE_PROMPT"].format(
                        MODALITY=data_modality,
                        DESC=semantic_desc,
                        SAMPLE_DATA="\n".join(sample_data),
                        SOURCE_COL=col_name,
                    )

                    llm_response = self.llm_client.invoke(
                        modality="TEXT",
                        is_remote=True,
                        prompt=prompt,
                        response_model=response_model,
                    )
                    if query_name not in population_specs:
                        population_specs[query_name] = []
                    population_specs[query_name].extend(llm_response)

        # Serialize PopulationSpec objects to dicts
        if enable_cache:
            cache_data = {
                query_name: [spec.model_dump() for spec in specs]
                for query_name, specs in population_specs.items()
            }
            with open(cache_path, 'w') as f:
                json.dump(cache_data, f, indent=2)
            print(f"Stored feature space to checkpoint: {cache_path}")

        return population_specs

    def _generate_classification_rules(
            self,
            query_name: str,
            df_full: pd.DataFrame,):
        # Fetch the data.
        X_train, X_test, Y_train, Y_test = self._data_split(
            query_name,
            df_full,
            train_size=min(50, len(df_full)//2),
        )

        clf = RuleClassifier(
            max_rules=10,
            min_precision=0.5,
            cv_folds=3,
            use_precision_constraint=False,
            use_tree_rules=True,
            enable_feedback_loop=True,
            max_feedback_iterations=5,
            debug=True
        )

        clf.fit(X_train, Y_train)

        print(f"\n{'='*80}")
        print(f"Learned Rules ({len(clf.rules)} rules)")
        print('='*80)
        clf.print_dnf()

        # Export and return rules
        pos_rules, neg_rules = clf.export_rules()

        print(f"\nExported {len(pos_rules)} positive rules to rewritten_pos_rules")
        print(f"Exported {len(neg_rules)} negative rules to rewritten_neg_rules")

        # Extract the indices of the positive predictions.
        Y_test_pred = clf.predict(X_test)
        predicted_labels = Y_train.tolist() + Y_test_pred.tolist()
        true_labels = Y_train.tolist() + Y_test.tolist()
        eval_result = self._evaluate_list(predicted_labels, true_labels)

        return pos_rules, neg_rules


    def _data_split(self, 
                    query_name: str,
                    df_full: pd.DataFrame, 
                    train_size: int,) \
    -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        random.seed(42)

        # Split data and label.
        ground_truth_set = self.ground_truth[query_name]
        Y = df_full["label"]
        X = df_full.drop(columns=["label"])

        # TODO: Revise the data sampling strategy.
        train_indices = X.sample(n=train_size, random_state=42).index
        test_indices = X.index.difference(train_indices)

        X_train = X.loc[train_indices]
        X_test = X.loc[test_indices]
        Y_train = Y.loc[train_indices]
        Y_test = Y.loc[test_indices]


        # Evaluate the label distribution in the training set.
        pos_count = sum(Y_train)
        neg_count = len(Y_train) - pos_count

        print(f"Training set label distribution for {query_name}:")
        print(f"Positive samples: {pos_count}")
        print(f"Negative samples: {neg_count}")
        print(f"Positive ratio: {pos_count / (pos_count + neg_count):.2%}")
        print(f"Oracle positive ratio: {len(ground_truth_set) / len(self.database):.2%}")

        return X_train, X_test, Y_train, Y_test


