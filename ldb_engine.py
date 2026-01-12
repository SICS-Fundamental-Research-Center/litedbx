import pandas as pd
import random
from pathlib import Path
from typing import List, Tuple, Dict

from llm_client import LiteLLMWrapper, BooleanFeatureResponse
from ldb_classifier import RuleClassifier
from data_structures import UCQ, PopulationSpecs
from evaluation import evaluate_set, evaluate_list
from semantic_ops import sem_mapping, sem_multi_mapping
from feature_gen import generate_feature_space
from rule_filter import (
    prefilter_by_static_rules,
    prefilter_by_proxies,
    filter_by_rewritten_rules
)
import logging
logger = logging.getLogger(__name__)


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

        logger.info("Initializing LDBEngine...")
        self.WD = Path(__file__).parent
        self.dataset_path = self.WD / "data" / dataset_name
        self.ckpt_home = self.dataset_path / ".ckpt"
        self.ckpt_home.mkdir(parents=True, exist_ok=True)
        self.ground_truth_paths = self.dataset_path / "ground_truth"
        self.database = \
            pd.read_csv(self.dataset_path / "data_full.csv").reset_index(drop=True)
        logger.info(f"Loaded database {dataset_name} with {len(self.database)} rows.")

        self.workloads = workloads
        self.ground_truth = self._init_ground_truth(list(workloads.keys()))
        logger.info("Loaded ground truths.")
        self.llm_client = LiteLLMWrapper()
        logger.info("Initialized LDBEngine.")


    async def apply(
            self, 
            queries: List[str],
            enable_proxies: bool = False) -> None:

        ckpt_prefix = f"PROXY_" if enable_proxies else "NOPXY_"

        # Step 1: Prefilter easy samples using static and semantic rules.
        remaining_dfs = {}
        for query_name in queries:

            workload = self.workloads[query_name]

            # Step 1.1: Filter the data based on static rules.
            static_view = prefilter_by_static_rules(
                self.database, self.workloads, query_name, self.ckpt_home, enable_cache=True
            )
            remaining_dfs[query_name] = static_view.reset_index(drop=True)
            logger.info(f"After static prefiltering: {len(static_view)}/{len(self.database)} rows remain.")

            # Step 1.2: Filter the data based on semantic rules.
            if not enable_proxies:
                continue
            sem_view = await prefilter_by_proxies(
                static_view, self.workloads, query_name, self.llm_client, 
                self.ckpt_home, ckpt_prefix=ckpt_prefix, enable_cache=True
            )
            sem_view = sem_view[workload.select_cols]
            remaining_dfs[query_name] = sem_view.reset_index(drop=True)
            logger.info(f"After semantic prefiltering: {len(sem_view)}/{len(static_view)} rows remain.")

        # Step 2: Generate LLM-based pseudo-labels for the remaining samples.
        for query_name, remaining_df in remaining_dfs.items():
            workload = self.workloads[query_name]
            for cq in workload.rules:
                for col_name, semantic_desc in cq.sem_rules:
                    new_col_name = f"{col_name}_pseudo_label"
                    remaining_df = await sem_mapping(
                        remaining_df,
                        col_name,
                        new_col_name,
                        semantic_desc,
                        BooleanFeatureResponse,
                        self.llm_client,
                        self.ckpt_home,
                        ckpt_prefix=ckpt_prefix,
                        enable_cache=True
                    )
                    cq.backup_rules.append(
                        (new_col_name, "Eq", True)
                    )
        
        # Step 3: Generate feature space for each queries.
        #   TODO: Generate feature space JOINTLY for all queries.
        population_specs = await generate_feature_space(
            self.workloads,
            remaining_dfs,
            self.llm_client,
            PopulationSpecs,
            self.ckpt_home,
            enable_cache=True
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
                remaining_df = await sem_multi_mapping(
                    remaining_df,
                    PopulationSpecs(value=pre_selected_specs),
                    self.llm_client,
                    self.ckpt_home,
                    ckpt_prefix=ckpt_prefix,
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
            remaining_df = remaining_dfs[query_name]
            workload = self.workloads[query_name]
            filtered_df = filter_by_rewritten_rules(remaining_df, workload)
            logger.info(f"After filtering by rewritten rules: {len(filtered_df)}/{len(remaining_df)} rows retrieved.")

            filtered_df.to_csv(self.dataset_path / f"retrieved_{query_name}.csv", index=False)
            assert self.ground_truth[query_name] is not None, f"Ground truth for {query_name} not found."

            result_set = set(
                tuple(row)
                for row in filtered_df[workload.select_cols].itertuples(index=False, name=None)
            )
            gt_set = self.ground_truth[query_name]
            _ = evaluate_set(result_set, gt_set)


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
        eval_result = evaluate_list(predicted_labels, true_labels)

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

        logger.info(f"Training set label distribution for {query_name}:")
        logger.info(f"Positive samples: {pos_count}")
        logger.info(f"Negative samples: {neg_count}")
        logger.info(f"Positive ratio: {pos_count / (pos_count + neg_count):.2%}")
        logger.info(f"Oracle positive ratio: {len(ground_truth_set) / len(self.database):.2%}")

        return X_train, X_test, Y_train, Y_test


