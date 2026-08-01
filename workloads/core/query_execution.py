# pylint: disable=missing-function-docstring,too-many-locals
# pylint: disable=invalid-name,import-outside-toplevel,unused-argument
# pylint: disable=too-many-arguments,too-many-positional-arguments
# pylint: disable=too-many-instance-attributes,consider-using-enumerate
# pylint: disable=consider-using-generator,too-many-lines,too-many-statements
"""Query rewrite, execution, and incremental evaluation helpers."""

import logging
import math
from time import time
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from data_structure import LdbDataManager, SemCQ
from llm import LdbLLMClient
from workloads.core.coreset_maintainer import CoresetMaintainer
from workloads.core.rewrite_candidates import (
    EXPANDED_FOREST,
    ForestConfig,
    RewriteCandidate,
    build_forest_configs,
    build_rewrite_candidates,
    select_candidate_index,
    select_forest_config,
)
from workloads.core.semantic_features import feature_key, predicate_key
from workloads.utils import (
    apply_rules,
    clf_to_rules,
    encode_features,
    loss_by_selectivity,
    train_classifier,
)

logger = logging.getLogger(__name__)


class QueryExecution:
    """Methods for query rewrite/execution and incremental evaluation."""

    def __init__(
        self,
        coreset_maintainer: CoresetMaintainer,
        data_manager: LdbDataManager,
        queries: dict[str, SemCQ],
        config: dict[str, Any],
        llm_client: LdbLLMClient,
        dynamic_setting: list[float],
        b_rew: int,
        b_lab: int,
        delta: float,
    ) -> None:
        self._coreset_maintainer = coreset_maintainer
        self.data_manager = data_manager
        self.queries = queries
        self.config = config
        self.llm_client = llm_client
        self.dynamic_setting = dynamic_setting
        self.b_rew = b_rew
        self.b_lab = b_lab
        self.delta = delta

    async def execute_queries(
        self,
        enable_rewrite: bool,
        enable_enrich: bool,
        debug: bool = False,
    ) -> dict:
        if not enable_rewrite:
            await self._execute_without_rewrite(debug=debug)
            return {}
        if not enable_enrich:
            self._execute_without_enrichment(debug=debug)
            return {}
        _, execution_trace = self._rewrite_and_execute_query(debug=debug)
        return execution_trace

    def _rewrite_and_execute_query(
        self, debug: bool = False
    ) -> tuple[dict, dict]:
        assert self.data_manager.trimmed_feature_names, (
            "No available features to be selected."
        )
        candidates = build_rewrite_candidates(
            feature_count=len(self.data_manager.trimmed_feature_names),
            minimum_feature_count=self._minimum_required_feature_count(),
        )
        execution_trace = {}
        rules_trace = {}
        for index, candidate in enumerate(candidates):
            execution_results = self._init_execution_results()
            for q_name in self.queries:
                query_results = self._execute_rewrite_candidate(
                    q_name=q_name,
                    candidate=candidate,
                    rules_trace=rules_trace,
                    debug=debug,
                )
                self._record_execution_results(
                    execution_results, q_name, query_results
                )
            execution_results["L_avg"] = sum(
                execution_results["L_static"].values()
            ) / len(self.queries)
            execution_trace[index] = execution_results

        best_results = self._compose_best_per_query(execution_trace, candidates)
        execution_trace[len(execution_trace)] = best_results
        for q_name in self.queries:
            self.data_manager.rewrite_rules[q_name] = {
                "active_external_features": list(
                    best_results["features"][q_name]
                ),
                "ucq": list(best_results["rules"][q_name]),
            }
            logger.info(
                "Selected rewrite for query %s with features %s; "
                "L_static = %.4f",
                q_name,
                best_results["features"][q_name],
                best_results["L_static"][q_name],
            )
        return best_results, execution_trace

    def _minimum_required_feature_count(self) -> int:
        """Keep every query's direct semantic predicate in each rewrite."""
        required_keys = {
            predicate_key(predicate)
            for query in self.queries.values()
            for predicate in query.Ps
        }
        required_names = {
            spec.target_col
            for feature_space in self.data_manager.enriched_features.values()
            for spec in feature_space
            if feature_key(spec) in required_keys
        }
        required_positions = [
            index
            for index, name in enumerate(
                self.data_manager.trimmed_feature_names
            )
            if name in required_names
        ]
        if len(required_positions) != len(required_names):
            raise ValueError("Required semantic features were not selected.")
        return max(required_positions, default=-1) + 1

    def _compose_best_per_query(
        self,
        execution_trace: dict,
        candidates: list[RewriteCandidate],
    ) -> dict:
        """Compose one annotation-selected candidate for each query."""
        composite = self._init_execution_results()
        for q_name in self.queries:
            observed_size = self.data_manager.coresets[q_name]["observed_size"]
            best_index = select_candidate_index(
                candidates=candidates,
                estimated_losses=[
                    execution_trace[index]["L_static"][q_name]
                    for index in range(len(candidates))
                ],
                loss_resolution=annotation_loss_resolution(
                    labels=self.data_manager.coresets[q_name]["labels"].iloc[
                        :observed_size
                    ],
                    query_size=len(self.queries),
                    delta=self.delta,
                ),
                candidate_complexities=[
                    self._candidate_preference(
                        q_name, execution_trace[index][q_name]
                    )
                    for index in range(len(candidates))
                ],
            )
            best = execution_trace[best_index]
            self._record_execution_results(composite, q_name, best[q_name])
        composite["L_avg"] = sum(composite["L_static"].values()) / len(
            self.queries
        )
        return composite

    def _candidate_preference(
        self, q_name: str, query_results: dict
    ) -> tuple[int, float, int]:
        """Order tied candidates using rule size and annotation-only loss."""
        rules = query_results["rules"]
        predicate_count = sum(len(conjunction) for conjunction in rules)
        referenced_features = {
            predicate[0] for conjunction in rules for predicate in conjunction
        }
        coreset = self.data_manager.coresets[q_name]
        observed_size = coreset["observed_size"]
        observed_labels = coreset["labels"].iloc[:observed_size].astype(int)
        active_data = coreset["ldb_data"].select_active_features(
            query_results["features"]
        )
        observed_data = encode_features(active_data).iloc[:observed_size]
        rule_predictions = apply_rules(rules, observed_data).astype(int)
        pi = max(1e-6, min(1 - 1e-6, float(observed_labels.mean())))
        rule_loss = float(
            loss_by_selectivity(observed_labels, rule_predictions, pi)
        )
        return predicate_count, rule_loss, len(referenced_features)

    def _init_execution_results(self) -> dict[str, Any]:
        stats = [
            "candidate",
            "candidate_feature_count",
            "rules",
            "features",
            "pred_eval",
            "trans_eval",
            "L_rew",
            "penalty_rew",
            "L_LOO",
            "penalty_LOO",
            "L_obj",
            "L_subj",
            "L_static",
            "memory_cost",
            "stream_selectivities",
            "overall_selectivity",
            "stream_sizes",
            "total_size",
        ]
        execution_results: dict[str, Any] = {stat: {} for stat in stats}
        execution_results["L_avg"] = float("inf")
        return execution_results

    def _execute_rewrite_candidate(
        self,
        q_name: str,
        candidate: RewriteCandidate,
        rules_trace: dict[str, list],
        debug: bool,
    ) -> dict[str, Any]:
        active_external_features = self._active_external_features(
            q_name, candidate
        )

        coreset = self.data_manager.coresets[q_name]
        observed_size = coreset["observed_size"]
        all_train_X = coreset["ldb_data"].select_active_features(
            active_external_features
        )
        all_train_Y = coreset["labels"].astype(int)
        test_X = self.data_manager.sigma_satisfied_data[0][q_name][
            "ldb_data"
        ].select_active_features(active_external_features)

        observed_X = all_train_X.iloc[:observed_size]
        observed_Y = all_train_Y.iloc[:observed_size]
        forest_configs = build_forest_configs()
        forest_errors = [
            compute_subjective_error(
                X=observed_X,
                Y=observed_Y,
                sample_weight=None,
                forest_config=config,
                query_size=len(self.queries),
                data_size=observed_size,
                delta=self.delta,
                loo_step=self.config["loo_step"],
            )
            for config in forest_configs
        ]
        evaluated_labels = max(
            1, math.ceil(observed_size / self.config["loo_step"])
        )
        forest_config = select_forest_config(
            configs=forest_configs,
            estimated_losses=[
                error + penalty for error, penalty in forest_errors
            ],
            loss_resolution=1.0 / evaluated_labels,
        )
        forest_index = forest_configs.index(forest_config)
        L_LOO, penalty_LOO = forest_errors[forest_index]

        train_X = all_train_X
        train_Y = all_train_Y
        sample_weight = None

        memory_cost = (
            self.memory_cost(train_X)
            + self.memory_cost(test_X)
            + self.memory_cost(train_Y)
        )
        rules_trace.setdefault(q_name, [])
        pred_Y_li, rules = self._propagate_and_extract_rules(
            train_X=train_X,
            train_Y=train_Y,
            sample_weight=sample_weight,
            test_X=test_X,
            forest_config=forest_config,
            rule_evidence_size=observed_size,
            allowed_rule_features=set(
                coreset["ldb_data"].base_features + active_external_features
            ),
            debug=debug,
        )
        rules_trace[q_name].append(rules)
        self.data_manager.sigma_satisfied_data[0][q_name][
            "propagated_labels"
        ] = pred_Y_li[0]

        trans_Y = apply_rules(rules, encode_features(test_X))
        pred_eval_results = self.data_manager.eval_query_quality(
            q_name=q_name,
            selected_cols=self.queries[q_name].selected,
            stream_idx=0,
            pred_labels=pred_Y_li,
        )
        trans_eval_results = self.data_manager.eval_query_quality(
            q_name=q_name,
            selected_cols=self.queries[q_name].selected,
            stream_idx=0,
            pred_labels=[trans_Y],
        )

        rule_feature_count = len(
            {predicate[0] for conjunction in rules for predicate in conjunction}
        )
        estimated_selectivity = coreset["estimated_selectivity"]
        if estimated_selectivity is None:
            estimated_selectivity = float(observed_Y.mean())
        L_rew, penalty_rew = compute_objective_error(
            pred_Y=pred_Y_li[0],
            trans_Y=trans_Y,
            b_rew=self.b_rew,
            schema_arity=max(1, rule_feature_count),
            query_size=len(self.queries),
            selected_data_size=len(pred_Y_li[0]) + observed_size,
            delta=self.delta,
            selectivity=estimated_selectivity,
        )
        penalty_rew *= 0.01
        L_obj = L_rew + penalty_rew

        penalty_LOO *= 0.01
        L_subj = L_LOO + penalty_LOO
        L_static = L_obj + L_subj

        if debug:
            logger.info(
                "Estimated %s for %s: L_obj=%.4f, L_subj=%.4f",
                candidate.kind,
                q_name,
                L_obj,
                L_subj,
            )
        stream_stat = (
            self.data_manager.sigma_satisfied_data.compute_stream_stat(q_name)
        )
        return {
            "candidate": candidate.kind,
            "candidate_feature_count": candidate.feature_count,
            "rules": rules,
            "features": active_external_features,
            "pred_eval": pred_eval_results,
            "trans_eval": trans_eval_results,
            "L_rew": L_rew,
            "penalty_rew": penalty_rew,
            "L_LOO": L_LOO,
            "penalty_LOO": penalty_LOO,
            "L_obj": L_obj,
            "L_subj": L_subj,
            "L_static": L_static,
            "memory_cost": memory_cost,
            "stream_selectivities": stream_stat["stream_selectivities"],
            "overall_selectivity": stream_stat["overall_selectivity"],
            "stream_sizes": stream_stat["stream_sizes"],
            "total_size": stream_stat["total_size"],
        }

    def _active_external_features(
        self, q_name: str, candidate: RewriteCandidate
    ) -> list[str]:
        """Resolve materialized features used by one candidate."""
        enriched = self.data_manager.enriched_features[q_name]
        feature_count = candidate.feature_count or 0
        return [
            spec.target_col
            for spec in enriched
            if spec.target_col
            in self.data_manager.trimmed_feature_names[:feature_count]
        ]

    def _propagate_and_extract_rules(
        self,
        train_X: pd.DataFrame,
        train_Y: pd.Series,
        sample_weight: np.ndarray | None,
        test_X: pd.DataFrame,
        forest_config: ForestConfig = EXPANDED_FOREST,
        rule_evidence_size: int | None = None,
        allowed_rule_features: set[str] | None = None,
        debug: bool = False,
    ) -> tuple[list[pd.Series], list]:
        if train_X.shape[1] == 0:
            return [pd.Series(1, index=test_X.index, dtype=int)], []

        clf, pred_Y_li = perform_label_propagation(
            train_X,
            train_Y,
            [test_X],
            sample_weight=sample_weight,
            forest_config=forest_config,
        )
        encoded_train_X = encode_features(train_X)
        evidence_size = (
            len(train_X) if rule_evidence_size is None else rule_evidence_size
        )
        if not 0 < evidence_size <= len(train_X):
            raise ValueError(
                "Rule evidence size must be within the training data."
            )
        rule_weight = (
            None
            if sample_weight is None
            else np.asarray(sample_weight)[:evidence_size]
        )
        rules = clf_to_rules(
            clf,
            feature_names=train_X.columns.tolist(),
            disjunction_budget=self.b_rew,
            X_train=encoded_train_X.iloc[:evidence_size].to_numpy(),
            y_train=train_Y.iloc[:evidence_size].to_numpy(),
            X_reference=encode_features(test_X).to_numpy(),
            y_reference=pred_Y_li[0].to_numpy(),
            sample_weight=rule_weight,
            allowed_rule_features=allowed_rule_features,
            debug=debug,
        )
        return pred_Y_li, rules

    @staticmethod
    def _record_execution_results(
        execution_results: dict[str, Any],
        q_name: str,
        query_results: dict[str, Any],
    ) -> None:
        execution_results[q_name] = query_results
        for key, value in query_results.items():
            execution_results[key][q_name] = value

    def _execute_without_enrichment(self, debug: bool = False) -> None:
        for q_name in self.queries:
            active_external_features = []
            train_X = self.data_manager.coresets[q_name][
                "ldb_data"
            ].select_active_features(active_external_features)
            train_Y = self.data_manager.coresets[q_name]["labels"].astype(int)
            sample_weight = None
            test_X = self.data_manager.sigma_satisfied_data[0][q_name][
                "ldb_data"
            ].select_active_features(active_external_features)
            memory_cost = (
                self.memory_cost(train_X)
                + self.memory_cost(test_X)
                + self.memory_cost(train_Y)
            )
            pred_Y_li, rules = self._propagate_and_extract_rules(
                train_X=train_X,
                train_Y=train_Y,
                sample_weight=sample_weight,
                test_X=test_X,
                debug=debug,
            )
            self.data_manager.sigma_satisfied_data[0][q_name][
                "propagated_labels"
            ] = pred_Y_li[0]

            trans_Y = apply_rules(rules, encode_features(test_X))
            trans_eval_results = self.data_manager.eval_query_quality(
                q_name=q_name,
                selected_cols=self.queries[q_name].selected,
                stream_idx=0,
                pred_labels=[trans_Y],
            )

            print("=" * 30)
            print(trans_eval_results)
            print(f"Memory cost for query {q_name}: {memory_cost}")
            print("=" * 30)

    async def _execute_without_rewrite(self, debug: bool = False) -> None:
        from data_structure.llm_resp_templates import BooleanFeatureResponse

        assert len(self.data_manager.trimmed_feature_names) > 0, (
            "No available features to be selected."
        )

        for q_name in self.queries:
            active_external_features = [
                spec.target_col
                for spec in self.data_manager.enriched_features[q_name]
                if spec.target_col in self.data_manager.trimmed_feature_names
            ]
            test_X = self.data_manager.sigma_satisfied_data[0][q_name][
                "ldb_data"
            ].select_active_features(active_external_features)

            sem_query = self.queries[q_name]
            prompt = (
                "Determine if the provided data item satisfies the following "
                "conditions: "
                + " AND ".join([pred.succ_cond for pred in sem_query.Ps])
            )

            data_items = [
                [", ".join([f"{col}: {val}" for col, val in row.items()])]
                for _, row in test_X.iterrows()
            ]

            resp = await self.llm_client.invoke_parallel(
                is_remote=False,
                modality="Text",
                prompt=prompt,
                data_items=data_items,
                response_model=BooleanFeatureResponse,
            )
            pred_Y = pd.Series([r.value for r in resp])

            pred_eval_results = self.data_manager.eval_query_quality(
                q_name=q_name,
                selected_cols=sem_query.selected,
                stream_idx=0,
                pred_labels=[pred_Y],
            )
            print("=" * 30)
            print(pred_eval_results)
            print("=" * 30)

    async def incremental_processing(
        self, debug: bool = False
    ) -> tuple[bool, list]:
        eval_results = []
        rerun = False

        for inc_round in range(1, len(self.dynamic_setting)):
            start_time = time()
            eval_results_single_step = {
                q_name: {
                    "error_certificate": 0.0,
                    "pred_eval": {},
                    "trans_eval": {},
                }
                for q_name in self.queries
            }

            for q_name in self.queries:
                if self.data_manager.sigma_satisfied_data[inc_round][q_name][
                    "ldb_data"
                ].df.empty:
                    logger.info(
                        "Inc-Round %s: No sigma-satisfied data available.",
                        inc_round,
                    )
                    break

                step_result = await self._incremental_query_step(
                    q_name=q_name,
                    inc_round=inc_round,
                    eval_results=eval_results,
                    debug=debug,
                )
                if step_result is None:
                    rerun = True
                    return rerun, []

                eval_results_single_step[q_name] = step_result

            end_time = time()
            eval_results.append(
                {
                    "inc_round": inc_round,
                    "inc_ratio": self.dynamic_setting[inc_round],
                    "eval_results": eval_results_single_step,
                    "eval_time": end_time - start_time,
                }
            )
        return False, eval_results

    async def _incremental_query_step(
        self,
        q_name: str,
        inc_round: int,
        eval_results: list,
        debug: bool,
    ) -> dict[str, Any] | None:

        active_external_features = self.data_manager.rewrite_rules[q_name][
            "active_external_features"
        ]

        await self.data_manager.sync_sigma_satisfied_data_features(
            q_name=q_name,
            tag="init",
            stream_idx=inc_round,
            enable_cache=True,
        )

        self._coreset_maintainer.expand_query_coreset(
            q_name=q_name, inc_round=inc_round
        )

        train_X = self.data_manager.coresets[q_name][
            "ldb_data"
        ].select_active_features(active_external_features)
        train_Y = self.data_manager.coresets[q_name]["labels"].astype(int)
        sample_weight = None
        test_X_li = [
            self.data_manager.sigma_satisfied_data[i][q_name][
                "ldb_data"
            ].select_active_features(active_external_features)
            if not self.data_manager.sigma_satisfied_data[i][q_name][
                "ldb_data"
            ].df.empty
            else pd.DataFrame(columns=train_X.columns)
            for i in range(inc_round + 1)
        ]
        _, pred_Y_li = perform_label_propagation(
            train_X, train_Y, test_X_li, sample_weight=sample_weight
        )

        self.data_manager.sigma_satisfied_data[inc_round][q_name][
            "propagated_labels"
        ] = pred_Y_li[-1]
        prev_prop_Y_li: list[pd.Series] = []
        for i in range(inc_round + 1):
            stream_record = self.data_manager.sigma_satisfied_data[i][q_name]
            if stream_record["ldb_data"].df.empty:
                prev_prop_Y_li.append(pd.Series(dtype=int))
                continue
            propagated_labels = stream_record["propagated_labels"]
            if propagated_labels is None:
                raise ValueError(
                    f"Propagated labels are missing for query '{q_name}' "
                    f"in stream-{i}."
                )
            prev_prop_Y_li.append(propagated_labels)

        data_err, pred_err = compute_inc_error_certificate(
            prev_prop_Y_li=prev_prop_Y_li,
            prop_Y_li=pred_Y_li,
        )
        err_certificate = data_err + pred_err
        prev_err_certificate = (
            eval_results[-1]["eval_results"][q_name]["error_certificate"]
            if eval_results
            else 0.0
        )
        err_certificate += prev_err_certificate * 0.5
        reoptimization_threshold = compute_inc_reoptimization_threshold(
            prev_prop_Y_li=prev_prop_Y_li,
            delta=self.delta,
        )
        logger.info(
            "Inc-Round %s, Query %s: Est. error certificate: %.4f "
            "(adaptive threshold: %.4f)",
            inc_round,
            q_name,
            err_certificate,
            reoptimization_threshold,
        )

        if err_certificate >= reoptimization_threshold:
            return None

        trans_Y_li = [
            apply_rules(
                self.data_manager.rewrite_rules[q_name]["ucq"],
                encode_features(test_X_li[i]),
            )
            for i in range(inc_round + 1)
        ]

        return {
            "error_certificate": err_certificate,
            "reoptimization_threshold": reoptimization_threshold,
            "data_err": data_err,
            "pred_err": pred_err,
            "pred_eval": self.data_manager.eval_query_quality(
                q_name=q_name,
                selected_cols=self.queries[q_name].selected,
                stream_idx=inc_round,
                pred_labels=pred_Y_li,
            ),
            "trans_eval": self.data_manager.eval_query_quality(
                q_name=q_name,
                selected_cols=self.queries[q_name].selected,
                stream_idx=inc_round,
                pred_labels=trans_Y_li,
            ),
        }

    def memory_cost(self, obj: pd.DataFrame | pd.Series) -> float:
        usage = obj.memory_usage(deep=True)
        if isinstance(usage, pd.Series):
            return float(usage.sum())
        return float(usage)


def compute_objective_error(
    pred_Y: pd.Series,
    trans_Y: pd.Series,
    b_rew: int,
    schema_arity: int,
    query_size: int,
    selected_data_size: int,
    delta: float,
    selectivity: float,
) -> tuple[float, float]:
    """Compute rewriting loss and penalty.

    Args:
        pred_Y: Predicted labels
        trans_Y: Translated labels
        schema_arity: Number of distinct features referenced by the rules
        query_size: Number of queries
        selected_data_size: Size of selected data
        delta: Delta parameter for penalty calculation
        selectivity: Query selectivity estimated from the annotation design

    Returns:
        Tuple of (L_rew, penalty)
    """
    pi = max(1e-6, min(1 - 1e-6, selectivity))

    # Compute rewriting loss
    L_rew = loss_by_selectivity(pred_Y, trans_Y, pi)

    # Compute the penalty
    Gamma_rew = max(pi, 1 - pi) / min(pi, 1 - pi)
    d_VC = b_rew * schema_arity * math.log(b_rew)

    penalty = Gamma_rew * math.sqrt(
        (d_VC * math.log(selected_data_size) + math.log(2 * query_size / delta))
        / selected_data_size
    )

    logger.info(
        "penalty: %.4f, Gamma_rew: %.4f, d_VC: %s, "
        "selected_data_size: %s, query_size: %s, delta: %s",
        penalty,
        Gamma_rew,
        d_VC,
        selected_data_size,
        query_size,
        delta,
    )
    logger.info(
        "pi: %.4f, sum(pred_Y): %s, len(pred_Y): %s",
        pi,
        sum(pred_Y),
        len(pred_Y),
    )

    return L_rew, penalty


def annotation_loss_resolution(
    labels: pd.Series, query_size: int, delta: float
) -> float:
    """Return the confidence radius for annotation-estimated loss."""
    if len(labels) == 0:
        return 0.0
    pi = max(1e-6, min(1 - 1e-6, float(labels.mean())))
    gamma = max(pi, 1 - pi) / min(pi, 1 - pi)
    return gamma * math.sqrt(
        math.log(2 * query_size / delta) / (2 * len(labels))
    )


def compute_subjective_error(
    X: pd.DataFrame,
    Y: pd.Series,
    query_size: int,
    data_size: int,
    delta: float,
    loo_step: int = 10,
    sample_weight: np.ndarray | None = None,
    forest_config: ForestConfig = EXPANDED_FOREST,
) -> tuple[float, float]:
    """Compute LOO error and penalty for subjective error estimation.

    Args:
        X: Feature DataFrame
        Y: Labels
        schema_arity: Number of features in schema
        query_size: Number of queries
        data_size: Size of data
        delta: Delta parameter
        loo_step: Step size for LOO
        sample_weight: Optional training weights aligned with X and Y

    Returns:
        Tuple of (L_LOO, penalty)
    """

    # LOO error could only compute with at least 2 sample.
    if len(X) < 2:
        logger.warning(
            "Not enough data for LOO error computation. "
            "Returning L_LOO=0.0 and penalty=0.0."
        )
        return 0.0, 0.0

    # If no features are available, there is nothing to train on.
    # Treat this as the trivial baseline instead of fitting sklearn on a
    # zero-column matrix.
    if X.shape[1] == 0:
        logger.warning(
            "No features available for LOO error computation. "
            "Returning L_LOO=0.0 and penalty=0.0."
        )
        return 0.0, 0.0

    Y_loo = []
    Y_true = []
    weights = (
        None
        if sample_weight is None
        else np.asarray(sample_weight, dtype=float)
    )
    if weights is not None and len(weights) != len(X):
        raise ValueError("sample_weight must align with X and Y.")

    X_encoded = encode_features(X)
    for i in range(0, len(X_encoded), loo_step):
        X_train = X_encoded.drop(index=i)
        Y_train = Y.drop(index=i)
        X_test = X_encoded.iloc[[i]]
        train_weight = None if weights is None else np.delete(weights, i)

        clf = train_classifier(
            X_train,
            Y_train,
            n_estimators=forest_config.n_estimators,
            max_depth=forest_config.max_depth,
            min_samples_leaf=forest_config.min_samples_leaf,
            sample_weight=train_weight,
        )

        pred = clf.predict(X_test)[0]
        Y_loo.append(pred)
        Y_true.append(Y.iloc[i])

    Y_true_series = pd.Series(Y_true)
    Y_loo_series = pd.Series(Y_loo)

    # Compute LOO loss score
    pi = sum(Y) / len(Y)
    pi = max(
        1e-6, min(1 - 1e-6, pi)
    )  # Ensure pi is in (0, 1) to avoid extreme penalties
    L_LOO = loss_by_selectivity(Y_true_series, Y_loo_series, pi)

    # Compute the penalty
    Gamma_LOO = max(pi, 1 - pi) / min(pi, 1 - pi)

    penalty = Gamma_LOO * math.sqrt(
        math.log(2 * query_size / delta) / (2 * data_size)
    )

    return L_LOO, penalty


# ============================================================================
# Label Propagation Utilities
# ============================================================================


def perform_label_propagation(
    train_X: pd.DataFrame,
    train_Y: pd.Series,
    test_X_li: list[pd.DataFrame],
    sample_weight: np.ndarray | None = None,
    forest_config: ForestConfig = EXPANDED_FOREST,
    debug: bool = False,
) -> tuple[RandomForestClassifier, list[pd.Series]]:

    # Train the classifier on the training data.
    train_X_proc = encode_features(train_X)
    clf = train_classifier(
        train_X_proc,
        train_Y,
        n_estimators=forest_config.n_estimators,
        max_depth=forest_config.max_depth,
        min_samples_leaf=forest_config.min_samples_leaf,
        sample_weight=sample_weight,
    )

    pred_Y_li = []
    for i in range(len(test_X_li)):
        if test_X_li[i].empty:
            pred_Y_li.append(pd.Series(dtype=int))
            continue
        test_X_proc = encode_features(test_X_li[i])
        pred_Y_li.append(
            pd.Series(clf.predict(test_X_proc), index=test_X_li[i].index)
        )

    return clf, pred_Y_li


def compute_inc_reoptimization_threshold(
    prev_prop_Y_li: list[pd.Series], delta: float
) -> float:
    """Derive a drift threshold from stream growth and shared sample size."""
    if not 0 < delta < 1:
        raise ValueError("Delta must be between 0 and 1.")
    if len(prev_prop_Y_li) < 2:
        return float("inf")

    current_size = sum(len(labels) for labels in prev_prop_Y_li)
    previous_size = current_size - len(prev_prop_Y_li[-1])
    if current_size == 0 or previous_size == 0:
        return float("inf")

    data_fraction = len(prev_prop_Y_li[-1]) / current_size
    confidence_margin = math.sqrt(math.log(2 / delta) / (2 * previous_size))
    return data_fraction + confidence_margin


def compute_inc_error_certificate(
    prev_prop_Y_li: list[pd.Series],
    prop_Y_li: list[pd.Series],
) -> tuple[float, float]:
    assert len(prev_prop_Y_li) == len(prop_Y_li), (
        f"Length of prev_prop_Y_li and prop_Y_li must be the same. "
        f"Got {len(prev_prop_Y_li)} and {len(prop_Y_li)}."
    )

    if len(prop_Y_li) == 1:
        return 0.0, 0.0  # The first iteration introduces no error.

    # Data error = |D_{added}| / |D_{total}|.
    curr_data_size = sum(
        [len(prev_prop_Y_li[i]) for i in range(len(prev_prop_Y_li))]
    )
    prev_data_size = curr_data_size - len(prev_prop_Y_li[-1])
    new_data_size = len(prev_prop_Y_li[-1])
    if curr_data_size == 0:
        return 0.0, 0.0
    data_err = new_data_size / curr_data_size
    if prev_data_size == 0:
        return data_err, 0.0

    # Prediction error = |Err_{shared}| / |D_{shared}|.
    pred_err = 0.0
    for i in range(len(prev_prop_Y_li) - 1):
        pred_err += sum(prev_prop_Y_li[i] != prop_Y_li[i])
    pred_err /= prev_data_size

    return data_err, pred_err
