# pylint: disable=missing-function-docstring,too-many-locals
# pylint: disable=invalid-name,import-outside-toplevel,unused-argument
# pylint: disable=too-many-arguments,too-many-positional-arguments
# pylint: disable=too-many-instance-attributes,consider-using-enumerate
# pylint: disable=consider-using-generator
"""Query rewrite, execution, and incremental evaluation helpers."""

import logging
import math
from time import time
from typing import Any

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from data_structure import LdbDataManager, SemCQ
from llm import LdbLLMClient
from workloads.core.coreset_maintainer import CoresetMaintainer
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
        eta: float,
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
        self.eta = eta

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
        best_static_error = float("inf")
        best_statistics = {}
        execution_trace = {}
        rules_trace = {}

        assert len(self.data_manager.trimmed_feature_names) > 0, (
            "No available features to be selected."
        )

        for i in range(len(self.data_manager.trimmed_feature_names) + 1):
            accumulated_error = 0
            execution_results = self._init_execution_results()

            for q_name in self.queries:
                query_results = self._execute_rewrite_candidate(
                    q_name=q_name,
                    feature_count=i,
                    rules_trace=rules_trace,
                    debug=debug,
                )
                accumulated_error += query_results["L_static"]
                self._record_execution_results(
                    execution_results, q_name, query_results
                )

            average_error = accumulated_error / len(self.queries)
            execution_results["L_avg"] = average_error
            execution_trace[i] = execution_results

            if average_error < best_static_error:
                best_static_error = average_error
                best_statistics = execution_results

        for q_name, rules in rules_trace.items():
            trimmed_feature_names = self.data_manager.trimmed_feature_names
            active_feature_names = [
                spec.target_col
                for spec in self.data_manager.enriched_features[q_name]
                if spec.target_col in trimmed_feature_names
            ]
            self.data_manager.rewrite_rules[q_name] = {
                "active_external_features": active_feature_names,
                "ucq": rules[-1],
            }

        return best_statistics, execution_trace

    def _init_execution_results(self) -> dict[str, Any]:
        stats = [
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
        ]
        execution_results: dict[str, Any] = {stat: {} for stat in stats}
        execution_results["L_avg"] = float("inf")
        return execution_results

    def _required_sigma_labels(self, q_name: str, stream_idx: int) -> pd.Series:
        labels = self.data_manager.sigma_satisfied_data[stream_idx][q_name][
            "labels"
        ]
        if labels is None:
            raise ValueError(
                f"Labels are missing for query '{q_name}' "
                f"in stream-{stream_idx}."
            )
        return labels

    def _execute_rewrite_candidate(
        self,
        q_name: str,
        feature_count: int,
        rules_trace: dict[str, list],
        debug: bool,
    ) -> dict[str, Any]:
        active_external_features = [
            spec.target_col
            for spec in self.data_manager.enriched_features[q_name]
            if spec.target_col
            in self.data_manager.trimmed_feature_names[:feature_count]
        ]
        train_X = self.data_manager.coresets[q_name][
            "ldb_data"
        ].select_active_features(active_external_features)
        train_Y = self.data_manager.coresets[q_name]["labels"].astype(int)
        test_X = self.data_manager.sigma_satisfied_data[0][q_name][
            "ldb_data"
        ].select_active_features(active_external_features)
        test_Y = self._required_sigma_labels(q_name, 0).astype(int)

        memory_cost = (
            self.memory_cost(train_X)
            + self.memory_cost(test_X)
            + self.memory_cost(train_Y)
            + self.memory_cost(test_Y)
        )

        rules_trace.setdefault(q_name, [])
        pred_Y_li, rules = self._propagate_and_extract_rules(
            train_X=train_X,
            train_Y=train_Y,
            test_X=test_X,
            test_Y=test_Y,
            debug=debug,
        )
        rules_trace[q_name].append(rules)
        self.data_manager.sigma_satisfied_data[0][q_name][
            "propagated_labels"
        ] = pred_Y_li[0]

        start_execute = time()
        trans_Y = apply_rules(rules, encode_features(test_X))
        end_execute = time()
        print("=" * 20)
        print(f"Online execution time: {end_execute - start_execute}")
        print("=" * 20)

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

        observed_size = self.data_manager.coresets[q_name]["observed_size"]
        L_rew, penalty_rew = compute_objective_error(
            pred_Y=pred_Y_li[0],
            trans_Y=trans_Y,
            b_rew=self.b_rew,
            schema_arity=len(train_X.columns.tolist()),
            query_size=len(self.queries),
            selected_data_size=len(pred_Y_li[0]) + self.b_lab,
            delta=self.delta,
        )
        L_obj = L_rew + penalty_rew

        L_LOO, penalty_LOO = compute_subjective_error(
            X=self.data_manager.sigma_satisfied_data[0][q_name]["ldb_data"]
            .select_active_features(active_external_features)
            .iloc[:observed_size],
            Y=self._required_sigma_labels(q_name, 0).astype(int)[
                :observed_size
            ],
            query_size=len(self.queries),
            data_size=observed_size,
            delta=self.delta,
            loo_step=self.config["loo_step"],
        )
        L_subj = L_LOO + penalty_LOO
        L_static = L_obj + L_subj

        if debug:
            logger.info(
                "Estimated for %s with %s external features: "
                "L_obj = %.4f (L_rew=%.4f, penalty=%.4f), "
                "L_subj = %.4f (L_LOO=%.4f, penalty=%.4f)",
                q_name,
                feature_count + 1,
                L_obj,
                L_rew,
                penalty_rew,
                L_subj,
                L_LOO,
                penalty_LOO,
            )

        return {
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
        }

    def _propagate_and_extract_rules(
        self,
        train_X: pd.DataFrame,
        train_Y: pd.Series,
        test_X: pd.DataFrame,
        test_Y: pd.Series,
        debug: bool,
    ) -> tuple[list[pd.Series], list]:
        if train_X.shape[1] == 0:
            return [pd.Series(1, index=test_Y.index, dtype=int)], []

        clf, pred_Y_li = perform_label_propagation(
            train_X, train_Y, [test_X], [test_Y]
        )
        rules = clf_to_rules(
            clf,
            feature_names=train_X.columns.tolist(),
            disjunction_budget=self.b_rew,
            X_train=encode_features(train_X).to_numpy(),
            y_train=train_Y.to_numpy(),
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
            test_X = self.data_manager.sigma_satisfied_data[0][q_name][
                "ldb_data"
            ].select_active_features(active_external_features)
            test_Y = self._required_sigma_labels(q_name, 0).astype(int)

            memory_cost = (
                self.memory_cost(train_X)
                + self.memory_cost(test_X)
                + self.memory_cost(train_Y)
                + self.memory_cost(test_Y)
            )
            pred_Y_li, rules = self._propagate_and_extract_rules(
                train_X=train_X,
                train_Y=train_Y,
                test_X=test_X,
                test_Y=test_Y,
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
            q_name=q_name, inc_round=inc_round, debug=debug
        )

        train_X = self.data_manager.coresets[q_name][
            "ldb_data"
        ].select_active_features(active_external_features)
        train_Y = self.data_manager.coresets[q_name]["labels"].astype(int)
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
        test_Y_li: list[pd.Series] = []
        for i in range(inc_round + 1):
            stream_record = self.data_manager.sigma_satisfied_data[i][q_name]
            if stream_record["ldb_data"].df.empty:
                test_Y_li.append(pd.Series(dtype=int))
                continue
            labels = stream_record["labels"]
            if labels is None:
                raise ValueError(
                    f"Labels are missing for query '{q_name}' in stream-{i}."
                )
            test_Y_li.append(labels.astype(int))
        _, pred_Y_li = perform_label_propagation(
            train_X, train_Y, test_X_li, test_Y_li
        )

        self.data_manager.sigma_satisfied_data[inc_round][q_name][
            "propagated_labels"
        ] = pred_Y_li[-1]
        observed_data_size = self.data_manager.coresets[q_name]["observed_size"]
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

        err_certificate = compute_inc_error_certificate(
            label_Y=self.data_manager.coresets[q_name]["labels"][
                :observed_data_size
            ].astype(int),
            prev_prop_Y_li=prev_prop_Y_li,
            prop_Y_li=pred_Y_li,
        )
        prev_err_certificate = (
            eval_results[-1]["eval_results"][q_name]["error_certificate"]
            if eval_results
            else 0.0
        )
        err_certificate += prev_err_certificate * 0.5
        logger.info(
            "Inc-Round %s, Query %s: Est. error certificate: %.4f",
            inc_round,
            q_name,
            err_certificate,
        )

        if err_certificate >= self.eta:
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
) -> tuple[float, float]:
    """Compute rewriting loss and penalty.

    Args:
        pred_Y: Predicted labels
        trans_Y: Translated labels
        schema_arity: Number of features in schema
        query_size: Number of queries
        selected_data_size: Size of selected data
        delta: Delta parameter for penalty calculation

    Returns:
        Tuple of (L_rew, penalty)
    """
    pi = sum(pred_Y) / len(pred_Y)
    pi = max(
        1e-6, min(1 - 1e-6, pi)
    )  # Ensure pi is in (0, 1) to avoid extreme penalties

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


def compute_subjective_error(
    X: pd.DataFrame,
    Y: pd.Series,
    query_size: int,
    data_size: int,
    delta: float,
    loo_step: int = 10,
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

    X_encoded = encode_features(X)
    for i in range(0, len(X_encoded), loo_step):
        X_train = X_encoded.drop(index=i)
        Y_train = Y.drop(index=i)
        X_test = X_encoded.iloc[[i]]

        clf = train_classifier(X_train, Y_train)

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
    test_Y_li: list[pd.Series],
    debug: bool = False,
) -> tuple[RandomForestClassifier, list[pd.Series]]:

    # Train the classifier on the training data.
    train_X_proc = encode_features(train_X)
    clf = train_classifier(train_X_proc, train_Y, n_estimators=3)

    assert len(test_X_li) == len(test_Y_li), (
        "test_X_li and test_Y_li must have the same length."
    )

    pred_Y_li = []
    for i in range(len(test_X_li)):
        if test_X_li[i].empty:
            pred_Y_li.append(pd.Series(dtype=int))
            continue
        test_X_proc = encode_features(test_X_li[i])
        pred_Y_li.append(
            pd.Series(clf.predict(test_X_proc), index=test_Y_li[i].index)
        )

    return clf, pred_Y_li


def compute_inc_error_certificate(
    label_Y: pd.Series,
    prev_prop_Y_li: list[pd.Series],
    prop_Y_li: list[pd.Series],
) -> float:
    assert len(prev_prop_Y_li) == len(prop_Y_li), (
        f"Length of prev_prop_Y_li and prop_Y_li must be the same. "
        f"Got {len(prev_prop_Y_li)} and {len(prop_Y_li)}."
    )

    if len(prop_Y_li) == 1:
        return 0.0  # The first iteration introduces no error.

    pi = sum(label_Y) / len(label_Y)
    piE = sum(
        [sum(prev_prop_Y_li[i]) for i in range(len(prev_prop_Y_li))]
    ) / sum([len(prev_prop_Y_li[i]) for i in range(len(prev_prop_Y_li))])
    Gamma = max(pi, 1 - pi) / (min(pi, 1 - pi) + 1e-6)
    GammaE = max(piE, 1 - piE) / (min(piE, 1 - piE) + 1e-6)

    curr_data_size = sum(
        [len(prev_prop_Y_li[i]) for i in range(len(prev_prop_Y_li))]
    )
    prev_data_size = curr_data_size - len(prev_prop_Y_li[-1])
    new_data_size = len(prev_prop_Y_li[-1])
    data_err = new_data_size / curr_data_size / 1.5

    pred_err = 0.0
    for i in range(len(prev_prop_Y_li) - 1):
        pred_err += sum(prev_prop_Y_li[i] != prop_Y_li[i])
    pred_err /= prev_data_size

    err_certificate = (Gamma + GammaE) * (data_err + pred_err)

    return err_certificate
