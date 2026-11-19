# pylint: disable=invalid-name,too-many-instance-attributes
# pylint: disable=missing-function-docstring,duplicate-code
"""Engine-facing LiteDBX workload facade."""

import copy
import logging
import os
from dataclasses import dataclass, replace
from pathlib import Path
from time import time
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import hypergeom

from data_structure import LdbDataManager, SemCQ, SemUCQ
from data_structure.annotation_sampling import AnnotationSelection
from data_structure.ldb_data import LdbData
from llm import LdbLLMClient
from workloads.config_schema import validate_workload_config
from workloads.core.coreset_maintainer import CoresetMaintainer
from workloads.core.feature_pipeline import FeaturePipeline
from workloads.core.preprocessing import Preprocessing
from workloads.core.query_execution import QueryExecution
from workloads.core.reporting import Reporting
from workloads.core.ucq_exec import flatten_sem_queries
from workloads.utils import (
    apply_rules,
    encode_features,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LiteDBXRuntimeSnapshot:
    """Captured LiteDBX runtime state for export and resumed evaluation.

    Holds the data/annotation state together with the trained components
    built on it, so downstream work (e.g. certificate estimation under a
    reduced label budget) can resume without rebuilding the feature space,
    coreset, or query machinery.
    """

    data_manager: LdbDataManager
    annotation_selections: dict[str, AnnotationSelection]
    feature_pipeline: FeaturePipeline
    coreset_maintainer: CoresetMaintainer
    query_execution: QueryExecution


def experiment_checkpoint_path(
    exp_group: str,
    scenario: str,
    exp_patch: dict,
    dynamic_setting: list[float],
) -> Path:
    """Build an experiment checkpoint path from its complete identity."""
    path = Path(__file__).parent.parent / ".data_ckpt" / exp_group / scenario
    if exp_patch:
        exp_term = "_".join(
            f"{key}={value}" for key, value in exp_patch.items()
        )
        path /= exp_term
    return path / "_".join(str(step) for step in dynamic_setting)


class LdbWorkload:
    """LiteDBX workload API consumed by :class:`ldb_engine.LdbEngine`."""

    def __init__(
        self,
        data_dir: str,
        scenario: str,
        queries: dict[str, SemUCQ],
        config: dict,
    ) -> None:
        self.llm_client = LdbLLMClient()

        self.data_dir = data_dir
        self.scenario = scenario
        # All-UCQ workload contract (plain filter queries arrive wrapped
        # in self-named single-branch UCQs): flatten to engine-level
        # branch SemCQs here; the engine and every downstream component
        # only ever see the flat dict, keyed by engine query name.
        flat_queries, ucq_map = flatten_sem_queries(queries)
        self.queries: dict[str, SemCQ] = flat_queries
        self.ucq_map: dict[str, list[str]] = ucq_map
        self.ucqs: dict[str, SemUCQ] = dict(queries)
        validate_workload_config(config)
        self.config = dict(config)
        self.enable_cache = True
        self._load_config_values()

        self.CKPT_path = (
            Path(__file__).parent.parent
            / ".data_ckpt"
            / self.scenario
            / "_".join(str(step) for step in self.dynamic_setting)
        )
        self._ensure_query_ckpts()

        self.usage_statistics = self._init_usage_statistics(
            self.dynamic_setting
        )

        self._init_components()

    def _init_components(self) -> None:
        self._preprocessing = Preprocessing(
            llm_client=self.llm_client,
            data_manager=self.data_manager,
            queries=self.queries,
            ckpt_path=self.CKPT_path,
            usage_statistics=self.usage_statistics,
            enable_cache=self.enable_cache,
        )
        (
            self._feature_pipeline,
            self._coreset_maintainer,
            self._query_execution,
        ) = self._build_runtime_components(
            self.data_manager, self.enable_cache)
        self._reporting = Reporting(self.usage_statistics)

    def _build_runtime_components(
        self, data_manager: LdbDataManager, enable_cache: bool
    ) -> tuple[FeaturePipeline, CoresetMaintainer, QueryExecution]:
        """Build feature/coreset/query components bound to a data manager.

        Used both for the live runtime (``self.data_manager``) and for the
        isolated probe snapshot, whose components must operate on the
        deep-copied data manager rather than the live one.
        """
        feature_pipeline = FeaturePipeline(
            data_manager=data_manager,
            queries=self.queries,
            ckpt_path=self.CKPT_path,
            llm_client=self.llm_client,
            usage_statistics=self.usage_statistics,
            random_seed=self.random_seed,
            b_lab=self.b_lab,
            b_se=self.b_se,
            b_fs=self.b_fs,
            enable_hitl=self.enable_hitl,
            enable_cache=enable_cache,
        )
        coreset_maintainer = CoresetMaintainer(
            data_manager=data_manager,
            config=self.config,
            enable_conf_pred=self.enable_conf_pred,
            enable_conf_struct=self.enable_conf_struct,
        )
        query_execution = QueryExecution(
            coreset_maintainer=coreset_maintainer,
            data_manager=data_manager,
            queries=self.queries,
            config=self.config,
            llm_client=self.llm_client,
            dynamic_setting=self.dynamic_setting,
            b_rew=self.b_rew,
            b_lab=self.b_lab,
            delta=self.delta,
            enable_subj=self.enable_subj,
            enable_obj=self.enable_obj,
            enable_cache=enable_cache,
        )
        return feature_pipeline, coreset_maintainer, query_execution

    @staticmethod
    def _init_usage_statistics(dynamic_setting: list[float]) -> list[dict]:
        return [
            {
                item: {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                    "prompt_cost": 0.0,
                    "completion_cost": 0.0,
                    "total_cost": 0.0,
                }
                for item in [
                    "sigma_augmentation",
                    "feature_space_init",
                    "feature_space_refine",
                    "materialize_labeled_full",
                    "materialize_unlabeled_full",
                    "materialize_unlabeled_inc",
                ]
            }
            for _ in range(len(dynamic_setting))
        ]

    # ------------------------------------------------------------------
    # Public engine-facing API
    # ------------------------------------------------------------------

    async def run_partial_evaluation(
        self, debug: bool = False
    ) -> tuple[
        dict,
        dict[str, Any],
        LiteDBXRuntimeSnapshot,
        dict[str, set[tuple]],
        dict[str, pd.DataFrame],
    ]:
        # float phase seconds + model-swap accounting (list intervals)
        durations: dict[str, Any] = {}
        # epoch bounds of each timed phase, for swap-interval exclusion
        # (user order 2026-11-18: engine-initiated model swaps are
        # servicing overhead, not query time)
        phase_bounds: dict[str, tuple[float, float]] = {}

        phase_start = time()
        self.data_manager.init_data_stream()
        self.data_manager.init_sigma_satisfied_data()
        if os.environ.get("LDB_DISABLE_PREFILTER") == "1":
            # R3.D13 variant run: skip the LLM prefilter UCQ stage and mirror
            # the zero-suggestion downstream state (no base features eligible)
            # so schema columns stay out of rewrite candidates — identical to
            # queries where the prefilter suggests nothing.
            for q_name in self.queries:
                self.data_manager.relevant_base_features[q_name] = []
        else:
            self._preprocessing.refine_sigma_satisfied_data()
        durations["preprocessing"] = time() - phase_start
        phase_bounds["preprocessing"] = (phase_start, time())

        phase_start = time()
        step_start = phase_start
        await self._feature_pipeline.construct_feature_space()
        annotation_start = time()
        annotation_selections = await self.data_manager.acquire_annotation(
            b_lab=self.b_lab,
            seed=self.random_seed,
            use_hitl=self.enable_hitl,
            enable_cache=self.enable_cache,
        )
        annotation_end = time()
        durations["coreset_annotation"] = annotation_end - annotation_start
        # Snapshot the pre-coreset probe state for certificate estimation.
        # The snapshot owns a deep copy of the data manager plus fresh
        # components bound to that copy, so certificate estimation runs in
        # full isolation and never mutates the live runtime state.
        probe_data_manager = copy.deepcopy(self.data_manager)
        (
            probe_feature_pipeline,
            probe_coreset_maintainer,
            probe_query_execution,
        ) = self._build_runtime_components(probe_data_manager, False)
        probe_snapshot = LiteDBXRuntimeSnapshot(
            data_manager=probe_data_manager,
            annotation_selections=copy.deepcopy(annotation_selections),
            feature_pipeline=probe_feature_pipeline,
            coreset_maintainer=probe_coreset_maintainer,
            query_execution=probe_query_execution,
        )
        await self.data_manager.init_coreset(
            annotation_selections=annotation_selections,
            use_hitl=self.enable_hitl,
            enable_cache=self.enable_cache,
        )
        step_end = time()
        durations["coreset_feature_space"] = step_end - step_start
        phase_bounds["coreset_feature_space"] = (step_start, step_end)

        step_start = step_end
        await self._feature_pipeline.sync_coreset_features(tag="init")
        step_end = time()
        durations["coreset_sync"] = step_end - step_start
        phase_bounds["coreset_sync"] = (step_start, step_end)

        step_start = step_end
        await self._feature_pipeline.rank_and_trim_feature_space()
        step_end = time()
        durations["query_trim"] = step_end - step_start

        step_start = step_end
        if self.enable_coreset_expansion:
            self._coreset_maintainer.expand_coresets(inc_round=0)
        step_end = time()
        durations["coreset_expand"] = step_end - step_start
        durations["coreset_total"] = step_end - phase_start

        phase_start = time()
        step_start = phase_start
        (
            retrieved_results,
            retrieved_frames,
            execution_trace,
        ) = await self._query_execution.execute_queries(
            enable_rewrite=self.enable_rewrite,
            enable_enrich=self.enable_enrich,
            debug=debug,
        )
        step_end = time()
        durations["query_rewrite"] = step_end - step_start
        durations["query_total"] = step_end - phase_start
        phase_bounds["query_rewrite"] = (step_start, step_end)
        phase_bounds["query_total"] = (phase_start, step_end)

        # Swap-interval exclusion (user order 2026-11-18): gross stays
        # keyed as before; each phase overlapped by an engine-initiated
        # servem.sh swap additionally reports net + overlap, and the
        # swap intervals + probe overhead ride along for the run summary.
        swap_mgr = getattr(self.llm_client, "auto_swap", None)
        if swap_mgr is not None:
            if getattr(swap_mgr, "probe_seconds", 0.0):
                durations["ensure_probe_total_s"] = round(
                    swap_mgr.probe_seconds, 3)
            if swap_mgr.intervals:
                durations["model_swap_total_s"] = swap_mgr.total_seconds()
                durations["model_swap_intervals"] = list(swap_mgr.intervals)
                for phase, (t0, t1) in phase_bounds.items():
                    overlap = swap_mgr.overlap_seconds(t0, t1)
                    if overlap > 0:
                        durations[f"{phase}__swap_overlap_s"] = round(
                            overlap, 3)
                        durations[f"{phase}__net_s"] = round(
                            durations[phase] - overlap, 3)

        return (
            execution_trace,
            durations,
            probe_snapshot,
            retrieved_results,
            retrieved_frames,
        )

    def final_retrieved_tables(
        self,
        retrieved_tables: dict[str, pd.DataFrame],
        rerun: bool,
        incremental_results: list,
    ) -> dict[str, pd.DataFrame]:
        """Replace the partial-eval tables when incremental evaluation ran.

        The answer basis is the FINAL accumulated retrieval state: the
        last inc round's trans_eval frames aggregate streams 0..k (see
        SigmaSatisfiedData._collect_quality_sets), so the last entry
        wins. ``rerun=True`` means a step demanded full re-execution and
        incremental_processing returned no valid results ([]); the
        partial-eval tables stand. Frames carry whatever columns the
        per-query Sigma frames have (base + enriched llm_label_*); all
        consumers are name-based, so no schema normalization happens
        (user order 2026-11-19).
        """
        if rerun or not incremental_results:
            return retrieved_tables
        last_round = incremental_results[-1]["eval_results"]
        return {
            q_name: last_round[q_name]["trans_eval"]["retrieved_frame"]
            for q_name in self.queries
        }

    def evaluate_ucqs(
        self, tables: dict[str, pd.DataFrame]
    ) -> dict[str, dict[str, Any]]:
        """Assemble and score every SemUCQ answer against its gt_path."""
        results: dict[str, dict[str, Any]] = {}
        for name, ucq in self.ucqs.items():
            missing = [b for b in ucq.queries if b not in tables]
            if missing:
                raise KeyError(
                    f"SemUCQ {name!r} needs branch tables {missing} that "
                    "were not exported."
                )
            answer = ucq.assembler({b: tables[b] for b in ucq.queries})
            metrics = ucq.evaluator(answer, ucq.gt_path)
            logger.info(
                "SemUCQ '%s' answer evaluated: kind=%s quality=%s",
                name,
                metrics.get("kind"),
                metrics.get("quality", metrics.get("f1")),
            )
            results[name] = {"answer": answer, "metrics": metrics}
        return results

    @staticmethod
    def error_bound(B: float, pi: float) -> float:
        if B is None or pi is None:
            raise ValueError("Both B and pi must be non-negative numbers")
        if B < 0 or B > 1 or pi < 0 or pi > 1:
            raise ValueError("Both B and pi must be in the range [0, 1]")
        if 0 <= B < pi:
            return 2 * (pi - B) / (2 * pi - B)
        return 0.0

    def compute_error_bound(
        self, q_name: str, stream_idx: int, B: float
    ) -> float:
        """Compute the error bound for a given query and stream index."""
        pi = self.data_manager.sigma_satisfied_data[stream_idx][q_name][
            "accumulated_true_selectivity"
        ]
        if pi is None:
            raise ValueError(
                f"True selectivity for query '{q_name}' in stream "
                f"{stream_idx} is not available."
            )
        logger.info(
            "Computing error bound for query '%s' in stream %d: "
            "B=%.4f, pi=%.4f",
            q_name,
            stream_idx,
            B,
            pi,
        )
        return self.error_bound(B, pi)

    async def estimate_error_certificates(
        self,
        probe_snapshot: LiteDBXRuntimeSnapshot,
        final_retrieved_data: dict[str, set[tuple]],
        selection_ratio: float,
    ) -> dict[str, float]:
        """Estimate the full-scope 0-1 error certificate."""

        error_certificates = {}

        # Update the annotation selections.
        (
            annotation_selections,
            annotation_data_T,
            annotation_labels_T,
            annotation_data_V,
            annotation_labels_V,
        ) = self.split_annotations(
            ldb_snapshot=probe_snapshot,
            selection_ratio=selection_ratio,
        )

        # Use the new annotation to construct the remaining workflow of LiteDBX.
        await probe_snapshot.data_manager.init_coreset(
            annotation_selections=annotation_selections,
            use_hitl=self.enable_hitl,
            enable_cache=self.enable_cache,
        )
        await probe_snapshot.feature_pipeline.sync_coreset_features(tag="init")
        await probe_snapshot.feature_pipeline.rank_and_trim_feature_space()
        if self.enable_coreset_expansion:
            probe_snapshot.coreset_maintainer.expand_coresets(inc_round=0)
        (
            probe_retrieved_data,
            _,  # retrieved frames: unused, certificate math is set-based
            _,
        ) = await probe_snapshot.query_execution.execute_queries(
            enable_rewrite=self.enable_rewrite,
            enable_enrich=self.enable_enrich,
            debug=False,
        )

        for q_name in self.queries:
            # Sync the new features for annotated samples and apply the UCQ.
            (
                pred_labels_T,
                pred_labels_V,
            ) = await self.sync_annotations_and_apply_rules(
                q_name=q_name,
                ldb_snapshot=probe_snapshot,
                annotation_data_T=annotation_data_T[q_name],
                annotation_data_V=annotation_data_V[q_name],
            )

            error_certificates[q_name] = await self.estimate_error_certificate(
                q_name=q_name,
                ldb_snapshot=probe_snapshot,
                annotation_labels_T=annotation_labels_T[q_name],
                annotation_labels_V=annotation_labels_V[q_name],
                pred_labels_T=pred_labels_T,
                pred_labels_V=pred_labels_V,
                probe_retrieved_data=probe_retrieved_data[q_name],
                final_retrieved_data=final_retrieved_data[q_name],
            )

        return error_certificates

    async def estimate_error_certificate(
        self,
        q_name: str,
        ldb_snapshot: LiteDBXRuntimeSnapshot,
        annotation_labels_T: pd.Series,
        annotation_labels_V: pd.Series,
        pred_labels_T: pd.Series,
        pred_labels_V: pd.Series,
        probe_retrieved_data: set[tuple],
        final_retrieved_data: set[tuple],
    ) -> float:
        """
        Return:
            Bi := min{ 1, di + Ui }

        Where:
            Sigma(D) := post-refinement candidate set (candidate_size); the
                        set the learned rule operates on and the only set the
                        verification split V samples, so the capture-
                        recapture is scoped to it. Discarded rows fall outside
                        this universe; refinement is expected to be sound
                        (introduced_fn = 0) so the candidate F1 the bound
                        addresses equals the global F1 the evaluation reports.
            di := INDICATOR( Sigma(D), hat(qT) != qT ) / |Sigma(D)|
            Ui := (e^init + K^+) / |Sigma(D)|
            e_init := SUM(T, INDICATOR( hat(qT) != Y ) )
            xi := SUM( V, INDICATOR( hat(qV) != Y ) )
            alphai := delta / |Q^S|
            Mi := |Sigma(D) setminus T|
            hi := |V|

            X(k) := random variable sampled from Hypergeom(Mi, k, hi)
                    0 <= k <= Mi
            F(X(k)) := P(X(k) <= xi)
            K+ := the max k such that F(X(k)) > alphai.
        """

        if len(annotation_labels_T) != len(pred_labels_T):
            raise ValueError(
                f"Length mismatch for training labels and predictions: "
                f"{len(annotation_labels_T)} vs {len(pred_labels_T)}"
            )
        if len(annotation_labels_V) != len(pred_labels_V):
            raise ValueError(
                f"Length mismatch for verification labels and predictions: "
                f"{len(annotation_labels_V)} vs {len(pred_labels_V)}"
            )

        sigma_D = ldb_snapshot.data_manager.sigma_satisfied_data[0][q_name]
        # Certifiable universe = post-refinement candidate set (candidate_size):
        # the set the learned rule operates on and the verification split V
        # samples. Invariant to the annotation row-shuffling that empties
        # ldb_data/selected_data by snapshot time. Discarded rows fall outside
        # this universe; refinement is expected to be sound (introduced_fn = 0)
        # so the candidate F1 the bound addresses equals the global F1 the
        # evaluation reports. introduced_fn is logged per query so any unsound
        # refinement is visible in the run output.
        sigma_D_size = sigma_D["candidate_size"]

        d = len(probe_retrieved_data ^ final_retrieved_data) / sigma_D_size
        e_init = (annotation_labels_T != pred_labels_T).sum()
        x = (annotation_labels_V != pred_labels_V).sum()
        alpha = self.delta / len(self.queries)
        Mi = sigma_D_size - len(annotation_labels_T)
        hi = len(annotation_labels_V)

        if Mi <= 0 or float(hypergeom.cdf(x, M=Mi, n=0, N=hi)) <= alpha:
            k_plus = 0
        else:
            lb, ub = 0, Mi
            while lb < ub:
                mid = (lb + ub + 1) // 2
                if float(hypergeom.cdf(x, M=Mi, n=mid, N=hi)) > alpha:
                    lb = mid
                else:
                    ub = mid - 1
            k_plus = lb

        U = (e_init + k_plus) / sigma_D_size

        logger.info(
            "Error certificate estimation for query '%s': "
            "d=%.4f, e_init=%d, x=%d, alpha=%.4f, Mi=%d, "
            "hi=%d, k_plus=%d, U=%.4f",
            q_name,
            d,
            e_init,
            x,
            alpha,
            Mi,
            hi,
            k_plus,
            U,
        )


        return float(min(1.0, d + U))

    def acquire_annotated_samples(
        self,
        q_name: str,
        ldb_snapshot: LiteDBXRuntimeSnapshot,
        annotation_selection: AnnotationSelection,
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Acquire the annotated samples for each query from the snapshot."""

        annotated_indices = annotation_selection.indices
        record = ldb_snapshot.data_manager.sigma_satisfied_data[0][q_name]
        record_data = record["ldb_data"].df.copy()
        record_labels = record["labels"]
        if record_labels is None:
            raise ValueError(
                f"Labels for query '{q_name}' are not available in the "
                f"snapshot."
            )

        return record_data.loc[annotated_indices], record_labels.loc[
            annotated_indices
        ]

    def split_annotations(
        self,
        ldb_snapshot: LiteDBXRuntimeSnapshot,
        selection_ratio: float,
    ) -> tuple[
        dict[str, AnnotationSelection],
        dict[str, LdbData],
        dict[str, pd.Series],
        dict[str, LdbData],
        dict[str, pd.Series],
    ]:
        """Split each annotation selection into train and verify parts."""

        rng = np.random.default_rng(self.random_seed)
        annotation_data_T: dict[str, LdbData] = {}
        annotation_labels_T: dict[str, pd.Series] = {}
        annotation_data_V: dict[str, LdbData] = {}
        annotation_labels_V: dict[str, pd.Series] = {}

        for q_name, selection in list(
            ldb_snapshot.annotation_selections.items()
        ):
            data_q, labels_q = self.acquire_annotated_samples(
                q_name=q_name,
                ldb_snapshot=ldb_snapshot,
                annotation_selection=selection,
            )

            total = len(selection.indices)
            selection_budget = min(int(total * selection_ratio + 0.5), total)

            ldb_data_config = ldb_snapshot.data_manager.complete_dataset.config

            if selection_budget >= total:
                # Nothing held out: all rows train, verify split is empty.
                annotation_data_T[q_name] = LdbData(
                    df=data_q, config=ldb_data_config
                )
                annotation_labels_T[q_name] = labels_q
                annotation_data_V[q_name] = LdbData(
                    df=data_q.iloc[:0], config=ldb_data_config
                )
                annotation_labels_V[q_name] = labels_q.iloc[:0]
                continue

            # Uniformly choose which rows survive the verification split.
            kept_positions = rng.choice(
                total, size=selection_budget, replace=False
            )
            kept = set(selection.indices[kept_positions])

            # Split the annotation rows into the kept (training) and discarded
            # (verification) partitions, aligned with the surviving rows.
            keep_mask = data_q.index.isin(kept)
            annotation_data_T[q_name] = LdbData(
                df=data_q[keep_mask], config=ldb_data_config
            )
            annotation_labels_T[q_name] = labels_q[keep_mask]
            annotation_data_V[q_name] = LdbData(
                df=data_q[~keep_mask], config=ldb_data_config
            )
            annotation_labels_V[q_name] = labels_q[~keep_mask]

            # Rebuild each stratum, retaining only surviving anchor/random rows.
            trimmed_strata = tuple(
                replace(
                    stratum,
                    anchor_indices=stratum.anchor_indices[
                        stratum.anchor_indices.isin(kept)
                    ],
                    random_indices=stratum.random_indices[
                        stratum.random_indices.isin(kept)
                    ],
                )
                for stratum in selection.strata
            )

            # Preserve the (already shuffled) order of the original selection.
            ldb_snapshot.annotation_selections[q_name] = replace(
                selection,
                indices=selection.indices[selection.indices.isin(kept)],
                strata=trimmed_strata,
            )

        return (
            ldb_snapshot.annotation_selections,
            annotation_data_T,
            annotation_labels_T,
            annotation_data_V,
            annotation_labels_V,
        )

    async def sync_annotations_and_apply_rules(
        self,
        q_name: str,
        ldb_snapshot: LiteDBXRuntimeSnapshot,
        annotation_data_T: LdbData,
        annotation_data_V: LdbData,
    ) -> tuple[pd.Series, pd.Series]:

        enriched_features = ldb_snapshot.data_manager.enriched_features[q_name]
        ucq = ldb_snapshot.data_manager.rewrite_rules[q_name]["ucq"]

        await annotation_data_T.sync_with_enriched_features(
            enriched_features=enriched_features,
            llm_client=self.llm_client,
            is_remote=False,
        )
        await annotation_data_V.sync_with_enriched_features(
            enriched_features=enriched_features,
            llm_client=self.llm_client,
            is_remote=False,
        )

        pred_labels_T = apply_rules(
            rules=ucq,
            df=encode_features(annotation_data_T.df),
        )

        pred_labels_V = apply_rules(
            rules=ucq,
            df=encode_features(annotation_data_V.df),
        )

        return pred_labels_T, pred_labels_V

    async def run_incremental_evaluation(
        self, error_certificates: dict[str, float]
    ) -> tuple[bool, list[Any]]:
        if len(self.dynamic_setting) <= 1:
            logger.info(
                "No incremental evaluation setting found. Skipping inc_eval."
            )
            return False, []

        rerun, result = await self._query_execution.incremental_processing(
            error_certificates=error_certificates
        )
        if not result:
            logger.info("No incremental processing results to report.")
        return rerun, result

    def report_results(
        self, execution_trace: dict[str, Any], results: list[Any]
    ) -> None:
        self._reporting.report_evaluation_trace(execution_trace)
        if results:
            self._reporting.report_dynamic_results(results)
        self._reporting.report_usage_statistics()

    def build_result_payload(
        self,
        execution_trace: dict[str, Any],
        phase_durations: dict[str, Any],
        incremental_results: list[Any],
        retrieved_tables: dict[str, pd.DataFrame] | None = None,
        ucq_results: dict[str, dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        payload = {
            "scenario": self.scenario,
            "queries": list(self.queries.keys()),
            "dynamic_setting": self.dynamic_setting,
            "execution_trace": execution_trace,
            "phase_durations": phase_durations,
            "incremental_results": incremental_results,
            "usage_statistics": self.usage_statistics[0],
        }
        if retrieved_tables is not None:
            payload["retrieved_tables"] = retrieved_tables
        if ucq_results is not None:
            payload["ucq_results"] = ucq_results
        return payload

    def inject_exp_setting(self, exp_group: str, exp_patch: dict) -> None:
        assert exp_group != "", (
            "Exp group cannot be empty when injecting exp setting."
        )
        assert exp_patch is not None, (
            "Exp patch cannot be None when injecting exp setting."
        )
        updated_config = {**self.config, **exp_patch}
        validate_workload_config(updated_config)
        self.config = updated_config
        for key, value in exp_patch.items():
            logger.info("Exp patch applied: %s=%s", key, value)

        self._load_config_values()

        self.CKPT_path = experiment_checkpoint_path(
            exp_group=exp_group,
            scenario=self.scenario,
            exp_patch=exp_patch,
            dynamic_setting=self.dynamic_setting,
        )
        self._ensure_query_ckpts()
        self.data_manager.set_ckpt_path(self.CKPT_path)
        self._init_components()

    def set_cache_enabled(self, enable_cache: bool) -> None:
        """Set the process cache policy before workload execution."""
        if self.enable_cache == enable_cache:
            return
        self.enable_cache = enable_cache
        logger.info(
            "Cache reads and writes are %s for this workload.",
            "enabled" if enable_cache else "disabled",
        )
        self._init_components()

    def _load_config_values(self) -> None:
        self.random_seed = self.config["random_seed"]
        self.b_lab = self.config["b_lab"]
        self.b_se = self.config["b_se"]
        self.b_rew = self.config["b_rew"]
        self.b_fs = self.config["b_fs"]
        self.delta = self.config["delta"]
        self.enable_hitl = self.config["enable_hitl"]
        self.enable_conf_pred = self.config["enable_conf_pred"]
        self.enable_conf_struct = self.config["enable_conf_struct"]
        self.enable_enrich = self.config["enable_enrich"]
        self.enable_rewrite = self.config["enable_rewrite"]
        self.enable_coreset_expansion = self.config["enable_coreset_expansion"]
        self.enable_subj = self.config.get("enable_subj", True)
        self.enable_obj = self.config.get("enable_obj", True)
        self.dynamic_setting = self.config["dynamic_setting"]
        self.data_manager = LdbDataManager(
            data_dir=self.data_dir,
            scenario=self.scenario,
            queries=self.queries,
            llm_client=self.llm_client,
            dynamic_steps=self.dynamic_setting,
        )

        if not (self.enable_subj or self.enable_obj):
            raise ValueError(
                "At least one of enable_subj or enable_obj must be True."
            )

    def _ensure_query_ckpts(self) -> None:
        for q_name in self.queries:
            (self.CKPT_path / q_name).mkdir(parents=True, exist_ok=True)

    def update_statistics(self, key: str, value: dict) -> None:
        assert key in self.usage_statistics[0], f"Invalid statistics key: {key}"
        for stat_key, stat_value in value.items():
            self.usage_statistics[0][key][stat_key] += stat_value

    def _update_statistics(self, key: str, value: dict) -> None:
        self.update_statistics(key, value)
