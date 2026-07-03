"""LiteDBX engine orchestration."""

import logging
from time import time
from typing import Any

from workloads.ldb_workload import LdbWorkload

logger = logging.getLogger(__name__)


class LdbEngine:
    """Coordinate partial and incremental evaluation for one workload."""

    def __init__(self, workload: LdbWorkload) -> None:
        """Initialize the engine with a workload instance."""
        self.workload = workload
        self.dynamic_setting = workload.dynamic_setting

    def _log_launch(self) -> None:
        """Log the initial workload execution context."""
        logger.info(
            "Launching LiteDBX Engine for queries: %s in scenario: %s "
            "with dynamic setting: %s.",
            list(self.workload.queries.keys()),
            self.workload.scenario,
            self.dynamic_setting,
        )
        logger.info(
            "Start PEval with %s%%-data partition.",
            self.dynamic_setting[0] * 100,
        )

    @staticmethod
    def _log_phase_durations(durations: dict[str, float]) -> None:
        """Log measured phase durations."""
        phases = [
            "preprocessing",
            "coreset_feature_space",
            "coreset_sync",
            "coreset_expand",
            "coreset_total",
            "query_trim",
            "query_rewrite",
            "query_total",
        ]
        for phase in phases:
            if phase not in durations:
                logger.warning("Duration for phase '%s' is missing.", phase)
                continue
            logger.debug(
                "Duration for phase '%s': %.2f seconds",
                phase,
                durations[phase],
            )

    def _report_results(
        self, execution_trace: dict[str, Any], results: list[Any]
    ) -> None:
        """Report evaluation, incremental, and usage results."""
        # LdbWorkload exposes reporting hooks as protected methods today.
        # Keep that coupling local until the workload API grows public wrappers.
        # pylint: disable=protected-access
        self.workload._report_evaluation_trace(execution_trace)

        if results:
            self.workload._report_dynamic_results(results)

        self.workload._report_usage_statistics()

    async def p_eval(
        self, debug: bool = False
    ) -> tuple[dict[str, Any], dict[str, float]]:
        """Run partial evaluation and return the trace and durations."""
        self._log_launch()
        durations: dict[str, float] = {}

        # Phase 1: preprocessing.
        phase_start = time()
        self.workload.data_manager.init_data_stream()
        self.workload.data_manager.init_sigma_satisfied_data()
        self.workload.refine_sigma_satisfied_data()
        durations["preprocessing"] = time() - phase_start

        # Phase 2: feature space and coreset construction.
        phase_start = time()
        step_start = phase_start
        await self.workload.construct_feature_space(debug=debug)
        step_end = time()
        durations["coreset_feature_space"] = step_end - step_start

        step_start = step_end
        await self.workload.sync_with_enriched_features(tag="init")
        step_end = time()
        durations["coreset_sync"] = step_end - step_start

        step_start = step_end
        self.workload.expand_coresets(inc_round=0)
        step_end = time()
        durations["coreset_expand"] = step_end - step_start
        durations["coreset_total"] = step_end - phase_start

        # Phase 3: schema selection and query rewriting.
        phase_start = time()
        step_start = phase_start
        await self.workload.rank_and_trim_feature_space()
        step_end = time()
        durations["query_trim"] = step_end - step_start

        execution_trace: dict[str, Any] = {}
        step_start = step_end
        if not self.workload.enable_rewrite:
            await self.workload.rewrite_and_execute_query_noRew()
        elif not self.workload.enable_enrich:
            self.workload.rewrite_and_execute_query_noEnr()
        else:
            _, execution_trace = self.workload.rewrite_and_execute_query()
        step_end = time()
        durations["query_rewrite"] = step_end - step_start
        durations["query_total"] = step_end - phase_start

        self._log_phase_durations(durations)
        return execution_trace, durations

    async def inc_eval(self) -> tuple[bool, list[Any]]:
        """Run incremental evaluation when the dynamic setting requires it."""
        if len(self.dynamic_setting) <= 1:
            logger.info(
                "No incremental evaluation setting found. Skipping inc_eval."
            )
            return False, []

        rerun, result = await self.workload.incremental_processing()

        if not result:
            logger.info("No incremental processing results to report.")

        return rerun, result

    async def execute(self, debug: bool = False) -> dict[str, Any]:
        """Run the workload and return all engine result payloads."""
        execution_trace, phase_durations = await self.p_eval(debug=debug)
        _, results = await self.inc_eval()

        self._report_results(execution_trace, results)

        return {
            "scenario": self.workload.scenario,
            "queries": list(self.workload.queries.keys()),
            "dynamic_setting": self.dynamic_setting,
            "execution_trace": execution_trace,
            "phase_durations": phase_durations,
            "incremental_results": results,
            "usage_statistics": self.workload.usage_statistics[0],
        }
