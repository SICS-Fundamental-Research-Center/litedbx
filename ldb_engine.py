# pylint: disable=too-few-public-methods,duplicate-code
"""LiteDBX engine orchestration."""

import logging
from typing import Any

from workloads.ldb_workload import LdbWorkload

logger = logging.getLogger(__name__)


class LdbEngine:
    """Coordinate evaluation for one workload."""

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

    async def execute(self, debug: bool = False) -> dict[str, Any]:
        """Run the workload and return all engine result payloads."""
        self._log_launch()
        (
            execution_trace,
            phase_durations,
        ) = await self.workload.run_partial_evaluation(debug=debug)
        self._log_phase_durations(phase_durations)

        (
            _,
            incremental_results,
        ) = await self.workload.run_incremental_evaluation()
        self.workload.report_results(execution_trace, incremental_results)

        return self.workload.build_result_payload(
            execution_trace=execution_trace,
            phase_durations=phase_durations,
            incremental_results=incremental_results,
        )
