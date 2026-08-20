# pylint: disable=too-few-public-methods,duplicate-code
"""LiteDBX engine orchestration."""

import logging
from pathlib import Path
from typing import Any

import yaml

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
            logger.info(
                "Duration for phase '%s': %.2f seconds",
                phase,
                durations[phase],
            )

    async def execute(
        self, 
        debug: bool = False,
        certificate: bool = False
    ) -> dict[str, Any]:
        """Run the workload and return all engine result payloads."""
        self._log_launch()
        (
            execution_trace,
            phase_durations,
            probe_snapshot,
            retrieved_results,
        ) = await self.workload.run_partial_evaluation(debug=debug)
        self._log_phase_durations(phase_durations)

        error_certificates = {}
        error_bounds = {}
        if certificate and retrieved_results:
            error_certificates_path = (
                Path(__file__).parent / ".data_ckpt" / "error_certificates.yaml"
            )
            error_certificates_log = self.load_error_certificates_log(
                path=error_certificates_path
            )
            scenario = self.workload.scenario
            first_fraction = self.dynamic_setting[0]
            query_key = f"[{', '.join(self.workload.queries)}]@{first_fraction}"
            error_certificates = error_certificates_log.get(scenario, {}).get(
                query_key, {}
            )
            if not error_certificates:
                error_certificates = (
                    await self.workload.estimate_error_certificates(
                        probe_snapshot=probe_snapshot,
                        final_retrieved_data=retrieved_results,
                        selection_ratio=0.7,
                    )
                )
                error_certificates_log.setdefault(scenario, {})[query_key] = (
                    error_certificates
                )
                with error_certificates_path.open(
                    "w", encoding="utf-8"
                ) as cache_file:
                    yaml.safe_dump(
                        error_certificates_log, cache_file, sort_keys=False
                    )
            logger.info(
                "Estimated error certificates for queries: %s",
                error_certificates,
            )

        if certificate and error_certificates:
            for q_name, B in error_certificates.items():
                error_bounds[q_name] = self.workload.compute_error_bound(
                    q_name=q_name,
                    stream_idx=0,
                    B=B,
                )
            avg_bound = sum(error_bounds.values()) / len(error_bounds)
            logger.info(
                "Computed error bounds for queries: %s",
                error_bounds,
            )
            logger.info(
                "Average error bound across queries: %.4f", avg_bound
            )

        (
            _,
            incremental_results,
        ) = await self.workload.run_incremental_evaluation(
            error_certificates=error_certificates
        )
        self.workload.report_results(execution_trace, incremental_results)

        return self.workload.build_result_payload(
            execution_trace=execution_trace,
            phase_durations=phase_durations,
            incremental_results=incremental_results,
        )

    def load_error_certificates_log(
        self, path: Path
    ) -> dict[str, dict[str, dict[str, float]]]:
        """Load error certificates from the cache if available."""

        path.parent.mkdir(parents=True, exist_ok=True)

        if path.exists():
            with path.open(encoding="utf-8") as cache_file:
                return yaml.safe_load(cache_file) or {}
        return {}
