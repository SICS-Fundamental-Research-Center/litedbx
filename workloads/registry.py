"""Shared workload registry and construction helpers."""

from typing import Any

from workloads.ldb_workload import LdbWorkload
from workloads.scenarios import animals, ecomm, medical, mmqa, movie

WORKLOAD_FACTORIES = {
    "medical": medical.get_workload,
    "movie": movie.get_workload,
    "ecomm": ecomm.get_workload,
    "mmqa": mmqa.get_workload,
    "animals": animals.get_workload,
}


def available_workloads() -> list[str]:
    """Return the sorted list of supported workload names."""
    return sorted(WORKLOAD_FACTORIES)


def build_workload(
    workload_name: str,
    queries: list[str],
    config_override: dict[str, Any] | None = None,
    exp_group: str = "default",
    enable_cache: bool = True,
) -> LdbWorkload:
    """Build a workload and apply experiment overrides."""
    workload_func = WORKLOAD_FACTORIES.get(workload_name)
    if workload_func is None:
        valid = ", ".join(available_workloads())
        raise ValueError(
            f"Invalid workload: {workload_name}. Available workloads: {valid}"
        )

    workload = workload_func(queries)
    effective_override = config_override or {}
    if exp_group != "default":
        workload.inject_exp_setting(
            exp_group=exp_group, exp_patch=effective_override
        )
    workload.set_cache_enabled(enable_cache)
    return workload
