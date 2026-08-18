"""Validation for the shared workload configuration contract."""

from collections.abc import Mapping
from numbers import Real
from typing import Any

UNIVERSAL_CONFIG_KEYS = frozenset(
    {
        "random_seed",
        "b_lab",
        "b_se",
        "b_rew",
        "b_fs",
        "k_neighbors",
        "enable_hitl",
        "enable_conf_struct",
        "enable_conf_pred",
        "enable_enrich",
        "enable_rewrite",
        "enable_coreset_expansion",
        "enable_subj",
        "enable_obj",
        "loo_step",
        "delta",
        "dynamic_setting",
    }
)


def validate_workload_config(config: Mapping[str, Any]) -> None:
    """Reject incomplete, unknown, or query-specific workload settings."""
    keys = set(config)
    missing = UNIVERSAL_CONFIG_KEYS - keys
    unknown = keys - UNIVERSAL_CONFIG_KEYS
    if missing:
        raise ValueError(f"Missing workload config keys: {sorted(missing)}")
    if unknown:
        raise ValueError(f"Unknown workload config keys: {sorted(unknown)}")

    nested = [
        key for key, value in config.items() if isinstance(value, Mapping)
    ]
    if nested:
        raise ValueError(
            "Workload config values must be universal, not query-specific: "
            f"{sorted(nested)}"
        )

    _require_int(config, "random_seed")
    for key in ("b_lab", "b_se", "b_rew", "b_fs"):
        _require_int(config, key, minimum=0)
    _require_int(config, "k_neighbors", minimum=1)
    _require_int(config, "loo_step", minimum=1)

    for key in (
        "enable_hitl",
        "enable_conf_struct",
        "enable_conf_pred",
        "enable_enrich",
        "enable_rewrite",
        "enable_coreset_expansion",
    ):
        if not isinstance(config[key], bool):
            raise ValueError(f"Workload config {key!r} must be boolean.")

    delta = config["delta"]
    if (
        isinstance(delta, bool)
        or not isinstance(delta, Real)
        or not 0 < float(delta) < 1
    ):
        raise ValueError("Workload config 'delta' must be between 0 and 1.")
    dynamic_setting = config["dynamic_setting"]
    if not isinstance(dynamic_setting, list) or not dynamic_setting:
        raise ValueError(
            "Workload config 'dynamic_setting' must be a nonempty list."
        )
    if any(
        isinstance(step, bool)
        or not isinstance(step, Real)
        or not 0 < float(step) <= 1
        for step in dynamic_setting
    ):
        raise ValueError("Dynamic steps must be numeric values in (0, 1].")
    if any(
        left >= right
        for left, right in zip(
            dynamic_setting, dynamic_setting[1:], strict=False
        )
    ):
        raise ValueError("Dynamic steps must be strictly increasing.")


def _require_int(
    config: Mapping[str, Any], key: str, minimum: int | None = None
) -> None:
    value = config[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"Workload config {key!r} must be an integer.")
    if minimum is not None and value < minimum:
        raise ValueError(f"Workload config {key!r} must be at least {minimum}.")
