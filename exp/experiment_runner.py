"""Run LiteDBX experiment configurations and export collected metrics."""

import csv
import itertools
import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from exp.model_hosting import ensure_models, release_models
from ldb_engine import LdbEngine
from workloads.registry import build_workload

logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT_DIR / "exp" / "results"


@dataclass
class TaskResult:
    """Result metadata and collected rows for one experiment task."""

    config_name: str
    task_name: str
    started_at: str
    ended_at: str
    rows: list[dict[str, Any]] | None = None


@dataclass(frozen=True)
class TaskRunContext:
    """Shared state for one task execution."""

    config_name: str
    task: dict[str, Any]
    models: list[str]
    debug: bool


def load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML mapping from disk."""
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def _csv_cell(value: Any) -> Any:
    """Return a CSV-safe scalar representation for structured values."""
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True)
    return value


def save_csv(
    path: Path, rows: list[dict[str, Any]], fieldnames: list[str]
) -> None:
    """Write rows to a CSV file with the given field order."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(
            {field: _csv_cell(row.get(field, "")) for field in fieldnames}
            for row in rows
        )


def selected_execution_trace_iterations(
    trace_spec: dict[str, Any], execution_trace: dict[str, Any]
) -> list[Any]:
    """Return execution trace iteration keys selected by the collect spec."""
    iterations = trace_spec.get("iterations", "ALL")
    if iterations == "ALL":
        return sorted(execution_trace.keys())
    if isinstance(iterations, list):
        return iterations
    raise ValueError("execution_trace.iterations must be ALL or a list")


def execution_trace_features(trace_spec: dict[str, Any]) -> list[Any]:
    """Return feature names selected by the collect spec."""
    features = trace_spec.get("features", [])
    if not isinstance(features, list):
        raise ValueError("execution_trace.features must be a list")
    return features


def get_nested_field_value(
    payload: dict[str, Any], field: str, query_name: str, iter_num: Any
) -> Any:
    """Return a possibly dotted field value from an execution trace payload."""
    value: Any = payload
    traversed = []
    for part in field.split("."):
        traversed.append(part)
        if not isinstance(value, dict):
            path = ".".join(traversed[:-1])
            raise ValueError(
                f"Cannot access nested field '{field}' for query "
                f"{query_name} in execution trace iteration {iter_num}: "
                f"'{path}' is not a mapping"
            )
        if part not in value:
            path = ".".join(traversed)
            raise ValueError(
                f"Missing field '{path}' for query {query_name} in "
                f"execution trace iteration: {iter_num}"
            )
        value = value[part]
    return value


def query_trace_for_iteration(
    execution_trace: dict[str, Any], iter_num: Any, query_name: str
) -> dict[str, Any]:
    """Return the per-query trace payload for one iteration."""
    if iter_num not in execution_trace:
        raise ValueError(f"Missing execution trace iteration: {iter_num}")

    iter_trace = execution_trace[iter_num]
    if not isinstance(iter_trace, dict):
        raise ValueError(
            f"Execution trace iteration {iter_num} must be a mapping"
        )
    if query_name not in iter_trace:
        raise ValueError(
            f"Missing query {query_name} in execution "
            f"trace iteration: {iter_num}"
        )

    query_trace = iter_trace[query_name]
    if not isinstance(query_trace, dict):
        raise ValueError(
            f"Execution trace for query {query_name} in iteration "
            f"{iter_num} must be a mapping"
        )
    return query_trace


def collect_execution_trace_info(
    query_name: str,
    trace_spec: Any,
    execution_trace: dict[str, Any],
) -> list[dict[str, Any]]:
    """Collect rows from engine_result.execution_trace only."""
    if trace_spec is None:
        return []
    if not isinstance(trace_spec, dict):
        raise ValueError("execution_trace collect spec must be a mapping")
    if not isinstance(execution_trace, dict):
        raise ValueError("engine_result.execution_trace must be a mapping")

    iterations = selected_execution_trace_iterations(
        trace_spec, execution_trace
    )
    features = execution_trace_features(trace_spec)
    if not features:
        return []

    execution_trace_info = []

    for iter_num in iterations:
        query_trace = query_trace_for_iteration(
            execution_trace, iter_num, query_name
        )
        feature_values = {
            feature: get_nested_field_value(
                query_trace, feature, query_name, iter_num
            )
            for feature in features
        }

        execution_trace_info.append(
            {
                "iter_num": iter_num,
                "query": query_name,
                **feature_values,
            }
        )

    return execution_trace_info


def parse_collect_spec(
    static_info: dict[str, Any],
    query_group: list[str],
    collect_specs: dict[str, Any],
    engine_result: dict[str, Any],
) -> list[dict[str, Any]]:
    """Expand structured collect config into result rows."""
    if not collect_specs:
        return []
    if not isinstance(collect_specs, dict):
        raise ValueError("collect must be a mapping")

    collected_info = []

    # Collect desired fields from the engine_result.
    for query in query_group:
        # Collect fields from execution trace.
        trace_spec = collect_specs.get("execution_trace", None)
        trace_info_list = collect_execution_trace_info(
            query_name=query,
            trace_spec=trace_spec,
            execution_trace=engine_result.get("execution_trace", {}),
        )
        collected_info.extend(
            [
                {
                    **static_info,
                    **trace_info,
                }
                for trace_info in trace_info_list
            ]
        )

    return collected_info


def iter_override_maps(task: dict[str, Any]) -> list[dict[str, Any]]:
    """Expand task override settings into concrete override maps."""
    override_specs = task.get("config_override", {})
    override_keys = list(override_specs.keys())
    override_values = [override_specs[k] for k in override_keys]

    for override_val in override_values:
        if not isinstance(override_val, list):
            raise ValueError(
                f"Override values must be lists, got: {override_val}"
            )

    if not override_values:
        return [{}]
    return [
        dict(zip(override_keys, override_combo, strict=True))
        for override_combo in itertools.product(*override_values)
    ]


async def run_query_group(
    context: TaskRunContext,
    workload_name: str,
    query_group: list[str],
    override_map: dict[str, Any],
) -> list[dict[str, Any]]:
    """Run one workload query group and return collected rows."""
    exp_group = context.task.get("exp_group", "default")
    workload = build_workload(
        workload_name,
        query_group,
        override_map,
        exp_group=exp_group,
    )
    engine = LdbEngine(workload)
    engine_result = await engine.execute(
        debug=bool(context.task.get("debug", context.debug))
    )

    collect_specs = context.task.get("collect", {})
    if not collect_specs:
        return []

    static_info = {
        "config": context.config_name,
        "task": context.task.get("name", context.config_name),
        "workload": workload_name,
        **{k: str(v) for k, v in override_map.items()},
    }
    return parse_collect_spec(
        static_info=static_info,
        query_group=query_group,
        collect_specs=collect_specs,
        engine_result=engine_result,
    )


async def run_task(context: TaskRunContext) -> TaskResult:
    """Run one expanded experiment task."""
    started_at = datetime.now(UTC).isoformat()
    rows: list[dict[str, Any]] = []

    try:
        await ensure_models(context.models)
        for obj in context.task.get("objects", []):
            workload_name = obj["workload"]
            for query_group in obj["queries"]:
                for override_map in iter_override_maps(context.task):
                    rows.extend(
                        await run_query_group(
                            context,
                            workload_name,
                            query_group,
                            override_map,
                        )
                    )
    finally:
        release_models(
            context.task.get("models", []),
            context.task.get("release_model_after_task", False),
        )

    ended_at = datetime.now(UTC).isoformat()
    return TaskResult(
        config_name=context.config_name,
        task_name=context.task.get("name", context.config_name),
        started_at=started_at,
        ended_at=ended_at,
        rows=rows if rows else None,
    )


async def run_config(
    config_path: Path, debug: bool = False
) -> list[TaskResult]:
    """Run all tasks declared in one experiment config."""
    config = load_yaml(config_path)
    tasks = config.get("tasks", [])
    results: list[TaskResult] = []
    for task in tasks:
        task_context = TaskRunContext(
            config_name=config_path.stem,
            task=task,
            models=task.get("models", []),
            debug=debug,
        )
        results.append(await run_task(task_context))
    return results


def export_results(config_path: Path, results: list[TaskResult]) -> Path:
    """Export collected task results to CSV."""
    rows: list[dict[str, Any]] = []
    for result in results:
        if result.rows:
            rows.extend(result.rows)

    fieldnames: list[str] = []
    for row in rows:
        for field in row:
            if field not in fieldnames:
                fieldnames.append(field)

    out_path = RESULTS_DIR / f"{config_path.stem}.csv"
    save_csv(out_path, rows, fieldnames)
    return out_path
