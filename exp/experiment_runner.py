"""Run LiteDBX experiment configurations and export collected metrics."""

import csv
import itertools
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


def save_csv(
    path: Path, rows: list[dict[str, Any]], fieldnames: list[str]
) -> None:
    """Write rows to a CSV file with the given field order."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def collect_execution_trace_info(
    query_name: str,
    trace_spec: Any,
    execution_trace: dict[str, Any],
) -> list[dict[str, Any]]:
    """Collect rows from engine_result.execution_trace only."""
    if trace_spec is None:
        return []

    iterations = trace_spec.get("iterations", "ALL")
    features = trace_spec.get("features", [])
    if iterations == "ALL":
        iterations = sorted(execution_trace.keys())
    if features == []:
        return []

    execution_trace_info = []

    for iter_num in iterations:
        assert iter_num in execution_trace, (
            f"Missing execution trace iteration: {iter_num}"
        )
        assert query_name in execution_trace[iter_num], (
            f"Missing query {query_name} in execution "
            f"trace iteration: {iter_num}"
        )

        execution_trace_info.append(
            {
                "iter_num": iter_num,
                **{
                    feature: execution_trace[iter_num]
                    .get(query_name)
                    .get(feature)
                    for feature in features
                },
            }
        )

    return execution_trace_info


def parse_collect_spec(
    workload: str,
    query_group: list[str],
    collect_specs: dict[str, Any],
    engine_result: dict[str, Any],
    override_map: dict[str, Any],
) -> list[dict[str, Any]]:
    """Expand structured collect config into result rows."""
    if not collect_specs:
        return []

    collected_info = []

    # Collect static fields from the config and overrides.
    static_info = {
        "workload": workload,
        **{k: str(v) for k, v in override_map.items()},
    }

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

    return parse_collect_spec(
        workload=workload_name,
        query_group=query_group,
        collect_specs=collect_specs,
        engine_result=engine_result,
        override_map=override_map,
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
