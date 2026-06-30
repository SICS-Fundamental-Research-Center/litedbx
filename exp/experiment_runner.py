from __future__ import annotations

import argparse
import asyncio
import csv
import itertools
import logging
import os
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from ldb_engine import LdbEngine
from workloads import medical, movie, ecomm, mmqa, animals

logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parent.parent
EXP_DIR = ROOT_DIR / "exp"
RESULTS_DIR = EXP_DIR / "results"
SERVEM = ROOT_DIR / "servem.sh"
MODEL_STATE_DIR = Path(os.environ.get("HOST_STATE_DIR", "/tmp/litedbx-model-hosting"))
DEFAULT_MODEL_PORTS = {
    "llama3-8b": 8000,
    "qwen3-4b": 8001,
    "qwen3-vl-8b": 8002,
    "llava-v1.6-7b": 8003,
    "qwen3-30b-fp8": 8004,
    "qwen3-vl-30b": 8005,
    "qwen3-vl-2b": 8006,
    "qwen3-vl-4b": 8007,
}

WORKLOAD_MAPPING = {
    "medical": medical.get_workload,
    "movie": movie.get_workload,
    "ecomm": ecomm.get_workload,
    "mmqa": mmqa.get_workload,
    "animals": animals.get_workload,
}


@dataclass
class TaskResult:
    config_name: str
    task_name: str
    status: str
    started_at: str
    ended_at: str
    task_config: dict[str, Any]
    rows: list[dict[str, Any]] | None = None
    error: str | None = None


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def save_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def start_model_session(model_key: str) -> None:
    subprocess.run(["bash", str(SERVEM), "start", model_key], check=True, cwd=ROOT_DIR)


def stop_model_session(model_key: str) -> None:
    subprocess.run(["bash", str(SERVEM), "stop", model_key], check=True, cwd=ROOT_DIR)


def session_name(model_key: str) -> str:
    if model_key.startswith("llava") or "-vl-" in model_key:
        return f"vllmv-{model_key}"
    return f"vllm-{model_key}"


def state_file(model_key: str) -> Path:
    return MODEL_STATE_DIR / f"{model_key}.env"


def read_state_env(model_key: str) -> dict[str, str]:
    path = state_file(model_key)
    if not path.exists():
        return {}
    state: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        state[key.strip()] = value.strip()
    return state


def model_port(model_key: str) -> int | None:
    state = read_state_env(model_key)
    if "MODEL_PORT" in state:
        try:
            return int(state["MODEL_PORT"])
        except ValueError:
            return None
    return DEFAULT_MODEL_PORTS.get(model_key)


def model_ready(model_key: str) -> bool:
    port = model_port(model_key)
    if port is None:
        return False
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=2) as resp:
            return resp.status == 200
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


def check_model_running(model_key: str) -> bool:
    proc = subprocess.run(
        ["tmux", "has-session", "-t", session_name(model_key)],
        cwd=ROOT_DIR,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return proc.returncode == 0


async def wait_for_model_ready(model_key: str, timeout_s: int = 600) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if model_ready(model_key):
            logger.info("Model ready: %s", model_key)
            return
        await asyncio.sleep(2)
    raise TimeoutError(f"Timed out waiting for model readiness: {model_key}")


async def ensure_models(models: list[str]) -> None:
    for model_key in models:
        if check_model_running(model_key):
            logger.info("Model already running: %s", model_key)
        else:
            logger.info("Starting model: %s", model_key)
            start_model_session(model_key)
        await wait_for_model_ready(model_key)


def release_models(models: list[str], release_after: bool) -> None:
    if not release_after:
        return
    for model_key in models:
        logger.info("Releasing model: %s", model_key)
        stop_model_session(model_key)


def build_workload(workload_name: str, queries: list[str], config_override: dict[str, Any]):
    workload_func = WORKLOAD_MAPPING.get(workload_name)
    if workload_func is None:
        raise ValueError(f"Invalid workload: {workload_name}")

    workload = workload_func(queries=queries)
    if config_override:
        workload.inject_exp_setting(
            exp_group=config_override.get("exp_group", "experiment"),
            exp_patch={k: v for k, v in config_override.items() if k != "exp_group"},
        )
    return workload



def expand_value(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    return [value]


def iter_selected_iterations(execution_trace: dict[str, Any], selector: str) -> list[tuple[int, Any]]:
    items = sorted(execution_trace.items(), key=lambda kv: int(kv[0]))
    if selector == "*":
        return [(int(k), v) for k, v in items]
    selected = []
    for token in selector.split(","):
        token = token.strip()
        if not token:
            continue
        idx = int(token)
        if str(idx) in execution_trace:
            selected.append((idx, execution_trace[str(idx)]))
    return selected


def collect_path_values(
    source: dict[str, Any],
    parts: list[str],
) -> list[tuple[dict[str, Any], Any]]:
    states: list[tuple[dict[str, Any], Any]] = [({}, source)]
    for part in parts:
        next_states: list[tuple[dict[str, Any], Any]] = []
        for metadata, value in states:
            if part == "*":
                if not isinstance(value, dict):
                    continue
                for child_key, child_value in sorted(value.items(), key=lambda kv: int(kv[0])):
                    child_metadata = dict(metadata)
                    if "iter_num" not in child_metadata:
                        child_metadata["iter_num"] = int(child_key)
                    else:
                        child_metadata[part] = child_key
                    next_states.append((child_metadata, child_value))
                continue

            if isinstance(value, dict) and part in value:
                next_states.append((metadata, value[part]))
        states = next_states
    return states


def parse_collect_spec(collect: list[str], engine_result: dict[str, Any], base_row: dict[str, Any]) -> list[dict[str, Any]]:
    rows = [base_row]
    for spec in collect:
        parts = spec.split(".")
        if not parts or any(part == "" for part in parts):
            raise ValueError(f"Unsupported collect spec: {spec}")

        output_name = parts[-1]
        next_rows: list[dict[str, Any]] = []
        for row in rows:
            for metadata, value in collect_path_values(engine_result, parts):
                if isinstance(value, dict) and row.get("query") in value:
                    value = value[row["query"]]
                new_row = dict(row)
                new_row.update(metadata)
                new_row[output_name] = value
                next_rows.append(new_row)
        rows = next_rows
    return rows


async def run_task(config_name: str, task: dict[str, Any]) -> TaskResult:
    started_at = datetime.now(timezone.utc).isoformat()
    rows: list[dict[str, Any]] = []
    try:
        models = task.get("models", [])
        await ensure_models(models)

        workload_entries = task.get("objects", [])
        override_specs = task.get("config_override", {})
        override_keys = list(override_specs.keys())
        override_values = [expand_value(override_specs[k]) for k in override_keys]
        override_combinations = list(itertools.product(*override_values)) if override_values else [()]

        for obj in workload_entries:
            workload_name = obj["workload"]
            query_groups = obj["queries"]
            for query_group in query_groups:
                for override_combo in override_combinations:
                    override_map = dict(zip(override_keys, override_combo))
                    override_map["exp_group"] = task.get("exp_group", "experiment")

                    workload = build_workload(workload_name, query_group, override_map)
                    engine = LdbEngine(workload)
                    engine_result = await engine.execute(debug=bool(task.get("debug", False)))

                    base_row = {
                        "config": config_name,
                        "task": task.get("name", config_name),
                        "workload": workload_name,
                        "query": ",".join(query_group),
                        "model": ",".join(models),
                        **override_map,
                    }
                    rows.extend(parse_collect_spec(task.get("collect", []), engine_result, base_row))

        status = "ok"
        error = None
    except Exception as exc:  # pragma: no cover
        status = "error"
        error = repr(exc)
        logger.exception("Task failed: %s", task.get("name", "<unnamed>"))
    finally:
        release_models(task.get("models", []), task.get("release_model_after_task", False))

    ended_at = datetime.now(timezone.utc).isoformat()
    return TaskResult(
        config_name=config_name,
        task_name=task.get("name", config_name),
        status=status,
        started_at=started_at,
        ended_at=ended_at,
        task_config=task,
        rows=rows if rows else None,
        error=error,
    )


async def run_config(config_path: Path) -> list[TaskResult]:
    config = load_yaml(config_path)
    tasks = config.get("tasks", [])
    results: list[TaskResult] = []
    for task in tasks:
        results.append(await run_task(config_path.stem, task))
    return results


def list_configs() -> list[Path]:
    return sorted(p for p in EXP_DIR.glob("*.yaml") if p.name != "config.yaml")


def export_results(config_path: Path, results: list[TaskResult]) -> Path:
    rows: list[dict[str, Any]] = []
    for result in results:
        if result.rows:
            rows.extend(result.rows)

    out_path = RESULTS_DIR / f"{config_path.stem}.csv"
    fieldnames = ["workload", "query", "model", "b_se", "iter_num", "memory_cost"]
    rows = [
        {
            "workload": row.get("workload"),
            "query": row.get("query"),
            "model": row.get("model"),
            "b_se": row.get("b_se"),
            "iter_num": row.get("iter_num"),
            "memory_cost": row.get("memory_cost"),
        }
        for row in rows
        if "workload" in row and "query" in row and "model" in row
    ]
    save_csv(out_path, rows, fieldnames)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="LiteDBX experiment platform")
    parser.add_argument("--config", action="append", help="Experiment config file under exp/")
    parser.add_argument("--list", action="store_true", help="List available experiment configs")
    args = parser.parse_args()

    if args.list:
        for path in list_configs():
            print(path.name)
        return

    config_names = args.config or [p.name for p in list_configs()]
    if not config_names:
        raise SystemExit("No experiment configs found under exp/")

    for config_name in config_names:
        config_path = Path(config_name)
        if not config_path.exists():
            config_path = EXP_DIR / config_name
        if not config_path.exists():
            raise SystemExit(f"Config not found: {config_path}")
        results = asyncio.run(run_config(config_path))
        out_path = export_results(config_path, results)
        print(out_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
