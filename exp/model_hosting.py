"""Local model-hosting helpers for experiment runs."""

import asyncio
import logging
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
SERVEM = ROOT_DIR / "servem.sh"
logger = logging.getLogger(__name__)


def start_model_session(model_key: str) -> None:
    """Start the configured model serving session."""
    subprocess.run(
        ["bash", str(SERVEM), "start", model_key], check=True, cwd=ROOT_DIR
    )


def stop_model_session(model_key: str) -> None:
    """Stop the configured model serving session."""
    subprocess.run(
        ["bash", str(SERVEM), "stop", model_key], check=True, cwd=ROOT_DIR
    )


def model_port(model_key: str) -> int | None:
    """Return the serving port reported by servem.sh."""
    proc = subprocess.run(
        ["bash", str(SERVEM), "state", model_key, "MODEL_PORT"],
        check=False,
        cwd=ROOT_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    if proc.returncode != 0:
        return None
    try:
        return int(proc.stdout.strip())
    except ValueError:
        return None


def model_ready(model_key: str) -> bool:
    """Return whether the model server responds to the models endpoint."""
    port = model_port(model_key)
    if port is None:
        return False
    try:
        with urllib.request.urlopen(
            f"http://127.0.0.1:{port}/v1/models", timeout=2
        ) as resp:
            return resp.status == 200
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


def check_model_running(model_key: str) -> bool:
    """Return whether servem.sh reports a model as running."""
    proc = subprocess.run(
        ["bash", str(SERVEM), "status", model_key],
        check=False,
        cwd=ROOT_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    return proc.returncode == 0 and " is running " in proc.stdout


async def wait_for_model_ready(model_key: str, timeout_s: int = 600) -> None:
    """Wait until a model server is ready or the timeout expires."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if model_ready(model_key):
            logger.info("Model ready: %s", model_key)
            return
        await asyncio.sleep(2)
    raise TimeoutError(f"Timed out waiting for model readiness: {model_key}")


async def ensure_models(models: list[str]) -> None:
    """Ensure each requested model is running and ready."""
    for model_key in models:
        if check_model_running(model_key):
            logger.info("Model already running: %s", model_key)
        else:
            logger.info("Starting model: %s", model_key)
            start_model_session(model_key)
        await wait_for_model_ready(model_key)


def release_models(models: list[str], release_after: bool) -> None:
    """Stop models after a task when configured to do so."""
    if not release_after:
        return
    for model_key in models:
        logger.info("Releasing model: %s", model_key)
        stop_model_session(model_key)
