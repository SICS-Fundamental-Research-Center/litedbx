"""Engine-initiated local model swap (user order 2026-11-18, simplified
2026-11-19): before each LOCAL call batch, check the endpoint serves the
configured model; if not, block and run ``servem.sh swap`` (which itself
waits for the model to become ready). Swap wall-clock intervals are
recorded so run timing can exclude them. LDB_AUTO_SWAP=0 disables.
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import time
import urllib.request
from pathlib import Path

logger = logging.getLogger("llm.auto_swap")

REPO_ROOT = Path(__file__).resolve().parents[1]

# Annotation scheduling order (user order 2026-11-18): a query's
# materialization work is grouped by modality so each modality's calls
# run back-to-back -- one serving layout per group, at most one swap
# between groups. Text last: prefilter/rewrite stages are Text, so
# trailing Text means no swap-back after the media groups finish.
MODALITY_RANK = {
    "Image": 0, "VectorImage": 0,
    "Audio": 1, "VectorAudio": 1,
    "Text": 2, "VectorText": 2,
}

_SERVEM_LINE = re.compile(r'^\s*MODEL_PATHS\[([^\]]+)\]="([^"]+)"')


class SwapFailed(RuntimeError):
    """Raised when ``servem.sh swap`` exits non-zero."""


def parse_servem_keys(servem_path: Path) -> dict[str, str]:
    """Reverse map {model_path: servem_key} from servem.sh MODEL_PATHS.

    Paths are stored WITHOUT the leading slash: config entries look
    like ``hosted_vllm//ssd_data/models/...`` whose split-on-``//``
    yields a slash-less path, and that side drives the lookup.
    """
    mapping: dict[str, str] = {}
    for line in Path(servem_path).read_text(encoding="utf-8").splitlines():
        m = _SERVEM_LINE.match(line)
        if m:
            mapping[m.group(2).lstrip("/")] = m.group(1)
    return mapping


class AutoSwapManager:
    def __init__(self, repo_root: Path | None = None):
        self.repo_root = Path(repo_root) if repo_root else REPO_ROOT
        self.servem = self.repo_root / "servem.sh"
        self.enabled = os.environ.get("LDB_AUTO_SWAP", "1") != "0"
        # swap wall-clock intervals (epoch seconds), newest last
        self.intervals: list[dict] = []
        # cumulative time spent probing /models (ensure overhead when no
        # swap fires; swap time itself lives in `intervals`)
        self.probe_seconds = 0.0
        self._path_to_key = parse_servem_keys(self.servem)

    @staticmethod
    def probe(api_base: str, timeout: float = 3.0) -> list[str] | None:
        """Served model ids at an OpenAI-compatible /models endpoint.

        None = endpoint dead/unreachable; [] = up but empty.
        """
        try:
            with urllib.request.urlopen(
                f"{api_base.rstrip('/')}/models", timeout=timeout
            ) as r:
                data = json.load(r)
            return [m.get("id", "") for m in data.get("data", [])]
        except Exception:  # noqa: BLE001 - any probe failure means "dead"
            return None

    @staticmethod
    def _serves(served: list[str] | None, expect: str) -> bool:
        return served is not None and any(expect in s for s in served)

    def ensure(self, model: str, api_base: str) -> bool:
        """If api_base does not serve ``model``, block and swap to it.

        Returns True when the endpoint serves the model (or swapped to
        it). Unknown-to-servem models return False and the call proceeds
        (and fails) naturally. Remote entries are skipped (not
        localhost).
        """
        if not self.enabled:
            return True
        if not str(api_base).startswith("http://localhost"):
            return True
        expect = model.split("//", 1)[-1].lstrip("/")
        t0 = time.time()
        served = self.probe(api_base)
        self.probe_seconds += time.time() - t0
        if self._serves(served, expect):
            return True
        key = self._path_to_key.get(expect)
        if key is None:
            logger.warning(
                "auto-swap: %s not served @%s and no servem.sh key maps "
                "to it; proceeding without swap", expect, api_base)
            return False
        logger.info("auto-swap -> %s (SERVEM_EAGER=1)", key)
        self._swap(key)
        return True

    def _swap(self, key: str) -> dict:
        """Run ``servem.sh swap <key>``; record the wall-clock interval.

        Blocks until servem returns (it waits for the model to answer
        HTTP on its port). The interval is recorded even on failure --
        the time was spent either way; the caller raises on rc != 0.
        """
        env = {**os.environ, "SERVEM_EAGER": "1"}
        t0 = time.time()
        proc = subprocess.run(
            ["bash", str(self.servem), "swap", key],
            cwd=self.repo_root, env=env, capture_output=True, text=True,
            timeout=1800,
        )
        t1 = time.time()
        interval = {"key": key, "t_start": t0, "t_end": t1,
                    "seconds": round(t1 - t0, 3)}
        self.intervals.append(interval)
        if proc.returncode != 0:
            tail = (proc.stdout + proc.stderr)[-2000:]
            raise SwapFailed(
                f"servem.sh swap {key} rc={proc.returncode}: {tail}")
        logger.info("auto-swap %s done in %.1fs", key, interval["seconds"])
        return interval

    def overlap_seconds(self, t0: float, t1: float) -> float:
        """Seconds of [t0, t1] covered by any swap interval."""
        total = 0.0
        for iv in self.intervals:
            lo = max(t0, iv["t_start"])
            hi = min(t1, iv["t_end"])
            if hi > lo:
                total += hi - lo
        return total

    def total_seconds(self) -> float:
        return sum(iv["seconds"] for iv in self.intervals)


def _smoke() -> None:
    """No-GPU smoke checks (run: python llm/auto_swap.py)."""
    # 1. servem key parse + production lookup path: config
    # "hosted_vllm//ssd_data/..." splits to a slash-less path; must hit
    # the servem key (regression: a leading-slash mismatch silently
    # disabled swapping in the first live run)
    keys = parse_servem_keys(REPO_ROOT / "servem.sh")
    assert "qwen3-omni-30b-awq" in set(keys.values()), keys
    cfg_style = "hosted_vllm//ssd_data/models/Qwen3-30B-A3B-Instruct-2507-FP8"
    assert keys[cfg_style.split("//", 1)[-1].lstrip("/")] == "qwen3-30b-fp8"
    print(f"[1] servem reverse-map OK: {len(keys)} models")

    # 2. dead-port probe
    assert AutoSwapManager.probe("http://localhost:59999/v1") is None
    print("[2] dead-port probe -> None OK")

    # 3. kill switch: no probe, no swap
    mgr = AutoSwapManager()
    mgr.enabled = False
    assert mgr.ensure("hosted_vllm//no/such/model",
                      "http://localhost:59999/v1") is True
    print("[3] LDB_AUTO_SWAP=0 short-circuit OK")

    # 4. unknown model on a dead endpoint: False, no swap attempted
    mgr = AutoSwapManager()
    assert mgr.ensure("hosted_vllm//no/such/model",
                      "http://localhost:59999/v1") is False
    assert mgr.intervals == []
    print("[4] unknown-model no-swap OK")

    # 5. ensure performs the (mocked) blocking swap when served model
    # differs -- never touches real GPUs
    mgr = AutoSwapManager()
    swapped = []

    def fake_swap(key):
        t0 = time.time()
        iv = {"key": key, "t_start": t0, "t_end": t0 + 60.0,
              "seconds": 60.0}
        mgr.intervals.append(iv)
        swapped.append(key)
        return iv

    mgr._swap = fake_swap
    some_path = next(iter(parse_servem_keys(REPO_ROOT / "servem.sh")))
    assert mgr.ensure("hosted_vllm//" + some_path,
                      "http://localhost:59999/v1") is True
    assert swapped == [mgr._path_to_key[some_path]]
    print("[5] ensure -> blocking swap OK (mocked)")

    # 6. overlap math
    mgr = AutoSwapManager()
    mgr.intervals = [
        {"t_start": 10.0, "t_end": 20.0, "seconds": 10.0, "key": "a"},
        {"t_start": 40.0, "t_end": 50.0, "seconds": 10.0, "key": "b"},
    ]
    assert mgr.overlap_seconds(5, 15) == 5.0
    assert mgr.overlap_seconds(12, 18) == 6.0
    assert mgr.overlap_seconds(10, 45) == 15.0
    assert mgr.overlap_seconds(25, 30) == 0.0
    assert mgr.total_seconds() == 20.0
    print("[6] overlap math OK")

    # 7. modality rank covers Vector* forms, stable-sort order
    specs = ["Text", "VectorAudio", "Image", "VectorText", "Audio",
             "VectorImage"]
    ranked = sorted(specs, key=lambda m: MODALITY_RANK.get(m, 99))
    assert ranked == ["Image", "VectorImage", "VectorAudio", "Audio",
                      "Text", "VectorText"], ranked
    print("[7] modality grouping rank OK")

    print("auto_swap smoke: all checks passed")


if __name__ == "__main__":
    _smoke()
