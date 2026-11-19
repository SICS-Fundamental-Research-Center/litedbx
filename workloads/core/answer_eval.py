"""Answer-layer evaluation: one F1 core + three (df, gt_path) interfaces.

  eval_rowset         F1 core (set input, single-column engine GT file)
  eval_filter         F1 of an assembled answer table
  eval_rel_err        scalar relative error (vendored SemBench S2)
  eval_group_rel_err  group-count relative error (vendored SemBench S3)

Aggregation scoring lives in workloads/core/sembench_eval.py (provenance
header + >>>/<<< verbatim regions); the checker is
revision/exp/R1.D3.1/4-sembench-reuse/_check_verbatim.py. Legacy
calibers (eval_scalar / eval_set) moved to their only consumers (the
_finalize* drivers, the animals scenario) on 2026-11-19.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from workloads.core.sembench_eval import (
    SEMBENCH_GENERIC,
    SEMBENCH_MOVIE,
)

INF = float("inf")


def eval_rowset(
    retrieved: set[tuple] | list[tuple],
    gt_path: Path,
    selected_col: str,
) -> dict[str, Any]:
    """Row-set F1 of a retrieved set against a single-column engine GT
    file. Bit-exact replica of the engine's _quality_metrics math (the
    night driver's parity check against engine trans_eval). NaN rows
    stay elements instead of SemBench's dropna -- disclosed divergence,
    unreachable while SELECT columns are NOT-NULL id columns."""
    gt = {tuple(row) for row in pd.read_csv(gt_path)[[selected_col]].values}
    retrieved = {tuple(t) for t in retrieved}
    tp = len(gt.intersection(retrieved))
    fp = len(retrieved - gt)
    fn = len(gt - retrieved)
    if fp == 0 and fn == 0 and tp == 0:
        precision = recall = f1 = 1.0
    else:
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
    return {
        "kind": "rowset",
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "n_gt": len(gt),
        "n_retrieved": len(retrieved),
        "gt_empty": not gt,
    }


def eval_filter(
    answer: pd.DataFrame, gt_path: Path, selected_col: str
) -> dict[str, Any]:
    """F1 of an assembled answer table (engine GT file format)."""
    retrieved = {tuple(row) for row in answer[[selected_col]].values}
    return eval_rowset(retrieved, gt_path, selected_col)


def eval_rel_err(answer: pd.DataFrame, gt_path: Path) -> dict[str, Any]:
    """Scalar relative error; scoring by vendored SemBench (region S2).

    quality = 1/(1+rel_err) is the aggregate_table_generator transform
    (1/(1+inf) -> 0.0); inf maps to None to keep JSON finite.
    """
    m = SEMBENCH_GENERIC._generic_aggregation_evaluation(
        answer, pd.read_csv(gt_path)
    )
    return {
        "kind": "rel_err",
        "gt_path": str(gt_path),
        "rel_err": None if m.relative_error == INF else m.relative_error,
        "quality": 1.0 / (1.0 + m.relative_error),
        "mape": (
            None
            if m.mean_absolute_percentage_error == INF
            else m.mean_absolute_percentage_error
        ),
        "abs_err": None if m.absolute_error == INF else m.absolute_error,
    }


def eval_group_rel_err(
    answer: pd.DataFrame, gt_path: Path, key_col: str, count_col: str
) -> dict[str, Any]:
    """Group-count relative error; scoring by vendored SemBench (region
    S3). Named columns adapt to SemBench's positional (key, count)
    convention; a missing column raises KeyError (config error)."""
    gt_df = pd.read_csv(gt_path)
    m = SEMBENCH_MOVIE._evaluate_sentiment_counts(
        answer[[key_col, count_col]], gt_df[[key_col, count_col]]
    )
    return {
        "kind": "group_rel_err",
        "gt_path": str(gt_path),
        "rel_err": None if m.relative_error == INF else m.relative_error,
        "quality": 1.0 / (1.0 + m.relative_error),
        "mape": (
            None
            if m.mean_absolute_percentage_error == INF
            else m.mean_absolute_percentage_error
        ),
        "abs_error": None if m.absolute_error == INF else m.absolute_error,
    }
