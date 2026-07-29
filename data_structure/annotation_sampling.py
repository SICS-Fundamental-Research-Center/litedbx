"""Annotation sampling strategies."""

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

MIN_EXPECTED_STRATUM_SAMPLES = 2.0
SUPPORTED_STRATEGIES = {
    "random",
    "local_llm_adaptive",
}
SamplingMode = Literal[
    "uniform",
    "stratified_random",
]


def automatic_annotation_strategy(
    has_semantic_proxy: bool = False,
) -> str:
    """Choose the sampling design from capabilities available to the query."""
    if has_semantic_proxy:
        return "local_llm_adaptive"
    return "random"



@dataclass(frozen=True)
class AnnotationSelection:
    """Selected annotation rows plus their sampling design."""

    indices: pd.Index
    mode: SamplingMode


def select_annotation_sample(
    data: pd.DataFrame,
    pseudo_labels: pd.Series | None,
    labeling_budget: int,
    strategy: str,
    seed: int,
) -> AnnotationSelection:
    """Select rows and retain the design needed for unbiased estimates."""
    if strategy not in SUPPORTED_STRATEGIES:
        raise ValueError(
            f"Unsupported annotation strategy: {strategy}. "
            f"Expected one of {sorted(SUPPORTED_STRATEGIES)}."
        )
    if not 0 <= labeling_budget <= len(data):
        raise ValueError("Labeling budget must fit the candidate data.")
    if labeling_budget == 0:
        return AnnotationSelection(data.index[:0], "uniform")
    if strategy == "random":
        return AnnotationSelection(
            data.sample(n=labeling_budget, random_state=seed).index,
            "uniform",
        )
    if pseudo_labels is None:
        raise ValueError(
            "Local-LLM-adaptive annotation requires pseudo labels."
        )

    local_labels = coerce_bool_labels(
        pseudo_labels.reindex(data.index), "local pseudo labels"
    )
    positive_indices = local_labels[local_labels].index
    negative_indices = local_labels[~local_labels].index
    target_positive_rate = len(positive_indices) / len(local_labels)
    minority_rate = min(target_positive_rate, 1 - target_positive_rate)
    if labeling_budget * minority_rate >= MIN_EXPECTED_STRATUM_SAMPLES:
        return AnnotationSelection(
            data.sample(n=labeling_budget, random_state=seed).index,
            "uniform",
        )
    target_positive_rate = 0.5

    positive_count = _positive_sample_count(
        labeling_budget=labeling_budget,
        positive_size=len(positive_indices),
        negative_size=len(negative_indices),
        target_positive_rate=target_positive_rate,
    )
    negative_count = labeling_budget - positive_count
    selected = positive_indices[:0]
    if positive_count:
        stratum = _select_stratum(
            data=data.loc[positive_indices],
            count=positive_count,
            seed=seed,
        )
        selected = stratum
    if negative_count:
        stratum = _select_stratum(
            data=data.loc[negative_indices],
            count=negative_count,
            seed=seed + 1,
        )
        selected = selected.append(stratum)
    shuffled = data.loc[selected].sample(frac=1, random_state=seed + 2).index
    return AnnotationSelection(shuffled, "stratified_random")


def _select_stratum(
    data: pd.DataFrame,
    count: int,
    seed: int,
) -> pd.Index:
    """Randomly select rows from one pseudo-label stratum."""
    if count >= len(data):
        return data.index
    return data.sample(n=count, random_state=seed).index


def _positive_sample_count(
    labeling_budget: int,
    positive_size: int,
    negative_size: int,
    target_positive_rate: float,
) -> int:
    """Allocate a fixed budget while covering both nonempty strata."""
    if positive_size == 0:
        return 0
    if negative_size == 0:
        return labeling_budget

    proportional = int(np.floor(labeling_budget * target_positive_rate + 0.5))
    if labeling_budget >= 2:
        proportional = int(np.clip(proportional, 1, labeling_budget - 1))
    positive_count = min(proportional, positive_size)
    negative_count = min(labeling_budget - positive_count, negative_size)
    positive_count += labeling_budget - positive_count - negative_count
    return positive_count


def coerce_bool_labels(labels: pd.Series, name: str) -> pd.Series:
    """Convert cached or in-memory binary labels to booleans."""
    if labels.isna().any():
        raise ValueError(f"{name} contain missing values.")
    if pd.api.types.is_bool_dtype(labels):
        return labels.astype(bool)
    if pd.api.types.is_numeric_dtype(labels):
        if (~labels.isin([0, 1])).any():
            raise ValueError(f"{name} contain values other than 0 and 1.")
        return labels.astype(bool)

    mapped = (
        labels.astype(str)
        .str.strip()
        .str.lower()
        .map({"true": True, "false": False, "1": True, "0": False})
    )
    if mapped.isna().any():
        raise ValueError(f"{name} contain non-boolean values.")
    return mapped.astype(bool)
