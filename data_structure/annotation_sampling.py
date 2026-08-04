# pylint: disable=too-many-arguments,too-many-positional-arguments
# pylint: disable=too-many-locals
"""Annotation sampling strategies."""

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

MIN_EXPECTED_STRATUM_SAMPLES = 2.0
SUPPORTED_STRATEGIES = {
    "random",
    "local_llm_adaptive",
    "local_llm_hybrid_diverse",
}
SamplingMode = Literal[
    "uniform",
    "stratified_random",
    "stratified_hybrid",
]


def automatic_annotation_strategy(
    diversity_columns: list[str], has_semantic_proxy: bool = False
) -> str:
    """Choose the sampling design from capabilities available to the query."""
    if diversity_columns:
        return "local_llm_hybrid_diverse"
    if has_semantic_proxy:
        return "local_llm_adaptive"
    return "random"


@dataclass(frozen=True)
class AnnotationStratumSample:
    """One proxy stratum with certainty and random sample components."""

    population_size: int
    anchor_indices: pd.Index
    random_indices: pd.Index


@dataclass(frozen=True)
class AnnotationSelection:
    """Selected annotation rows plus their sampling design."""

    indices: pd.Index
    mode: SamplingMode
    strata: tuple[AnnotationStratumSample, ...] = ()


def select_annotation_sample(
    data: pd.DataFrame,
    pseudo_labels: pd.Series | None,
    labeling_budget: int,
    strategy: str,
    seed: int,
    diversity_columns: list[str] | None = None,
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
        raise ValueError("Local-LLM annotation requires pseudo labels.")

    local_labels = coerce_bool_labels(
        pseudo_labels.reindex(data.index), "local pseudo labels"
    )
    positive_indices = local_labels[local_labels].index
    negative_indices = local_labels[~local_labels].index
    target_positive_rate = len(positive_indices) / len(local_labels)
    use_text_diversity = strategy == "local_llm_hybrid_diverse"
    if strategy == "local_llm_adaptive":
        minority_rate = min(target_positive_rate, 1 - target_positive_rate)
        if labeling_budget * minority_rate >= MIN_EXPECTED_STRATUM_SAMPLES:
            return AnnotationSelection(
                data.sample(n=labeling_budget, random_state=seed).index,
                "uniform",
            )

    positive_count = _positive_sample_count(
        labeling_budget=labeling_budget,
        positive_size=len(positive_indices),
        negative_size=len(negative_indices),
        target_positive_rate=0.5,
    )
    negative_count = labeling_budget - positive_count
    selected = data.index[:0]
    strata: list[AnnotationStratumSample] = []
    if positive_count:
        stratum = _select_stratum(
            data=data.loc[positive_indices],
            count=positive_count,
            seed=seed,
            diversity_columns=diversity_columns,
            use_text_diversity=use_text_diversity,
        )
        strata.append(stratum)
        selected = stratum.anchor_indices.append(stratum.random_indices)
    if negative_count:
        stratum = _select_stratum(
            data=data.loc[negative_indices],
            count=negative_count,
            seed=seed + 1,
            diversity_columns=diversity_columns,
            use_text_diversity=use_text_diversity,
        )
        strata.append(stratum)
        selected = selected.append(
            stratum.anchor_indices.append(stratum.random_indices)
        )
    shuffled = data.loc[selected].sample(frac=1, random_state=seed + 2).index
    mode: SamplingMode = (
        "stratified_hybrid" if use_text_diversity else "stratified_random"
    )
    return AnnotationSelection(shuffled, mode, tuple(strata))


def _select_stratum(
    data: pd.DataFrame,
    count: int,
    seed: int,
    diversity_columns: list[str] | None,
    use_text_diversity: bool,
) -> AnnotationStratumSample:
    """Select one proxy stratum and retain its known sampling components."""
    if count >= len(data):
        return AnnotationStratumSample(len(data), data.index, data.index[:0])
    if not use_text_diversity:
        indices = data.sample(n=count, random_state=seed).index
        return AnnotationStratumSample(len(data), data.index[:0], indices)

    # Keep one deterministic coverage point while reserving the rest of the
    # stratum budget for probability samples used by design-based estimates.
    anchor_count = min(1, count - 1)
    anchors = _select_diverse_rows(
        data=data,
        count=anchor_count,
        seed=seed,
        diversity_columns=diversity_columns or [],
    )
    random_indices = (
        data.drop(index=anchors)
        .sample(n=count - anchor_count, random_state=seed)
        .index
    )
    return AnnotationStratumSample(len(data), anchors, random_indices)


def annotation_sample_weights(
    population_size: int, selection: AnnotationSelection
) -> pd.Series:
    """Return normalized inverse-inclusion weights for released annotations."""
    if len(selection.indices) == 0:
        return pd.Series(index=selection.indices, dtype=float)
    if population_size <= 0:
        raise ValueError("Population size must be positive.")

    weights = pd.Series(index=selection.indices, dtype=float)
    if selection.mode == "uniform":
        weights.loc[:] = population_size / len(selection.indices)
    else:
        for stratum in selection.strata:
            weights.loc[stratum.anchor_indices] = 1.0
            remaining_size = stratum.population_size - len(
                stratum.anchor_indices
            )
            if len(stratum.random_indices):
                weights.loc[stratum.random_indices] = remaining_size / len(
                    stratum.random_indices
                )
            elif remaining_size:
                raise ValueError(
                    "A partially sampled stratum needs probability samples."
                )

    if weights.isna().any() or (weights <= 0).any():
        raise ValueError("Annotation sampling design produced invalid weights.")
    return weights / float(weights.mean())


def _select_diverse_rows(
    data: pd.DataFrame,
    count: int,
    seed: int,
    diversity_columns: list[str],
) -> pd.Index:
    """Select representative, nonredundant text rows from one stratum."""
    if count == 0:
        return data.index[:0]
    missing = [column for column in diversity_columns if column not in data]
    if missing:
        raise ValueError(f"Missing annotation diversity columns: {missing}")
    text = (
        data[diversity_columns]
        .fillna("")
        .astype(str)
        .agg(" ".join, axis=1)
        .str.strip()
    )
    if not text.str.len().any():
        return data.sample(n=count, random_state=seed).index

    try:
        vectors = csr_matrix(
            TfidfVectorizer(
                lowercase=True,
                stop_words="english",
                ngram_range=(1, 2),
                max_features=8192,
            ).fit_transform(text)
        )
    except ValueError:
        return data.sample(n=count, random_state=seed).index

    centroid = np.asarray(vectors.mean(axis=0))
    centroid_similarity = cosine_similarity(vectors, centroid).ravel()
    first = int(np.argmax(centroid_similarity))
    selected_positions = [first]
    max_similarity = cosine_similarity(vectors, vectors[first]).ravel()
    max_similarity[first] = np.inf
    while len(selected_positions) < count:
        next_position = int(np.argmin(max_similarity))
        selected_positions.append(next_position)
        similarity = cosine_similarity(vectors, vectors[next_position]).ravel()
        max_similarity = np.maximum(max_similarity, similarity)
        max_similarity[selected_positions] = np.inf
    return data.index.take(selected_positions)


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
