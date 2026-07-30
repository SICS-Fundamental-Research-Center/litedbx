"""Logical feature grouping and budget-aware selection."""

# The selector keeps grouping and ranking state together for readability.
# pylint: disable=too-many-locals

from collections import defaultdict
from dataclasses import dataclass

from data_structure import PopulationSpec

from .semantic_features import (
    BudgetFeatureKey,
    SemanticFeatureKey,
    budget_feature_key,
    feature_key,
)


@dataclass(frozen=True)
class FeatureSelection:
    """Result of selecting logical features under one global budget."""

    features_by_query: dict[str, list[PopulationSpec]]
    ranked_feature_names: list[str]
    selected_feature_names: list[str]
    selected_keys: frozenset[BudgetFeatureKey]


def select_feature_groups(
    features_by_query: dict[str, list[PopulationSpec]],
    importance_by_query: dict[str, dict[str, float]],
    required_exact_keys: set[SemanticFeatureKey],
    budget: int,
) -> FeatureSelection:
    """Rank logical feature groups and retain at most ``budget`` groups."""
    if budget < 0:
        raise ValueError("External feature budget cannot be negative.")
    if features_by_query.keys() != importance_by_query.keys():
        raise ValueError("Feature spaces and importance tables must align.")

    specs_by_key: dict[BudgetFeatureKey, list[PopulationSpec]] = {}
    importance_by_key: defaultdict[BudgetFeatureKey, float] = defaultdict(float)
    for q_name, feature_space in features_by_query.items():
        for spec in feature_space:
            key = budget_feature_key(spec, required_exact_keys)
            specs_by_key.setdefault(key, []).append(spec)
            importance_by_key[key] += importance_by_query[q_name].get(
                spec.target_col, 0.0
            )

    required_keys = {
        budget_feature_key(spec, required_exact_keys)
        for feature_space in features_by_query.values()
        for spec in feature_space
        if feature_key(spec) in required_exact_keys
    }
    if len(required_keys) > budget:
        raise ValueError(
            "External feature budget is smaller than the number of "
            "distinct semantic predicates."
        )

    required_order = [key for key in specs_by_key if key in required_keys]
    optional_order = sorted(
        (key for key in specs_by_key if key not in required_keys),
        key=importance_by_key.__getitem__,
        reverse=True,
    )
    ranked_keys = required_order + optional_order
    selected_keys = frozenset(ranked_keys[:budget])

    ranked_feature_names = _feature_names(ranked_keys, specs_by_key)
    selected_feature_names = _feature_names(
        [key for key in ranked_keys if key in selected_keys], specs_by_key
    )
    selected_features = {
        q_name: [
            spec
            for spec in feature_space
            if budget_feature_key(spec, required_exact_keys) in selected_keys
        ]
        for q_name, feature_space in features_by_query.items()
    }
    return FeatureSelection(
        features_by_query=selected_features,
        ranked_feature_names=ranked_feature_names,
        selected_feature_names=selected_feature_names,
        selected_keys=selected_keys,
    )


def _feature_names(
    keys: list[BudgetFeatureKey],
    specs_by_key: dict[BudgetFeatureKey, list[PopulationSpec]],
) -> list[str]:
    """Flatten aliases while preserving logical ranking order."""
    return list(
        dict.fromkeys(
            spec.target_col for key in keys for spec in specs_by_key[key]
        )
    )
