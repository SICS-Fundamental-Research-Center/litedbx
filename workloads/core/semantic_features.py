"""Semantic-feature identity and materialization helpers."""

import hashlib
import json
import logging
from typing import Literal, cast

from data_structure import PopulationSpec, SemCQ

logger = logging.getLogger(__name__)

SemanticFeatureKey = tuple[str, str, str]
BudgetFeatureKey = tuple[str, ...]


def normalize_prompt(prompt: str) -> str:
    """Normalize prompt text for stable semantic identity matching."""
    return " ".join(prompt.split()).casefold()


def feature_key(spec: PopulationSpec) -> SemanticFeatureKey:
    """Return the logical identity of a materialized semantic feature."""
    return (
        spec.source_col,
        normalize_prompt(spec.prompt),
        spec.feature_type,
    )


def budget_feature_key(
    spec: PopulationSpec,
    required_keys: set[SemanticFeatureKey],
) -> BudgetFeatureKey:
    """Return the logical identity used by the external-feature budget."""
    exact_key = feature_key(spec)
    if exact_key in required_keys:
        return ("semantic", *exact_key)
    return (
        "derived",
        spec.source_col,
        spec.target_col,
        spec.feature_type,
    )


def predicate_key(predicate) -> SemanticFeatureKey:
    """Return the logical feature identity required by a predicate."""
    return (
        predicate.field,
        normalize_prompt(predicate.prompt),
        "bool",
    )


def required_feature_keys(queries: dict[str, SemCQ]) -> set[SemanticFeatureKey]:
    """Collect direct semantic-predicate feature identities."""
    return {
        predicate_key(predicate)
        for query in queries.values()
        for predicate in query.Ps
    }


def ensure_semantic_features(
    q_name: str,
    query: SemCQ,
    feature_space: list[PopulationSpec],
) -> list[PopulationSpec]:
    """Ensure every semantic predicate has one materialized feature."""
    existing = {feature_key(spec) for spec in feature_space}
    for predicate in query.Ps:
        key = predicate_key(predicate)
        if key in existing:
            continue
        digest = hashlib.sha1(
            json.dumps(
                {
                    "field": predicate.field,
                    "prompt": predicate.prompt.strip(),
                },
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()[:12]
        spec = PopulationSpec(
            source_col=predicate.field,
            source_modality=cast(
                Literal["Text", "Image", "VectorText", "VectorImage"],
                predicate.modality,
            ),
            target_col=f"llm_label_semantic_{digest}",
            prompt=predicate.prompt,
            feature_type="bool",
        )
        feature_space.append(spec)
        existing.add(key)
        logger.info(
            "Added semantic-predicate feature %s for query %s.",
            spec.target_col,
            q_name,
        )
    return feature_space
