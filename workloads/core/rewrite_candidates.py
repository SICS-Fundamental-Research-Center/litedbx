"""Typed configuration and selection for query rewrite candidates."""

from dataclasses import dataclass
from typing import Literal

CandidateKind = Literal["expanded_forest"]


@dataclass(frozen=True)
class ForestConfig:
    """Random-forest parameters shared by fitting and LOO evaluation."""

    n_estimators: int
    max_depth: int
    min_samples_leaf: int = 1


@dataclass(frozen=True)
class RewriteCandidate:
    """One independently evaluated rewrite strategy."""

    kind: CandidateKind
    feature_count: int | None = None


EXPANDED_FOREST = ForestConfig(
    n_estimators=50,
    max_depth=3,
    min_samples_leaf=2,
)


def build_forest_configs(
    maximum: ForestConfig = EXPANDED_FOREST,
) -> list[ForestConfig]:
    """Build universal depth candidates up to the configured forest cap."""
    return [
        ForestConfig(
            n_estimators=maximum.n_estimators,
            max_depth=depth,
            min_samples_leaf=maximum.min_samples_leaf,
        )
        for depth in range(1, maximum.max_depth + 1)
    ]


def select_forest_config(
    configs: list[ForestConfig],
    estimated_losses: list[float],
    loss_resolution: float,
) -> ForestConfig:
    """Choose the shallowest forest statistically tied on annotation loss."""
    if len(configs) != len(estimated_losses):
        raise ValueError("Forest configurations and losses must align.")
    if not configs:
        raise ValueError("At least one forest configuration is required.")
    if loss_resolution < 0:
        raise ValueError("Loss resolution cannot be negative.")
    minimum_loss = min(estimated_losses)
    eligible = [
        index
        for index, loss in enumerate(estimated_losses)
        if loss <= minimum_loss + loss_resolution
    ]
    return min(
        (configs[index] for index in eligible),
        key=lambda config: (
            config.max_depth,
            config.min_samples_leaf,
            config.n_estimators,
        ),
    )


def build_rewrite_candidates(
    feature_count: int,
    minimum_feature_count: int = 0,
) -> list[RewriteCandidate]:
    """Build expanded-forest candidates over all feature-prefix sizes."""
    if not 0 <= minimum_feature_count <= feature_count:
        raise ValueError("Minimum feature count must fit the feature space.")
    return [
        RewriteCandidate("expanded_forest", count)
        for count in range(minimum_feature_count, feature_count + 1)
    ]


def select_candidate_index(
    candidates: list[RewriteCandidate],
    estimated_losses: list[float],
    loss_resolution: float = 0.0,
    candidate_complexities: list[tuple[float, ...]] | None = None,
) -> int:
    """Select a candidate using only query structure and annotation evidence."""
    if len(candidates) != len(estimated_losses):
        raise ValueError("Candidates and estimated losses must align.")
    if not candidates:
        raise ValueError("At least one rewrite candidate is required.")
    if loss_resolution < 0:
        raise ValueError("Loss resolution cannot be negative.")
    if candidate_complexities is None:
        candidate_complexities = [
            (0, index) for index in range(len(candidates))
        ]
    if len(candidate_complexities) != len(candidates):
        raise ValueError("Candidates and complexities must align.")
    eligible = list(range(len(candidates)))
    minimum_preference = min(
        candidate_complexities[index][0] for index in eligible
    )
    indistinguishable = [
        index
        for index in eligible
        if candidate_complexities[index][0]
        <= minimum_preference + loss_resolution
    ]
    return min(
        indistinguishable,
        key=lambda index: (
            candidate_complexities[index],
            estimated_losses[index],
            index,
        ),
    )
