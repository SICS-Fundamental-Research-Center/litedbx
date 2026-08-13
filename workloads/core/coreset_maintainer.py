# pylint: disable=missing-function-docstring,invalid-name
# pylint: disable=too-many-arguments,too-many-positional-arguments
# pylint: disable=too-many-locals,logging-fstring-interpolation,fixme
"""Coreset selection and maintenance for LiteDBX workloads."""

import logging
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from data_structure import LdbDataManager
from workloads.utils import (
    class_balanced_sample_weights,
    encode_features,
    norm_features,
    train_classifier,
    weight_features,
)

logger = logging.getLogger(__name__)


class CoresetMaintainer:
    """Maintain workload coresets and confidence thresholds."""

    def __init__(
        self,
        data_manager: LdbDataManager,
        config: dict,
        enable_conf_pred: bool,
        enable_conf_struct: bool,
    ) -> None:
        self.data_manager = data_manager
        self.config = config
        self.enable_conf_pred = enable_conf_pred
        self.enable_conf_struct = enable_conf_struct

    def expand_coresets(self, inc_round: int = 0) -> None:
        for q_name in self.data_manager.coresets:
            self.expand_query_coreset(q_name=q_name, inc_round=inc_round)

    def expand_query_coreset(self, q_name: str, inc_round: int = 0) -> None:
        coreset = self.data_manager.coresets[q_name]
        sigma_record = self.data_manager.sigma_satisfied_data[inc_round][q_name]
        labeled_x = coreset["ldb_data"].exclude_fk_and_id()
        labeled_y = coreset["labels"]
        unlabeled_x = sigma_record["ldb_data"].exclude_fk_and_id()

        if unlabeled_x.empty:
            logger.info(
                "No unlabeled samples remain for query %s in stream-%s; "
                "skipping coreset expansion.",
                q_name,
                inc_round,
            )
            return

        if not labeled_x.columns.equals(unlabeled_x.columns):
            labeled_x = labeled_x.loc[
                :, labeled_x.columns.isin(unlabeled_x.columns)
            ]
            if set(labeled_x.columns) != set(unlabeled_x.columns):
                raise ValueError(
                    f"Schema mismatch after alignment for query '{q_name}' "
                    f"in stream-{inc_round}. "
                    f"Labeled columns: {labeled_x.columns.tolist()}, "
                    f"Unlabeled columns: {unlabeled_x.columns.tolist()}"
                )
            unlabeled_x = unlabeled_x.loc[:, labeled_x.columns]

        mode = "empirical" if inc_round == 0 else "inc"
        annotated_selectivity = float(labeled_y.mean())
        estimated_selectivity = coreset["estimated_selectivity"]
        selectivity = (
            estimated_selectivity
            if estimated_selectivity is not None
            else annotated_selectivity
        )
        selectivity_source = (
            "estimated" if estimated_selectivity is not None else "annotated"
        )
        logger.info(
            "Using %s coreset selectivity %.4f for query %s "
            "(annotated=%.4f, estimated=%s).",
            selectivity_source,
            selectivity,
            q_name,
            annotated_selectivity,
            estimated_selectivity,
        )
        confidence_training_weights = class_balanced_sample_weights(
            labels=labeled_y,
            base_weight=coreset["annotation_weights"],
        )
        selected_x_idx, selected_y, new_lb, new_ub = select_coreset(
            labeled_X=labeled_x,
            labeled_Y=labeled_y,
            unlabeled_X=unlabeled_x,
            selectivity=selectivity,
            sample_weight=confidence_training_weights,
            k_neighbors=self.config["k_neighbors"],
            mode=mode,
            lb=coreset["lb"],
            ub=coreset["ub"],
            enable_conf_pred=self.enable_conf_pred,
            enable_conf_struct=self.enable_conf_struct,
        )
        coreset["lb"] = new_lb
        coreset["ub"] = new_ub

        selected_x = (
            sigma_record["ldb_data"]
            .df.iloc[selected_x_idx]
            .copy()
            .reset_index(drop=True)
        )
        coreset["ldb_data"].df = pd.concat(
            [coreset["ldb_data"].df, selected_x],
            ignore_index=True,
        )
        coreset["labels"] = pd.concat(
            [labeled_y, selected_y], ignore_index=True
        )
        coreset["annotation_weights"] = pd.concat(
            [coreset["annotation_weights"],
             pd.Series(1.0, index=range(len(selected_y)))],
            ignore_index=True,
        )


        logger.info(
            "Inc-Round %s: Expanded coreset for query %s: added %s "
            "samples. New coreset size: %s",
            inc_round,
            q_name,
            len(selected_x),
            len(coreset["ldb_data"].df),
        )


def select_coreset(
    labeled_X: pd.DataFrame,
    labeled_Y: pd.Series,
    unlabeled_X: pd.DataFrame,
    k_neighbors: int = 5,
    mode: Literal["balanced", "empirical", "inc"] = "empirical",
    lb: float = float("inf"),
    ub: float = float("-inf"),
    enable_conf_pred: bool = True,
    enable_conf_struct: bool = True,
    selectivity: float | None = None,
    sample_weight: np.ndarray | None = None,
) -> tuple[np.ndarray, pd.Series, float, float]:

    # Compute the confidence scores for all unlabeled samples.
    if selectivity is None:
        selectivity = float(labeled_Y.mean())
    if not 0.0 <= selectivity <= 1.0:
        raise ValueError("Selectivity must be between 0 and 1.")
    confs = est_conf(
        labeled_X,
        labeled_Y,
        unlabeled_X,
        selectivity,
        k_neighbors,
        sample_weight=sample_weight,
        enable_conf_pred=enable_conf_pred,
        enable_conf_struct=enable_conf_struct,
    )
    new_lb = confs.min()
    new_ub = confs.max()

    # Sort samples by confidence (descending - highest confs first)
    sorted_indices = np.argsort(-confs)

    # Determine how many samples are predicted as positive vs negative
    # Top selectivity * len(unlabeled_X) are predicted as positive
    n_predicted_pos = int(len(unlabeled_X) * selectivity)
    predicted_pos_indices = sorted_indices[:n_predicted_pos]
    predicted_neg_indices = sorted_indices[n_predicted_pos:]

    if mode == "balanced":
        # Balanced selection: choose confident positives and negatives.
        select_step = int(len(unlabeled_X) * min(selectivity, 1 - selectivity))
        selected_pos_indices = predicted_pos_indices[:select_step]
        selected_neg_indices = predicted_neg_indices[-select_step:]
    elif mode == "empirical":
        # Selection with empirical distribution.
        # TODO: Determine the optimal selection ratio.
        ratio = 0.2
        selected_pos_indices = predicted_pos_indices[
            : int(n_predicted_pos * ratio)
        ]
        selected_neg_indices = predicted_neg_indices[
            -int((len(unlabeled_X) - n_predicted_pos) * ratio) :
        ]
    elif mode == "inc":
        selected_pos_indices = predicted_pos_indices[
            confs[predicted_pos_indices] >= ub
        ]
        selected_neg_indices = predicted_neg_indices[
            confs[predicted_neg_indices] <= lb
        ]
    else:
        raise ValueError(f"Unsupported selection mode: {mode}")

    # Combine selected indices (positives first, then negatives)
    selected_indices = np.concatenate(
        [selected_pos_indices, selected_neg_indices]
    )

    # Create labels aligned with the selected positive/negative indices.
    selected_Y = pd.Series(
        [1] * len(selected_pos_indices) + [0] * len(selected_neg_indices)
    )

    return selected_indices, selected_Y, new_lb, new_ub


def est_conf(
    labeled_X: pd.DataFrame,
    labeled_Y: pd.Series,
    unlabeled_X: pd.DataFrame,
    selectivity: float,
    k_neighbors: int,
    sample_weight: np.ndarray | None = None,
    enable_conf_pred: bool = True,
    enable_conf_struct: bool = True,
) -> np.ndarray:

    # Preprocess
    labeled_X_proc = encode_features(labeled_X)
    unlabeled_X_proc = encode_features(unlabeled_X)

    # If FK/ID removal leaves no usable features, sklearn confidence models
    # cannot be fit. Use the empirical selectivity as a neutral confidence so
    # coreset expansion can continue without changing the experiment design.
    if labeled_X_proc.shape[1] == 0 or unlabeled_X_proc.shape[1] == 0:
        logger.warning(
            "No features available for coreset confidence estimation. "
            "Returning empirical selectivity %.4f for %s unlabeled samples.",
            selectivity,
            len(unlabeled_X_proc),
        )
        return np.full(len(unlabeled_X_proc), selectivity, dtype=float)

    # Predication confidence.
    pred_probas, feat_importance = est_prediction_conf(
        labeled_X_proc=labeled_X_proc,
        labeled_Y=labeled_Y,
        unlabeled_X_proc=unlabeled_X_proc,
        sample_weight=sample_weight,
    )

    # Structural confidence.
    labeled_X_nw = norm_features(
        weight_features(labeled_X_proc, feat_importance)
    )
    unlabeled_X_nw = norm_features(
        weight_features(unlabeled_X_proc, feat_importance)
    )
    struct_probas = est_structural_conf(
        labeled_X_nw=labeled_X_nw,
        labeled_Y=labeled_Y,
        unlabeled_X_nw=unlabeled_X_nw,
        k_neighbors=k_neighbors,
    )

    # Combined confidence.
    combined_probas = selectivity * pred_probas * int(enable_conf_pred) + (
        1 - selectivity
    ) * struct_probas * int(enable_conf_struct)

    return combined_probas


def est_prediction_conf(
    labeled_X_proc: pd.DataFrame,
    labeled_Y: pd.Series,
    unlabeled_X_proc: pd.DataFrame,
    sample_weight: np.ndarray | None = None,
) -> tuple[np.ndarray, pd.DataFrame]:

    # TODO: Try feature selection.

    # Compute the predication_conf.
    # TODO: configureable parameters:
    #   - n_estimators
    #   - max_depth
    #   - random_seed
    clf = train_classifier(
        X=labeled_X_proc, Y=labeled_Y, sample_weight=sample_weight
    )

    # Handle case when all labels are the same (single class)
    unique_classes = np.unique(labeled_Y)
    if len(unique_classes) == 1:
        # All samples belong to the same class, return uniform probabilities
        single_class = unique_classes[0]
        probas = np.full(len(unlabeled_X_proc), single_class, dtype=float)
        logger.info(
            f"All labeled samples belong to class {single_class}. "
            f"Returning uniform probabilities: {single_class}"
        )
    else:
        # Normal case: multiple classes, use predict_proba
        probas = clf.predict_proba(unlabeled_X_proc)[:, 1]

    feat_importances = pd.DataFrame(
        {
            "feature": labeled_X_proc.columns.tolist(),
            "importance": clf.feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    return probas, feat_importances


def est_structural_conf(
    labeled_X_nw: pd.DataFrame,
    labeled_Y: pd.Series,
    unlabeled_X_nw: pd.DataFrame,
    k_neighbors: int,
) -> np.ndarray:

    labeled_feats = labeled_X_nw.to_numpy()
    unlabeled_feats = unlabeled_X_nw.to_numpy()

    dist_pos, dist_neg = _compute_knn_distances(
        labeled_feats=labeled_feats,
        unlabeled_feats=unlabeled_feats,
        labels=labeled_Y,
        k_neighbors=k_neighbors,
    )

    struct_prob = _distances_to_structural_prob(
        dist_pos=dist_pos, dist_neg=dist_neg
    )

    return struct_prob


def _compute_knn_distances(
    labeled_feats: np.ndarray,
    unlabeled_feats: np.ndarray,
    labels: pd.Series,
    k_neighbors: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute k-NN distances to positive and negative classes."""
    # Handle NaN values: replace with 0
    labeled_feats = np.nan_to_num(labeled_feats, nan=0.0)
    unlabeled_feats = np.nan_to_num(unlabeled_feats, nan=0.0)

    pos_idx = labels[labels == 1].index
    neg_idx = labels[labels == 0].index

    n_unlabeled = len(unlabeled_feats)

    # Initialize distances with infinity
    dist_pos = np.full((n_unlabeled, k_neighbors), np.inf)
    dist_neg = np.full((n_unlabeled, k_neighbors), np.inf)

    # Compute distances to positive samples if available
    if len(pos_idx) > 0:
        pos_samples = labeled_feats[pos_idx]
        n_neighbors = min(k_neighbors, len(pos_idx))
        knn_pos = NearestNeighbors(n_neighbors=n_neighbors).fit(pos_samples)
        dist_pos, _ = knn_pos.kneighbors(unlabeled_feats)

    # Compute distances to negative samples if available
    if len(neg_idx) > 0:
        neg_samples = labeled_feats[neg_idx]
        n_neighbors = min(k_neighbors, len(neg_idx))
        knn_neg = NearestNeighbors(n_neighbors=n_neighbors).fit(neg_samples)
        dist_neg, _ = knn_neg.kneighbors(unlabeled_feats)

    return dist_pos, dist_neg


def _distances_to_structural_prob(
    dist_pos: np.ndarray, dist_neg: np.ndarray
) -> np.ndarray:
    """Convert k-NN distances to geometric probabilities."""
    epsilon = 1e-6
    geo_score_pos = 1.0 / (dist_pos.mean(axis=1) + epsilon)
    geo_score_neg = 1.0 / (dist_neg.mean(axis=1) + epsilon)
    total_score = geo_score_pos + geo_score_neg
    return geo_score_pos / total_score
