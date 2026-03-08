import pandas as pd
import numpy as np
from typing import Tuple, Literal
from sklearn.neighbors import NearestNeighbors
from .utils import (
    encode_features,
    weight_features,
    norm_features, 
    train_classifier
)


def select_coreset(labeled_X: pd.DataFrame, labeled_Y: pd.Series,
        unlabeled_X: pd.DataFrame,
        k_neighbors: int=5,
        mode: Literal["balanced", "emperical"] = "emperical") -> Tuple[np.ndarray, pd.Series]:

    # Compute the confidence scores for all unlabeled samples.
    selectivity = sum(labeled_Y) / len(labeled_Y)
    confs = est_conf(labeled_X, labeled_Y, unlabeled_X, selectivity, k_neighbors)

    # Sort samples by confidence (descending - highest confs first)
    sorted_indices = np.argsort(-confs)

    # Determine how many samples are predicted as positive vs negative
    # Top selectivity * len(unlabeled_X) are predicted as positive
    n_predicted_pos = int(len(unlabeled_X) * selectivity)
    predicted_pos_indices = sorted_indices[:n_predicted_pos]
    predicted_neg_indices = sorted_indices[n_predicted_pos:]

    if mode == "balanced":
        # Balanced selection: select top select_step from positives and bottom select_step from negatives
        select_step = int(len(unlabeled_X) * min(selectivity, 1 - selectivity))
        selected_pos_indices = predicted_pos_indices[:select_step]
        selected_neg_indices = predicted_neg_indices[-select_step:]
    elif mode == "emperical":
        # Selection with emperical distribution: select according to the predicted ratio
        # TODO: Determine the optimal selection ratio.
        ratio = 0.2
        selected_pos_indices = predicted_pos_indices[:int(n_predicted_pos * ratio)]
        selected_neg_indices = predicted_neg_indices[-int((len(unlabeled_X) - n_predicted_pos) * ratio):]
    else:
        raise ValueError(f"Unsupported selection mode: {mode}")

    # Combine selected indices (positives first, then negatives)
    selected_indices = np.concatenate([selected_pos_indices, selected_neg_indices])

    # Create labels aligned with the indices (first select_step are positive, rest are negative)
    selected_Y = pd.Series([1] * len(selected_pos_indices) + [0] * len(selected_neg_indices))

    return selected_indices, selected_Y


def est_conf(labeled_X: pd.DataFrame, labeled_Y: pd.Series, 
        unlabeled_X: pd.DataFrame, 
        selectivity: float,
        k_neighbors: int) -> np.ndarray:
    
    # Preprocess
    labeled_X_proc = encode_features(labeled_X)
    unlabeled_X_proc = encode_features(unlabeled_X)

    # Predication confidence.
    pred_probas, feat_importance = est_prediction_conf(
        labeled_X_proc=labeled_X_proc,
        labeled_Y=labeled_Y,
        unlabeled_X_proc=unlabeled_X_proc,
    )

    # Structural confidence.
    labeled_X_nw = norm_features(weight_features(labeled_X_proc, feat_importance))
    unlabeled_X_nw = norm_features(weight_features(unlabeled_X_proc, feat_importance))
    struct_probas = est_structural_conf(
        labeled_X_nw=labeled_X_nw,
        labeled_Y=labeled_Y,
        unlabeled_X_nw=unlabeled_X_nw,
        k_neighbors=k_neighbors
    )

    # Combined confidence.
    combined_probas = selectivity * pred_probas + (1 - selectivity) * struct_probas

    return combined_probas


def est_prediction_conf(
        labeled_X_proc: pd.DataFrame, 
        labeled_Y: pd.Series, 
        unlabeled_X_proc: pd.DataFrame) -> Tuple[np.ndarray, pd.DataFrame]:

    # TODO: Try feature selection.
    
    # Compute the predication_conf.
    # TODO: configureable parameters:
    #   - n_estimators
    #   - max_depth
    #   - random_seed
    clf = train_classifier(X=labeled_X_proc, Y=labeled_Y)

    probas = clf.predict_proba(unlabeled_X_proc)[:, 1]
    feat_importances = pd.DataFrame({
        "feature": labeled_X_proc.columns.tolist(),
        "importance": clf.feature_importances_
    }).sort_values("importance", ascending=False)

    return probas, feat_importances


def est_structural_conf(
        labeled_X_nw: pd.DataFrame, 
        labeled_Y: pd.Series,
        unlabeled_X_nw: pd.DataFrame,
        k_neighbors: int) -> np.ndarray:

    labeled_feats = labeled_X_nw.to_numpy()
    unlabeled_feats = unlabeled_X_nw.to_numpy()

    dist_pos, dist_neg = _compute_knn_distances(
        labeled_feats=labeled_feats,
        unlabeled_feats=unlabeled_feats,
        labels=labeled_Y,
        k_neighbors=k_neighbors
    )

    struct_prob = _distances_to_structural_prob(dist_pos=dist_pos, dist_neg=dist_neg)

    return struct_prob


def _compute_knn_distances(
        labeled_feats: np.ndarray, unlabeled_feats: np.ndarray,
        labels: pd.Series, k_neighbors: int) -> Tuple[np.ndarray, np.ndarray]:
    """Compute k-NN distances to positive and negative classes."""
    pos_idx = labels[labels == 1].index
    neg_idx = labels[labels == 0].index

    assert len(pos_idx) > 0 and len(neg_idx) > 0, \
        "Both positive and negative samples must be present."

    pos_samples = labeled_feats[pos_idx]
    neg_samples = labeled_feats[neg_idx]

    n_neighbors = min(k_neighbors, len(pos_idx), len(neg_idx))
    knn_pos = NearestNeighbors(n_neighbors=n_neighbors).fit(pos_samples)
    knn_neg = NearestNeighbors(n_neighbors=n_neighbors).fit(neg_samples)

    dist_pos, _ = knn_pos.kneighbors(unlabeled_feats)
    dist_neg, _ = knn_neg.kneighbors(unlabeled_feats)

    return dist_pos, dist_neg


def _distances_to_structural_prob(dist_pos: np.ndarray, dist_neg: np.ndarray) -> np.ndarray:
    """Convert k-NN distances to geometric probabilities."""
    epsilon = 1e-6
    geo_score_pos = 1.0 / (dist_pos.mean(axis=1) + epsilon)
    geo_score_neg = 1.0 / (dist_neg.mean(axis=1) + epsilon)
    total_score = geo_score_pos + geo_score_neg
    return geo_score_pos / total_score



