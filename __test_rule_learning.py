"""
Few-Shot Binary Classification Framework

Simple, configurable framework with modular optimization methods.
"""

import pandas as pd
import numpy as np
import logging
import time
import argparse
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable
import sys

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# Config
# =============================================================================
@dataclass
class Config:
    # Data (required)
    data_path: str

    # Data (optional with defaults)
    visible_samples: int = 50
    random_seed: int = 42

    # Feature selection
    top_k: int = 10

    # Self-training
    self_training_mode: str = 'Conf'  # Options: 'Conf', 'ConfGeo', 'Geo
    n_rounds: int = 3  # TODO: compute risk value in each round to determine the risk threshold.
    confidence_threshold: float = 0.95
    max_samples_per_round: int = 10
    balance_classes: bool = True
    report_each_round_f1: bool = False
    report_each_round_risk: bool = False

    # Geometric self-training parameters
    geo_k_neighbors: int = 5  # k for k-NN (max neighbors, will be capped by minority class size)
    geo_initial_weight: float = 0.6  # Initial weight for geometric scoring (decreases over rounds)
    geo_decision_boundary: float = 0.5  # Decision boundary for classification (default: 0.5)

    # Classifier
    n_estimators: int = 100
    max_depth: int = 10

    # Methods to run (will be set dynamically)
    methods: List[List[str]] = field(default_factory=list)


# =============================================================================
# Modular Optimization Methods
# =============================================================================
def apply_feature_selection(vis_X: pd.DataFrame, vis_Y: pd.Series, config: Config) -> List[str]:
    """Select top-k features based on importance."""
    clf_temp = train_classifier(vis_X, vis_Y, config)
    importances = pd.DataFrame({
        'feature': vis_X.columns,
        'importance': clf_temp.feature_importances_
    }).sort_values('importance', ascending=False)
    features = importances.head(config.top_k)['feature'].tolist()
    logger.info(f"  [Feature Selection] Top-{config.top_k}: {len(features)} features")
    return features


# =============================================================================
# Helper functions for self-training
# =============================================================================
def _compute_feature_weights(vis_X_proc: pd.DataFrame, features: List[str],
                             vis_Y: pd.Series, config: Config) -> np.ndarray:
    """Compute feature importance weights."""
    clf_temp = train_classifier(vis_X_proc[features], vis_Y, config)
    feature_importances = clf_temp.feature_importances_

    feature_weights = np.ones(len(features))
    for i, feat in enumerate(features):
        if feat in vis_X_proc.columns:
            idx = list(vis_X_proc.columns).index(feat)
            if idx < len(feature_importances):
                feature_weights[i] = np.sqrt(feature_importances[idx])
    return feature_weights


def _compute_knn_distances(vis_X_weighted: np.ndarray, inv_X_weighted: np.ndarray,
                           vis_Y: pd.Series, k_neighbors: int) -> tuple:
    """Compute k-NN distances to positive and negative classes."""
    from sklearn.neighbors import NearestNeighbors

    pos_idx = vis_Y[vis_Y == 1].index
    neg_idx = vis_Y[vis_Y == 0].index

    if len(pos_idx) == 0 or len(neg_idx) == 0:
        return None, None

    pos_samples = vis_X_weighted[pos_idx]
    neg_samples = vis_X_weighted[neg_idx]

    n_neighbors = min(k_neighbors, len(pos_idx), len(neg_idx))
    knn_pos = NearestNeighbors(n_neighbors=n_neighbors).fit(pos_samples)
    knn_neg = NearestNeighbors(n_neighbors=n_neighbors).fit(neg_samples)

    dist_pos, _ = knn_pos.kneighbors(inv_X_weighted)
    dist_neg, _ = knn_neg.kneighbors(inv_X_weighted)

    return dist_pos, dist_neg


def _distances_to_geo_prob(dist_pos: np.ndarray, dist_neg: np.ndarray) -> np.ndarray:
    """Convert k-NN distances to geometric probabilities."""
    epsilon = 1e-6
    geo_score_pos = 1.0 / (dist_pos.mean(axis=1) + epsilon)
    geo_score_neg = 1.0 / (dist_neg.mean(axis=1) + epsilon)
    total_score = geo_score_pos + geo_score_neg
    return geo_score_pos / total_score


def _prepare_weighted_features(vis_X_proc: pd.DataFrame, inv_X_proc: pd.DataFrame,
                                features: List[str], vis_Y: pd.Series, config: Config) -> tuple:
    """Prepare normalized and feature-weighted datasets for geometric methods."""
    from sklearn.preprocessing import StandardScaler

    feature_weights = _compute_feature_weights(vis_X_proc, features, vis_Y, config)

    scaler = StandardScaler()
    vis_X_norm = scaler.fit_transform(vis_X_proc[features])
    inv_X_norm = scaler.transform(inv_X_proc[features])

    vis_X_weighted = vis_X_norm * feature_weights
    inv_X_weighted = inv_X_norm * feature_weights

    return vis_X_weighted, inv_X_weighted


def _compute_geometric_predictions(vis_X_proc: pd.DataFrame, inv_X_proc: pd.DataFrame,
                                    features: List[str], vis_Y: pd.Series, config: Config) -> tuple:
    """Compute predictions and probabilities using geometric k-NN method."""
    vis_X_weighted, inv_X_weighted = _prepare_weighted_features(vis_X_proc, inv_X_proc, features, vis_Y, config)

    dist_pos, dist_neg = _compute_knn_distances(vis_X_weighted, inv_X_weighted, vis_Y, config.geo_k_neighbors)
    if dist_pos is None or dist_neg is None:
        return None, None, None

    geo_prob_pos = _distances_to_geo_prob(dist_pos, dist_neg)
    predictions = (geo_prob_pos >= config.geo_decision_boundary).astype(int)
    confidence = np.abs(geo_prob_pos - config.geo_decision_boundary)

    return predictions, geo_prob_pos, confidence


def _select_samples_confidence(predictions: np.ndarray, conf_mask_pos: np.ndarray,
                                conf_mask_neg: np.ndarray, high_conf_mask: np.ndarray,
                                config: Config, rng: np.random.Generator) -> np.ndarray:
    """Select samples using confidence thresholding with random selection."""
    if config.balance_classes:
        pos_idx = np.where(conf_mask_pos)[0]
        neg_idx = np.where(conf_mask_neg)[0]

        if len(pos_idx) == 0 or len(neg_idx) == 0:
            n = min(high_conf_mask.sum(), config.max_samples_per_round)
            return rng.choice(np.where(high_conf_mask)[0], n, replace=False)
        else:
            n_per_class = min(len(pos_idx), len(neg_idx), config.max_samples_per_round // 2)
            if n_per_class == 0:
                n_per_class = min(len(pos_idx), len(neg_idx))
            selected_pos = rng.choice(pos_idx, n_per_class, replace=False)
            selected_neg = rng.choice(neg_idx, n_per_class, replace=False)
            return np.concatenate([selected_pos, selected_neg])
    else:
        n = min(high_conf_mask.sum(), config.max_samples_per_round)
        return rng.choice(np.where(high_conf_mask)[0], n, replace=False)


def _select_samples_balanced(predictions: np.ndarray, conf_scores: np.ndarray,
                             config: Config) -> np.ndarray:
    """Select samples with optional class balancing based on confidence scores."""
    pos_pred_idx = np.where(predictions == 1)[0]
    neg_pred_idx = np.where(predictions == 0)[0]

    if config.balance_classes:
        if len(pos_pred_idx) == 0 or len(neg_pred_idx) == 0:
            available_idx = pos_pred_idx if len(pos_pred_idx) > 0 else neg_pred_idx
            n = min(len(available_idx), config.max_samples_per_round)
            if n == 0:
                return np.array([])
            return available_idx[np.argsort(conf_scores[available_idx])[-n:]]
        else:
            n_per_class = min(len(pos_pred_idx), len(neg_pred_idx), config.max_samples_per_round // 2)
            if n_per_class == 0:
                n_per_class = min(len(pos_pred_idx), len(neg_pred_idx))
            selected_pos = pos_pred_idx[np.argsort(conf_scores[pos_pred_idx])[-n_per_class:]]
            selected_neg = neg_pred_idx[np.argsort(conf_scores[neg_pred_idx])[-n_per_class:]]
            return np.concatenate([selected_pos, selected_neg])
    else:
        n = min(len(predictions), config.max_samples_per_round)
        return np.argsort(conf_scores)[-n:]


def _update_datasets(vis_X: pd.DataFrame, vis_Y: pd.Series, vis_X_proc: pd.DataFrame,
                     inv_X: pd.DataFrame, inv_X_proc: pd.DataFrame,
                     selected: np.ndarray, predictions: np.ndarray,
                     log_labels: bool = False) -> tuple:
    """Update datasets by moving selected samples from invisible to visible."""
    vis_X_new = pd.concat([vis_X, inv_X.iloc[selected]], ignore_index=True)
    vis_Y_new = pd.concat([vis_Y, pd.Series(predictions[selected], dtype=vis_Y.dtype)], ignore_index=True).astype(vis_Y.dtype)
    vis_X_proc_new = pd.concat([vis_X_proc, inv_X_proc.iloc[selected]], ignore_index=True)
    inv_X_new = inv_X.drop(inv_X.index[selected])
    inv_X_proc_new = inv_X_proc.drop(inv_X_proc.index[selected])

    if log_labels:
        new_labels = pd.Series(predictions[selected], dtype=vis_Y.dtype)
        logger.info(f"  Adding {len(new_labels)} new labels: {new_labels.value_counts().to_dict()}")

    return vis_X_new, vis_Y_new, vis_X_proc_new, inv_X_new, inv_X_proc_new


def _compute_round_f1(vis_X: pd.DataFrame, vis_Y: pd.Series, inv_X: pd.DataFrame,
                      inv_Y: Optional[pd.Series], features: List[str], config: Config) -> float:
    """Compute F1 score for current round using all visible and invisible samples."""
    from sklearn.metrics import f1_score

    if inv_Y is None:
        raise ValueError("inv_Y must be provided to compute F1 score")

    # Preprocess and train classifier on current visible set
    vis_X_proc = preprocess_features(vis_X)
    inv_X_proc = preprocess_features(inv_X)

    clf = train_classifier(vis_X_proc[features], vis_Y, config)
    inv_Y_pred = clf.predict(inv_X_proc[features])

    # Compute F1 using all samples
    all_Y_pred = np.concatenate([vis_Y.to_numpy(), inv_Y_pred])
    all_Y_true = np.concatenate([vis_Y.to_numpy(), inv_Y.to_numpy()])

    return float(f1_score(all_Y_true, all_Y_pred))


def _compute_loss(pi: float, true_label: int, predicted_label: int) -> float:
    """Compute loss for a single prediction.

    Args:
        pi: Proportion of positive samples in visible set
        true_label: Actual label (0 or 1)
        predicted_label: Predicted label (0 or 1)

    Returns:
        Loss value (0 if correct, otherwise weighted penalty)
    """
    if true_label == predicted_label:
        return 0.0

    max_pi = max(pi, 1 - pi)

    if true_label == 1 and predicted_label == 0:
        # False negative
        return max_pi / pi if pi > 0 else float('inf')
    else:
        # False positive (true_label == 0, predicted_label == 1)
        return max_pi / (1 - pi) if pi < 1 else float('inf')


def _compute_round_risk_base(vis_X: pd.DataFrame, vis_Y: pd.Series, inv_X: pd.DataFrame,
                              inv_Y: Optional[pd.Series], features: List[str], config: Config,
                              predict_fn: Callable, **predict_fn_kwargs) -> float:
    """Base risk computation function that uses a provided prediction function.

    Risk is computed by evicting each visible sample one at a time,
    training on the remaining samples, and predicting the evicted sample.

    Args:
        vis_X: Current visible features (BEFORE adding new samples)
        vis_Y: Current visible labels (BEFORE adding new samples)
        inv_X: Current invisible features (unused, for interface compatibility)
        inv_Y: Current invisible labels (for validation only)
        features: List of feature names to use
        config: Configuration object
        predict_fn: Function that takes (vis_X_remaining, vis_Y_remaining, evicted_X_proc,
                                       features, config, **predict_fn_kwargs) and returns prediction
        **predict_fn_kwargs: Additional keyword arguments to pass to predict_fn

    Returns:
        Total risk value (sum of individual losses)
    """
    if inv_Y is None:
        raise ValueError("inv_Y must be provided to compute risk value")

    if len(vis_X) == 0:
        return 0.0

    vis_X_proc = preprocess_features(vis_X)

    # Calculate pi once for the current vis_X
    pi = vis_Y.sum() / len(vis_Y)
    total_risk = 0.0

    # Evict each sample one at a time
    for idx in range(len(vis_X)):
        # Split vis_X into remaining and evicted
        vis_X_remaining = vis_X_proc.drop(index=vis_X_proc.index[idx])
        vis_Y_remaining = vis_Y.drop(index=vis_Y.index[idx])

        evicted_X_proc = vis_X_proc.iloc[[idx]]
        evicted_Y = vis_Y.iloc[idx]

        # Skip if remaining samples don't have both classes
        if vis_Y_remaining.sum() == 0 or vis_Y_remaining.sum() == len(vis_Y_remaining):
            continue

        # Use the provided prediction function
        try:
            evicted_pred = predict_fn(vis_X_remaining, vis_Y_remaining, evicted_X_proc,
                                      features, config, **predict_fn_kwargs)
            if evicted_pred is None:
                continue
        except Exception:
            # Skip if prediction fails
            continue

        # Calculate loss
        loss = _compute_loss(pi, int(evicted_Y), evicted_pred)
        total_risk += loss

    Gamma = max(pi, 1-pi) / min(pi, 1-pi) if min(pi, 1-pi) > 0 else float('inf')
    delta = 0.5
    workload_size = 1
    total_risk += Gamma * np.sqrt(
        np.log(2 * workload_size / delta) / (2 * len(vis_X))
    )

    return total_risk


def _predict_with_classifier(vis_X_remaining: pd.DataFrame, vis_Y_remaining: pd.Series,
                             evicted_X_proc: pd.DataFrame, features: List[str],
                             config: Config, **kwargs) -> int:
    """Prediction function using classifier only."""
    clf = train_classifier(vis_X_remaining[features], vis_Y_remaining, config)
    return int(clf.predict(evicted_X_proc[features])[0])


def _predict_with_geometric(vis_X_remaining: pd.DataFrame, vis_Y_remaining: pd.Series,
                            evicted_X_proc: pd.DataFrame, features: List[str],
                            config: Config, **kwargs) -> Optional[int]:
    """Prediction function using geometric k-NN only."""
    predictions, _, _ = _compute_geometric_predictions(
        vis_X_remaining, evicted_X_proc, features, vis_Y_remaining, config
    )
    return int(predictions[0]) if predictions is not None else None


def _predict_with_combined(vis_X_remaining: pd.DataFrame, vis_Y_remaining: pd.Series,
                           evicted_X_proc: pd.DataFrame, features: List[str],
                           config: Config, round_idx: int = 0, **kwargs) -> Optional[int]:
    """Prediction function combining classifier and geometric."""
    # Get classifier prediction
    clf = train_classifier(vis_X_remaining[features], vis_Y_remaining, config)
    proba = clf.predict_proba(evicted_X_proc[features])[0, 1]

    # Get geometric prediction
    _, geo_prob_pos, _ = _compute_geometric_predictions(
        vis_X_remaining, evicted_X_proc, features, vis_Y_remaining, config
    )

    if geo_prob_pos is None:
        return None

    geo_prob = geo_prob_pos[0]

    # Combine classifier probability with geometric probability
    geo_weight = config.geo_initial_weight * (1 - round_idx / config.n_rounds)
    combined_prob = geo_weight * geo_prob + (1 - geo_weight) * proba

    # Get prediction based on combined probability
    return int(combined_prob >= config.geo_decision_boundary)


def _compute_round_risk_conf(vis_X: pd.DataFrame, vis_Y: pd.Series, inv_X: pd.DataFrame,
                              inv_Y: Optional[pd.Series], features: List[str], config: Config) -> float:
    """Compute risk value for confidence-based self-training."""
    return _compute_round_risk_base(vis_X, vis_Y, inv_X, inv_Y, features, config,
                                     _predict_with_classifier)


def _compute_round_risk_geo(vis_X: pd.DataFrame, vis_Y: pd.Series, inv_X: pd.DataFrame,
                             inv_Y: Optional[pd.Series], features: List[str], config: Config) -> float:
    """Compute risk value for geometric-only self-training."""
    return _compute_round_risk_base(vis_X, vis_Y, inv_X, inv_Y, features, config,
                                     _predict_with_geometric)


def _compute_round_risk_conf_geo(vis_X: pd.DataFrame, vis_Y: pd.Series, inv_X: pd.DataFrame,
                                  inv_Y: Optional[pd.Series], features: List[str],
                                  config: Config, round_idx: int) -> float:
    """Compute risk value for combined confidence + geometric self-training."""
    return _compute_round_risk_base(vis_X, vis_Y, inv_X, inv_Y, features, config,
                                     _predict_with_combined, round_idx=round_idx)


def apply_self_training_conf(vis_X: pd.DataFrame, vis_Y: pd.Series, inv_X: pd.DataFrame,
                             features: List[str], config: Config, inv_Y: Optional[pd.Series] = None) -> tuple:
    """Apply self-training to augment training data using confidence scores.

    Returns:
        tuple: (vis_X, vis_Y, inv_X, f1_history, risk_history) where inv_X maintains its original index.
               f1_history is a list of F1 scores per round (empty if report_each_round_f1 is False).
               risk_history is a list of risk values per round (empty if report_each_round_risk is False).
               The caller can use inv_X.index to sync inv_Y.
    """
    vis_X_proc = preprocess_features(vis_X)
    inv_X_proc = preprocess_features(inv_X)
    rng = np.random.default_rng(config.random_seed)
    f1_history = []
    risk_history = []

    # Track original visible samples (before adding pseudo-labels)
    orig_vis_size = len(vis_X)

    logger.info(f"  [Self-Training] Pos={vis_Y.sum()}, Neg={len(vis_Y) - vis_Y.sum()}")

    for round_idx in range(config.n_rounds):
        logger.info(f"  Round {round_idx + 1}/{config.n_rounds} | Vis: {len(vis_X)} Inv: {len(inv_X)}")

        # Compute risk value for this round if enabled
        if config.report_each_round_risk and inv_Y is not None:
            risk = _compute_round_risk_conf(vis_X, vis_Y, inv_X, inv_Y, features, config)
            risk_history.append(risk)
            logger.info(f"  Round {round_idx + 1} Risk: {risk:.4f} Size of vis_X: {len(vis_X)}")


        clf = train_classifier(vis_X_proc[features], vis_Y, config)
        probas = clf.predict_proba(inv_X_proc[features])[:, 1]
        predictions = clf.predict(inv_X_proc[features])

        # Apply confidence threshold and select samples
        conf_mask_pos = (probas >= config.confidence_threshold)
        conf_mask_neg = (probas <= (1 - config.confidence_threshold))
        high_conf_mask = conf_mask_pos | conf_mask_neg

        if high_conf_mask.sum() == 0:
            logger.info("  No samples above threshold. Stopping.")
            break

        selected = _select_samples_confidence(predictions, conf_mask_pos, conf_mask_neg,
                                               high_conf_mask, config, rng)

        # Update datasets
        vis_X, vis_Y, vis_X_proc, inv_X, inv_X_proc = _update_datasets(
            vis_X, vis_Y, vis_X_proc, inv_X, inv_X_proc, selected, predictions
        )

        # Compute F1 score for this round if enabled
        if config.report_each_round_f1 and inv_Y is not None:
            # Sync inv_Y with inv_X after dropping selected samples
            current_inv_Y = inv_Y.iloc[selected].reset_index(drop=True)
            inv_Y = inv_Y.drop(inv_Y.index[selected])

            # Use only original visible samples (exclude pseudo-labeled samples)
            vis_X_orig = vis_X.iloc[:orig_vis_size]
            vis_Y_orig = vis_Y.iloc[:orig_vis_size]

            # Add pseudo-labeled samples back to invisible set for evaluation
            vis_X_pseudo = vis_X.iloc[orig_vis_size:]
            vis_Y_pseudo = vis_Y.iloc[orig_vis_size:]

            inv_X_eval = pd.concat([inv_X, vis_X_pseudo], ignore_index=True)
            inv_Y_eval = pd.concat([inv_Y, vis_Y_pseudo], ignore_index=True)

            f1 = _compute_round_f1(vis_X_orig, vis_Y_orig, inv_X_eval, inv_Y_eval, features, config)
            f1_history.append(f1)
            logger.info(f"  Round {round_idx + 1} F1: {f1:.4f}")

        if len(inv_X) == 0:
            logger.info("  No more invisible samples. Stopping.")
            break

    return vis_X, vis_Y, inv_X, f1_history, risk_history


def apply_self_training_geo(vis_X: pd.DataFrame, vis_Y: pd.Series, inv_X: pd.DataFrame,
                            features: List[str], config: Config, inv_Y: Optional[pd.Series] = None) -> tuple:
    """Apply geometric self-training using ONLY k-NN distances, no confidence scores.

    This method selects samples based purely on geometric proximity to labeled samples
    using k-NN distances weighted by feature importance.

    Returns:
        tuple: (vis_X, vis_Y, inv_X, f1_history, risk_history) where inv_X maintains its original index.
               f1_history is a list of F1 scores per round (empty if report_each_round_f1 is False).
               risk_history is a list of risk values per round (empty if report_each_round_risk is False).
               The caller can use inv_X.index to sync inv_Y.
    """
    vis_X_proc = preprocess_features(vis_X)
    inv_X_proc = preprocess_features(inv_X)
    f1_history = []
    risk_history = []

    # Track original visible samples (before adding pseudo-labels)
    orig_vis_size = len(vis_X)

    logger.info(f"  [Self-Training: Geo-Only] Pos={vis_Y.sum()}, Neg={len(vis_Y) - vis_Y.sum()}")

    for round_idx in range(config.n_rounds):
        logger.info(f"  Round {round_idx + 1}/{config.n_rounds} | Vis: {len(vis_X)} Inv: {len(inv_X)}")

        # Compute risk value for this round if enabled
        if config.report_each_round_risk and inv_Y is not None:
            risk = _compute_round_risk_geo(vis_X, vis_Y, inv_X, inv_Y, features, config)
            risk_history.append(risk)
            logger.info(f"  Round {round_idx + 1} Risk: {risk:.4f} Size of vis_X: {len(vis_X)}")

        # Compute geometric predictions
        predictions, _, confidence = _compute_geometric_predictions(
            vis_X_proc, inv_X_proc, features, vis_Y, config
        )

        if predictions is None:
            logger.info("  Not enough samples of both classes. Stopping.")
            break

        # Select samples with highest geometric confidence
        selected = _select_samples_balanced(predictions, confidence, config)

        if len(selected) == 0:
            logger.info("  No samples available. Stopping.")
            break

        # Update datasets
        vis_X, vis_Y, vis_X_proc, inv_X, inv_X_proc = _update_datasets(
            vis_X, vis_Y, vis_X_proc, inv_X, inv_X_proc, selected, predictions, log_labels=True
        )

        # Compute F1 score for this round if enabled
        if config.report_each_round_f1 and inv_Y is not None:
            inv_Y = inv_Y.drop(inv_Y.index[selected])

            # Use only original visible samples (exclude pseudo-labeled samples)
            vis_X_orig = vis_X.iloc[:orig_vis_size]
            vis_Y_orig = vis_Y.iloc[:orig_vis_size]

            # Add pseudo-labeled samples back to invisible set for evaluation
            vis_X_pseudo = vis_X.iloc[orig_vis_size:]
            vis_Y_pseudo = vis_Y.iloc[orig_vis_size:]

            inv_X_eval = pd.concat([inv_X, vis_X_pseudo], ignore_index=True)
            inv_Y_eval = pd.concat([inv_Y, vis_Y_pseudo], ignore_index=True)

            f1 = _compute_round_f1(vis_X_orig, vis_Y_orig, inv_X_eval, inv_Y_eval, features, config)
            f1_history.append(f1)
            logger.info(f"  Round {round_idx + 1} F1: {f1:.4f}")

        if len(inv_X) == 0:
            logger.info("  No more invisible samples. Stopping.")
            break

    return vis_X, vis_Y, inv_X, f1_history, risk_history


def apply_self_training_conf_geo(vis_X: pd.DataFrame, vis_Y: pd.Series, inv_X: pd.DataFrame,
                                 features: List[str], config: Config, inv_Y: Optional[pd.Series] = None) -> tuple:
    """Apply self-training combining confidence scores and geometric k-NN.

    Returns:
        tuple: (vis_X, vis_Y, inv_X, f1_history, risk_history) where inv_X maintains its original index.
               f1_history is a list of F1 scores per round (empty if report_each_round_f1 is False).
               risk_history is a list of risk values per round (empty if report_each_round_risk is False).
               The caller can use inv_X.index to sync inv_Y.
    """
    vis_X_proc = preprocess_features(vis_X)
    inv_X_proc = preprocess_features(inv_X)
    f1_history = []
    risk_history = []

    # Track original visible samples (before adding pseudo-labels)
    orig_vis_size = len(vis_X)

    logger.info(f"  [Self-Training: Geometric] Pos={vis_Y.sum()}, Neg={len(vis_Y) - vis_Y.sum()}")

    for round_idx in range(config.n_rounds):
        logger.info(f"  Round {round_idx + 1}/{config.n_rounds} | Vis: {len(vis_X)} Inv: {len(inv_X)}")

        # Compute risk value for this round if enabled
        if config.report_each_round_risk and inv_Y is not None:
            risk = _compute_round_risk_conf_geo(vis_X, vis_Y, inv_X, inv_Y, features, config, round_idx)
            risk_history.append(risk)
            logger.info(f"  Round {round_idx + 1} Risk: {risk:.4f} Size of vis_X: {len(vis_X)}")

        # Get classifier predictions
        clf = train_classifier(vis_X_proc[features], vis_Y, config)
        probas = clf.predict_proba(inv_X_proc[features])[:, 1]

        # Get geometric predictions
        _, geo_prob_pos, _ = _compute_geometric_predictions(
            vis_X_proc, inv_X_proc, features, vis_Y, config
        )

        if geo_prob_pos is None:
            logger.info("  Not enough samples of both classes. Stopping.")
            break

        # Combine classifier probability with geometric probability
        geo_weight = config.geo_initial_weight * (1 - round_idx / config.n_rounds)
        combined_prob = geo_weight * geo_prob_pos + (1 - geo_weight) * probas

        # Get predictions based on combined probability
        predictions = (combined_prob >= config.geo_decision_boundary).astype(int)
        confidence = np.abs(combined_prob - config.geo_decision_boundary)

        # Select samples with highest confidence
        selected = _select_samples_balanced(predictions, confidence, config)

        if len(selected) == 0:
            logger.info("  No samples available. Stopping.")
            break

        # Update datasets
        vis_X, vis_Y, vis_X_proc, inv_X, inv_X_proc = _update_datasets(
            vis_X, vis_Y, vis_X_proc, inv_X, inv_X_proc, selected, predictions, log_labels=True
        )

        # Compute F1 score for this round if enabled
        if config.report_each_round_f1 and inv_Y is not None:
            inv_Y = inv_Y.drop(inv_Y.index[selected])

            # Use only original visible samples (exclude pseudo-labeled samples)
            vis_X_orig = vis_X.iloc[:orig_vis_size]
            vis_Y_orig = vis_Y.iloc[:orig_vis_size]

            # Add pseudo-labeled samples back to invisible set for evaluation
            vis_X_pseudo = vis_X.iloc[orig_vis_size:]
            vis_Y_pseudo = vis_Y.iloc[orig_vis_size:]

            inv_X_eval = pd.concat([inv_X, vis_X_pseudo], ignore_index=True)
            inv_Y_eval = pd.concat([inv_Y, vis_Y_pseudo], ignore_index=True)

            f1 = _compute_round_f1(vis_X_orig, vis_Y_orig, inv_X_eval, inv_Y_eval, features, config)
            f1_history.append(f1)
            logger.info(f"  Round {round_idx + 1} F1: {f1:.4f}")

        if len(inv_X) == 0:
            logger.info("  No more invisible samples. Stopping.")
            break

    return vis_X, vis_Y, inv_X, f1_history, risk_history


def apply_self_training(vis_X: pd.DataFrame, vis_Y: pd.Series, inv_X: pd.DataFrame,
                        features: List[str], config: Config, inv_Y: Optional[pd.Series] = None) -> tuple:
    """Dispatch to appropriate self-training method based on config.

    Returns:
        tuple: (vis_X, vis_Y, inv_X, f1_history, risk_history) where inv_X maintains its original index.
               f1_history is a list of F1 scores per round (empty if report_each_round_f1 is False).
               risk_history is a list of risk values per round (empty if report_each_round_risk is False).
               The caller can use inv_X.index to sync inv_Y.
    """
    if config.self_training_mode == 'Conf':
        return apply_self_training_conf(vis_X, vis_Y, inv_X, features, config, inv_Y)
    elif config.self_training_mode == 'ConfGeo':
        return apply_self_training_conf_geo(vis_X, vis_Y, inv_X, features, config, inv_Y)
    elif config.self_training_mode == 'Geo':
        return apply_self_training_geo(vis_X, vis_Y, inv_X, features, config, inv_Y)
    else:
        raise ValueError(f"Unknown self_training_mode: {config.self_training_mode}. "
                        f"Must be 'Conf' or 'ConfGeo' or 'Geo'")


# =============================================================================
# Core Functions
# =============================================================================
def preprocess_features(X: pd.DataFrame) -> pd.DataFrame:
    """Encode categorical variables."""
    from sklearn.preprocessing import LabelEncoder
    X_proc = X.copy()
    for col in X_proc.select_dtypes(include=['object']).columns:
        X_proc[col] = LabelEncoder().fit_transform(X_proc[col].astype(str))
    return X_proc


def train_classifier(vis_X: pd.DataFrame, vis_Y: pd.Series, config: Config):
    """Train Random Forest classifier."""
    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(
        n_estimators=config.n_estimators,
        max_depth=config.max_depth,
        random_state=config.random_seed,
        class_weight='balanced'
    )
    clf.fit(vis_X, vis_Y)
    return clf


def evaluate_predictions(vis_Y: pd.Series, inv_Y: pd.Series, inv_Y_pred: np.ndarray) -> Dict[str, Any]:
    """Calculate F1, precision, recall."""
    from sklearn.metrics import f1_score, precision_score, recall_score
    all_Y_pred = np.concatenate([vis_Y.to_numpy(), inv_Y_pred])
    all_Y_true = np.concatenate([vis_Y.to_numpy(), inv_Y.to_numpy()])
    return {
        'f1': f1_score(all_Y_true, all_Y_pred),
        'precision': precision_score(all_Y_true, all_Y_pred),
        'recall': recall_score(all_Y_true, all_Y_pred)
    }


# =============================================================================
# Classification Pipeline
# =============================================================================
def run_classification(vis_X: pd.DataFrame, vis_Y: pd.Series, inv_X: pd.DataFrame,
                       inv_Y: pd.Series, config: Config, methods: List[str]) -> Dict[str, Any]:
    """Run classification with specified method combinations."""
    start_time = time.time()

    # Preprocess
    vis_X_proc = preprocess_features(vis_X)
    inv_X_proc = preprocess_features(inv_X)

    # Store original data before self-training (for reference)
    inv_X_proc_orig = inv_X_proc.copy()
    vis_Y_orig = vis_Y.copy()
    inv_Y_orig = inv_Y.copy()

    # Step 1: Feature selection (if enabled)
    if 'FS' in methods:
        features = apply_feature_selection(vis_X_proc, vis_Y, config)
    else:
        features = vis_X_proc.columns.tolist()
        logger.info(f"  [No Feature Selection] Using all {len(features)} features")

    # Step 2: Self-training (if enabled)
    f1_history = []
    risk_history = []
    if any(m.startswith('ST') for m in methods):
        # Determine self-training mode from methods list
        st_mode = config.self_training_mode
        for m in methods:
            if m.startswith('ST_'):
                st_mode = m.split('_', 1)[1]
                break

        # Temporarily set the mode for this run
        original_mode = config.self_training_mode
        config.self_training_mode = st_mode

        vis_X, vis_Y, inv_X, f1_history, risk_history = apply_self_training(
            vis_X, vis_Y, inv_X, features, config, inv_Y
        )

        # Sync inv_Y with inv_X using the remaining indices
        inv_Y = inv_Y.loc[inv_X.index].reset_index(drop=True)
        inv_X = inv_X.reset_index(drop=True)

        # Restore original mode
        config.self_training_mode = original_mode

        vis_X_proc = preprocess_features(vis_X)
        inv_X_proc = preprocess_features(inv_X)

    # Track final training size
    final_train_size = len(vis_X)

    # Step 3: Final training and prediction
    clf = train_classifier(vis_X_proc[features], vis_Y, config)
    predictions = clf.predict(inv_X_proc_orig[features])


    # Evaluate
    metrics = evaluate_predictions(vis_Y_orig, inv_Y_orig, predictions)
    elapsed = time.time() - start_time

    logger.info(f"  Results -> F1: {metrics['f1']:.4f}, P: {metrics['precision']:.4f}, R: {metrics['recall']:.4f} (Time: {elapsed:.2f}s)\n")

    return {
        'metrics': metrics,
        'features': features,
        'time': elapsed,
        'train_size': final_train_size,
        'f1_history': f1_history,
        'risk_history': risk_history
    }


# =============================================================================
# Experiment Runner
# =============================================================================
def run_experiment(vis_X: pd.DataFrame, vis_Y: pd.Series, inv_X: pd.DataFrame,
                   inv_Y: pd.Series, config: Config) -> List[Dict[str, Any]]:
    """Run all specified method combinations and return results."""
    results = []

    # Get baseline for improvement calculation
    baseline_result = run_classification(
        vis_X.copy(), vis_Y.copy(), inv_X.copy(), inv_Y.copy(),
        config, methods=[]
    )
    baseline_f1 = baseline_result['metrics']['f1']

    # Run each method combination
    for methods in config.methods:
        method_name = " + ".join(methods) if methods else "Baseline"

        logger.info(f"\n[METHOD] {method_name}")
        try:
            result = run_classification(
                vis_X.copy(), vis_Y.copy(), inv_X.copy(), inv_Y.copy(),
                config, methods=methods
            )

            # Handle F1 history: find best round
            f1_history = result.get('f1_history', [])
            best_round = -1  # -1 means no F1 history or empty
            if f1_history:
                best_round = int(np.argmax(f1_history)) + 1  # +1 for 1-based indexing

            result_entry = {
                'Method': method_name,
                'F1': result['metrics']['f1'],
                'P': result['metrics']['precision'],
                'R': result['metrics']['recall'],
                'Feats': len(result['features']),
                'Time': result['time'],
                'TrainSize': result['train_size'],
                'Improvement': (result['metrics']['f1'] - baseline_f1) / baseline_f1 * 100,
                'BestRound': best_round
            }
            logger.info(f"[DEBUG] Appending result: Method='{result_entry['Method']}', F1={result_entry['F1']}")
            results.append(result_entry)
        except Exception as e:
            logger.error(f"[ERROR] Exception running method {method_name}: {e}")
            import traceback
            traceback.print_exc()

    return results


def print_report(all_results: Dict[str, List[Dict]]) -> None:
    """Print centralized report across all workloads."""
    logger.info("\n\n" + "="*129)
    logger.info(" " * 45 + "EXPERIMENT REPORT")
    logger.info("="*129)

    # Prepare data for summary table
    summary_data = []
    for dataset_name, results in all_results.items():
        for result in results:
            summary_data.append({
                'Dataset': dataset_name,
                'Method': result['Method'],
                'F1': result['F1'],
                'Precision': result['P'],
                'Recall': result['R'],
                'Improvement': result['Improvement'],
                'Feats': result['Feats'],
                'Time': result['Time'],
                'TrainSize': result['TrainSize']
            })

    df = pd.DataFrame(summary_data)

    # Get all unique methods across all datasets to maintain consistency
    all_methods = df['Method'].unique().tolist()
    # Sort methods: Baseline first, then alphabetically
    all_methods_sorted = ['Baseline'] + sorted([m for m in all_methods if m != 'Baseline'])
    method_order = all_methods_sorted

    # Calculate average by method
    avg_df = df.groupby('Method').agg({
        'F1': 'mean',
        'Precision': 'mean',
        'Recall': 'mean',
        'Improvement': 'mean',
        'Feats': 'mean',
        'Time': 'mean',
        'TrainSize': 'mean'
    }).reset_index()
    avg_df['Dataset'] = 'AVERAGE'

    # Combine individual results and average
    df_combined = pd.concat([df, avg_df], ignore_index=True)

    # Sort by dataset and maintain method order
    dataset_order = sorted([d for d in df['Dataset'].unique() if d != 'AVERAGE']) + ['AVERAGE']
    df_combined['Dataset'] = pd.Categorical(df_combined['Dataset'], categories=dataset_order, ordered=True)
    df_combined['Method'] = pd.Categorical(df_combined['Method'], categories=method_order, ordered=True)
    df_combined = df_combined.sort_values(['Dataset', 'Method'])

    # Print table
    logger.info(f"\n{'Dataset':<25} {'Method':<20} {'F1':>8} {'Precision':>10} {'Recall':>8} {'Improvement':>12} {'TrainSize':>10} {'Feats':>8} {'Time(s)':>8}")
    logger.info("-" * 129)

    current_dataset = None
    for _, row in df_combined.iterrows():
        if current_dataset is not None and row['Dataset'] != current_dataset:
            logger.info("-" * 129)
        current_dataset = row['Dataset']

        logger.info(f"{row['Dataset']:<25} {row['Method']:<20} {row['F1']:>8.4f} {row['Precision']:>10.4f} {row['Recall']:>8.4f} {row['Improvement']:>10.2f}% {int(row['TrainSize']):>10} {int(row['Feats']):>8} {row['Time']:>8.2f}")

    logger.info("="*129 + "\n")


# =============================================================================
# Argument Parser & Main
# =============================================================================
def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Few-Shot Binary Classification with Self-Training',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run predefined workflows (default behavior)
  python __test_rule_learning.py

  # Run single dataset with custom parameters
  python __test_rule_learning.py --workload medical.Q1 --methods FS ST_ConfGeo

  # Experiment with geometric ST parameters
  python __test_rule_learning.py --workload medical.Q1 --methods ST_ConfGeo --geo-k 10 --geo-weight 0.8 --geo-boundary 0.5

  # Compare multiple methods
  python __test_rule_learning.py --workload medical.Q1 --methods Baseline FS ST_ConfGeo "FS + ST_ConfGeo"

  # Run grid search to find best parameters for all workloads
  python __test_rule_learning.py --grid-search

Available methods:
  Baseline              : No optimization
  FS                    : Feature Selection
  ST_Conf         : Self-Training (confidence-based)
  ST_ConfGeo          : Self-Training (confidence&geometric-based)
  FS + ST_Conf    : Feature Selection + ST (confidence)
  FS + ST_ConfGeo     : Feature Selection + ST (geometric)
        """
    )

    # Data arguments
    parser.add_argument('--workload', type=str, help='Workload name for reporting')
    parser.add_argument('--visible-samples', type=int, default=50, help='Number of visible samples (default: 50)')
    parser.add_argument('--random-seed', type=int, default=42, help='Random seed (default: 42)')

    # Methods
    parser.add_argument('--methods', type=str, nargs='+', default=['Baseline', 'FS', 'ST_Conf', 'ST_ConfGeo', 'FS + ST_Conf', 'FS + ST_ConfGeo'],
                        help='Methods to compare (space-separated). Use quotes for methods with spaces.')

    # Feature selection
    parser.add_argument('--top-k', type=int, default=10, help='Top-k features to select (default: 10)')

    # Self-training
    parser.add_argument('--st-rounds', type=int, default=3, help='Number of self-training rounds (default: 3)')
    parser.add_argument('--max-samples', type=int, default=10, help='Max samples per round (default: 10)')
    parser.add_argument('--balance-classes', action='store_true', default=True, help='Balance classes when selecting samples')

    # ConfGeo ST parameters
    parser.add_argument('--geo-k', type=int, default=5, help='k for k-NN in ConfGeo ST (default: 5)')
    parser.add_argument('--geo-weight', type=float, default=0.6, help='Initial ConfGeo weight (default: 0.6)')
    parser.add_argument('--geo-boundary', type=float, default=0.5, help='Decision boundary for ConfGeo ST (default: 0.5)')

    # Classifier
    parser.add_argument('--n-estimators', type=int, default=100, help='Random forest estimators (default: 100)')
    parser.add_argument('--max-depth', type=int, default=10, help='Random forest max depth (default: 10)')

    # Grid search
    parser.add_argument('--grid-search', action='store_true', help='Run grid search to find best parameters for all workloads')
    parser.add_argument('--optimized-grid', type=str, metavar='WORKLOAD', help='Run optimized grid search with F1/risk tracing (format: WORKLOAD:GRID_TYPE, e.g., medical.Q1:quick)')
    parser.add_argument('--grid-output', type=str, metavar='FILE', help='Output file for grid search results (JSON format)')
    parser.add_argument('--n-jobs', type=int, default=-1, help='Number of parallel jobs for grid search (default: -1 for all cores, use 1 for sequential)')

    return parser.parse_args()


def create_config_from_args(args):
    """Create Config object from command-line arguments."""
    # Parse methods
    method_map = {
        'Baseline': [],
        'FS': ['FS'],
        'ST_Conf': ['ST_Conf'],
        'ST_ConfGeo': ['ST_ConfGeo'],
        'FS + ST_Conf': ['FS', 'ST_Conf'],
        'FS + ST_ConfGeo': ['FS', 'ST_ConfGeo'],
    }

    methods = [method_map[m] for m in args.methods]

    # Detect dataset name from filename if not provided
    _dataset, _query = args.workload.split('.')
    _data_path: str = f"data/{_dataset}/.ckpt/NOPXY__{_query}_full.csv"

    return Config(
        data_path=_data_path,
        visible_samples=args.visible_samples,
        random_seed=args.random_seed,
        top_k=args.top_k,
        n_rounds=args.st_rounds,
        max_samples_per_round=args.max_samples,
        balance_classes=args.balance_classes,
        geo_k_neighbors=args.geo_k,
        geo_initial_weight=args.geo_weight,
        geo_decision_boundary=args.geo_boundary,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        methods=methods
    ), args.workload


# =============================================================================
# Grid Search
# =============================================================================
def grid_search(workload_name: str, data_path: str, param_grid: Dict[str, List[Any]],
                 base_methods: Optional[List[List[str]]] = None, metric: str = 'f1',
                 n_jobs: int = -1) -> Dict[str, Any]:
    """
    Perform grid search to find best configuration for a workload.

    Args:
        workload_name: Name of the workload (e.g., 'medical.Q1')
        data_path: Path to the data file
        param_grid: Dictionary of parameters to search
            {
                'geo_k_neighbors': [5, 10, 15],
                'geo_initial_weight': [0.3, 0.6, 0.8],
                'geo_decision_boundary': [0.4, 0.5, 0.6],
                'max_samples_per_round': [10, 20, 30],
                'n_rounds': [3, 5, 7]
            }
        base_methods: List of method combinations to test (default: ST_ConfGeo only)
        metric: Metric to optimize ('f1', 'precision', 'recall')
        n_jobs: Number of parallel jobs (default: -1 for all cores, use 1 for sequential)

    Returns:
        Dictionary with best_config, best_score, and all_results
    """
    from itertools import product
    from tqdm import tqdm
    from joblib import Parallel, delayed

    if base_methods is None:
        base_methods = [['ST_ConfGeo']]

    logger.info(f"\n{'='*80}")
    logger.info(f"GRID SEARCH: {workload_name}")
    logger.info(f"{'='*80}")
    logger.info(f"Parameters to search: {list(param_grid.keys())}")
    logger.info(f"Optimizing: {metric}")
    logger.info(f"Total combinations: {np.prod([len(v) for v in param_grid.values()])}")
    logger.info(f"Parallel jobs: {n_jobs if n_jobs > 0 else 'All available cores'}")
    logger.info(f"{'='*80}\n")

    # Load and split data
    df = pd.read_csv(data_path)
    Y, X = df['label'], df.drop(columns=['label', 'patient_id'])

    # Use fixed seed for reproducibility
    np.random.seed(42)
    vis_idx = X.sample(n=50).index
    inv_idx = X.index.difference(vis_idx)

    vis_X, inv_X = X.loc[vis_idx].reset_index(drop=True), X.loc[inv_idx].reset_index(drop=True)
    vis_Y, inv_Y = Y.loc[vis_idx].reset_index(drop=True), Y.loc[inv_idx].reset_index(drop=True)

    # Get baseline F1
    baseline_result = run_classification(
        vis_X.copy(), vis_Y.copy(), inv_X.copy(), inv_Y.copy(),
        Config(data_path=data_path, visible_samples=50, random_seed=42),
        methods=[]
    )
    baseline_f1 = baseline_result['metrics']['f1']

    # Generate all parameter combinations
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    all_combinations = list(product(*param_values))

    # Single combination evaluation function
    def evaluate_combination(idx_combination):
        idx, combination = idx_combination
        try:
            # Create config for this combination
            config = Config(
                data_path=data_path,
                visible_samples=50,
                random_seed=42,
                methods=base_methods,
                **{param_names[i]: combination[i] for i in range(len(param_names))}
            )

            # Run experiment
            result = run_classification(
                vis_X.copy(), vis_Y.copy(), inv_X.copy(), inv_Y.copy(),
                config, methods=base_methods[0]
            )

            score = result['metrics'][metric]
            improvement = (score - baseline_f1) / baseline_f1 * 100

            # Store result
            result_dict = {
                'iteration': idx,
                'params': {param_names[i]: combination[i] for i in range(len(param_names))},
                'f1': result['metrics']['f1'],
                'precision': result['metrics']['precision'],
                'recall': result['metrics']['recall'],
                'train_size': result['train_size'],
                'improvement': improvement
            }

            return result_dict, score
        except Exception as e:
            logger.warning(f"Combination {idx} failed: {e}")
            return None

    # Run grid search
    logger.info(f"Testing {len(all_combinations)} combinations with n_jobs={n_jobs}...\n")

    # Prepare all tasks
    tasks = [(idx, combo) for idx, combo in enumerate(all_combinations, 1)]

    # Execute with progress bar
    from tqdm import tqdm
    results = Parallel(n_jobs=n_jobs)(
        delayed(evaluate_combination)(task)
        for task in tqdm(tasks, desc="Grid Search", unit="comb")
    )

    # Extract results (filter out any None values from failures)
    all_results = [r[0] for r in results if r is not None]
    scores = [r[1] for r in results if r is not None]

    # Find best
    best_idx = np.argmax(scores)
    best_result = all_results[best_idx]
    best_score = scores[best_idx]

    # Print summary
    logger.info(f"\n{'='*80}")
    logger.info(f"GRID SEARCH COMPLETE: {workload_name}")
    logger.info(f"{'='*80}")
    logger.info(f"Best {metric}: {best_score:.4f}")
    logger.info(f"Improvement over baseline: {best_result['improvement']:+.2f}%")
    logger.info(f"Best parameters:")
    for param, value in best_result['params'].items():
        logger.info(f"  {param}: {value}")
    logger.info(f"Train size: {best_result['train_size']}")
    logger.info(f"{'='*80}\n")

    return {
        'workload': workload_name,
        'baseline_f1': baseline_f1,
        'best_score': best_score,
        'best_improvement': best_result['improvement'],
        'best_config': best_result['params'],
        'best_train_size': best_result['train_size'],
        'all_results': all_results
    }

def optimized_grid_search(workload_name: str, data_path: str, param_grid: Dict[str, List[Any]],
                           methods_list: List[List[str]], metric: str = 'f1',
                           n_jobs: int = -1, output_file: Optional[str] = None) -> Dict[str, Any]:
    """
    Optimized grid search with parallelized Phase 1 (F1 tracing)
    and parallelized Phase 2 (risk tracing).
    """
    from itertools import product
    from datetime import datetime
    from tqdm import tqdm
    from joblib import Parallel, delayed
    import json

    logger.info(f"\n{'='*80}")
    logger.info(f"OPTIMIZED GRID SEARCH: {workload_name}")
    logger.info(f"{'='*80}")
    logger.info(f"Methods to test: {['+'.join(m) for m in methods_list]}")
    logger.info(f"Parameters to search: {list(param_grid.keys())}")
    logger.info(f"Optimizing: {metric}")
    total_combos = np.prod([len(v) for v in param_grid.values()]) * len(methods_list)
    logger.info(f"Total combinations: {total_combos}")
    logger.info(f"Parallel jobs: {n_jobs if n_jobs > 0 else 'All available cores'}")
    logger.info(f"{'='*80}\n")

    # Load and split data
    df = pd.read_csv(data_path)
    Y, X = df['label'], df.drop(columns=['label', 'patient_id'])

    # Use fixed seed for reproducibility
    np.random.seed(42)
    vis_idx = X.sample(n=50).index
    inv_idx = X.index.difference(vis_idx)

    vis_X, inv_X = X.loc[vis_idx].reset_index(drop=True), X.loc[inv_idx].reset_index(drop=True)
    vis_Y, inv_Y = Y.loc[vis_idx].reset_index(drop=True), Y.loc[inv_idx].reset_index(drop=True)

    # Get baseline F1
    baseline_result = run_classification(
        vis_X.copy(), vis_Y.copy(), inv_X.copy(), inv_Y.copy(),
        Config(data_path=data_path, visible_samples=50, random_seed=42),
        methods=[]
    )
    baseline_f1 = baseline_result['metrics']['f1']
    logger.info(f"Baseline F1: {baseline_f1:.4f}\n")

    # Generate all parameter combinations
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    all_param_combinations = list(product(*param_values))

    # Create all tasks: (method, params) combinations
    all_tasks = []
    for method in methods_list:
        for params in all_param_combinations:
            all_tasks.append((method, params))

    # Phase 1: Grid search with F1 tracing
    def evaluate_with_f1_tracing(task_info):
        method, params = task_info
        try:
            # Create config with report_each_round_f1=True
            config = Config(
                data_path=data_path,
                visible_samples=50,
                random_seed=42,
                report_each_round_f1=True,
                **{param_names[i]: params[i] for i in range(len(param_names))}
            )

            # Run experiment
            result = run_classification(
                vis_X.copy(), vis_Y.copy(), inv_X.copy(), inv_Y.copy(),
                config, methods=method
            )

            f1_history = result.get('f1_history', [])
            assert f1_history, f"No F1 history for method {'+'.join(method)} with params {params}"

            # Find best round
            best_round_idx = int(np.argmax(f1_history))  # 0-indexed
            best_f1 = f1_history[best_round_idx]
            best_n_rounds = best_round_idx + 1  # Convert to 1-indexed

            return {
                'method': method,
                'method_name': '+'.join(method),
                'params': {param_names[i]: params[i] for i in range(len(param_names))},
                'f1_history': f1_history,
                'best_f1': best_f1,
                'best_n_rounds': best_n_rounds,
                'final_f1': result['metrics']['f1'],
                'precision': result['metrics']['precision'],
                'recall': result['metrics']['recall'],
                'train_size': result['train_size']
            }
        except Exception as e:
            logger.warning(f"Task failed for method {'+'.join(method)} with params {params}: {e}")
            return None

    # Run Phase 1
    logger.info(f"Phase 1: Testing {len(all_tasks)} combinations with F1 tracing...\n")

    results = Parallel(n_jobs=n_jobs)(
        delayed(evaluate_with_f1_tracing)(task)
        for task in tqdm(all_tasks, desc="Phase 1: F1 Grid Search", unit="task")
    )

    # Filter out failed results
    phase1_results = [r for r in results if r is not None]

    # Find best config for each method
    method_best_configs = {}
    for method in methods_list:
        method_name = '+'.join(method)
        method_results = [r for r in phase1_results if r['method_name'] == method_name]

        if not method_results:
            logger.warning(f"No successful results for method {method_name}")
            continue

        # Find result with highest best_f1
        best = max(method_results, key=lambda x: x['best_f1'])
        method_best_configs[method_name] = best

        logger.info(f"\nBest config for {method_name}:")
        logger.info(f"  Best F1: {best['best_f1']:.4f} at round {best['best_n_rounds']}")
        logger.info(f"  Parameters:")
        for k, v in best['params'].items():
            logger.info(f"    {k}: {v}")

    # ------------------------------------------------------------------
    # Phase 2: Risk evaluation (PARALLELIZED)
    # ------------------------------------------------------------------
    def evaluate_with_risk(method_name, best_config):
        try:
            # Create config with best_n_rounds and report_each_round_risk=True
            config_params = best_config['params'].copy()
            config_params['n_rounds'] = best_config['best_n_rounds']
            config_params['report_each_round_risk'] = True

            config = Config(
                data_path=data_path,
                visible_samples=50,
                random_seed=42,
                **config_params
            )

            # Run experiment
            result = run_classification(
                vis_X.copy(), vis_Y.copy(), inv_X.copy(), inv_Y.copy(),
                config, methods=best_config['method']
            )

            risk_history = result.get('risk_history', [])

            risk_results[method_name] = {
                'best_config': best_config,
                'risk_history': risk_history,
                'final_metrics': {
                    'f1': result['metrics']['f1'],
                    'precision': result['metrics']['precision'],
                    'recall': result['metrics']['recall']
                }
            }

            logger.info(f"\nRisk trace for {method_name}:")
            for round_idx, risk in enumerate(risk_history):
                logger.info(f"  Round {round_idx + 1}: {risk:.4f}")

        except Exception as e:
            logger.error(f"Phase 2 failed for {method_name}: {e}")
            return None

    logger.info(f"\n{'='*80}")
    logger.info("Phase 2: Computing risk traces for best configurations...\n")

    phase2_tasks = list(method_best_configs.items())

    phase2_results = Parallel(n_jobs=n_jobs)(
        delayed(evaluate_with_risk)(method_name, best_config)
        for method_name, best_config in tqdm(
            phase2_tasks,
            desc="Phase 2: Risk Evaluation",
            unit="method"
        )
    )

    risk_results = {
        r['method_name']: {
            'best_config': r['best_config'],
            'risk_history': r['risk_history'],
            'final_metrics': r['final_metrics']
        }
        for r in phase2_results
        if r is not None
    }

    # Logging after parallel execution
    for method_name, r in risk_results.items():
        logger.info(f"\nRisk trace for {method_name}:")
        for i, risk in enumerate(r['risk_history']):
            logger.info(f"  Round {i + 1}: {risk:.4f}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    summary = {
        'workload': workload_name,
        'timestamp': datetime.now().isoformat(),
        'baseline_f1': baseline_f1,
        'methods': {}
    }

    for method_name, r in risk_results.items():
        best = r['best_config']
        summary['methods'][method_name] = {
            'best_parameters': best['params'],
            'best_n_rounds': best['best_n_rounds'],
            'best_f1': best['best_f1'],
            'f1_trace': best['f1_history'],
            'risk_trace': r['risk_history'],
            'final_metrics': r['final_metrics']
        }

    logger.info(f"\n{'='*80}")
    logger.info(f"OPTIMIZED GRID SEARCH COMPLETE: {workload_name}")
    logger.info(f"{'='*80}")

    for method_name, m in summary['methods'].items():
        logger.info(f"\n{method_name}:")
        logger.info(f"  Best F1: {m['best_f1']:.4f} (round {m['best_n_rounds']})")
        logger.info(
            f"  Improvement: {(m['best_f1'] - baseline_f1) / baseline_f1 * 100:+.2f}%"
        )
        logger.info("  Best parameters:")
        for k, v in m['best_parameters'].items():
            if k != 'n_rounds':
                logger.info(f"    {k}: {v}")
        logger.info(f"  Final F1: {m['final_metrics']['f1']:.4f}")
        logger.info(f"  Final P: {m['final_metrics']['precision']:.4f}")
        logger.info(f"  Final R: {m['final_metrics']['recall']:.4f}")

    logger.info(f"\n{'='*80}\n")

    if output_file:
        with open(output_file, 'w') as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Results saved to: {output_file}\n")

    return summary


def run_optimized_grid_search(workload_name: str, data_path: str,
                               grid_type: str = 'comprehensive', n_jobs: int = -1,
                               output_file: Optional[str] = None) -> Dict[str, Any]:
    """
    Convenience function to run optimized grid search with predefined parameter grids.

    Args:
        workload_name: Name of the workload (e.g., 'medical.Q1')
        data_path: Path to the data file
        grid_type: Type of parameter grid ('quick', 'comprehensive', 'severe_imbalance', etc.)
        n_jobs: Number of parallel jobs
        output_file: Path to save results (optional)

    Returns:
        Summary dictionary with best configs and traces
    """
    # Define parameter grids
    param_grids = {
        'quick': {
            'geo_k_neighbors': [5, 10],
            'geo_initial_weight': [0.5, 0.8],
            'geo_decision_boundary': [0.4, 0.5],
            'max_samples_per_round': [10, 20],
            'n_rounds': [10]  # Upper bound
        },
        'comprehensive': {
            'geo_k_neighbors': [10],
            'geo_initial_weight': [0.5],
            'geo_decision_boundary': [0.1],
            'max_samples_per_round': [40],
            'n_rounds': [20]  # Upper bound
        },
        # 'comprehensive': {
        #     'geo_k_neighbors': [5, 10, 15, 20],
        #     'geo_initial_weight': [0.1, 0.3, 0.5, 0.7, 0.9],
        #     'geo_decision_boundary': [0.1, 0.3, 0.5, 0.7, 0.9],
        #     'max_samples_per_round': [10, 20, 30, 40, 50],
        #     'n_rounds': [20]  # Upper bound
        # },
        'severe_imbalance': {
            'geo_k_neighbors': [5, 10, 15],
            'geo_initial_weight': [0.5, 0.7, 0.9],
            'geo_decision_boundary': [0.3, 0.4, 0.5],
            'max_samples_per_round': [10, 15, 20],
            'n_rounds': [10]  # Upper bound
        },
        'balanced': {
            'geo_k_neighbors': [5, 10, 15],
            'geo_initial_weight': [0.3, 0.5, 0.7],
            'geo_decision_boundary': [0.4, 0.5, 0.6],
            'max_samples_per_round': [15, 25, 35],
            'n_rounds': [10]  # Upper bound
        },
        'high_baseline': {
            'geo_k_neighbors': [5, 10],
            'geo_initial_weight': [0.3, 0.5],
            'geo_decision_boundary': [0.5, 0.6],
            'max_samples_per_round': [20, 30, 40],
            'n_rounds': [10]  # Upper bound
        }
    }

    # Define methods to test
    methods_list = [
        # ['ST_Conf'],
        # ['ST_Geo'],
        # ['ST_ConfGeo'],
        # ['FS', 'ST_Conf'],
        # ['FS', 'ST_Geo'],
        ['FS', 'ST_ConfGeo'],
    ]

    if grid_type not in param_grids:
        raise ValueError(f"Unknown grid_type: {grid_type}. Choose from: {list(param_grids.keys())}")

    param_grid = param_grids[grid_type]

    return optimized_grid_search(
        workload_name=workload_name,
        data_path=data_path,
        param_grid=param_grid,
        methods_list=methods_list,
        n_jobs=n_jobs,
        output_file=output_file
    )


def grid_search_workloads(n_jobs: int = -1):
    """
    Run grid search on multiple workloads with predefined parameter grids.

    Args:
        n_jobs: Number of parallel jobs (default: -1 for all cores)
    """
    workloads = [
        ('medical.Q1', 'data/medical/.ckpt/NOPXY__Q1_full.csv'),
        ('medical.Q3', 'data/medical/.ckpt/NOPXY__Q3_full.csv'),
        ('medical.Q8', 'data/medical/.ckpt/NOPXY__Q8_full.csv'),
    ]

    # Define parameter grids for different scenarios
    param_grids = {
        # For severely imbalanced datasets (Q1: ~10% positive)
        'severe_imbalance': {
            'geo_k_neighbors': [5, 10, 15],
            'geo_initial_weight': [0.5, 0.7, 0.9],
            'geo_decision_boundary': [0.3, 0.4, 0.5],
            'max_samples_per_round': [10, 15, 20],
            'n_rounds': [3, 5]
        },

        # For balanced datasets (Q8: ~45% positive)
        'balanced': {
            'geo_k_neighbors': [5, 10, 15],
            'geo_initial_weight': [0.3, 0.5, 0.7],
            'geo_decision_boundary': [0.4, 0.5, 0.6],
            'max_samples_per_round': [15, 25, 35],
            'n_rounds': [3, 5, 7]
        },

        # For high baseline / reverse imbalance (Q3: ~79% positive)
        'high_baseline': {
            'geo_k_neighbors': [5, 10],
            'geo_initial_weight': [0.3, 0.5],
            'geo_decision_boundary': [0.5, 0.6],
            'max_samples_per_round': [20, 30, 40],
            'n_rounds': [5, 7]
        },

        # Quick search (fewer combinations)
        'quick': {
            'geo_k_neighbors': [5, 10],
            'geo_initial_weight': [0.5, 0.8],
            'geo_decision_boundary': [0.4, 0.5],
            'max_samples_per_round': [10, 20],
            'n_rounds': [3, 5]
        },

        # A comprehensive grid
        'comprehensive': {
            'geo_k_neighbors': [5, 10, 15, 20],
            'geo_initial_weight': [0.1, 0.3, 0.5, 0.7, 0.9],
            'geo_decision_boundary': [0.1, 0.3, 0.5, 0.7, 0.9],
            'max_samples_per_round': [10, 20, 30, 40, 50],
            'n_rounds': [5]
        }

    }

    # Map workloads to appropriate parameter grids
    workload_grids = {
        'medical.Q1': 'comprehensive',
        'medical.Q3': 'comprehensive',
        'medical.Q8': 'comprehensive'
    }
    # workload_grids = {
    #     'medical.Q1': 'severe_imbalance',
    #     'medical.Q3': 'high_baseline',
    #     'medical.Q8': 'balanced'
    # }

    all_search_results = {}

    # Define methods to test
    method_combinations = [
        # [],
        # ['FS'],
        # ['ST_Conf'],
        # ['ST_ConfGeo'],
        # ['ST_Geo'],
        # ['FS', 'ST_Conf'],
        # ['FS', 'ST_ConfGeo']
        ['FS', 'ST_Geo']
    ]

    for workload_name, data_path in workloads:
        grid_type = workload_grids[workload_name]
        param_grid = param_grids[grid_type]

        # Test each method combination separately
        for methods in method_combinations:
            method_name = '+'.join(methods) if methods else 'Baseline'
            logger.info(f"\nTesting {workload_name} with method: {method_name}")

            result = grid_search(
                workload_name=f"{workload_name}_{method_name}",
                data_path=data_path,
                param_grid=param_grid,
                base_methods=[methods],  # Wrap in list since grid_search expects list of method lists
                metric='f1',
                n_jobs=n_jobs
            )

            all_search_results[f"{workload_name}_{method_name}"] = result

    # Print comparison table
    print_grid_search_summary(all_search_results)


def print_grid_search_summary(all_results: Dict[str, Dict[str, Any]]) -> None:
    """Print summary of grid search results across all workloads."""
    logger.info("\n" + "="*100)
    logger.info("GRID SEARCH SUMMARY".center(70))
    logger.info("="*100)

    # Group by workload
    workloads = {}
    for key, result in all_results.items():
        # Split key like "medical.Q1_Baseline" into workload and method
        if '_' in key:
            parts = key.split('_', 1)
            workload = parts[0]
            method = parts[1]
        else:
            workload = key
            method = 'Unknown'

        if workload not in workloads:
            workloads[workload] = []
        workloads[workload].append({
            'method': method,
            'baseline': result['baseline_f1'],
            'best_f1': result['best_score'],
            'improvement': result['best_improvement'],
            'config': result['best_config'],
            'train_size': result['best_train_size']
        })

    # Print each workload section
    for workload_name in sorted(workloads.keys()):
        results = workloads[workload_name]
        baseline = results[0]['baseline']

        # Sort by F1 score descending
        results.sort(key=lambda x: x['best_f1'], reverse=True)

        logger.info(f"\n{workload_name} (Baseline F1: {baseline:.4f})")
        logger.info("-" * 100)
        logger.info(f"{'Method':<25} {'F1':>8} {'Improvement':>12} {'K':>4} {'Weight':>6} {'Boundary':>8} {'Samples':>8} {'Rounds':>6} {'Train':>6}")
        logger.info("-" * 100)

        for r in results:
            method = r['method']
            f1 = r['best_f1']
            imp = r['improvement']
            config = r['config']
            train_size = r['train_size']

            logger.info(f"{method:<25} {f1:>8.4f} {imp:>+11.2f}% {config['geo_k_neighbors']:>4} "
                       f"{config['geo_initial_weight']:>6.1f} {config['geo_decision_boundary']:>8.1f} "
                       f"{config['max_samples_per_round']:>8} {config['n_rounds']:>6} {train_size:>6}")

    logger.info("\n" + "="*100 + "\n")


def run_predefined_workloads():
    """Run predefined workflow experiments."""
    # Define workloads with methods to run
    # Methods:
    #   [] = Baseline
    #   ['FS'] = Feature Selection
    #   ['ST_Conf'] = Self-Training (confidence-based)
    #   ['ST_ConfGeo'] = Self-Training (geometric-based)
    #   ['FS', 'ST_Conf'] = FS + ST (confidence)
    #   ['FS', 'ST_ConfGeo'] = FS + ST (geometric)
    workloads = [
        # Compare all self-training methods on each dataset
        ("medical_Q1", Config(
            data_path="data/medical/.ckpt/NOPXY__Q1_full.csv",
            visible_samples=50,
            random_seed=42,
            geo_k_neighbors=10,
            geo_initial_weight=0.5,
            geo_decision_boundary=0.1,
            max_samples_per_round=40,
            n_rounds=1,
            report_each_round_f1=True,
            methods=[
                # [],
                # ['ST_Geo'],
                # ['ST_ConfGeo'],
                # ['FS', 'ST_Geo'],
                ['FS', 'ST_ConfGeo']
            ]
        )),
        # ("medical_Q3", Config(
        #     data_path="data/medical/.ckpt/NOPXY__Q3_full.csv",
        #     visible_samples=50,
        #     random_seed=42,
        #     geo_k_neighbors=5,
        #     geo_initial_weight=0.1,
        #     geo_decision_boundary=0.1,
        #     max_samples_per_round=20,
        #     n_rounds=4,
        #     methods=[
        #         [],
        #         ['FS'],
        #         ['ST_Conf'],
        #         ['ST_ConfGeo'],
        #         ['FS', 'ST_Conf'],
        #         ['FS', 'ST_ConfGeo']
        #     ]
        # )),
        # ("medical_Q8", Config(
        #     data_path="data/medical/.ckpt/NOPXY__Q8_full.csv",
        #     visible_samples=50,
        #     random_seed=42,
        #     geo_k_neighbors=5,
        #     geo_initial_weight=0.7,
        #     geo_decision_boundary=0.1,
        #     max_samples_per_round=50,
        #     n_rounds=16,
        #     methods=[
        #         [],
        #         ['FS'],
        #         ['ST_Conf'],
        #         ['ST_ConfGeo'],
        #         ['FS', 'ST_Conf'],
        #         ['FS', 'ST_ConfGeo']
        #     ]
        # )),
    ]

    # Store all results
    all_results = {}

    # Run experiments
    for i, (dataset_name, config) in enumerate(workloads, 1):
        logger.info(f"\n\n{'#'*80}")
        logger.info(f"# WORKLOAD {i}/{len(workloads)}: {dataset_name}")
        logger.info(f"{'#'*80}\n")

        # Load and split data
        df = pd.read_csv(config.data_path)
        Y, X = df['label'], df.drop(columns=['label', 'patient_id'])

        np.random.seed(config.random_seed)
        vis_idx = X.sample(n=config.visible_samples).index
        inv_idx = X.index.difference(vis_idx)

        vis_X, inv_X = X.loc[vis_idx].reset_index(drop=True), X.loc[inv_idx].reset_index(drop=True)
        vis_Y, inv_Y = Y.loc[vis_idx].reset_index(drop=True), Y.loc[inv_idx].reset_index(drop=True)

        logger.info(f"Data: {len(vis_X)} visible, {len(inv_X)} invisible")
        logger.info(f"Methods: {config.methods}\n")

        # Run experiment
        results = run_experiment(vis_X, vis_Y, inv_X, inv_Y, config)
        all_results[dataset_name] = results

    # Print centralized report
    print_report(all_results)


def run_custom_experiment(args):
    """Run custom experiment with command-line parameters."""
    config, dataset_name = create_config_from_args(args)

    logger.info(f"\n\n{'#'*80}")
    logger.info(f"# CUSTOM EXPERIMENT: {dataset_name}")
    logger.info(f"{'#'*80}\n")
    logger.info(f"Configuration:")
    logger.info(f"  Data: {config.data_path}")
    logger.info(f"  Visible samples: {config.visible_samples}")
    logger.info(f"  Methods: {args.methods}")
    logger.info(f"  ST rounds: {config.n_rounds}")
    logger.info(f"  Max samples/round: {config.max_samples_per_round}")
    logger.info(f"  Geo k-neighbors: {config.geo_k_neighbors}")
    logger.info(f"  Geo initial weight: {config.geo_initial_weight}")
    logger.info(f"  Top-k features: {config.top_k}")
    logger.info(f"  Random seed: {config.random_seed}\n")

    # Load and split data
    df = pd.read_csv(config.data_path)
    Y, X = df['label'], df.drop(columns=['label', 'patient_id'])

    np.random.seed(config.random_seed)
    vis_idx = X.sample(n=config.visible_samples).index
    inv_idx = X.index.difference(vis_idx)

    vis_X, inv_X = X.loc[vis_idx].reset_index(drop=True), X.loc[inv_idx].reset_index(drop=True)
    vis_Y, inv_Y = Y.loc[vis_idx].reset_index(drop=True), Y.loc[inv_idx].reset_index(drop=True)

    logger.info(f"Data: {len(vis_X)} visible, {len(inv_X)} invisible")
    logger.info(f"Methods: {config.methods}\n")

    # Run experiment
    results = run_experiment(vis_X, vis_Y, inv_X, inv_Y, config)

    # Print report
    print_report({dataset_name: results})


if __name__ == "__main__":

    # Check if any command-line arguments are provided (excluding script name)
    args = parse_args()

    # Check for --optimized-grid flag
    if args.optimized_grid:
        # Parse workload:grid_type format
        parts = args.optimized_grid.split(':')
        workload_name = parts[0]
        grid_type = parts[1] if len(parts) > 1 else 'quick'

        # Map workload name to data path
        workload_paths = {
            'medical.Q1': 'data/medical/.ckpt/NOPXY__Q1_full.csv',
            'medical.Q3': 'data/medical/.ckpt/NOPXY__Q3_full.csv',
            'medical.Q8': 'data/medical/.ckpt/NOPXY__Q8_full.csv',
        }

        if workload_name not in workload_paths:
            logger.error(f"Unknown workload: {workload_name}. Choose from: {list(workload_paths.keys())}")
            sys.exit(1)

        data_path = workload_paths[workload_name]
        logger.info(f"Running optimized grid search for {workload_name} with grid type '{grid_type}'...\n")

        run_optimized_grid_search(
            workload_name=workload_name,
            data_path=data_path,
            grid_type=grid_type,
            n_jobs=args.n_jobs,
            output_file=args.grid_output
        )
    # Check for --grid-search flag
    elif args.grid_search:
        logger.info("Running grid search on all workloads...\n")
        grid_search_workloads(n_jobs=args.n_jobs)
    elif len(sys.argv) == 1 or (len(sys.argv) > 1 and '--workload' not in sys.argv):
        # No arguments or no --workload flag: run predefined workflows
        if len(sys.argv) > 1:
            logger.info("No --workload argument provided. Running predefined workflows.\n")
        run_predefined_workloads()
    else:
        # Custom experiment mode
        if args.workload is None:
            logger.error("Error: --workload argument is required for custom experiments.")
            logger.info("Use --help to see usage information.")
            sys.exit(1)
        run_custom_experiment(args)
