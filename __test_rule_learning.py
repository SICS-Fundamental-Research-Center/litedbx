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
from typing import List, Dict, Any, Optional
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


def apply_self_training_conf(vis_X: pd.DataFrame, vis_Y: pd.Series, inv_X: pd.DataFrame,
                                    features: List[str], config: Config) -> tuple:
    """Apply self-training to augment training data.

    Returns:
        tuple: (vis_X, vis_Y, inv_X) where inv_X maintains its original index.
               The caller can use inv_X.index to sync inv_Y.
    """
    vis_X_proc = preprocess_features(vis_X)
    inv_X_proc = preprocess_features(inv_X)
    rng = np.random.default_rng(config.random_seed)

    logger.info(f"  [Self-Training] Pos={vis_Y.sum()}, Neg={len(vis_Y) - vis_Y.sum()}")

    for round_idx in range(config.n_rounds):
        logger.info(f"  Round {round_idx + 1}/{config.n_rounds} | Vis: {len(vis_X)} Inv: {len(inv_X)}")

        clf = train_classifier(vis_X_proc[features], vis_Y, config)
        probas = clf.predict_proba(inv_X_proc[features])[:, 1]
        predictions = clf.predict(inv_X_proc[features])

        conf_mask_pos = (probas >= config.confidence_threshold)
        conf_mask_neg = (probas <= (1 - config.confidence_threshold))
        high_conf_mask = conf_mask_pos | conf_mask_neg

        if high_conf_mask.sum() == 0:
            logger.info("  No samples above threshold. Stopping.")
            break

        # Select samples
        if config.balance_classes:
            pos_idx = np.where(conf_mask_pos)[0]
            neg_idx = np.where(conf_mask_neg)[0]

            if len(pos_idx) == 0 or len(neg_idx) == 0:
                n = min(high_conf_mask.sum(), config.max_samples_per_round)
                selected = rng.choice(np.where(high_conf_mask)[0], n, replace=False)
            else:
                n_per_class = min(len(pos_idx), len(neg_idx), config.max_samples_per_round // 2)
                if n_per_class == 0:
                    n_per_class = min(len(pos_idx), len(neg_idx))
                selected_pos = rng.choice(pos_idx, n_per_class, replace=False)
                selected_neg = rng.choice(neg_idx, n_per_class, replace=False)
                selected = np.concatenate([selected_pos, selected_neg])
        else:
            n = min(high_conf_mask.sum(), config.max_samples_per_round)
            selected = rng.choice(np.where(high_conf_mask)[0], n, replace=False)

        # Update datasets 
        vis_X = pd.concat([vis_X, inv_X.iloc[selected]], ignore_index=True)
        vis_Y = pd.concat([vis_Y, pd.Series(predictions[selected])], ignore_index=True)
        vis_X_proc = pd.concat([vis_X_proc, inv_X_proc.iloc[selected]], ignore_index=True)
        inv_X = inv_X.drop(inv_X.index[selected])
        inv_X_proc = inv_X_proc.drop(inv_X_proc.index[selected])

        if len(inv_X) == 0:
            logger.info("  No more invisible samples. Stopping.")
            break

    return vis_X, vis_Y, inv_X


def apply_self_training_geo(vis_X: pd.DataFrame, vis_Y: pd.Series, inv_X: pd.DataFrame,
                          features: List[str], config: Config) -> tuple:
    """Apply geometric self-training using ONLY clustering (k-NN distances), no confidence scores.

    This method selects samples based purely on geometric proximity to labeled samples
    using k-NN distances weighted by feature importance.

    Returns:
        tuple: (vis_X, vis_Y, inv_X) where inv_X maintains its original index.
               The caller can use inv_X.index to sync inv_Y.
    """
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import StandardScaler

    vis_X_proc = preprocess_features(vis_X)
    inv_X_proc = preprocess_features(inv_X)

    logger.info(f"  [Self-Training: Geo-Only] Pos={vis_Y.sum()}, Neg={len(vis_Y) - vis_Y.sum()}")

    for round_idx in range(config.n_rounds):
        logger.info(f"  Round {round_idx + 1}/{config.n_rounds} | Vis: {len(vis_X)} Inv: {len(inv_X)}")

        # Get feature importances from current classifier
        clf_temp = train_classifier(vis_X_proc[features], vis_Y, config)
        feature_importances = clf_temp.feature_importances_

        # Create feature weights based on importance
        feature_weights = np.ones(len(features))
        for i, feat in enumerate(features):
            if feat in vis_X_proc.columns:
                idx = list(vis_X_proc.columns).index(feat)
                if idx < len(feature_importances):
                    feature_weights[i] = np.sqrt(feature_importances[idx])

        # Normalize features
        scaler = StandardScaler()
        vis_X_norm = scaler.fit_transform(vis_X_proc[features])
        inv_X_norm = scaler.transform(inv_X_proc[features])

        # Apply feature importance weights
        vis_X_weighted = vis_X_norm * feature_weights
        inv_X_weighted = inv_X_norm * feature_weights

        # Separate visible samples by class
        pos_idx = vis_Y[vis_Y == 1].index
        neg_idx = vis_Y[vis_Y == 0].index

        if len(pos_idx) == 0 or len(neg_idx) == 0:
            logger.info("  Not enough samples of both classes. Stopping.")
            break

        # Compute k-NN distances to each class
        pos_samples = vis_X_weighted[pos_idx]
        neg_samples = vis_X_weighted[neg_idx]

        n_neighbors = min(config.geo_k_neighbors, len(pos_idx), len(neg_idx))
        knn_pos = NearestNeighbors(n_neighbors=n_neighbors).fit(pos_samples)
        knn_neg = NearestNeighbors(n_neighbors=n_neighbors).fit(neg_samples)

        dist_pos, _ = knn_pos.kneighbors(inv_X_weighted)
        dist_neg, _ = knn_neg.kneighbors(inv_X_weighted)

        # Calculate geometric scores: inverse of average distance
        # Closer to positive class → higher positive score
        # Closer to negative class → higher negative score
        epsilon = 1e-6
        geo_score_pos = 1.0 / (dist_pos.mean(axis=1) + epsilon)
        geo_score_neg = 1.0 / (dist_neg.mean(axis=1) + epsilon)

        # Convert scores to probabilities using softmax-like normalization
        total_score = geo_score_pos + geo_score_neg
        geo_prob_pos = geo_score_pos / total_score

        # Make predictions based PURELY on geometric proximity
        predictions = (geo_prob_pos >= config.geo_decision_boundary).astype(int)

        # Select samples with HIGHEST geometric confidence
        # Geometric confidence = distance from decision boundary
        geo_confidence = np.abs(geo_prob_pos - config.geo_decision_boundary)

        if config.balance_classes:
            # Separate by predicted class
            pos_pred_idx = np.where(predictions == 1)[0]
            neg_pred_idx = np.where(predictions == 0)[0]

            if len(pos_pred_idx) == 0 or len(neg_pred_idx) == 0:
                # If only one class predicted, select from that class
                available_idx = pos_pred_idx if len(pos_pred_idx) > 0 else neg_pred_idx
                n = min(len(available_idx), config.max_samples_per_round)
                if n == 0:
                    logger.info("  No samples available. Stopping.")
                    break
                selected = available_idx[np.argsort(geo_confidence[available_idx])[-n:]]
                logger.info(f"  Selected: {n} samples (one class only)")
            else:
                # Balance classes: select equal number from each
                n_per_class = min(len(pos_pred_idx), len(neg_pred_idx), config.max_samples_per_round // 2)
                if n_per_class == 0:
                    n_per_class = min(len(pos_pred_idx), len(neg_pred_idx))
                # Select samples with HIGHEST geometric confidence within each class
                selected_pos = pos_pred_idx[np.argsort(geo_confidence[pos_pred_idx])[-n_per_class:]]
                selected_neg = neg_pred_idx[np.argsort(geo_confidence[neg_pred_idx])[-n_per_class:]]
                selected = np.concatenate([selected_pos, selected_neg])
                logger.info(f"  Selected: {len(selected_pos)} pos + {len(selected_neg)} neg")
        else:
            n = min(len(inv_X), config.max_samples_per_round)
            selected = np.argsort(geo_confidence)[-n:]
            logger.info(f"  Selected: {n} samples")

        # Update datasets (do NOT reset index for inv_X/inv_X_proc)
        vis_X = pd.concat([vis_X, inv_X.iloc[selected]], ignore_index=True)
        new_labels = pd.Series(predictions[selected], dtype=vis_Y.dtype)
        logger.info(f"  Adding {len(new_labels)} new labels: {new_labels.value_counts().to_dict()}")
        vis_Y = pd.concat([vis_Y, new_labels], ignore_index=True).astype(vis_Y.dtype)
        vis_X_proc = pd.concat([vis_X_proc, inv_X_proc.iloc[selected]], ignore_index=True)
        inv_X = inv_X.drop(inv_X.index[selected])
        inv_X_proc = inv_X_proc.drop(inv_X_proc.index[selected])

        if len(inv_X) == 0:
            logger.info("  No more invisible samples. Stopping.")
            break

    return vis_X, vis_Y, inv_X


def apply_self_training_conf_geo(vis_X: pd.DataFrame, vis_Y: pd.Series, inv_X: pd.DataFrame,
                                   features: List[str], config: Config) -> tuple:
    """Apply geometric self-training using k-NN and feature importance weighting.

    Returns:
        tuple: (vis_X, vis_Y, inv_X) where inv_X maintains its original index.
               The caller can use inv_X.index to sync inv_Y.
    """
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import StandardScaler

    vis_X_proc = preprocess_features(vis_X)
    inv_X_proc = preprocess_features(inv_X)

    logger.info(f"  [Self-Training: Geometric] Pos={vis_Y.sum()}, Neg={len(vis_Y) - vis_Y.sum()}")

    for round_idx in range(config.n_rounds):
        logger.info(f"  Round {round_idx + 1}/{config.n_rounds} | Vis: {len(vis_X)} Inv: {len(inv_X)}")

        # Get feature importances from classifier
        clf_temp = train_classifier(vis_X_proc[features], vis_Y, config)
        feature_importances = clf_temp.feature_importances_

        # Create feature mapping (handle case where features might be subset)
        feature_weights = np.ones(len(features))
        for i, feat in enumerate(features):
            if feat in vis_X_proc.columns:
                idx = list(vis_X_proc.columns).index(feat)
                if idx < len(feature_importances):
                    feature_weights[i] = np.sqrt(feature_importances[idx])

        # Normalize features
        scaler = StandardScaler()
        vis_X_norm = scaler.fit_transform(vis_X_proc[features])
        inv_X_norm = scaler.transform(inv_X_proc[features])

        # Apply feature importance weights
        vis_X_weighted = vis_X_norm * feature_weights
        inv_X_weighted = inv_X_norm * feature_weights

        # Train classifier for predictions
        clf = train_classifier(vis_X_proc[features], vis_Y, config)
        probas = clf.predict_proba(inv_X_proc[features])[:, 1]

        # Separate visible samples by class
        pos_idx = vis_Y[vis_Y == 1].index
        neg_idx = vis_Y[vis_Y == 0].index

        if len(pos_idx) == 0 or len(neg_idx) == 0:
            logger.info("  Not enough samples of both classes. Stopping.")
            break

        # Compute nearest neighbors for geometric consistency
        pos_samples = vis_X_weighted[pos_idx]
        neg_samples = vis_X_weighted[neg_idx]

        # Find k-nearest neighbors (k = min(config.geo_k_neighbors, minority class size))
        n_neighbors = min(config.geo_k_neighbors, len(pos_idx), len(neg_idx))
        knn_pos = NearestNeighbors(n_neighbors=n_neighbors).fit(pos_samples)
        knn_neg = NearestNeighbors(n_neighbors=n_neighbors).fit(neg_samples)

        dist_pos, _ = knn_pos.kneighbors(inv_X_weighted)
        dist_neg, _ = knn_neg.kneighbors(inv_X_weighted)

        # Compute geometric scores: inverse of average distance
        epsilon = 1e-6
        geo_score_pos = 1.0 / (dist_pos.mean(axis=1) + epsilon)
        geo_score_neg = 1.0 / (dist_neg.mean(axis=1) + epsilon)

        # Normalize scores
        total_score = geo_score_pos + geo_score_neg
        geo_prob_pos = geo_score_pos / total_score

        # Combine classifier probability with geometric probability
        # Geometric weight decreases over rounds (more reliance on classifier as it improves)
        geo_weight = config.geo_initial_weight * (1 - round_idx / config.n_rounds)
        combined_prob = geo_weight * geo_prob_pos + (1 - geo_weight) * probas

        # Get predictions based on combined probability
        predictions = (combined_prob >= config.geo_decision_boundary).astype(int)

        # Select samples with highest confidence (distance from decision boundary)
        # For geometric method, we select top-k instead of using strict threshold
        conf_scores = np.abs(combined_prob - config.geo_decision_boundary)

        # TODO: Improve balancing logic
        # TODO; HUMAN IN THE LOOP
        if config.balance_classes:
            # Separate by predicted class
            pos_pred_idx = np.where(predictions == 1)[0]
            neg_pred_idx = np.where(predictions == 0)[0]

            if len(pos_pred_idx) == 0 or len(neg_pred_idx) == 0:
                # If only one class predicted, select from that class
                available_idx = pos_pred_idx if len(pos_pred_idx) > 0 else neg_pred_idx
                n = min(len(available_idx), config.max_samples_per_round)
                if n == 0:
                    logger.info("  No samples available. Stopping.")
                    break
                selected = available_idx[np.argsort(conf_scores[available_idx])[-n:]]
                logger.info(f"  Selected: {n} samples (one class only)")
            else:
                # Balance classes: select equal number from each
                n_per_class = min(len(pos_pred_idx), len(neg_pred_idx), config.max_samples_per_round // 2)
                if n_per_class == 0:
                    n_per_class = min(len(pos_pred_idx), len(neg_pred_idx))
                # Select samples with highest confidence within each class
                selected_pos = pos_pred_idx[np.argsort(conf_scores[pos_pred_idx])[-n_per_class:]]
                selected_neg = neg_pred_idx[np.argsort(conf_scores[neg_pred_idx])[-n_per_class:]]
                selected = np.concatenate([selected_pos, selected_neg])
                logger.info(f"  Selected: {len(selected_pos)} pos + {len(selected_neg)} neg")
        else:
            n = min(len(inv_X), config.max_samples_per_round)
            selected = np.argsort(conf_scores)[-n:]
            logger.info(f"  Selected: {n} samples")

        # Update datasets
        vis_X = pd.concat([vis_X, inv_X.iloc[selected]], ignore_index=True)
        new_labels = pd.Series(predictions[selected], dtype=vis_Y.dtype)
        logger.info(f"  Adding {len(new_labels)} new labels: {new_labels.value_counts().to_dict()}")
        vis_Y = pd.concat([vis_Y, new_labels], ignore_index=True).astype(vis_Y.dtype)
        vis_X_proc = pd.concat([vis_X_proc, inv_X_proc.iloc[selected]], ignore_index=True)
        inv_X = inv_X.drop(inv_X.index[selected])
        inv_X_proc = inv_X_proc.drop(inv_X_proc.index[selected])

        if len(inv_X) == 0:
            logger.info("  No more invisible samples. Stopping.")
            break

    return vis_X, vis_Y, inv_X


def apply_self_training(vis_X: pd.DataFrame, vis_Y: pd.Series, inv_X: pd.DataFrame,
                        features: List[str], config: Config) -> tuple:
    """Dispatch to appropriate self-training method based on config.

    Returns:
        tuple: (vis_X, vis_Y, inv_X) where inv_X maintains its original index.
               The caller can use inv_X.index to sync inv_Y.
    """
    if config.self_training_mode == 'Conf':
        return apply_self_training_conf(vis_X, vis_Y, inv_X, features, config)
    elif config.self_training_mode == 'ConfGeo':
        return apply_self_training_conf_geo(vis_X, vis_Y, inv_X, features, config)
    elif config.self_training_mode == 'Geo':
        return apply_self_training_geo(vis_X, vis_Y, inv_X, features, config)
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

        vis_X, vis_Y, inv_X = apply_self_training(
            vis_X, vis_Y, inv_X, features, config
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
        'train_size': final_train_size
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

            result_entry = {
                'Method': method_name,
                'F1': result['metrics']['f1'],
                'P': result['metrics']['precision'],
                'R': result['metrics']['recall'],
                'Feats': len(result['features']),
                'Time': result['time'],
                'TrainSize': result['train_size'],
                'Improvement': (result['metrics']['f1'] - baseline_f1) / baseline_f1 * 100
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
            'n_rounds': [4, 6, 8, 10, 12, 14, 16]
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
            geo_initial_weight=0.7,
            geo_decision_boundary=0.3,
            max_samples_per_round=10,
            n_rounds=4,
            methods=[
                [],
                ['FS'],
                ['ST_Conf'],
                ['ST_ConfGeo'],
                ['FS', 'ST_Conf'],
                ['FS', 'ST_ConfGeo']
            ]
        )),
        ("medical_Q3", Config(
            data_path="data/medical/.ckpt/NOPXY__Q3_full.csv",
            visible_samples=50,
            random_seed=42,
            geo_k_neighbors=5,
            geo_initial_weight=0.1,
            geo_decision_boundary=0.1,
            max_samples_per_round=20,
            n_rounds=4,
            methods=[
                [],
                ['FS'],
                ['ST_Conf'],
                ['ST_ConfGeo'],
                ['FS', 'ST_Conf'],
                ['FS', 'ST_ConfGeo']
            ]
        )),
        ("medical_Q8", Config(
            data_path="data/medical/.ckpt/NOPXY__Q8_full.csv",
            visible_samples=50,
            random_seed=42,
            geo_k_neighbors=5,
            geo_initial_weight=0.7,
            geo_decision_boundary=0.1,
            max_samples_per_round=50,
            n_rounds=16,
            methods=[
                [],
                ['FS'],
                ['ST_Conf'],
                ['ST_ConfGeo'],
                ['FS', 'ST_Conf'],
                ['FS', 'ST_ConfGeo']
            ]
        )),
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

    # Check for --grid-search flag
    if args.grid_search:
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
