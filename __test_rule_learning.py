"""
Few-Shot Binary Classification Framework

Simple, configurable framework with modular optimization methods.
"""

import pandas as pd
import numpy as np
import logging
import time
from dataclasses import dataclass, field
from typing import List, Callable, Dict, Any

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
    self_training_mode: str = 'confidence'  # Options: 'confidence', 'geometric'
    n_rounds: int = 3
    confidence_threshold: float = 0.95
    max_samples_per_round: int = 10
    balance_classes: bool = True

    # Geometric self-training parameters
    geo_k_neighbors: int = 5  # k for k-NN (max neighbors, will be capped by minority class size)
    geo_initial_weight: float = 0.6  # Initial weight for geometric scoring (decreases over rounds)

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


def apply_self_training_confidence(vis_X: pd.DataFrame, vis_Y: pd.Series, inv_X: pd.DataFrame,
                                    inv_Y: pd.Series, features: List[str], config: Config) -> tuple:
    """Apply self-training to augment training data."""
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
        inv_X = inv_X.drop(inv_X.index[selected]).reset_index(drop=True)
        inv_Y = inv_Y.drop(inv_Y.index[selected]).reset_index(drop=True)
        inv_X_proc = inv_X_proc.drop(inv_X_proc.index[selected]).reset_index(drop=True)

        if len(inv_X) == 0:
            logger.info("  No more invisible samples. Stopping.")
            break

    return vis_X, vis_Y, inv_X, inv_Y


def apply_self_training_geometric(vis_X: pd.DataFrame, vis_Y: pd.Series, inv_X: pd.DataFrame,
                                   inv_Y: pd.Series, features: List[str], config: Config) -> tuple:
    """Apply geometric self-training using k-NN and feature importance weighting."""
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
        predictions = (combined_prob >= 0.5).astype(int)

        # Select samples with highest confidence (distance from 0.5)
        # For geometric method, we select top-k instead of using strict threshold
        conf_scores = np.abs(combined_prob - 0.5)

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
        inv_X = inv_X.drop(inv_X.index[selected]).reset_index(drop=True)
        inv_Y = inv_Y.drop(inv_Y.index[selected]).reset_index(drop=True)
        inv_X_proc = inv_X_proc.drop(inv_X_proc.index[selected]).reset_index(drop=True)

        if len(inv_X) == 0:
            logger.info("  No more invisible samples. Stopping.")
            break

    return vis_X, vis_Y, inv_X, inv_Y


def apply_self_training(vis_X: pd.DataFrame, vis_Y: pd.Series, inv_X: pd.DataFrame,
                        inv_Y: pd.Series, features: List[str], config: Config) -> tuple:
    """Dispatch to appropriate self-training method based on config."""
    if config.self_training_mode == 'confidence':
        return apply_self_training_confidence(vis_X, vis_Y, inv_X, inv_Y, features, config)
    elif config.self_training_mode == 'geometric':
        return apply_self_training_geometric(vis_X, vis_Y, inv_X, inv_Y, features, config)
    else:
        raise ValueError(f"Unknown self_training_mode: {config.self_training_mode}. "
                        f"Must be 'confidence' or 'geometric'")


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

    # Track initial training size
    initial_train_size = len(vis_X)

    # Preprocess
    vis_X_proc = preprocess_features(vis_X)
    inv_X_proc = preprocess_features(inv_X)

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

        vis_X, vis_Y, inv_X, inv_Y = apply_self_training(
            vis_X, vis_Y, inv_X, inv_Y, features, config
        )

        # Restore original mode
        config.self_training_mode = original_mode

        vis_X_proc = preprocess_features(vis_X)
        inv_X_proc = preprocess_features(inv_X)

    # Track final training size
    final_train_size = len(vis_X)

    # Step 3: Final training and prediction
    clf = train_classifier(vis_X_proc[features], vis_Y, config)
    predictions = clf.predict(inv_X_proc[features])

    # Evaluate
    metrics = evaluate_predictions(vis_Y, inv_Y, predictions)
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
        result = run_classification(
            vis_X.copy(), vis_Y.copy(), inv_X.copy(), inv_Y.copy(),
            config, methods=methods
        )

        results.append({
            'Method': method_name,
            'F1': result['metrics']['f1'],
            'P': result['metrics']['precision'],
            'R': result['metrics']['recall'],
            'Feats': len(result['features']),
            'Time': result['time'],
            'TrainSize': result['train_size'],
            'Improvement': (result['metrics']['f1'] - baseline_f1) / baseline_f1 * 100
        })

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

    # Get method order from first dataset to maintain consistency
    method_order = df[df['Dataset'] == df['Dataset'].unique()[0]]['Method'].tolist()

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
# Main
# =============================================================================
if __name__ == "__main__":
    # Define workloads with methods to run
    # Methods:
    #   [] = Baseline
    #   ['FS'] = Feature Selection
    #   ['ST_confidence'] = Self-Training (confidence-based)
    #   ['ST_geometric'] = Self-Training (geometric-based)
    #   ['FS', 'ST_confidence'] = FS + ST (confidence)
    #   ['FS', 'ST_geometric'] = FS + ST (geometric)
    workloads = [
        # Compare all self-training methods on each dataset
        ("medical_Q1", Config(
            data_path="data/medical/.ckpt/NOPXY__Q1_full.csv",
            visible_samples=50,
            random_seed=42,
            methods=[
                [],
                ['FS'],
                ['ST_confidence'],
                ['ST_geometric'],
                ['FS', 'ST_confidence'],
                ['FS', 'ST_geometric']
            ]
        )),
        ("medical_Q3", Config(
            data_path="data/medical/.ckpt/NOPXY__Q3_full.csv",
            visible_samples=50,
            random_seed=42,
            methods=[
                [],
                ['FS'],
                ['ST_confidence'],
                ['ST_geometric'],
                ['FS', 'ST_confidence'],
                ['FS', 'ST_geometric']
            ]
        )),
        ("medical_Q8", Config(
            data_path="data/medical/.ckpt/NOPXY__Q8_full.csv",
            visible_samples=50,
            random_seed=42,
            methods=[
                [],
                ['FS'],
                ['ST_confidence'],
                ['ST_geometric'],
                ['FS', 'ST_confidence'],
                ['FS', 'ST_geometric']
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
