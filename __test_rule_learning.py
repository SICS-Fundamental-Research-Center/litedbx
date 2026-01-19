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
    n_rounds: int = 3
    confidence_threshold: float = 0.95
    max_samples_per_round: int = 10
    balance_classes: bool = True

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


def apply_self_training(vis_X: pd.DataFrame, vis_Y: pd.Series, inv_X: pd.DataFrame,
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

    # Step 1: Feature selection (if enabled)
    if 'FS' in methods:
        features = apply_feature_selection(vis_X_proc, vis_Y, config)
    else:
        features = vis_X_proc.columns.tolist()
        logger.info(f"  [No Feature Selection] Using all {len(features)} features")

    # Step 2: Self-training (if enabled)
    if 'ST' in methods:
        vis_X, vis_Y, inv_X, inv_Y = apply_self_training(
            vis_X, vis_Y, inv_X, inv_Y, features, config
        )
        vis_X_proc = preprocess_features(vis_X)
        inv_X_proc = preprocess_features(inv_X)

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
        'time': elapsed
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
            'Improvement': (result['metrics']['f1'] - baseline_f1) / baseline_f1 * 100
        })

    return results


def print_report(all_results: Dict[str, List[Dict]]) -> None:
    """Print centralized report across all workloads."""
    logger.info("\n\n" + "="*120)
    logger.info(" " * 45 + "CENTRALIZED REPORT")
    logger.info("="*120)

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
                'Time': result['Time']
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
        'Time': 'mean'
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
    logger.info(f"\n{'Dataset':<25} {'Method':<20} {'F1':>8} {'Precision':>10} {'Recall':>8} {'Improvement':>12} {'Feats':>8} {'Time(s)':>8}")
    logger.info("-" * 120)

    current_dataset = None
    for _, row in df_combined.iterrows():
        if current_dataset is not None and row['Dataset'] != current_dataset:
            logger.info("-" * 120)
        current_dataset = row['Dataset']

        logger.info(f"{row['Dataset']:<25} {row['Method']:<20} {row['F1']:>8.4f} {row['Precision']:>10.4f} {row['Recall']:>8.4f} {row['Improvement']:>10.2f}% {int(row['Feats']):>8} {row['Time']:>8.2f}")

    logger.info("="*120 + "\n")


# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    # Define workloads with methods to run
    # Methods: [] = Baseline, ['FS'] = Feature Selection, ['ST'] = Self-Training, ['FS', 'ST'] = Both
    workloads = [
        ("medical_Q1", Config(
            data_path="data/medical/.ckpt/NOPXY__Q1_full.csv",
            visible_samples=50,
            random_seed=42,
            methods=[[], ['FS'], ['ST'], ['FS', 'ST']]  # Specify which method combinations to run
        )),
        ("medical_Q3", Config(
            data_path="data/medical/.ckpt/NOPXY__Q3_full.csv",
            visible_samples=50,
            random_seed=42,
            methods=[[], ['FS'], ['ST'], ['FS', 'ST']]
        )),
        ("medical_Q8", Config(
            data_path="data/medical/.ckpt/NOPXY__Q8_full.csv",
            visible_samples=50,
            random_seed=42,
            methods=[[], ['FS'], ['ST'], ['FS', 'ST']]
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
