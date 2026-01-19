"""
Few-Shot Binary Classification Framework

Simple, configurable framework with ablation study support.
"""

import pandas as pd
import numpy as np
import logging
import time
from dataclasses import dataclass

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# Config (all parameters in one place)
# =============================================================================
@dataclass
class Config:
    # Data (required)
    data_path: str

    # Data (optional with defaults)
    visible_samples: int = 50
    random_seed: int = 42

    # Feature selection
    use_feature_selection: bool = True
    top_k: int = 10

    # Self-training
    use_self_training: bool = True
    n_rounds: int = 3
    confidence_threshold: float = 0.95
    max_samples_per_round: int = 10
    balance_classes: bool = True

    # Classifier
    n_estimators: int = 100
    max_depth: int = 10


# =============================================================================
# Core Functions
# =============================================================================
def preprocess_features(X):
    """Encode categorical variables."""
    from sklearn.preprocessing import LabelEncoder
    X_proc = X.copy()
    for col in X_proc.select_dtypes(include=['object']).columns:
        X_proc[col] = LabelEncoder().fit_transform(X_proc[col].astype(str))
    return X_proc


def train_classifier(vis_X, vis_Y):
    """Train Random Forest."""
    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(
        n_estimators=Config.n_estimators,
        max_depth=Config.max_depth,
        random_state=Config.random_seed,
        class_weight='balanced'
    )
    clf.fit(vis_X, vis_Y)
    return clf


def evaluate_predictions(vis_Y, inv_Y, inv_Y_pred):
    """Calculate F1, precision, recall."""
    from sklearn.metrics import f1_score, precision_score, recall_score
    all_Y_pred = np.concatenate([vis_Y.values, inv_Y_pred])
    all_Y_true = np.concatenate([vis_Y.values, inv_Y.values])
    return {
        'f1': f1_score(all_Y_true, all_Y_pred),
        'precision': precision_score(all_Y_true, all_Y_pred),
        'recall': recall_score(all_Y_true, all_Y_pred)
    }


def few_shot_classify(vis_X, vis_Y, inv_X, config: Config, use_fs=True):
    """Few-shot classification with optional feature selection."""
    vis_X_proc = preprocess_features(vis_X)
    inv_X_proc = preprocess_features(inv_X)

    if use_fs:
        clf_temp = train_classifier(vis_X_proc, vis_Y)
        importances = pd.DataFrame({
            'feature': vis_X_proc.columns,
            'importance': clf_temp.feature_importances_
        }).sort_values('importance', ascending=False)
        features = importances.head(config.top_k)['feature'].tolist()
        logger.info(f"[FEATURE SELECTION] Top-{config.top_k}: {len(features)} features")
    else:
        features = vis_X_proc.columns.tolist()
        logger.info(f"[NO FEATURE SELECTION] Using all {len(features)} features")

    clf = train_classifier(vis_X_proc[features], vis_Y)
    predictions = clf.predict(inv_X_proc[features])

    return {'predictions': predictions, 'features': features}


def self_training(vis_X, vis_Y, inv_X, inv_Y, config: Config, use_fs=True):
    """Self-training with safety rails."""
    vis_X_proc = preprocess_features(vis_X)
    inv_X_proc = preprocess_features(inv_X)

    result_init = few_shot_classify(vis_X, vis_Y, inv_X, config, use_fs)
    features = result_init['features']

    # Use seeded RNG for reproducibility
    rng = np.random.default_rng(config.random_seed)

    logger.info(f"[SELF-TRAINING] Pos={vis_Y.sum()}, Neg={len(vis_Y) - vis_Y.sum()}")

    for round_idx in range(config.n_rounds):
        logger.info(f"\nRound {round_idx + 1}/{config.n_rounds} | Vis: {len(vis_X)} Inv: {len(inv_X)}")

        clf = train_classifier(vis_X_proc[features], vis_Y)
        probas = clf.predict_proba(inv_X_proc[features])[:, 1]
        predictions = clf.predict(inv_X_proc[features])

        metrics = evaluate_predictions(vis_Y, inv_Y, predictions)
        logger.info(f"[ROUND {round_idx + 1}] F1: {metrics['f1']:.4f}, P: {metrics['precision']:.4f}, R: {metrics['recall']:.4f}")

        conf_mask_pos = (probas >= config.confidence_threshold)
        conf_mask_neg = (probas <= (1 - config.confidence_threshold))
        high_conf_mask = conf_mask_pos | conf_mask_neg

        logger.debug(f"High-conf: {high_conf_mask.sum()} (Pos={conf_mask_pos.sum()}, Neg={conf_mask_neg.sum()})")

        if high_conf_mask.sum() == 0:
            logger.info("No samples above threshold. Stopping.")
            break

        # Select samples
        if config.balance_classes:
            pos_idx = np.where(conf_mask_pos)[0]
            neg_idx = np.where(conf_mask_neg)[0]

            if len(pos_idx) == 0 or len(neg_idx) == 0:
                n = min(high_conf_mask.sum(), config.max_samples_per_round)
                selected = rng.choice(np.where(high_conf_mask)[0], n, replace=False)
                logger.info(f"Selected: {n} samples (one class only)")
            else:
                n_per_class = min(len(pos_idx), len(neg_idx), config.max_samples_per_round // 2)
                if n_per_class == 0:
                    n_per_class = min(len(pos_idx), len(neg_idx))
                selected_pos = rng.choice(pos_idx, n_per_class, replace=False)
                selected_neg = rng.choice(neg_idx, n_per_class, replace=False)
                selected = np.concatenate([selected_pos, selected_neg])
                logger.info(f"Selected: {n_per_class} pos + {n_per_class} neg")
        else:
            n = min(high_conf_mask.sum(), config.max_samples_per_round)
            selected = rng.choice(np.where(high_conf_mask)[0], n, replace=False)
            logger.info(f"Selected: {n} samples")

        # Update datasets
        vis_X = pd.concat([vis_X, inv_X.iloc[selected]], ignore_index=True)
        vis_Y = pd.concat([vis_Y, pd.Series(predictions[selected])], ignore_index=True)
        vis_X_proc = pd.concat([vis_X_proc, inv_X_proc.iloc[selected]], ignore_index=True)
        inv_X = inv_X.drop(inv_X.index[selected]).reset_index(drop=True)
        inv_Y = inv_Y.drop(inv_Y.index[selected]).reset_index(drop=True)
        inv_X_proc = inv_X_proc.drop(inv_X_proc.index[selected]).reset_index(drop=True)

        if len(inv_X) == 0:
            logger.info("No more invisible samples. Stopping.")
            break

    # Final prediction
    logger.info("\n[FINAL PREDICTION]")
    clf_final = train_classifier(vis_X_proc[features], vis_Y)
    final_predictions = clf_final.predict(inv_X_proc[features])
    final_metrics = evaluate_predictions(vis_Y, inv_Y, final_predictions)

    return {'metrics': final_metrics, 'features': features}


# =============================================================================
# Ablation Study
# =============================================================================
def run_ablation(vis_X, vis_Y, inv_X, inv_Y, config: Config):
    """Run ablation study and return results (no printing)."""
    results = []

    # Baseline
    logger.info("[ABLATION] BASELINE (no optimizations)")
    start = time.time()
    res = few_shot_classify(vis_X, vis_Y, inv_X, config, use_fs=False)
    elapsed = time.time() - start
    m = evaluate_predictions(vis_Y, inv_Y, res['predictions'])
    results.append({'Method': 'Baseline', 'F1': m['f1'], 'P': m['precision'], 'R': m['recall'], 'Feats': len(res['features']), 'Time': elapsed})
    baseline_f1 = m['f1']
    logger.info(f"  F1: {m['f1']:.4f}, Precision: {m['precision']:.4f}, Recall: {m['recall']:.4f} (Time: {elapsed:.2f}s)\n")

    # + Feature Selection
    if config.use_feature_selection:
        logger.info(f"[ABLATION] + FEATURE SELECTION (top-{config.top_k})")
        start = time.time()
        res = few_shot_classify(vis_X, vis_Y, inv_X, config, use_fs=True)
        elapsed = time.time() - start
        m = evaluate_predictions(vis_Y, inv_Y, res['predictions'])
        results.append({'Method': '+ FS', 'F1': m['f1'], 'P': m['precision'], 'R': m['recall'], 'Feats': len(res['features']), 'Time': elapsed})
        logger.info(f"  F1: {m['f1']:.4f} (vs baseline: {(m['f1']-baseline_f1)/baseline_f1*100:+.2f}%, Time: {elapsed:.2f}s)\n")

    # + Self-Training
    if config.use_self_training:
        logger.info(f"[ABLATION] + SELF-TRAINING ({config.n_rounds} rounds)")
        start = time.time()
        res = self_training(vis_X.copy(), vis_Y.copy(), inv_X.copy(), inv_Y.copy(), config, use_fs=False)
        elapsed = time.time() - start
        m = res['metrics']
        results.append({'Method': '+ ST', 'F1': m['f1'], 'P': m['precision'], 'R': m['recall'], 'Feats': len(res['features']), 'Time': elapsed})
        logger.info(f"  F1: {m['f1']:.4f} (vs baseline: {(m['f1']-baseline_f1)/baseline_f1*100:+.2f}%, Time: {elapsed:.2f}s)\n")

    # + Feature selection + Self-Training
    if config.use_self_training and config.use_feature_selection:
        logger.info(f"[ABLATION] + FS & SELF-TRAINING ({config.n_rounds} rounds)")
        start = time.time()
        res = self_training(vis_X.copy(), vis_Y.copy(), inv_X.copy(), inv_Y.copy(), config, use_fs=True)
        elapsed = time.time() - start
        m = res['metrics']
        results.append({'Method': '+ FS & ST', 'F1': m['f1'], 'P': m['precision'], 'R': m['recall'], 'Feats': len(res['features']), 'Time': elapsed})
        logger.info(f"  F1: {m['f1']:.4f} (vs baseline: {(m['f1']-baseline_f1)/baseline_f1*100:+.2f}%, Time: {elapsed:.2f}s)\n")

    # Add improvement percentages
    for r in results:
        r['Improvement'] = (r['F1'] - baseline_f1) / baseline_f1 * 100

    return results


def print_centralized_report(all_results):
    """Print centralized ablation report across all workloads."""
    logger.info("\n\n" + "="*120)
    logger.info(" " * 45 + "CENTRALIZED ABLATION REPORT")
    logger.info("="*120)

    # Prepare data for summary table
    summary_data = []
    for dataset_name, methods in all_results.items():
        for method_result in methods:
            summary_data.append({
                'Dataset': dataset_name,
                'Method': method_result['Method'],
                'F1': method_result['F1'],
                'Precision': method_result['P'],
                'Recall': method_result['R'],
                'Improvement': method_result['Improvement'],
                'Feats': method_result['Feats'],
                'Time': method_result['Time']
            })

    # Convert to DataFrame for easier manipulation
    df = pd.DataFrame(summary_data)

    # Print full table
    logger.info(f"\n{'Dataset':<25} {'Method':<15} {'F1':>8} {'Precision':>10} {'Recall':>8} {'Improvement':>12} {'Feats':>8} {'Time(s)':>8}")
    logger.info("-" * 120)
    for _, row in df.iterrows():
        logger.info(f"{row['Dataset']:<25} {row['Method']:<15} {row['F1']:>8.4f} {row['Precision']:>10.4f} {row['Recall']:>8.4f} {row['Improvement']:>10.2f}% {row['Feats']:>8} {row['Time']:>8.2f}")
    logger.info("="*120)

    # Print average summary by method
    logger.info("\n" + "="*100)
    logger.info(" " * 35 + "AVERAGE PERFORMANCE BY METHOD")
    logger.info("="*100)
    avg_df = df.groupby('Method').agg({
        'F1': 'mean',
        'Precision': 'mean',
        'Recall': 'mean',
        'Improvement': 'mean',
        'Time': 'mean'
    }).reset_index()

    logger.info(f"\n{'Method':<20} {'F1':>8} {'Precision':>10} {'Recall':>8} {'Improvement':>12} {'Time(s)':>8}")
    logger.info("-" * 100)
    for _, row in avg_df.iterrows():
        logger.info(f"{row['Method']:<20} {row['F1']:>8.4f} {row['Precision']:>10.4f} {row['Recall']:>8.4f} {row['Improvement']:>10.2f}% {row['Time']:>8.2f}")
    logger.info("="*100 + "\n")


# =============================================================================
# Main
# =============================================================================
def run_experiment(config: Config, dataset_name: str):
    """Run ablation study for a single configuration and return results."""
    logger.info("="*80)
    logger.info(f"DATASET: {dataset_name}")
    logger.info("="*80)

    # Load data
    df = pd.read_csv(config.data_path)
    Y, X = df['label'], df.drop(columns=['label', 'patient_id'])

    # Split
    np.random.seed(config.random_seed)
    vis_idx = X.sample(n=config.visible_samples).index
    inv_idx = X.index.difference(vis_idx)

    vis_X, inv_X = X.loc[vis_idx].reset_index(drop=True), X.loc[inv_idx].reset_index(drop=True)
    vis_Y, inv_Y = Y.loc[vis_idx].reset_index(drop=True), Y.loc[inv_idx].reset_index(drop=True)

    logger.info(f"Data: {len(vis_X)} visible, {len(inv_X)} invisible")
    logger.info(f"Features for training:\n{list(X.columns)}\n")

    # Run ablation and return results
    results = run_ablation(vis_X, vis_Y, inv_X, inv_Y, config)
    return results


if __name__ == "__main__":
    # Define multiple workloads with descriptive names
    workloads = [
        ("medical_Q1", Config(
            data_path="data/medical/.ckpt/NOPXY__Q1_full.csv",
            visible_samples=50,
            random_seed=42
        )),
        ("medical_Q3", Config(
            data_path="data/medical/.ckpt/NOPXY__Q3_full.csv",
            visible_samples=50,
            random_seed=42
        )),
        ("medical_Q8", Config(
            data_path="data/medical/.ckpt/NOPXY__Q8_full.csv",
            visible_samples=50,
            random_seed=42
        )),
    ]

    # Store all results for centralized report
    all_results = {}

    # Run experiments for each workload
    for i, (dataset_name, config) in enumerate(workloads, 1):
        logger.info(f"\n\n{'#'*80}")
        logger.info(f"# WORKLOAD {i}/{len(workloads)}: {dataset_name}")
        logger.info(f"{'#'*80}\n")
        results = run_experiment(config, dataset_name)
        all_results[dataset_name] = results

    # Print centralized report
    print_centralized_report(all_results)
