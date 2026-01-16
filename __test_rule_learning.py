"""
Few-Shot Binary Classification Framework

Simple, configurable framework with ablation study support.
"""

import pandas as pd
import numpy as np
import logging
import time

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# Config (all parameters in one place)
# =============================================================================
class Config:
    # Data
    visible_samples = 50
    random_seed = 42
    data_path = "data/medical/.ckpt/NOPXY__Q1_full.csv"

    # Feature selection
    use_feature_selection = True
    top_k = 10

    # Self-training
    use_self_training = True
    n_rounds = 5
    confidence_threshold = 0.95
    max_samples_per_round = 10
    balance_classes = True

    # Classifier
    n_estimators = 100
    max_depth = 10


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


def few_shot_classify(vis_X, vis_Y, inv_X, use_fs=True):
    """Few-shot classification with optional feature selection."""
    vis_X_proc = preprocess_features(vis_X)
    inv_X_proc = preprocess_features(inv_X)

    if use_fs:
        clf_temp = train_classifier(vis_X_proc, vis_Y)
        importances = pd.DataFrame({
            'feature': vis_X_proc.columns,
            'importance': clf_temp.feature_importances_
        }).sort_values('importance', ascending=False)
        features = importances.head(Config.top_k)['feature'].tolist()
        logger.info(f"[FEATURE SELECTION] Top-{Config.top_k}: {len(features)} features")
    else:
        features = vis_X_proc.columns.tolist()
        logger.info(f"[NO FEATURE SELECTION] Using all {len(features)} features")

    clf = train_classifier(vis_X_proc[features], vis_Y)
    predictions = clf.predict(inv_X_proc[features])

    return {'predictions': predictions, 'features': features}


def self_training(vis_X, vis_Y, inv_X, inv_Y, use_fs=True):
    """Self-training with safety rails."""
    vis_X_proc = preprocess_features(vis_X)
    inv_X_proc = preprocess_features(inv_X)

    result_init = few_shot_classify(vis_X, vis_Y, inv_X, use_fs)
    features = result_init['features']

    logger.info(f"[SELF-TRAINING] Pos={vis_Y.sum()}, Neg={len(vis_Y) - vis_Y.sum()}")

    for round_idx in range(Config.n_rounds):
        logger.info(f"\nRound {round_idx + 1}/{Config.n_rounds} | Vis: {len(vis_X)} Inv: {len(inv_X)}")

        clf = train_classifier(vis_X_proc[features], vis_Y)
        probas = clf.predict_proba(inv_X_proc[features])[:, 1]
        predictions = clf.predict(inv_X_proc[features])

        metrics = evaluate_predictions(vis_Y, inv_Y, predictions)
        logger.info(f"[ROUND {round_idx + 1}] F1: {metrics['f1']:.4f}, P: {metrics['precision']:.4f}, R: {metrics['recall']:.4f}")

        conf_mask_pos = (probas >= Config.confidence_threshold)
        conf_mask_neg = (probas <= (1 - Config.confidence_threshold))
        high_conf_mask = conf_mask_pos | conf_mask_neg

        logger.debug(f"High-conf: {high_conf_mask.sum()} (Pos={conf_mask_pos.sum()}, Neg={conf_mask_neg.sum()})")

        if high_conf_mask.sum() == 0:
            logger.info("No samples above threshold. Stopping.")
            break

        # Select samples
        if Config.balance_classes:
            pos_idx = np.where(conf_mask_pos)[0]
            neg_idx = np.where(conf_mask_neg)[0]

            if len(pos_idx) == 0 or len(neg_idx) == 0:
                n = min(high_conf_mask.sum(), Config.max_samples_per_round)
                selected = np.random.choice(np.where(high_conf_mask)[0], n, replace=False)
                logger.info(f"Selected: {n} samples (one class only)")
            else:
                n_per_class = min(len(pos_idx), len(neg_idx), Config.max_samples_per_round // 2)
                if n_per_class == 0:
                    n_per_class = min(len(pos_idx), len(neg_idx))
                selected_pos = np.random.choice(pos_idx, n_per_class, replace=False)
                selected_neg = np.random.choice(neg_idx, n_per_class, replace=False)
                selected = np.concatenate([selected_pos, selected_neg])
                logger.info(f"Selected: {n_per_class} pos + {n_per_class} neg")
        else:
            n = min(high_conf_mask.sum(), Config.max_samples_per_round)
            selected = np.random.choice(np.where(high_conf_mask)[0], n, replace=False)
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
def run_ablation(vis_X, vis_Y, inv_X, inv_Y):
    """Run ablation study and print report."""
    results = []

    # Baseline
    logger.info("\n" + "="*80)
    logger.info("[ABLATION] BASELINE (no optimizations)")
    logger.info("="*80)
    start = time.time()
    res = few_shot_classify(vis_X, vis_Y, inv_X, use_fs=False)
    elapsed = time.time() - start
    m = evaluate_predictions(vis_Y, inv_Y, res['predictions'])
    results.append({'Method': 'Baseline', 'F1': m['f1'], 'P': m['precision'], 'R': m['recall'], 'Feats': len(res['features']), 'Time': elapsed})
    baseline_f1 = m['f1']
    logger.info(f"BASELINE F1: {m['f1']:.4f} (Time: {elapsed:.2f}s)\n")

    # + Feature Selection
    if Config.use_feature_selection:
        logger.info("="*80)
        logger.info(f"[ABLATION] + FEATURE SELECTION (top-{Config.top_k})")
        logger.info("="*80)
        start = time.time()
        res = few_shot_classify(vis_X, vis_Y, inv_X, use_fs=True)
        elapsed = time.time() - start
        m = evaluate_predictions(vis_Y, inv_Y, res['predictions'])
        results.append({'Method': f'+ FS', 'F1': m['f1'], 'P': m['precision'], 'R': m['recall'], 'Feats': len(res['features']), 'Time': elapsed})
        logger.info(f"+FS F1: {m['f1']:.4f} (vs baseline: {(m['f1']-baseline_f1)/baseline_f1*100:+.2f}%, Time: {elapsed:.2f}s)\n")

    # + Self-Training (on top of feature selection)
    if Config.use_self_training:
        logger.info("="*80)
        logger.info(f"[ABLATION] + SELF-TRAINING ({Config.n_rounds} rounds)")
        logger.info("="*80)

        # Use FS results as baseline for ST comparison
        if Config.use_feature_selection:
            prev_f1 = results[-1]['F1']
            use_fs_for_st = True
        else:
            prev_f1 = baseline_f1
            use_fs_for_st = False

        start = time.time()
        res = self_training(vis_X.copy(), vis_Y.copy(), inv_X.copy(), inv_Y.copy(), use_fs=use_fs_for_st)
        elapsed = time.time() - start
        m = res['metrics']
        results.append({'Method': '+ FS & ST', 'F1': m['f1'], 'P': m['precision'], 'R': m['recall'], 'Feats': len(res['features']), 'Time': elapsed})
        logger.info(f"+ST F1: {m['f1']:.4f} (vs previous: {(m['f1']-prev_f1)/prev_f1*100:+.2f}%, Time: {elapsed:.2f}s)\n")

    # Print report
    logger.info("\n" + "="*90)
    logger.info(" " * 35 + "ABLATION REPORT")
    logger.info("="*90)

    for r in results:
        r['Improvement'] = (r['F1'] - baseline_f1) / baseline_f1 * 100

    logger.info(f"\n{'Method':<20} {'F1':>8} {'Precision':>10} {'Recall':>8} {'Improvement':>15} {'# Feats':>10} {'Time(s)':>8}")
    logger.info("-" * 90)
    for r in results:
        logger.info(f"{r['Method']:<20} {r['F1']:>8.4f} {r['P']:>10.4f} {r['R']:>8.4f} {r['Improvement']:>12.2f}% {r['Feats']:>10} {r['Time']:>8.2f}")
    logger.info("="*90 + "\n")


# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    logger.info("="*80)
    logger.info("FEW-SHOT CLASSIFICATION - ABLATION STUDY")
    logger.info("="*80)

    # Load data
    df = pd.read_csv(Config.data_path)
    Y, X = df['label'], df.drop(columns=['label', 'patient_id'])

    # Split
    np.random.seed(Config.random_seed)
    vis_idx = X.sample(n=Config.visible_samples).index
    inv_idx = X.index.difference(vis_idx)

    vis_X, inv_X = X.loc[vis_idx].reset_index(drop=True), X.loc[inv_idx].reset_index(drop=True)
    vis_Y, inv_Y = Y.loc[vis_idx].reset_index(drop=True), Y.loc[inv_idx].reset_index(drop=True)

    logger.info(f"Data: {len(vis_X)} visible, {len(inv_X)} invisible")
    logger.info(f"Features for training:\n{list(X.columns)}\n")

    # Run ablation
    run_ablation(vis_X, vis_Y, inv_X, inv_Y)
