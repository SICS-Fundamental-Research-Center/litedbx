"""
Few-Shot Binary Classification Framework

A simple, modular framework for few-shot learning with LLM-extracted features.
"""

import pandas as pd
import numpy as np


def preprocess_features(X):
    """Encode categorical variables."""
    from sklearn.preprocessing import LabelEncoder

    X_processed = X.copy()
    categorical_cols = X_processed.select_dtypes(include=['object']).columns.tolist()

    for col in categorical_cols:
        le = LabelEncoder()
        X_processed[col] = le.fit_transform(X_processed[col].astype(str))

    if categorical_cols:
        print(f"[PREPROCESS] Encoded: {categorical_cols}")

    return X_processed


def train_classifier(vis_X, vis_Y):
    """Train a Random Forest classifier."""
    from sklearn.ensemble import RandomForestClassifier

    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        class_weight='balanced'
    )
    clf.fit(vis_X, vis_Y)
    return clf


def evaluate_predictions(vis_Y, inv_Y, inv_Y_pred):
    """Calculate F1, precision, and recall."""
    from sklearn.metrics import f1_score, precision_score, recall_score

    all_Y_pred = np.concatenate([vis_Y.values, inv_Y_pred])
    all_Y_true = np.concatenate([vis_Y.values, inv_Y.values])

    return {
        'f1': f1_score(all_Y_true, all_Y_pred),
        'precision': precision_score(all_Y_true, all_Y_pred),
        'recall': recall_score(all_Y_true, all_Y_pred)
    }


def few_shot_classify(vis_X, vis_Y, inv_X, top_k=10):
    """
    Complete few-shot classification pipeline.

    Args:
        vis_X: Visible features (labeled)
        vis_Y: Visible labels
        inv_X: Invisible features (unlabeled)
        top_k: Number of top features to use (default: 10)

    Returns:
        Dictionary with predictions, and selected features
    """
    # Preprocess
    vis_X_proc = preprocess_features(vis_X)
    inv_X_proc = preprocess_features(inv_X)

    # Train classifier to get feature importance
    clf_temp = train_classifier(vis_X_proc, vis_Y)
    importances = pd.DataFrame({
        'feature': vis_X_proc.columns,
        'importance': clf_temp.feature_importances_
    }).sort_values('importance', ascending=False)

    # Select top-k features
    best_features = importances.head(top_k)['feature'].tolist()

    print(f"[FEATURE SELECTION] Using top-{top_k} features")
    print(f"[FEATURE SELECTION] {best_features}\n")

    # Train final model
    vis_X_sel = vis_X_proc[best_features]
    inv_X_sel = inv_X_proc[best_features]
    clf = train_classifier(vis_X_sel, vis_Y)

    # Predict
    predictions = clf.predict(inv_X_sel)

    result = {
        'predictions': predictions,
        'classifier': clf,
        'top_k': top_k,
        'features': best_features,
        'importance': importances
    }

    return result


def self_training(vis_X, vis_Y, inv_X, inv_Y, n_rounds=3, confidence_threshold=0.95,
                  max_samples_per_round=10, balance_classes=True):
    """
    Self-training with safety rails.

    Args:
        vis_X: Initial visible features
        vis_Y: Initial visible labels
        inv_X: Invisible features
        inv_Y: Invisible labels (for evaluation only)
        n_rounds: Maximum iterations
        confidence_threshold: Minimum confidence to add sample (default: 0.95)
        max_samples_per_round: Max samples to add per round
        balance_classes: Force equal selection from both classes

    Returns:
        Dictionary with predictions, metrics, and training history
    """
    # Preprocess once at the beginning
    vis_X_proc = preprocess_features(vis_X)
    inv_X_proc = preprocess_features(inv_X)

    # Get features using initial training
    result_init = few_shot_classify(vis_X, vis_Y, inv_X, top_k=10)
    best_features = result_init['features']

    # Check initial class distribution
    print(f"[INIT] Visible class distribution: Positive={vis_Y.sum()}, Negative={len(vis_Y) - vis_Y.sum()}")

    history = []

    for round_idx in range(n_rounds):
        print(f"\n{'='*60}")
        print(f"SELF-TRAINING ROUND {round_idx + 1}/{n_rounds}")
        print(f"Visible samples: {len(vis_X)} | Invisible samples: {len(inv_X)}")
        print(f"{'='*60}\n")

        # Train on processed data with selected features
        vis_X_sel = vis_X_proc[best_features]
        inv_X_sel = inv_X_proc[best_features]
        clf = train_classifier(vis_X_sel, vis_Y)

        # Predict
        probas = clf.predict_proba(inv_X_sel)[:, 1]
        predictions = clf.predict(inv_X_sel)

        # Evaluate current performance
        metrics = evaluate_predictions(vis_Y, inv_Y, predictions)
        print(f"[ROUND {round_idx + 1}] F1: {metrics['f1']:.4f}, P: {metrics['precision']:.4f}, R: {metrics['recall']:.4f}")

        # Select high-confidence samples
        conf_mask_pos = (probas >= confidence_threshold)
        conf_mask_neg = (probas <= (1 - confidence_threshold))
        high_conf_mask = conf_mask_pos | conf_mask_neg

        print(f"[ROUND {round_idx + 1}] High confidence samples: {high_conf_mask.sum()} (Pos={conf_mask_pos.sum()}, Neg={conf_mask_neg.sum()})")

        if high_conf_mask.sum() == 0:
            print(f"[ROUND {round_idx + 1}] No samples above threshold. Stopping.")
            break

        # Class balancing (soft - allow imbalance when necessary)
        if balance_classes:
            pos_indices = np.where(conf_mask_pos)[0]
            neg_indices = np.where(conf_mask_neg)[0]

            # If one class has very few samples, take what we can get
            if len(pos_indices) == 0 or len(neg_indices) == 0:
                print(f"[ROUND {round_idx + 1}] Only one class available. Taking up to {max_samples_per_round} samples.")
                n_samples = min(high_conf_mask.sum(), max_samples_per_round)
                high_conf_indices = np.where(high_conf_mask)[0]
                selected_indices = np.random.choice(high_conf_indices, n_samples, replace=False)
            else:
                # Try to balance, but use minimum of available samples instead of strict cap
                n_per_class = min(len(pos_indices), len(neg_indices), max_samples_per_round // 2)

                if n_per_class == 0:
                    # Fallback: take at least some samples from each available class
                    n_per_class = min(len(pos_indices), len(neg_indices))

                selected_pos = np.random.choice(pos_indices, n_per_class, replace=False)
                selected_neg = np.random.choice(neg_indices, n_per_class, replace=False)
                selected_indices = np.concatenate([selected_pos, selected_neg])

            if len(pos_indices) > 0 and len(neg_indices) > 0:
                print(f"[ROUND {round_idx + 1}] Selected: {len(np.where(conf_mask_pos[selected_indices])[0])} positive + {len(np.where(conf_mask_neg[selected_indices])[0])} negative")
            else:
                print(f"[ROUND {round_idx + 1}] Selected: {len(selected_indices)} samples")
        else:
            n_samples = min(high_conf_mask.sum(), max_samples_per_round)
            high_conf_indices = np.where(high_conf_mask)[0]
            selected_indices = np.random.choice(high_conf_indices, n_samples, replace=False)
            print(f"[ROUND {round_idx + 1}] Selected: {n_samples} samples")

        # Add to visible sets (both original and processed)
        vis_X = pd.concat([vis_X, inv_X.iloc[selected_indices]], ignore_index=True)
        vis_Y = pd.concat([vis_Y, pd.Series(predictions[selected_indices])], ignore_index=True)
        vis_X_proc = pd.concat([vis_X_proc, inv_X_proc.iloc[selected_indices]], ignore_index=True)

        # Remove from invisible sets
        inv_X = inv_X.drop(inv_X.index[selected_indices]).reset_index(drop=True)
        inv_Y = inv_Y.drop(inv_Y.index[selected_indices]).reset_index(drop=True)
        inv_X_proc = inv_X_proc.drop(inv_X_proc.index[selected_indices]).reset_index(drop=True)

        # Store history
        history.append({
            'round': round_idx + 1,
            'n_visible': len(vis_X),
            'n_invisible': len(inv_X),
            'n_added': len(selected_indices),
            'metrics': metrics
        })

        if len(inv_X) == 0:
            print(f"[ROUND {round_idx + 1}] No more invisible samples. Stopping.")
            break

    # Final prediction
    print(f"\n{'='*60}")
    print("FINAL PREDICTION")
    print(f"{'='*60}\n")
    vis_X_sel = vis_X_proc[best_features]
    inv_X_sel = inv_X_proc[best_features]
    clf_final = train_classifier(vis_X_sel, vis_Y)
    final_predictions = clf_final.predict(inv_X_sel)
    final_metrics = evaluate_predictions(vis_Y, inv_Y, final_predictions)

    return {
        'predictions': final_predictions,
        'metrics': final_metrics,
        'history': history
    }


# =============================================================================
# Example Usage
# =============================================================================

if __name__ == "__main__":
    # Load data
    df = pd.read_csv("data/medical/.ckpt/NOPXY__Q1_full.csv")
    Y, X = df['label'], df.drop(columns=['label', 'patient_id'])

    # Split into visible/invisible
    np.random.seed(42)
    vis_indices = X.sample(n=50).index
    inv_indices = X.index.difference(vis_indices)

    vis_X, inv_X = X.loc[vis_indices].reset_index(drop=True), X.loc[inv_indices].reset_index(drop=True)
    vis_Y, inv_Y = Y.loc[vis_indices].reset_index(drop=True), Y.loc[inv_indices].reset_index(drop=True)

    print("="*60)
    print("FEW-SHOT CLASSIFICATION")
    print("="*60)
    print(f"Visible samples: {len(vis_X)}")
    print(f"Invisible samples: {len(inv_X)}")
    print(f"Available features: {list(X.columns)}\n")

    # Make predictions
    result = few_shot_classify(vis_X, vis_Y, inv_X, top_k=10)

    # Evaluate performance
    metrics = evaluate_predictions(vis_Y, inv_Y, result['predictions'])

    print("="*60)
    print("RESULTS")
    print("="*60)
    print(f"Top-{result['top_k']} features: {result['features']}")
    print(f"F1-score: {metrics['f1']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")

    # Optional: Try self-training (commented out by default)
    # Uncomment below to test self-training:
    
    print("\n" + "="*60)
    print("SELF-TRAINING EXPERIMENT")
    print("="*60)
    
    vis_X_st = vis_X.copy()
    vis_Y_st = vis_Y.copy()
    inv_X_st = inv_X.copy()
    inv_Y_st = inv_Y.copy()
    
    result_st = self_training(
        vis_X_st, vis_Y_st, inv_X_st, inv_Y_st,
        n_rounds=5,
        confidence_threshold=0.95,
        max_samples_per_round=10,
        balance_classes=True
    )
    
    print(f"\nSelf-Training F1: {result_st['metrics']['f1']:.4f}")
    print(f"Improvement: {(result_st['metrics']['f1'] - metrics['f1'])*100:+.2f}%")
