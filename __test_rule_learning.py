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
        Dictionary with predictions, metrics, and selected features
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
        'top_k': top_k,
        'features': best_features,
        'importance': importances
    }

    # Add metrics if labels available
    if 'label' in inv_X:
        result['metrics'] = evaluate_predictions(vis_Y, inv_X['label'], predictions)

    return result


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

    # Make predictions (REAL SCENARIO - no labels needed for invisible data)
    result = few_shot_classify(vis_X, vis_Y, inv_X, top_k=10)

    # Evaluate performance using predictions and ground truth
    metrics = evaluate_predictions(vis_Y, inv_Y, result['predictions'])

    print("="*60)
    print("RESULTS")
    print("="*60)
    print(f"Top-{result['top_k']} features: {result['features']}")
    print(f"F1-score: {metrics['f1']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
