import pandas as pd
import logging
from typing import Tuple
from sklearn.calibration import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score


logger = logging.getLogger(__name__)


def pred_and_eval(df: pd.DataFrame, labels: pd.Series) -> dict:
    # Train enriched classifier (all features)
    logger.info(f"Training enriched classifier with {len(df.columns)} features.")

    df_proc = df.copy()
    for col in df_proc.select_dtypes(include=['object']).columns:
        df_proc[col] = LabelEncoder().fit_transform(df_proc[col].astype(str))

    clf = _train_classifier(df_proc, labels)
    preds = clf.predict(df_proc)
    f1 = f1_score(labels, preds, zero_division=0)

    # Feature importance
    feature_importance = dict(zip(df.columns, clf.feature_importances_))
    # Sort by importance (descending)
    feature_importance = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))

    # Bad cases (misclassified samples) with prediction probabilities for uncertainty
    misclassified_mask = labels != preds
    pred_proba = clf.predict_proba(df_proc)[:, 1]  # Probability of positive class
    bad_cases = df[misclassified_mask].copy()

    # Preserve original index for looking up source data
    bad_cases['_original_index'] = bad_cases.index
    bad_cases['_true_label'] = labels[misclassified_mask].values
    bad_cases['_predicted_label'] = preds[misclassified_mask]
    bad_cases['_pred_proba'] = pred_proba[misclassified_mask]

    # Sort by uncertainty (probability closest to 0.5 = most uncertain)
    bad_cases['_uncertainty'] = abs(bad_cases['_pred_proba'] - 0.5)
    bad_cases = bad_cases.sort_values('_uncertainty', ascending=True)
    bad_cases = bad_cases.drop(columns=['_uncertainty'])

    logger.info(f"Enriched F1: {f1:.4f}")
    logger.info(f"Bad cases: {misclassified_mask.sum()} / {len(labels)} samples misclassified")

    return {
        'f1': f1,
        'feature_importance': feature_importance,
        'bad_cases': bad_cases,
    }
    

def _train_classifier(X: pd.DataFrame, Y: pd.Series) -> RandomForestClassifier:

    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        class_weight='balanced'
    )
    clf.fit(X, Y)

    return clf
