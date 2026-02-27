import pandas as pd
import numpy as np
import logging
from typing import Tuple
from sklearn.calibration import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler


logger = logging.getLogger(__name__)


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """Encode categorical variables."""
    df_proc = df.copy()
    for col in df_proc.select_dtypes(include=['object']).columns:
        encoded = LabelEncoder().fit_transform(df_proc[col].astype(str))
        df_proc[col] = pd.Series(encoded, index=df_proc.index, dtype=int)
    return df_proc


def norm_features(df: pd.DataFrame) -> pd.DataFrame:
    scaler = StandardScaler()
    df_norm = scaler.fit_transform(df)
    return pd.DataFrame(df_norm, columns=df.columns, index=df.index)


def weight_features(df: pd.DataFrame, feat_importance: pd.DataFrame) -> pd.DataFrame:
    df_proc = df.copy()

    feature_weight = np.ones(len(df.columns.tolist()))
    for tup in feat_importance.itertuples():
        feat, weight = tup[1], tup[2]
        idx = df.columns.tolist().index(feat)
        feature_weight[idx] = weight

    df_proc = df_proc * feature_weight

    return df_proc


def train_classifier(
        X: pd.DataFrame, Y: pd.Series,
        n_estimators=100,
        max_depth=10) -> RandomForestClassifier:

    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42,
        class_weight='balanced'
    )
    clf.fit(X, Y)

    return clf


def evaluate_classifier(Y_true: pd.Series, Y_pred: pd.Series) -> dict:
    # Calculate TP, FP, TN, FN
    TP = ((Y_true == 1) & (Y_pred == 1)).sum()
    FP = ((Y_true == 0) & (Y_pred == 1)).sum()
    TN = ((Y_true == 0) & (Y_pred == 0)).sum()
    FN = ((Y_true == 1) & (Y_pred == 0)).sum()

    assert TP + FP + TN + FN == len(Y_true), (
        "Sum of TP, FP, TN, FN must equal total samples. "
        f"Got {TP + FP + TN + FN} samples, expected {len(Y_true)}."
    )

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'TP': TP,
        'FP': FP,
        'TN': TN,
        'FN': FN,
    }

def compute_feature_importance(
        X: pd.DataFrame, Y: pd.Series,
        n_estimators=100,
        max_depth=10) -> pd.DataFrame:

    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42,
        class_weight='balanced'
    )
    clf.fit(X, Y)

    feat_importances = pd.DataFrame({
        "feature": X.columns.tolist(),
        "importance": clf.feature_importances_
    }).sort_values("importance", ascending=False)

    return feat_importances


def pred_and_eval(df: pd.DataFrame, labels: pd.Series) -> dict:
    # Train enriched classifier (all features)
    logger.info(f"Training enriched classifier with {len(df.columns)} features.")

    df_proc = encode_features(df)

    clf = train_classifier(df_proc, labels)
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


def clf_to_rules(
    clf: RandomForestClassifier,
    feature_names: list[str],
    disjunction_budget: int,
    y_train: np.ndarray,
    debug: bool = False
) -> list[list[Tuple[str, float, str]]]:

    assert len(feature_names) == clf.n_features_in_

    # ---- Estimate forest statistics ----
    leaf_sizes = []
    depths = []

    N = len(y_train)
    pos_total = np.sum(y_train == 1)
    pi = pos_total / N

    for tree in clf.estimators_:
        tree_ = tree.tree_
        leaf_mask = tree_.children_left == -1  # type: ignore
        leaf_sizes.extend(tree_.n_node_samples[leaf_mask])  # type: ignore
        depths.append(tree_.max_depth)  # type: ignore

    avg_leaf_size = np.mean(leaf_sizes)
    avg_depth = int(np.mean(depths))

    # ---- Adaptive parameters ----
    min_support = int(max(0.5 * avg_leaf_size, np.sqrt(N) * pi))
    min_support = max(min_support, 3)

    max_rule_length = min(6, avg_depth)

    # recall emphasis for rare class
    beta = 1.0 / np.sqrt(max(pi, 1e-6))

    length_penalty = 0.01 * (1 / avg_depth)

    candidates = []

    for tree in clf.estimators_:
        tree_ = tree.tree_

        children_left = tree_.children_left  # type: ignore
        children_right = tree_.children_right  # type: ignore
        feature = tree_.feature  # type: ignore
        threshold = tree_.threshold  # type: ignore
        value = tree_.value
        n_samples = tree_.n_node_samples  # type: ignore

        def traverse(node_id: int, path: list):

            if children_left[node_id] == children_right[node_id]:

                counts = value[node_id][0]
                pos_count = counts[1]
                total = n_samples[node_id]

                if pos_count == 0:
                    return

                if total < min_support:
                    return

                if len(path) > max_rule_length:
                    return

                # ---- precision ----
                precision = (pos_count + 1.0) / (total + 2.0)

                # ---- recall ----
                recall = pos_count / pos_total if pos_total > 0 else 0

                # ---- F_beta score ----
                if precision + recall > 0:
                    f_beta = (
                        (1 + beta**2) * precision * recall /
                        (beta**2 * precision + recall)
                    )
                else:
                    f_beta = 0

                score = f_beta * np.log1p(total)

                # slight length regularization
                score -= length_penalty * len(path)

                candidates.append(
                    (path.copy(), precision, total, score)
                )
                return

            feat_name = feature_names[feature[node_id]]
            thresh = threshold[node_id]

            if children_left[node_id] != -1:
                path.append((feat_name, thresh, "<="))
                traverse(children_left[node_id], path)
                path.pop()

            if children_right[node_id] != -1:
                path.append((feat_name, thresh, ">"))
                traverse(children_right[node_id], path)
                path.pop()

        traverse(0, [])

    candidates.sort(key=lambda x: x[3], reverse=True)

    # deduplicate
    seen = set()
    selected = []
    for rule, prec, supp, score in candidates:
        key = tuple(rule)
        if key not in seen:
            seen.add(key)
            selected.append((rule, prec, supp, score))
        if len(selected) >= disjunction_budget:
            break

    rules = [r for r, _, _, _ in selected]

    if debug and selected:
        confs = [c for _, c, _, _ in selected]
        supps = [s for _, _, s, _ in selected]
        lens = [len(r) for r in rules]

        print(
            f"[Adaptive Rules] "
            f"N={N}, pi={pi:.3f}, "
            f"min_sup={min_support}, "
            f"max_len={max_rule_length}, "
            f"beta={beta:.2f} | "
            f"conf=({np.mean(confs):.3f}), "
            f"supp=({np.mean(supps):.1f}), "
            f"len=({np.mean(lens):.1f})"
        )

    return rules

def apply_rules(rules: list, df: pd.DataFrame, debug: bool = False) -> pd.Series:
    if not rules:
        return pd.Series(0, index=df.index)

    result = pd.Series(False, index=df.index)

    for rule in rules:
        assert rule, "Empty rule is not allowed."

        mask = pd.Series(True, index=df.index)
        for feat_name, thresh, op in rule:
            if op == '<=':
                mask &= (df[feat_name] <= thresh)
            else:
                mask &= (df[feat_name] > thresh)
        if debug:
            logger.info(f"Applied: {_visualize_rule(rule)}")
            logger.info(f"  Rule coverage: {mask.sum()} / {len(df)} samples")

        result |= mask

    return result.astype(int)


def loss_by_selectivity(Y_A: pd.Series, Y_B: pd.Series, pi: float) -> float:

    assert 0 < pi < 1, f"pi must be in (0, 1), got {pi}"
    assert len(Y_A) == len(Y_B), "Y_A and Y_B must have the same length"

    y_a = np.asarray(Y_A)
    y_b = np.asarray(Y_B)

    numerator = max(pi, 1 - pi)
    losses = np.zeros(len(y_a))

    # Case 1: Y_A[i] == 1 and Y_B[i] == 0
    mask_1_0 = (y_a == 1) & (y_b == 0)
    losses[mask_1_0] = numerator / pi

    # Case 2: Y_A[i] == 0 and Y_B[i] == 1
    mask_0_1 = (y_a == 0) & (y_b == 1)
    losses[mask_0_1] = numerator / (1 - pi)

    # Case 3: Y_A[i] == Y_B[i] (loss is 0, already initialized)

    avg_loss = losses.mean()

    return avg_loss
    


def _visualize_rule(rule: list) -> str:
    conditions = [f"{feat} {op} {thresh:.3f}" for feat, thresh, op in rule]
    return " AND ".join(conditions)

