import pandas as pd
import numpy as np
import logging
from typing import Tuple, List
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
    feature_names: List[str],
    disjunction_budget: int,
    y_train: np.ndarray,
    debug: bool = False
) -> List[List[Tuple[str, float, str]]]:

    assert len(feature_names) == clf.n_features_in_

    # ------------------------------------------------------------------
    # 1. Dataset & forest statistics
    # ------------------------------------------------------------------
    N = len(y_train)
    pos_total = int(np.sum(y_train == 1))
    pi = pos_total / max(N, 1)

    leaf_sizes, depths = [], []
    for tree in clf.estimators_:
        tree_ = tree.tree_
        is_leaf = tree_.children_left == -1
        leaf_sizes.extend(tree_.n_node_samples[is_leaf])
        depths.append(tree_.max_depth)

    avg_leaf_size = np.mean(leaf_sizes)
    avg_depth = max(int(np.mean(depths)), 1)

    # ------------------------------------------------------------------
    # 2. Adaptive hyperparameters
    # ------------------------------------------------------------------
    min_support = int(max(0.5 * avg_leaf_size, np.sqrt(N) * pi, 3))
    max_rule_length = min(6, avg_depth)

    beta = 1.0 / np.sqrt(max(pi, 1e-6))
    length_penalty = 0.01 / avg_depth

    # ------------------------------------------------------------------
    # 3. Canonicalization
    # ------------------------------------------------------------------
    def canonicalize_rule(path):
        intervals = {}

        for feat, thresh, op in path:
            if feat not in intervals:
                intervals[feat] = [-np.inf, np.inf]

            if op == "<=":
                intervals[feat][1] = min(intervals[feat][1], thresh)
            else:  # ">"
                intervals[feat][0] = max(intervals[feat][0], thresh)

        return tuple(
            (feat, low, high)
            for feat, (low, high) in sorted(intervals.items())
        )

    # ------------------------------------------------------------------
    # 4. Rule extraction
    # ------------------------------------------------------------------
    candidates = []

    for tree in clf.estimators_:
        tree_ = tree.tree_

        cl = tree_.children_left
        cr = tree_.children_right
        feat = tree_.feature
        thr = tree_.threshold
        val = tree_.value
        node_samples = tree_.n_node_samples

        def traverse(node_id: int, path: list):

            if cl[node_id] == cr[node_id]:  # leaf
                total = node_samples[node_id]
                pos = val[node_id][0][1]

                if pos == 0 or total < min_support or len(path) > max_rule_length:
                    return

                precision = (pos + 1.0) / (total + 2.0)
                recall = pos / pos_total if pos_total else 0.0

                denom = beta**2 * precision + recall
                f_beta = ((1 + beta**2) * precision * recall / denom) if denom else 0.0

                score = f_beta * np.log1p(total) - length_penalty * len(path)

                canon = canonicalize_rule(path)
                candidates.append((canon, score))
                return

            fname = feature_names[feat[node_id]]
            threshold = thr[node_id]

            path.append((fname, threshold, "<="))
            traverse(cl[node_id], path)
            path.pop()

            path.append((fname, threshold, ">"))
            traverse(cr[node_id], path)
            path.pop()

        traverse(0, [])

    # ------------------------------------------------------------------
    # 5. Subsumption-aware selection
    # ------------------------------------------------------------------
    def subsumes(rule_a, rule_b):
        """
        True if rule_a subsumes rule_b.
        Both are canonical interval tuples:
        (feature, low, high)
        """
        a_dict = {f: (l, h) for f, l, h in rule_a}
        b_dict = {f: (l, h) for f, l, h in rule_b}

        for feat, (low_b, high_b) in b_dict.items():
            if feat not in a_dict:
                return False
            low_a, high_a = a_dict[feat]
            if low_a > low_b or high_a < high_b:
                return False

        return True

    candidates.sort(key=lambda x: x[1], reverse=True)

    selected = []

    for rule, score in candidates:

        # Skip if already covered by a stronger rule
        if any(subsumes(sel_rule, rule) for sel_rule, _ in selected):
            continue

        # Remove weaker rules this rule subsumes
        selected = [
            (sel_rule, sel_score)
            for sel_rule, sel_score in selected
            if not subsumes(rule, sel_rule)
        ]

        selected.append((rule, score))

        if len(selected) >= disjunction_budget:
            break

    # ------------------------------------------------------------------
    # 6. Reconstruct readable rules
    # ------------------------------------------------------------------
    rules = []

    for rule_intervals, _ in selected:
        reconstructed = []
        for feat, low, high in rule_intervals:
            if low != -np.inf:
                reconstructed.append((feat, low, ">"))
            if high != np.inf:
                reconstructed.append((feat, high, "<="))
        rules.append(reconstructed)

    # ------------------------------------------------------------------
    # 7. Debug
    # ------------------------------------------------------------------
    if debug and rules:
        lengths = [len(r) for r in rules]
        print(
            f"[Adaptive Rules] "
            f"N={N}, pi={pi:.3f}, min_sup={min_support}, "
            f"max_len={max_rule_length}, beta={beta:.2f} | "
            f"n_rules={len(rules)}, avg_len={np.mean(lengths):.2f}"
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

