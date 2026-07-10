# ruff: noqa: B023
# pylint: disable=missing-function-docstring,invalid-name
# pylint: disable=too-many-arguments,too-many-positional-arguments
# pylint: disable=too-many-locals,too-many-branches,too-many-statements
# pylint: disable=consider-using-enumerate,logging-fstring-interpolation
# pylint: disable=cell-var-from-loop,unnecessary-lambda
"""Workload-level modeling, rule, and feature helpers."""

import logging

import numpy as np
import pandas as pd
from sklearn.calibration import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """Encode categorical variables."""
    df_proc = df.copy()
    for col in df_proc.select_dtypes(include=["object"]).columns:
        encoded = LabelEncoder().fit_transform(df_proc[col].astype(str))
        df_proc[col] = pd.Series(encoded, index=df_proc.index, dtype=int)
    return df_proc


def norm_features(df: pd.DataFrame) -> pd.DataFrame:
    scaler = StandardScaler()
    df_norm = scaler.fit_transform(df)
    return pd.DataFrame(df_norm, columns=df.columns, index=df.index)


def weight_features(
    df: pd.DataFrame, feat_importance: pd.DataFrame
) -> pd.DataFrame:
    df_proc = df.copy()

    feature_weight = np.ones(len(df.columns.tolist()))
    for tup in feat_importance.itertuples():
        feat, weight = tup[1], tup[2]
        idx = df.columns.tolist().index(feat)
        feature_weight[idx] = weight

    df_proc = df_proc * feature_weight

    return df_proc


def train_classifier(
    X: pd.DataFrame, Y: pd.Series, n_estimators=100, max_depth=10
) -> RandomForestClassifier:
    # Sklearn classifiers require integer labels.
    Y = Y.astype(int)
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42,
        class_weight="balanced",
    )
    clf.fit(X, Y)

    return clf


def evaluate_classifier(Y_true: pd.Series, Y_pred: pd.Series) -> dict:
    # Convert labels to integers for consistency
    Y_true = Y_true.astype(int)
    Y_pred = Y_pred.astype(int)

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
    f1 = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0
    )

    return {
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "TP": TP,
        "FP": FP,
        "TN": TN,
        "FN": FN,
    }


def compute_feature_importance(
    X: pd.DataFrame, Y: pd.Series, n_estimators=100, max_depth=10
) -> pd.DataFrame:

    if X.shape[1] == 0:
        logger.warning(
            "No features available for feature importance computation. "
            "Returning empty feature-importance table."
        )
        return pd.DataFrame({"feature": [], "importance": []})

    # Sklearn classifiers require integer labels.
    Y = Y.astype(int)

    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42,
        class_weight="balanced",
    )
    clf.fit(X, Y)

    feat_importances = pd.DataFrame(
        {"feature": X.columns.tolist(), "importance": clf.feature_importances_}
    ).sort_values("importance", ascending=False)

    return feat_importances


def clf_to_rules(
    clf: RandomForestClassifier,
    feature_names: list[str],
    disjunction_budget: int,
    X_train: np.ndarray,
    y_train: np.ndarray,
    debug: bool = False,
) -> list[list[tuple[str, float, str]]]:

    assert len(feature_names) == clf.n_features_in_

    # Convert labels to integers for consistency
    y_train = y_train.astype(int)

    N = len(y_train)
    pos_mask = y_train == 1
    neg_mask = y_train == 0

    # Fallback: if all samples are of the same class, return empty rule set
    if len(np.unique(y_train)) == 1:
        fallback_rules = []
        for feature_name in feature_names:
            if not feature_name.startswith("llm_label_"):
                continue
            fallback_rules.append(
                [(feature_name, 0.5, ">")]
            )  # Return the LLM predicated label.
        if len(fallback_rules) > 0:
            logger.warning(
                "Generated fallback rules based on LLM-predicated labels."
            )
        else:
            logger.warning(
                "All training samples belong to the same class (%s). "
                "Returning empty rule set.",
                y_train[0],
            )
        return fallback_rules

    _lambda = sum(neg_mask) / max(1, sum(pos_mask))

    # ------------------------------------------------------------
    # Canonicalization
    # ------------------------------------------------------------
    def canonicalize_rule(path):
        intervals = {}

        for feat, thresh, op in path:
            if feat not in intervals:
                intervals[feat] = [-np.inf, np.inf]

            if op == "<=":
                intervals[feat][1] = min(intervals[feat][1], thresh)
            else:
                intervals[feat][0] = max(intervals[feat][0], thresh)

        return tuple(
            (feat, low, high) for feat, (low, high) in sorted(intervals.items())
        )

    # ------------------------------------------------------------
    # Evaluate rule coverage on training data
    # ------------------------------------------------------------
    def evaluate_rule(rule):
        mask = np.ones(N, dtype=bool)

        for feat, low, high in rule:
            idx = feature_names.index(feat)

            if low != -np.inf:
                mask &= X_train[:, idx] > low
            if high != np.inf:
                mask &= X_train[:, idx] <= high

        return mask

    # ------------------------------------------------------------
    # Extract candidate rules from forest
    # ------------------------------------------------------------
    candidates = []

    for tree in clf.estimators_:
        tree_ = tree.tree_

        cl = tree_.children_left  # type: ignore
        cr = tree_.children_right  # type: ignore
        feat = tree_.feature  # type: ignore
        thr = tree_.threshold  # type: ignore
        val = tree_.value

        def traverse(node_id: int, path: list):

            if cl[node_id] == cr[node_id]:  # leaf
                pos = val[node_id][0][1]
                if pos == 0:
                    return

                canon = canonicalize_rule(path)
                candidates.append(canon)
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

    # Remove duplicates
    candidates = sorted(set(candidates), key=lambda r: str(r))

    # ------------------------------------------------------------
    # Greedy marginal coverage selection
    # ------------------------------------------------------------
    selected = []
    uncovered_pos = pos_mask.copy()

    for _ in range(disjunction_budget):
        best_rule = None
        best_gain = -1 * np.inf
        best_mask = None

        for rule in candidates:
            mask = evaluate_rule(rule)

            new_pos = np.sum(mask & uncovered_pos)
            new_neg = np.sum(mask & neg_mask)

            # Gain function (tunable)
            gain = new_pos - _lambda * new_neg

            if gain > best_gain:
                best_gain = gain
                best_rule = rule
                best_mask = mask

        if best_rule is None or best_gain <= 0:
            break

        selected.append(best_rule)

        # Remove covered positives
        assert best_mask is not None, (
            "best_mask should not be None when best_rule is selected."
        )
        uncovered_pos &= ~best_mask

        # Remove rule from candidates
        candidates.remove(best_rule)

        if np.sum(uncovered_pos) == 0:
            break

    # ------------------------------------------------------------
    # Reconstruct readable rules
    # ------------------------------------------------------------
    rules = []

    for rule in selected:
        reconstructed = []
        for feat, low, high in rule:
            if low != -np.inf:
                reconstructed.append((feat, low, ">"))
            if high != np.inf:
                reconstructed.append((feat, high, "<="))
        rules.append(reconstructed)

    if debug:
        print(f"[Marginal OR Rules] _lambda={_lambda}, selected={len(rules)}")

    return rules


def apply_rules(
    rules: list, df: pd.DataFrame, debug: bool = False
) -> pd.Series:
    if not rules:
        return pd.Series(
            1, index=df.index
        )  # Empty rule means accept all tuples.

    result = pd.Series(False, index=df.index)

    for rule in rules:
        assert rule, "Empty rule is not allowed."

        mask = pd.Series(True, index=df.index)
        for feat_name, thresh, op in rule:
            if op == "<=":
                mask &= df[feat_name] <= thresh
            else:
                mask &= df[feat_name] > thresh
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
