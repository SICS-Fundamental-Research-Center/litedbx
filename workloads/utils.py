# ruff: noqa: B023
# pylint: disable=missing-function-docstring,invalid-name
# pylint: disable=too-many-arguments,too-many-positional-arguments
# pylint: disable=too-many-locals,too-many-branches,too-many-statements
# pylint: disable=consider-using-enumerate,logging-fstring-interpolation
# pylint: disable=cell-var-from-loop,unnecessary-lambda
"""Workload-level modeling, rule, and feature helpers."""

import logging
from typing import Protocol, cast

import numpy as np
import pandas as pd
from sklearn.calibration import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

logger = logging.getLogger(__name__)


class _SklearnTree(Protocol):
    """Structural subset of sklearn's Cython ``Tree`` used for traversal."""

    children_left: np.ndarray
    children_right: np.ndarray
    feature: np.ndarray
    threshold: np.ndarray
    value: np.ndarray


def class_balanced_sample_weights(
    labels: pd.Series | np.ndarray,
    base_weight: pd.Series | np.ndarray | None = None,
) -> np.ndarray:
    """Balance classes while preserving relative design weights."""
    label_values = np.asarray(labels)
    if len(label_values) == 0:
        return np.array([], dtype=float)

    weights = (
        np.ones(len(label_values), dtype=float)
        if base_weight is None
        else np.asarray(base_weight, dtype=float).copy()
    )
    if len(weights) != len(label_values):
        raise ValueError("base_weight must align with labels.")
    if not np.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError("base_weight must be finite and positive.")

    classes = np.unique(label_values)
    for class_value in classes:
        class_mask = label_values == class_value
        class_mass = float(weights[class_mask].sum())
        weights[class_mask] /= len(classes) * class_mass
    return weights


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """Encode categorical variables."""
    df_proc = df.copy()
    for col in df_proc.select_dtypes(include=["object"]).columns:
        encoded = np.asarray(
            LabelEncoder().fit_transform(df_proc[col].astype(str)),
            dtype=np.int64,
        )
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
    X: pd.DataFrame,
    Y: pd.Series,
    n_estimators=100,
    max_depth=10,
    min_samples_leaf=1,
    sample_weight: pd.Series | np.ndarray | None = None,
    estimated_selectivity: float | None = None,
) -> RandomForestClassifier:
    # Sklearn classifiers require integer labels.
    Y = Y.astype(int)

    estimated_class_weight = None if \
        estimated_selectivity is None else {
        0: 1,
        1: (1 - estimated_selectivity) / estimated_selectivity
    }

    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=42,
        class_weight=estimated_class_weight
    )
    clf.fit(X, Y, sample_weight=sample_weight)

    return clf


def compute_feature_importance(
    X: pd.DataFrame,
    Y: pd.Series,
    n_estimators=100,
    max_depth=10,
    sample_weight: pd.Series | np.ndarray | None = None,
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
        class_weight=None if sample_weight is not None else "balanced",
    )
    clf.fit(X, Y, sample_weight=sample_weight)

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
    X_reference: np.ndarray | None = None,
    y_reference: np.ndarray | None = None,
    sample_weight: np.ndarray | None = None,
    allowed_rule_features: set[str] | None = None,
    debug: bool = False,
) -> list[list[tuple[str, float, str]]]:
    """Distill forest predictions into a bounded, coherent decision-tree UCQ."""
    assert len(feature_names) == clf.n_features_in_
    if disjunction_budget <= 0:
        return []
    if (X_reference is None) != (y_reference is None):
        raise ValueError(
            "X_reference and y_reference must either both be set or both be "
            "None."
        )

    train_y = np.asarray(y_train, dtype=int)
    weights = (
        np.ones(len(train_y), dtype=float)
        if sample_weight is None
        else np.asarray(sample_weight, dtype=float)
    )
    if len(weights) != len(train_y):
        raise ValueError("sample_weight must align with y_train.")

    reference_X = X_train if X_reference is None else X_reference
    reference_y = train_y if y_reference is None else np.asarray(y_reference)
    reference_y = reference_y.astype(int)
    if len(reference_X) != len(reference_y):
        raise ValueError("Reference features and labels must align.")
    if reference_X.shape[1] != len(feature_names):
        raise ValueError(
            "Reference features must align with the classifier schema."
        )
    if len(reference_y) == 0:
        return []

    eligible_positions = [
        index
        for index, feature_name in enumerate(feature_names)
        if allowed_rule_features is None
        or feature_name in allowed_rule_features
    ]
    if not eligible_positions:
        return [[]] if reference_y.mean() >= 0.5 else []

    surrogate_names = [feature_names[index] for index in eligible_positions]
    reference_features = np.asarray(reference_X)[:, eligible_positions]
    train_features = np.asarray(X_train)[:, eligible_positions]
    unique_reference_labels = np.unique(reference_y)
    if len(unique_reference_labels) == 1:
        return [[]] if unique_reference_labels[0] == 1 else []

    surrogate = DecisionTreeClassifier(
        max_leaf_nodes=disjunction_budget + 1,
        random_state=42,
    )
    surrogate.fit(reference_features, reference_y)

    def canonicalize_rule(path):
        intervals = {}
        for feature, threshold, operator in path:
            intervals.setdefault(feature, [-np.inf, np.inf])
            if operator == "<=":
                intervals[feature][1] = min(intervals[feature][1], threshold)
            else:
                intervals[feature][0] = max(intervals[feature][0], threshold)
        return tuple(
            (feature, low, high)
            for feature, (low, high) in sorted(intervals.items())
        )

    tree = cast(_SklearnTree, surrogate.tree_)
    positive_class_positions = np.flatnonzero(surrogate.classes_ == 1)
    if len(positive_class_positions) != 1:
        return []
    positive_class_position = int(positive_class_positions[0])
    candidate_rules = []

    def traverse(node_id: int, path: list) -> None:
        if tree.children_left[node_id] == tree.children_right[node_id]:
            predicted_class = int(np.argmax(tree.value[node_id][0]))
            if predicted_class == positive_class_position:
                candidate_rules.append(canonicalize_rule(path))
            return

        feature_name = surrogate_names[tree.feature[node_id]]
        threshold = tree.threshold[node_id]
        traverse(
            tree.children_left[node_id],
            path + [(feature_name, threshold, "<=")],
        )
        traverse(
            tree.children_right[node_id],
            path + [(feature_name, threshold, ">")],
        )

    traverse(0, [])
    candidate_rules = sorted(set(candidate_rules), key=str)

    feature_positions = {
        feature_name: index
        for index, feature_name in enumerate(surrogate_names)
    }

    def evaluate_rule(rule, data):
        mask = np.ones(len(data), dtype=bool)
        for feature, low, high in rule:
            position = feature_positions[feature]
            if low != -np.inf:
                mask &= data[:, position] > low
            if high != np.inf:
                mask &= data[:, position] <= high
        return mask

    if len(candidate_rules) > disjunction_budget:
        train_y_bool = train_y.astype(bool)

        def rule_priority(rule):
            reference_mask = evaluate_rule(rule, reference_features)
            benefit = int(
                np.count_nonzero(reference_mask & reference_y.astype(bool))
                - np.count_nonzero(reference_mask & ~reference_y.astype(bool))
            )
            train_mask = evaluate_rule(rule, train_features)
            annotation_error = float(
                np.average(train_mask != train_y_bool, weights=weights)
            )
            return (-benefit, annotation_error, len(rule), str(rule))

        candidate_rules = sorted(candidate_rules, key=rule_priority)[
            :disjunction_budget
        ]

    rules = []
    reference_prediction = np.zeros(len(reference_y), dtype=bool)
    for rule in candidate_rules:
        reference_prediction |= evaluate_rule(rule, reference_features)
        reconstructed = []
        for feature, low, high in rule:
            if low != -np.inf:
                reconstructed.append((feature, low, ">"))
            if high != np.inf:
                reconstructed.append((feature, high, "<="))
        rules.append(reconstructed)

    if debug:
        logger.info(
            "Distilled %s rule(s) with population disagreement %.6f.",
            len(rules),
            float(np.mean(reference_prediction != reference_y.astype(bool))),
        )
    return rules


def apply_rules(
    rules: list, df: pd.DataFrame, debug: bool = False
) -> pd.Series:
    if not rules:
        return pd.Series(0, index=df.index, dtype=int)
    if any(not rule for rule in rules):
        return pd.Series(
            1, index=df.index, dtype=int
        )  # An empty conjunction accepts all tuples.

    result = pd.Series(False, index=df.index)

    for rule in rules:
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
    if len(Y_A) == 0:
        return 0.0

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
