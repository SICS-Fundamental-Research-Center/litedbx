# ruff: noqa: B023
# pylint: disable=missing-function-docstring,invalid-name
# pylint: disable=too-many-arguments,too-many-positional-arguments
# pylint: disable=too-many-locals,too-many-branches,too-many-statements
# pylint: disable=consider-using-enumerate,logging-fstring-interpolation
# pylint: disable=cell-var-from-loop,unnecessary-lambda
"""Workload-level modeling, rule, and feature helpers."""

import logging
from itertools import combinations

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
) -> RandomForestClassifier:
    # Sklearn classifiers require integer labels.
    Y = Y.astype(int)
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=42,
        class_weight=None if sample_weight is not None else "balanced",
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

    assert len(feature_names) == clf.n_features_in_

    # Convert labels to integers for consistency
    y_train = y_train.astype(int)

    N = len(y_train)
    weights = (
        np.ones(N, dtype=float)
        if sample_weight is None
        else np.asarray(sample_weight, dtype=float)
    )
    if len(weights) != N:
        raise ValueError("sample_weight must align with y_train.")
    if (X_reference is None) != (y_reference is None):
        raise ValueError(
            "X_reference and y_reference must either both be set or both be "
            "None."
        )
    if X_reference is not None and X_reference.shape[1] != len(feature_names):
        raise ValueError(
            "Reference features must align with the classifier schema."
        )
    if y_reference is not None:
        assert X_reference is not None
        if len(X_reference) != len(y_reference):
            raise ValueError("Reference features and labels must align.")

    # Without a population reference, preserve a usable single-class fallback.
    if y_reference is None and len(np.unique(y_train)) == 1:
        fallback_rules = []
        for feature_name in feature_names:
            if not feature_name.startswith("llm_label_"):
                continue
            if (
                allowed_rule_features is not None
                and feature_name not in allowed_rule_features
            ):
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
    feature_positions = {
        feature_name: index for index, feature_name in enumerate(feature_names)
    }

    def evaluate_rule(rule, data):
        mask = np.ones(len(data), dtype=bool)

        for feat, low, high in rule:
            idx = feature_positions[feat]

            if low != -np.inf:
                mask &= data[:, idx] > low
            if high != np.inf:
                mask &= data[:, idx] <= high

        return mask

    def balanced_disagreement(target, prediction, row_weights=None):
        target = np.asarray(target, dtype=bool)
        if len(target) == 0:
            return 0.0
        prediction = np.asarray(prediction, dtype=bool)
        local_weights = (
            np.ones(len(target), dtype=float)
            if row_weights is None
            else np.asarray(row_weights, dtype=float)
        )
        positive_weight = local_weights[target].sum()
        negative_weight = local_weights[~target].sum()
        if positive_weight > 0 and negative_weight > 0:
            class_weights = np.where(
                target,
                0.5 / positive_weight,
                0.5 / negative_weight,
            )
            return float(
                (local_weights * class_weights * (target != prediction)).sum()
            )
        return float(np.average(target != prediction, weights=local_weights))

    # ------------------------------------------------------------
    # Extract candidate rules from forest
    # ------------------------------------------------------------
    candidates = []
    positive_class_positions = np.flatnonzero(clf.classes_ == 1)
    if len(positive_class_positions) != 1:
        return []
    positive_class_position = int(positive_class_positions[0])

    for tree in clf.estimators_:
        tree_ = tree.tree_

        cl = tree_.children_left  # type: ignore
        cr = tree_.children_right  # type: ignore
        feat = tree_.feature  # type: ignore
        thr = tree_.threshold  # type: ignore
        val = tree_.value

        def traverse(node_id: int, path: list):

            if cl[node_id] == cr[node_id]:  # leaf
                predicted_class_position = int(np.argmax(val[node_id][0]))
                if predicted_class_position != positive_class_position:
                    return
                if not path:
                    candidates.append(canonicalize_rule(path))
                    return

                # Include supported generalizations of each positive leaf.
                # A forest path may contain incidental predicates that make the
                # translated UCQ less faithful than a supported sub-conjunction.
                for size in range(1, len(path) + 1):
                    for subset in combinations(path, size):
                        candidate = canonicalize_rule(subset)
                        if allowed_rule_features is None or all(
                            feature in allowed_rule_features
                            for feature, _, _ in candidate
                        ):
                            candidates.append(candidate)
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
    # Select among forest-supported rules using released annotations. Forest
    # agreement resolves annotation-level ties without evaluation labels.
    # ------------------------------------------------------------
    selected = []
    reference_X = X_train if X_reference is None else X_reference
    reference_y = y_train if y_reference is None else y_reference.astype(int)
    reference_prediction = np.zeros(len(reference_y), dtype=bool)
    annotation_prediction = np.zeros(N, dtype=bool)
    current_reference_loss = balanced_disagreement(
        reference_y, reference_prediction
    )
    current_annotation_loss = balanced_disagreement(
        y_train, annotation_prediction, weights
    )
    loss_resolution = 1.0 / max(1, N)
    reference_resolution = 1.0 / max(1, len(reference_y))

    for _ in range(disjunction_budget):
        evaluations = []
        for rule in candidates:
            reference_mask = evaluate_rule(rule, reference_X)
            candidate_prediction = reference_prediction | reference_mask
            reference_loss = balanced_disagreement(
                reference_y, candidate_prediction
            )
            annotation_mask = evaluate_rule(rule, X_train)
            annotation_loss = balanced_disagreement(
                y_train,
                annotation_prediction | annotation_mask,
                weights,
            )
            annotation_improves = (
                annotation_loss < current_annotation_loss - 1e-12
            )
            reference_improves = (
                reference_loss < current_reference_loss - 1e-12
            )
            annotation_acceptable = (
                annotation_loss
                <= current_annotation_loss + loss_resolution
            )
            reference_acceptable = (
                reference_loss
                <= current_reference_loss + reference_resolution
            )
            if not (
                (annotation_improves and reference_acceptable)
                or (reference_improves and annotation_acceptable)
            ):
                continue
            evaluations.append(
                (
                    annotation_loss,
                    reference_loss,
                    len(rule),
                    str(rule),
                    rule,
                    reference_mask,
                    annotation_mask,
                )
            )

        if not evaluations:
            break
        minimum_loss = min(item[0] for item in evaluations)
        eligible = [
            item
            for item in evaluations
            if item[0] <= minimum_loss + loss_resolution
        ]
        best = min(eligible, key=lambda item: (item[2], item[1], item[3]))
        (
            current_annotation_loss,
            current_reference_loss,
            _,
            _,
            best_rule,
            reference_mask,
            annotation_mask,
        ) = best
        selected.append(best_rule)
        reference_prediction |= reference_mask
        annotation_prediction |= annotation_mask
        candidates.remove(best_rule)
        if current_annotation_loss <= 1e-12:
            break

    if not selected and len(reference_y) > 0 and reference_y.mean() >= 0.5:
        selected = [canonicalize_rule([])]

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
        logger.info(
            "Distilled %s rule(s) with annotation disagreement %.6f "
            "and population disagreement %.6f.",
            len(rules),
            current_annotation_loss,
            current_reference_loss,
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
