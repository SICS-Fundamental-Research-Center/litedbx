"""Utility functions for LdbWorkload class.

This module contains helper functions extracted from LdbWorkload to improve
code organization and maintainability.
"""

import logging
import pandas as pd
import numpy as np
import math
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from typing import Tuple
from data_structure import PopulationSpec
from common import (
    encode_features,
    loss_by_selectivity, 
    train_classifier,
    clf_to_rules,
    apply_rules,
    evaluate_classifier,
)


logger = logging.getLogger(__name__)


# ============================================================================
# Contrastive Learning Utilities
# ============================================================================

def build_contrastive_batch(
    sem_pred,
    pos_batch_data: list,
    neg_batch_data: list,
    pos_batch_indices: list,
    neg_batch_indices: list,
    previous_feedback: dict | None,
    labeled_data_df: pd.DataFrame,
) -> Tuple[list, list]:
    """Build data items and metadata for contrastive learning.

    Args:
        sem_pred: Semantic predicate
        pos_batch_data: List of positive sample data
        neg_batch_data: List of negative sample data
        pos_batch_indices: List of positive sample indices
        neg_batch_indices: List of negative sample indices
        previous_feedback: Feedback from previous iteration
        labeled_data_df: Labeled data DataFrame

    Returns:
        Tuple of (data_items, metadata)
    """
    data_items = pos_batch_data + neg_batch_data
    metadata = [
        {"label": True, "sample_id": int(i)} for i in pos_batch_indices
    ] + [
        {"label": False, "sample_id": int(i)} for i in neg_batch_indices
    ]

    # Add bad cases
    if previous_feedback and not previous_feedback['bad_cases'].empty:
        for _, row in previous_feedback['bad_cases'].head(5).iterrows():
            field_content = labeled_data_df.loc[
                int(row['_original_index']), sem_pred.field
            ]
            data_items.append(field_content)
            metadata.append({
                "label": bool(row['_true_label']),
                "sample_id": int(row['_original_index']),
                "is_bad_case": True,
                "misclassified_as": bool(row['_predicted_label'])
            })

    return data_items, metadata


def build_feature_generation_prompt(
    sem_pred,
    feature_space: list[PopulationSpec],
    previous_feedback: dict | None,
    iteration: int,
    feature_budget: int,
    prompt_template: str,
    data_df: pd.DataFrame | None = None,
    num_samples: int = 3,
) -> str:
    """Build prompt for feature generation with conditional sections.

    Args:
        sem_pred: Semantic predicate
        feature_space: Current feature space
        previous_feedback: Feedback from previous iteration
        iteration: Current iteration number
        feature_budget: Maximum number of features
        prompt_template: Prompt template string
        data_df: Optional DataFrame to show current schema and sample data
        num_samples: Number of sample rows to display

    Returns:
        Formatted prompt string
    """
    is_first = iteration == 0 or len(feature_space) == 0

    # Build schema and sample data section
    schema_sample_section = ""
    if data_df is not None and not data_df.empty:
        # List all columns
        columns_str = ", ".join(data_df.columns.tolist())
        schema_section = f"=== Current Data Schema ===\nColumns: {columns_str}\n"

        # Show sample data
        sample_rows = data_df.head(num_samples)
        sample_data_str = sample_rows.to_string(index=True)
        sample_section = f"\n=== Sample Data (first {num_samples} rows) ===\n{sample_data_str}\n"

        schema_sample_section = schema_section + sample_section

    if is_first:
        return prompt_template.format(
            MODALITY=sem_pred.modality,
            DESC=sem_pred.prompt,
            SOURCE_COL=sem_pred.field,
            FEATURE_BUDGET=feature_budget,
            SCHEMA_SAMPLE_SECTION=schema_sample_section,
            PREVIOUS_FEATURES_SECTION="",
            PERFORMANCE_FEEDBACK_SECTION="",
            INSTRUCTIONS_SECTION="Propose an initial set of discriminative features that can effectively distinguish positive from negative samples.",
            CONSTRAINTS_ADDITIONAL="",
        )

    # Build current features section
    current_features_str = "\n".join([
        f"  - {spec.target_col} ({spec.feature_type}): {spec.prompt[:80]}..."
        for spec in feature_space
    ])
    previous_features_section = f"\n\n=== Current Feature Space ===\n{current_features_str}\n"

    # Build feedback section
    if previous_feedback:
        importance_str = "\n".join([
            f"  - {feat}: {imp:.4f}"
            for feat, imp in list(previous_feedback['feature_importance'].items())[:10]
        ])
        performance_feedback_section = f"""
\n=== Performance Feedback ===
\n1. Feature Importance (sorted by importance):\n{importance_str}
\n2. F1 Score:\n   - With current features: {previous_feedback['f1']:.4f}\n
"""
        constraints = """- Make sure "to_remove" contains EXACTLY the feature names from the current feature space (check the "Current Feature Space" section above).
- Avoid proposing features that are already in the current feature space."""
    else:
        performance_feedback_section = ""
        constraints = "- Avoid proposing features that are already in the current feature space."

    return prompt_template.format(
        MODALITY=sem_pred.modality,
        DESC=sem_pred.prompt,
        SOURCE_COL=sem_pred.field,
        FEATURE_BUDGET=feature_budget,
        SCHEMA_SAMPLE_SECTION=schema_sample_section,
        PREVIOUS_FEATURES_SECTION=previous_features_section,
        PERFORMANCE_FEEDBACK_SECTION=performance_feedback_section,
        INSTRUCTIONS_SECTION="Propose features that will improve classification performance based on the feedback above.",
        CONSTRAINTS_ADDITIONAL=constraints,
    )



# ============================================================================
# Label and Ground Truth Utilities
# ============================================================================

def build_ground_truth_labels(
    data: list[pd.DataFrame],
    q_name: str,
    selected_columns: list[str],
    data_dir: str,
    debug: bool = False,
) -> list[pd.Series]:
    """Build ground truth labels for a query.

    Args:
        data: LdbData object
        q_name: Query name
        selected_columns: Columns to select
        data_dir: Data directory path
        debug: Enable debug mode

    Returns:
        List of series of boolean labels
    """
    ground_truth_df = pd.read_csv(f"{data_dir}/ground_truth/{q_name}.csv")
    ground_truth_df = ground_truth_df[selected_columns]
    ground_truth_set = set(tuple(row) for row in ground_truth_df.values)

    labels_li = []

    for d in data:
        labels = d[selected_columns].apply(
            lambda row: tuple(row) in ground_truth_set,
            axis=1
        ).reset_index(drop=True)
        labels_li.append(labels)

        if debug:
            true_rows = d[labels][selected_columns]
            true_rows_set = set(tuple(row) for row in true_rows.values)
            assert true_rows_set == ground_truth_set, \
                f"[DebugErr] Fail to build ground truth labels for query {q_name}."
            logger.info(f"Ground truth of {q_name}: {labels.sum()} positives / {len(labels)} samples. Oracle selectivity: {labels.sum() / len(labels):.4f}")

    return labels_li


# ============================================================================
# Error Estimation Utilities
# ============================================================================

def compute_objective_error(
    pred_Y: pd.Series,
    trans_Y: pd.Series,
    b_rew: int,
    schema_arity: int,
    query_size: int,
    selected_data_size: int,
    delta: float,
) -> Tuple[float, float]:
    """Compute rewriting loss and penalty.

    Args:
        pred_Y: Predicted labels
        trans_Y: Translated labels
        schema_arity: Number of features in schema
        query_size: Number of queries
        selected_data_size: Size of selected data
        delta: Delta parameter for penalty calculation

    Returns:
        Tuple of (L_rew, penalty)
    """
    pi = sum(pred_Y) / len(pred_Y)
    pi = max(1e-6, min(1 - 1e-6, pi))  # Ensure pi is in (0, 1) to avoid extreme penalties

    # Compute rewriting loss
    L_rew = loss_by_selectivity(pred_Y, trans_Y, pi)

    # Compute the penalty
    Gamma_rew = max(pi, 1-pi) / min(pi, 1-pi)
    d_VC = b_rew * schema_arity * math.log(b_rew)

    penalty = Gamma_rew * math.sqrt(
        (d_VC * math.log(selected_data_size) + math.log(2 * query_size / delta)) /
        selected_data_size
    )

    print(f"penalty: {penalty:.4f}, Gamma_rew: {Gamma_rew:.4f}, d_VC: {d_VC}, selected_data_size: {selected_data_size}, query_size: {query_size}, delta: {delta}")
    print(f"pi: {pi:.4f}, sum(pred_Y): {sum(pred_Y)}, len(pred_Y): {len(pred_Y)}")
    print("---" * 10)

    return L_rew, penalty


def compute_subjective_error(
    X: pd.DataFrame,
    Y: pd.Series,
    query_size: int,
    data_size: int,
    delta: float,
    loo_step: int = 10,
) -> Tuple[float, float]:
    """Compute LOO error and penalty for subjective error estimation.

    Args:
        X: Feature DataFrame
        Y: Labels
        schema_arity: Number of features in schema
        query_size: Number of queries
        data_size: Size of data
        delta: Delta parameter
        loo_step: Step size for LOO

    Returns:
        Tuple of (L_LOO, penalty)
    """

    # LOO error could only compute with at least 2 sample.
    if len(X) < 2:
        return 0.0, 0.0

    # If no features are available, there is nothing to train on.
    # Treat this as the trivial baseline instead of fitting sklearn on a
    # zero-column matrix.
    if X.shape[1] == 0:
        return 0.0, 0.0

    Y_loo = []
    Y_true = []

    X_encoded = encode_features(X)
    for i in range(0, len(X_encoded), loo_step):
        X_train = X_encoded.drop(index=i)
        Y_train = Y.drop(index=i)
        X_test = X_encoded.iloc[[i]]

        clf = train_classifier(X_train, Y_train)

        pred = clf.predict(X_test)[0]
        Y_loo.append(pred)
        Y_true.append(Y.iloc[i])

    Y_true_series = pd.Series(Y_true)
    Y_loo_series = pd.Series(Y_loo)

    # Compute LOO loss score
    pi = sum(Y) / len(Y)
    pi = max(1e-6, min(1 - 1e-6, pi))  # Ensure pi is in (0, 1) to avoid extreme penalties
    L_LOO = loss_by_selectivity(Y_true_series, Y_loo_series, pi)

    # Compute the penalty
    Gamma_LOO = max(pi, 1-pi) / min(pi, 1-pi)

    penalty = Gamma_LOO * math.sqrt(
        math.log(2 * query_size / delta) / (2 * data_size)
    )

    return L_LOO, penalty


# ============================================================================
# Reporting Utilities
# ============================================================================

def report_usage_statistics(usage_statistics: dict):
    """Report LLM usage statistics.

    Args:
        usage_statistics: Dictionary with usage stats for each phase
    """
    logger.info("=== LLM Usage Statistics ===")
    for item, stats in usage_statistics.items():
        logger.info((
            f"{item}: Prompt Tokens={stats['prompt_tokens']}, "
            f"Completion Tokens={stats['completion_tokens']}, "
            f"Total Tokens={stats['total_tokens']}, "
            f"Prompt Cost=${stats['prompt_cost']:.4f}, "
            f"Completion Cost=${stats['completion_cost']:.4f}, "
            f"Total Cost=${stats['total_cost']:.4f}"))

    total_cost = sum(stats['total_cost'] for stats in usage_statistics.values())
    logger.info(f"Total LLM Cost: ${total_cost:.4f}")


# ============================================================================
# Reporting Evaluation Trace
# ============================================================================

def report_evaluation_trace(execution_trace: dict):
    """Report the evaluation trace with detailed metrics and rules.

    Args:
        execution_trace: Dictionary containing execution results per iteration
    """
    def _format_rule(condition: tuple) -> str:
        """Format a single rule condition as 'feature op value'.

        Args:
            condition: Tuple of (feature, value, op)

        Returns:
            Formatted rule string
        """
        if len(condition) == 3:
            feature, value, op = condition
            # Format value to 2 decimal places
            if isinstance(value, (int, float)):
                value_str = f"{float(value):.2f}"
            else:
                value_str = str(value)
            return f"{feature} {op} {value_str}"
        return str(condition)


    def _format_rules(rules: list) -> str:
        """Format list of rules into readable string.

        Args:
            rules: List of rule conditions

        Returns:
            Formatted rules string
        """
        if not isinstance(rules, list):
            return str(rules)

        formatted = []
        for rule in rules:
            if isinstance(rule, list) and len(rule) > 0:
                # Join conditions with AND
                conditions = " AND ".join([_format_rule(c) for c in rule])
                formatted.append(f"  IF {conditions}")
            else:
                formatted.append(f"  {rule}")

        return "\n".join(formatted) if formatted else "  (no rules)"


    assert execution_trace is not None, "Execution trace is required for reporting."

    # Get all query names
    all_query_names = set()
    for results in execution_trace.values():
        for key in results.keys():
            if key not in ["rules", "features", "pred_eval", "trans_eval",
                           "L_rew", "penalty_rew", "L_LOO", "penalty_LOO",
                           "L_obj", "L_subj", "L_static", "L_avg", "memory_cost"]:
                all_query_names.add(key)

    # Find best iteration for each query (highest trans_f1) AND global best (lowest L_avg)
    best_trans_f1_iters = {}  # query_name -> (iter_idx, trans_f1)
    global_best_iter = min(execution_trace.keys(),
                          key=lambda i: execution_trace[i].get("L_avg", float('inf')))

    for q_name in all_query_names:
        best_trans_f1 = -1
        best_iter_for_trans = None

        for iter_idx, results in execution_trace.items():
            if "trans_eval" in results and q_name in results["trans_eval"]:
                trans_f1 = results["trans_eval"][q_name].get("f1", -1)
                if trans_f1 > best_trans_f1:
                    best_trans_f1 = trans_f1
                    best_iter_for_trans = iter_idx

        if best_iter_for_trans is not None:
            best_trans_f1_iters[q_name] = (best_iter_for_trans, best_trans_f1)

    # ========== SECTION 1: OVERVIEW TABLE ==========
    overview_data = []
    for iter_idx, results in execution_trace.items():
        for q_name in all_query_names:
            row = {
                "Iter": iter_idx,
                "NFeat": iter_idx,
                "Query": q_name,
            }

            if "pred_eval" in results and q_name in results["pred_eval"]:
                pred_eval = results["pred_eval"][q_name]
                row["pred_f1"] = f"{pred_eval.get('f1', 0):.2f}"
                row["pred_p"] = f"{pred_eval.get('precision', 0):.2f}"
                row["pred_r"] = f"{pred_eval.get('recall', 0):.2f}"

            if "trans_eval" in results and q_name in results["trans_eval"]:
                trans_eval = results["trans_eval"][q_name]
                row["trans_f1"] = f"{trans_eval.get('f1', 0):.2f}"
                row["trans_p"] = f"{trans_eval.get('precision', 0):.2f}"
                row["trans_r"] = f"{trans_eval.get('recall', 0):.2f}"

            row["L_rew"] = f"{results.get('L_rew', {}).get(q_name, 0):.2f}"
            row["penalty_rew"] = f"{results.get('penalty_rew', {}).get(q_name, 0):.2f}"
            row["L_LOO"] = f"{results.get('L_LOO', {}).get(q_name, 0):.2f}"
            row["penalty_LOO"] = f"{results.get('penalty_LOO', {}).get(q_name, 0):.2f}"
            row["L_obj"] = f"{results.get('L_obj', {}).get(q_name, 0):.2f}"
            row["L_subj"] = f"{results.get('L_subj', {}).get(q_name, 0):.2f}"
            row["L_static"] = f"{results.get('L_static', {}).get(q_name, 0):.2f}"
            row["memory_cost"] = f"{results.get('memory_cost', {}).get(q_name, 0):.2f}"

            overview_data.append(row)

    df_overview = pd.DataFrame(overview_data)
    col_order = ["Iter", "NFeat", "Query", "pred_f1", "pred_p", "pred_r",
                 "trans_f1", "trans_p", "trans_r",
                 "L_rew", "penalty_rew", "L_obj",
                 "L_LOO", "penalty_LOO", "L_subj", "L_static", "memory_cost"]
    col_order = [c for c in col_order if c in df_overview.columns]
    df_overview = df_overview[col_order]

    print("\n" + "="*150)
    print("OVERVIEW - Evaluation Metrics per Iteration")
    print("="*150)
    print(df_overview.to_string(index=False))
    print("="*150)

    # ========== SECTION 2: AVERAGE ERROR ==========
    avg_errors = [{
        "Iter": i,
        "NFeat": i,
        "L_avg": f"{results.get('L_avg', 0):.2f}"
    } for i, results in execution_trace.items()]

    print("\nAverage Error per Iteration:")
    print("-" * 40)
    print(pd.DataFrame(avg_errors).to_string(index=False))
    print("-" * 40)

    # ========== SECTION 3: BEST RULES PER QUERY ==========
    print("\n" + "="*100)
    print("BEST RULES PER QUERY")
    print("="*100)

    for q_name in sorted(all_query_names):
        print(f"\n{'='*80}")
        print(f"Query: {q_name}")
        print('='*80)

        # Show rules from iteration with highest trans_f1
        if q_name in best_trans_f1_iters:
            iter_idx, trans_f1 = best_trans_f1_iters[q_name]
            results = execution_trace[iter_idx]

            print(f"\n[Highest trans_f1={trans_f1:.2f}] @ Iter {iter_idx} (NFeat={iter_idx + 1})")

            if "features" in results and q_name in results["features"]:
                print(f"Features: {results['features'][q_name]}")

            if "rules" in results and q_name in results["rules"]:
                rules = results["rules"][q_name]
                print("Rules:")
                print(_format_rules(rules))

            if "trans_eval" in results and q_name in results["trans_eval"]:
                te = results["trans_eval"][q_name]
                print(f"Metrics: trans_f1={te.get('f1', 0):.2f}, "
                      f"L_static={results.get('L_static', {}).get(q_name, 0):.2f}")

        # Show rules from iteration with lowest L_avg (global best)
        results = execution_trace[global_best_iter]
        l_avg = results.get("L_avg", 0)

        print(f"\n[Lowest L_avg={l_avg:.2f}] @ Iter {global_best_iter} (NFeat={global_best_iter + 1})")

        if "features" in results and q_name in results["features"]:
            print(f"Features: {results['features'][q_name]}")

        if "rules" in results and q_name in results["rules"]:
            rules = results["rules"][q_name]
            print("Rules:")
            print(_format_rules(rules))

        if "trans_eval" in results and q_name in results["trans_eval"]:
            te = results["trans_eval"][q_name]
            print(f"Metrics: trans_f1={te.get('f1', 0):.2f}, "
                  f"L_static={results.get('L_static', {}).get(q_name, 0):.2f}")

    print("\n" + "="*100 + "\n")


# ============================================================================
# Label Propagation Utilities
# ============================================================================

def perform_label_propagation(
    train_X: pd.DataFrame,
    train_Y: pd.Series,
    test_X_li: list[pd.DataFrame],
    test_Y_li: list[pd.Series],
    debug: bool = False,
) -> Tuple[RandomForestClassifier, list[pd.Series]]:

    # Train the classifier on the training data.
    train_X_proc = encode_features(train_X)
    clf = train_classifier(train_X_proc, train_Y, n_estimators=3)

    assert len(test_X_li) == len(test_Y_li), "test_X_li and test_Y_li must have the same length."

    pred_Y_li = []
    for i in range(len(test_X_li)):
        if test_X_li[i].empty:
            pred_Y_li.append(pd.Series(dtype=int))
            continue
        test_X_proc = encode_features(test_X_li[i])
        pred_Y_li.append(pd.Series(clf.predict(test_X_proc), index=test_Y_li[i].index))

    return clf, pred_Y_li


def compute_inc_error_certificate(
        label_Y: pd.Series, prev_prop_Y_li: list[pd.Series], prop_Y_li: list[pd.Series]) -> float:
    assert len(prev_prop_Y_li) == len(prop_Y_li), (
        f"Length of prev_prop_Y_li and prop_Y_li must be the same. "
        f"Got {len(prev_prop_Y_li)} and {len(prop_Y_li)}."
    )

    if len(prop_Y_li) == 1:
        return 0.0   # The first iteration introduces no error.

    pi = sum(label_Y) / len(label_Y)
    piE = \
        sum([sum(prev_prop_Y_li[i]) for i in range(len(prev_prop_Y_li))]) / \
            sum([len(prev_prop_Y_li[i]) for i in range(len(prev_prop_Y_li))])
    Gamma = max(pi, 1-pi) / (min(pi, 1-pi) + 1e-6)
    GammaE = max(piE, 1-piE) / (min(piE, 1-piE) + 1e-6)

    curr_data_size = sum([len(prev_prop_Y_li[i]) for i in range(len(prev_prop_Y_li))])
    prev_data_size = curr_data_size - len(prev_prop_Y_li[-1])
    new_data_size = len(prev_prop_Y_li[-1])
    data_err = new_data_size / curr_data_size / 1.5

    pred_err = 0.0
    for i in range(len(prev_prop_Y_li) - 1):
        pred_err += sum(prev_prop_Y_li[i] != prop_Y_li[i])
    pred_err /= prev_data_size


    err_certificate = (Gamma + GammaE) * (data_err + pred_err)

    return err_certificate




def pred_and_eval(df: pd.DataFrame, labels: pd.Series) -> dict:
    # Train enriched classifier (all features)
    logger.info(f"Training enriched classifier with {len(df.columns)} features.")

    df_proc = encode_features(df)

    clf = DecisionTreeClassifier(
        max_depth=3,              
        min_samples_leaf=3,
        min_samples_split=3,
        class_weight="balanced",
        random_state=42
    )
    clf.fit(df_proc, labels)

    preds = clf.predict(df_proc)
    f1 = f1_score(labels, preds, zero_division=0)

    # Feature importance
    feature_importance = dict(zip(df.columns, clf.feature_importances_))
    # Sort by importance (descending)
    feature_importance = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))

    # Bad cases (misclassified samples) with prediction probabilities for uncertainty
    misclassified_mask = labels != preds
    proba = clf.predict_proba(df_proc)
    if len(np.unique(labels)) == 1:
        pred_proba = np.ones(len(df_proc)) * (1 if clf.classes_[0] else 0)
    else:
        pred_proba = np.asarray(proba)[:, 1]
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