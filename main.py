"""
Main entry point for the rule-based classifier.

Decoupled design:
- train_classifier(): Handles model training
- evaluate(): Independent evaluation function that only computes metrics from IDs
"""

from data_fetcher import get_data
from ldb_classifier import RuleClassifier
import pandas as pd


def evaluate(pred_ids, true_ids):
    """
    Independent evaluation function.

    Computes TP, TN, FP, FN, Precision, Recall, and F1-score from predicted and true IDs.

    Args:
        pred_ids: List of predicted sample IDs (predicted as class 1)
        true_ids: List of true positive sample IDs (actual class 1)

    Returns:
        Dictionary with metrics: tp, tn, fp, fn, precision, recall, f1_score
    """
    # Convert to sets for efficient computation
    pred_set = set(pred_ids)
    true_set = set(true_ids)

    # Compute confusion matrix elements
    tp = len(pred_set & true_set)  # Predicted positive, actually positive
    fp = len(pred_set - true_set)  # Predicted positive, actually negative
    fn = len(true_set - pred_set)  # Predicted negative, actually positive
    tn = 0  # Cannot compute TN without knowing the total universe

    # Compute metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'tn': tn,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score
    }


def train_rule_based_classifier(dataset: str, query: str, n_samples: int, debug: bool = False):
    """
    Train the rule-based classifier.

    Args:
        dataset: Dataset name (e.g., 'medical')
        query: Query identifier (e.g., 'q1', 'q3', 'q8')
        n_samples: Number of training samples
        debug: Print debug information

    Returns:
        Tuple of (trained classifier, X_train, y_train, X_test, y_test)
    """
    print(f"\n{'='*80}")
    print(f"Training: {dataset}, Query: {query}, Samples: {n_samples}")
    print('='*80)

    # Load data
    X_train, X_test, y_train, y_test = get_data(dataset, query, n_samples)

    print(f"\nDataset Info:")
    print(f"  Training size: {len(X_train)}")
    print(f"  Test size: {len(X_test)}")
    print(f"  Positive in train: {y_train.sum()} ({y_train.mean():.2%})")
    print(f"  Positive in test: {y_test.sum()} ({y_test.mean():.2%})")

    # Create and train classifier
    clf = RuleClassifier(
        max_rules=10,
        min_precision=0.5,
        cv_folds=3,
        use_precision_constraint=False,
        use_tree_rules=True,
        enable_feedback_loop=True,
        max_feedback_iterations=5,
        debug=debug
    )

    clf.fit(X_train, y_train)

    # Print learned rules
    print(f"\n{'='*80}")
    print(f"Learned Rules ({len(clf.rules)} rules)")
    print('='*80)
    clf.print_dnf()

    return clf, X_train, y_train, X_test, y_test


def main():
    """Run training and evaluation on multiple queries."""
    print("\n" + "="*80)
    print("RULE-BASED CLASSIFIER: TRAINING AND EVALUATION")
    print("="*80)

    results = {}

    # Evaluate on each query
    for query in [
        "q1", 
        # "q3", 
        # "q8"
    ]:
        try:
            # Step 1: Train classifier
            clf, X_train, y_train, X_test, y_test = train_rule_based_classifier(
                "medical", query, 50, debug=False
            )

            # Step 2: Make predictions
            y_train_pred = clf.predict(X_train)
            y_test_pred = clf.predict(X_test)

            # Step 3: Use independent evaluate function
            # Get IDs of predicted positives
            pred_train_ids = X_train.index[y_train_pred == 1].tolist()
            pred_test_ids = X_test.index[y_test_pred == 1].tolist()

            # Get IDs of true positives
            true_train_ids = X_train.index[y_train == 1].tolist()
            true_test_ids = X_test.index[y_test == 1].tolist()

            # Evaluate using independent function
            train_metrics = evaluate(pred_train_ids, true_train_ids)
            test_metrics = evaluate(pred_test_ids, true_test_ids)

            # Combined evaluation
            pred_combined_ids = pred_train_ids + pred_test_ids
            true_combined_ids = true_train_ids + true_test_ids
            combined_metrics = evaluate(pred_combined_ids, true_combined_ids)

            # Print results
            print(f"\n{'='*80}")
            print("Performance (via independent evaluate())")
            print('='*80)
            print(f"Train   - F1: {train_metrics['f1_score']:.4f}, Precision: {train_metrics['precision']:.4f}, Recall: {train_metrics['recall']:.4f}")
            print(f"Test    - F1: {test_metrics['f1_score']:.4f}, Precision: {test_metrics['precision']:.4f}, Recall: {test_metrics['recall']:.4f}")
            print(f"Overall  - F1: {combined_metrics['f1_score']:.4f}, Precision: {combined_metrics['precision']:.4f}, Recall: {combined_metrics['recall']:.4f}")

            results[query] = {
                'overall_f1': combined_metrics['f1_score'],
                'rules': clf.rules
            }

        except Exception as e:
            print(f"\nError processing {query}: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    for query, res in results.items():
        print(f"{query}: Overall F1 = {res['overall_f1']:.4f}, Rules = {len(res['rules'])}")

    print("\n" + "="*80)


def test():
    df = pd.read_csv("data/medical_q1.csv").reset_index(drop=True)
    pred_ids = df[(df["LLM_urgent_care_required"] == False) & (df["LLM_label"] == True)].index.tolist()
    true_ids = df[df["label"] == 1].index.tolist()
    print(evaluate(pred_ids, true_ids))


if __name__ == "__main__":
    main()
    # test()
