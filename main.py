"""
Main entry point for the rule-based classifier.

Usage:
    python main.py
"""

from data_fetcher import get_data
from ldb_classifier import RuleClassifier


def evaluate_rule_classifier(dataset: str, query: str, n_samples: int):
    """
    Train and evaluate the rule-based classifier.

    Args:
        dataset: Dataset name (e.g., 'medical')
        query: Query identifier (e.g., 'q1', 'q3', 'q8')
        n_samples: Number of training samples
    """
    print(f"\n{'='*80}")
    print(f"Dataset: {dataset}, Query: {query}, Samples: {n_samples}")
    print('='*80)

    # Load data
    X_train, X_test, y_train, y_test = get_data(dataset, query, n_samples)

    print(f"\nDataset Info:")
    print(f"  Training size: {len(X_train)}")
    print(f"  Test size: {len(X_test)}")
    print(f"  Positive in train: {y_train.sum()} ({y_train.mean():.2%})")
    print(f"  Positive in test: {y_test.sum()} ({y_test.mean():.2%})")

    # Create and train classifier
    # Enable feedback loop for iterative improvement
    clf = RuleClassifier(
        max_rules=10,                    # Maximum number of rules
        min_precision=0.5,               # Minimum precision for rule acceptance
        cv_folds=3,                      # Cross-validation folds
        use_precision_constraint=False,  # Don't enforce precision constraint
        use_tree_rules=True,             # Use tree-based rule generation
        enable_feedback_loop=True,       # Enable feedback-driven refinement
        max_feedback_iterations=5,       # Max refinement iterations
        debug=True                       # Set True for detailed output
    )

    clf.fit(X_train, y_train)

    # Print learned rules in DNF format
    print(f"\n{'='*80}")
    print(f"Learned Rules ({len(clf.rules)} rules)")
    print('='*80)
    clf.print_dnf()

    # Evaluate on train and test
    train_f1, train_prec, train_rec, train_acc = clf.score(X_train, y_train)
    test_f1, test_prec, test_rec, test_acc = clf.score(X_test, y_test)

    # Evaluate on combined dataset
    import pandas as pd
    X_combined = pd.concat([X_train, X_test], ignore_index=True)
    y_combined = pd.concat([y_train, y_test], ignore_index=True)
    overall_f1, overall_prec, overall_rec, overall_acc = clf.score(X_combined, y_combined)

    # Print results
    print(f"\n{'='*80}")
    print("Performance")
    print('='*80)
    print(f"Train   - F1: {train_f1:.4f}, Precision: {train_prec:.4f}, Recall: {train_rec:.4f}, Accuracy: {train_acc:.4f}")
    print(f"Test    - F1: {test_f1:.4f}, Precision: {test_prec:.4f}, Recall: {test_rec:.4f}, Accuracy: {test_acc:.4f}")
    print(f"Overall  - F1: {overall_f1:.4f}, Precision: {overall_prec:.4f}, Recall: {overall_rec:.4f}, Accuracy: {overall_acc:.4f}")

    return {
        'train_f1': train_f1,
        'test_f1': test_f1,
        'overall_f1': overall_f1,
        'rules': clf.rules
    }


def main():
    """Run evaluation on multiple queries."""
    print("\n" + "="*80)
    print("RULE-BASED CLASSIFIER EVALUATION")
    print("="*80)

    results = {}

    # Evaluate on each query
    for query in [
        "q1", 
        "q3", 
        "q8"
    ]:
        try:
            results[query] = evaluate_rule_classifier("medical", query, 50)
        except Exception as e:
            print(f"\nError processing {query}: {e}")

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    for query, res in results.items():
        print(f"{query}: Overall F1 = {res['overall_f1']:.4f}, Rules = {len(res['rules'])}")

    print("\n" + "="*80)


if __name__ == "__main__":
    main()
