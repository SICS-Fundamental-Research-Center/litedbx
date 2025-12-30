# Rule-Based Binary Classifier

An interpretable rule-based classifier for tabular data with feedback-driven refinement.

## Features

- **Interpretable Rules**: Human-readable IF-THEN format
- **Greedy Forward Selection**: Optimizes F1 score
- **Tree-Based Rule Generation**: Creates multi-condition rules from decision trees
- **Feedback-Driven Refinement**: Iteratively reduces false negatives
- **Cross-Validation**: Prevents overfitting

## Usage

### Basic Usage

```python
from data_fetcher import get_data
from ldb_classifier import RuleClassifier

# Load data
X_train, X_test, y_train, y_test = get_data("medical", "q1", 100)

# Create and train classifier
clf = RuleClassifier(
    max_rules=10,
    min_precision=0.5,
    cv_folds=3,
    use_tree_rules=True,
    enable_feedback_loop=True
)

clf.fit(X_train, y_train)

# Print learned rules
clf.print_rules()

# Evaluate
train_f1, train_prec, train_rec, train_acc = clf.score(X_train, y_train)
test_f1, test_prec, test_rec, test_acc = clf.score(X_test, y_test)

print(f"Train F1: {train_f1:.4f}")
print(f"Test F1: {test_f1:.4f}")
```

### Running the Demo

```bash
python main.py
```

## Parameters

- `max_rules`: Maximum number of rules to select (default: 10)
- `min_precision`: Minimum precision for rule consideration (default: 0.5)
- `cv_folds`: Number of cross-validation folds (default: 3)
- `use_precision_constraint`: Enforce min_precision constraint (default: False)
- `use_tree_rules`: Use tree-based rule generation (default: False)
- `enable_feedback_loop`: Enable feedback-driven refinement (default: False)
- `max_feedback_iterations`: Max refinement iterations (default: 3)
- `debug`: Print debug information (default: False)

## Rule Format

Rules are generated in interpretable IF-THEN format:

```
IF age >= 30 THEN predict 1
IF (x1 > 0.5 AND category == 'A') THEN predict 1
```

## Files

- `ldb_classifier.py`: Main classifier implementation (721 lines)
- `data_fetcher.py`: Data loading utilities
- `main.py`: Demo script
