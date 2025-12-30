"""
Rule-based binary classifier with cross-validation and feedback-driven refinement.

Key Features:
- Interpretable rules (IF-THEN format)
- Greedy forward selection with F1 optimization
- Tree-based rule generation for complex patterns
- Feedback-driven refinement to reduce false negatives
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional


# =============================================================================
# Rule Classes
# =============================================================================

class RuleCondition:
    """A single condition in a rule (e.g., age > 30)."""

    def __init__(self, feature: str, value, operator: str, feature_type: str):
        self.feature = feature
        self.value = value
        self.operator = operator  # 'le', 'ge', 'eq'
        self.feature_type = feature_type  # 'numerical', 'categorical'

    def matches(self, sample: pd.Series) -> bool:
        """Check if sample satisfies this condition."""
        if self.feature_type == 'categorical':
            return sample[self.feature] == self.value
        elif self.operator == 'le':
            return sample[self.feature] <= self.value
        elif self.operator == 'ge':
            return sample[self.feature] >= self.value
        return False

    def __str__(self):
        if self.operator == 'eq':
            return f"{self.feature} == {self.value}"
        elif self.operator == 'le':
            return f"{self.feature} <= {self.value}"
        elif self.operator == 'ge':
            return f"{self.feature} >= {self.value}"
        return f"{self.feature} {self.operator} {self.value}"

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        if not isinstance(other, RuleCondition):
            return False
        return (self.feature == other.feature and
                self.value == other.value and
                self.operator == other.operator and
                self.feature_type == other.feature_type)

    def __hash__(self):
        return hash((self.feature, self.value, self.operator, self.feature_type))


class Rule:
    """A rule with one or more conditions (AND logic)."""

    def __init__(self, conditions: List[RuleCondition]):
        self.conditions = conditions

    def get_covered_indices(self, X: pd.DataFrame) -> pd.Series:
        """Return boolean mask of samples covered by this rule."""
        covered = pd.Series([True] * len(X), index=X.index)

        for condition in self.conditions:
            covered &= X.apply(condition.matches, axis=1)

        return covered

    def __str__(self):
        if len(self.conditions) == 1:
            return f"IF {self.conditions[0]} THEN predict 1"
        else:
            conds = " AND ".join(str(c) for c in self.conditions)
            return f"IF ({conds}) THEN predict 1"

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        if not isinstance(other, Rule):
            return False
        return self.conditions == other.conditions

    def __hash__(self):
        return hash(tuple(self.conditions))


# =============================================================================
# Main Classifier
# =============================================================================

class RuleClassifier:
    """
    Rule-based binary classifier.

    Parameters:
        max_rules: Maximum number of rules to select
        min_precision: Minimum precision for rule consideration
        cv_folds: Number of cross-validation folds
        use_precision_constraint: Enforce min_precision constraint
        use_tree_rules: Use tree-based rule generation
        enable_feedback_loop: Enable feedback-driven refinement
        max_feedback_iterations: Max refinement iterations
        fp_penalty_ratio: Allowable FP/TP ratio for new rules
        min_f1_improvement: Minimum F1 improvement to continue
        debug: Print debug information
    """

    def __init__(
        self,
        max_rules: int = 10,
        min_precision: float = 0.5,
        cv_folds: int = 3,
        use_precision_constraint: bool = False,
        use_tree_rules: bool = False,
        enable_feedback_loop: bool = False,
        max_feedback_iterations: int = 3,
        fp_penalty_ratio: float = 0.5,
        min_f1_improvement: float = 0.01,
        debug: bool = False
    ):
        self.max_rules = max_rules
        self.min_precision = min_precision
        self.cv_folds = cv_folds
        self.use_precision_constraint = use_precision_constraint
        self.use_tree_rules = use_tree_rules
        self.enable_feedback_loop = enable_feedback_loop
        self.max_feedback_iterations = max_feedback_iterations
        self.fp_penalty_ratio = fp_penalty_ratio
        self.min_f1_improvement = min_f1_improvement
        self.debug = debug

        self.rules: List[Rule] = []
        self.best_cv_f1: float = 0.0

    # -------------------------------------------------------------------------
    # Rule Generation
    # -------------------------------------------------------------------------

    def generate_candidate_rules(self, X: pd.DataFrame) -> List[Rule]:
        """Generate all possible single-condition rules."""
        candidates = []

        for col in X.columns:
            if pd.api.types.is_numeric_dtype(X[col]):
                # Numerical: use unique values as thresholds
                thresholds = X[col].unique()
                for thresh in thresholds:
                    candidates.append(Rule([RuleCondition(col, thresh, 'le', 'numerical')]))
                    candidates.append(Rule([RuleCondition(col, thresh, 'ge', 'numerical')]))
            else:
                # Categorical: one rule per unique value
                for val in X[col].unique():
                    candidates.append(Rule([RuleCondition(col, val, 'eq', 'categorical')]))

        return candidates

    def generate_tree_based_rules(self, X: pd.DataFrame, y: pd.Series) -> List[Rule]:
        """Generate rules using a constrained decision tree."""
        from sklearn.tree import DecisionTreeClassifier

        # Encode categorical features
        X_encoded = X.copy()
        feature_mappings = {}

        for col in X_encoded.columns:
            if not pd.api.types.is_numeric_dtype(X_encoded[col]):
                unique_vals = X_encoded[col].unique()
                mapping = {val: idx for idx, val in enumerate(sorted(unique_vals))}
                feature_mappings[col] = mapping
                X_encoded[col] = X_encoded[col].map(mapping)

        # Train constrained tree
        min_samples = max(5, int(0.1 * len(X)))
        tree = DecisionTreeClassifier(
            max_depth=2,
            min_samples_leaf=min_samples,
            min_samples_split=2 * min_samples,
            criterion='gini',
            random_state=42
        )
        tree.fit(X_encoded, y)

        # Extract rules from tree
        return self._extract_rules_from_tree(tree, X, X_encoded, feature_mappings)

    def _extract_rules_from_tree(
        self, tree, X: pd.DataFrame, X_encoded: pd.DataFrame, feature_mappings: dict
    ) -> List[Rule]:
        """Extract rules from trained decision tree paths."""
        rules = []
        n_nodes = tree.tree_.node_count
        children_left = tree.tree_.children_left
        children_right = tree.tree_.children_right
        feature = tree.tree_.feature
        threshold = tree.tree_.threshold

        def extract_path(node_id: int, current_conditions: List[RuleCondition]):
            """Recursively extract rules from tree paths."""
            if children_left[node_id] == children_right[node_id]:
                # Leaf node - check if it predicts class 1
                if tree.tree_.value[node_id][0][1] > tree.tree_.value[node_id][0][0]:
                    if current_conditions:
                        rules.append(Rule(current_conditions.copy()))
                return

            # Internal node - continue traversing
            feat_name = X_encoded.columns[feature[node_id]]
            thresh = threshold[node_id]

            # Get original feature name if it was categorical
            orig_feat = feat_name
            if feat_name in feature_mappings:
                # Categorical feature - convert threshold back to category
                inv_mapping = {v: k for k, v in feature_mappings[feat_name].items()}
                cat_val = inv_mapping.get(int(thresh), thresh)
                left_cond = RuleCondition(orig_feat, cat_val, 'le', 'categorical')
                right_cond = RuleCondition(orig_feat, cat_val, 'ge', 'categorical')
            else:
                # Numerical feature
                left_cond = RuleCondition(orig_feat, thresh, 'le', 'numerical')
                right_cond = RuleCondition(orig_feat, thresh, 'ge', 'numerical')

            # Go left (feature <= threshold)
            extract_path(children_left[node_id], current_conditions + [left_cond])

            # Go right (feature > threshold)
            extract_path(children_right[node_id], current_conditions + [right_cond])

        extract_path(0, [])
        return rules

    # -------------------------------------------------------------------------
    # Rule Evaluation
    # -------------------------------------------------------------------------

    def evaluate_rule_set(
        self, rules: List[Rule], X: pd.DataFrame, y: pd.Series
    ) -> Tuple[float, float, float, float]:
        """Evaluate F1, precision, recall, accuracy for a rule set."""
        if not rules:
            return 0.0, 0.0, 0.0, y.mean()

        predictions = self.predict_with_rules(X, rules)

        tp = ((predictions == 1) & (y == 1)).sum()
        fp = ((predictions == 1) & (y == 0)).sum()
        tn = ((predictions == 0) & (y == 0)).sum()
        fn = ((predictions == 0) & (y == 1)).sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        accuracy = (tp + tn) / (tp + tn + fp + fn)

        return f1, precision, recall, accuracy

    def predict_with_rules(self, X: pd.DataFrame, rules: List[Rule]) -> np.ndarray:
        """Predict using rule set: 1 if ANY rule fires, else 0."""
        if not rules:
            return np.zeros(len(X), dtype=int)

        covered = pd.Series([False] * len(X), index=X.index)

        for rule in rules:
            covered |= rule.get_covered_indices(X)

        return np.array(covered.astype(int))

    # -------------------------------------------------------------------------
    # Feedback-Driven Refinement
    # -------------------------------------------------------------------------

    def _collect_feedback_statistics(
        self, X: pd.DataFrame, y: pd.Series, rule_set: List[Rule]
    ) -> dict:
        """Collect statistics for feedback-driven refinement."""
        predictions = self.predict_with_rules(X, rule_set)

        tp_idx = X.index[(predictions == 1) & (y == 1)]
        fp_idx = X.index[(predictions == 1) & (y == 0)]
        fn_idx = X.index[(predictions == 0) & (y == 1)]
        tn_idx = X.index[(predictions == 0) & (y == 0)]

        # Find uncovered positives (FN not covered by any rule)
        uncovered_positives = []
        if len(fn_idx) > 0:
            for idx in fn_idx:
                covered = False
                for rule in rule_set:
                    if rule.get_covered_indices(X.loc[[idx]]).iloc[0]:
                        covered = True
                        break
                if not covered:
                    uncovered_positives.append(idx)

        return {
            'tp_idx': tp_idx, 'fp_idx': fp_idx, 'fn_idx': fn_idx, 'tn_idx': tn_idx,
            'tp_count': len(tp_idx), 'fp_count': len(fp_idx),
            'fn_count': len(fn_idx), 'tn_count': len(tn_idx),
            'uncovered_positives': uncovered_positives,
            'predictions': predictions
        }

    def _mine_residual_rules(
        self, X: pd.DataFrame, y: pd.Series, feedback_stats: dict
    ) -> List[Rule]:
        """Mine corrective rules from false negative samples."""
        from sklearn.tree import DecisionTreeClassifier

        fn_idx = feedback_stats['fn_idx']

        if len(fn_idx) < 2:
            if self.debug:
                print(f"[FEEDBACK]   No FN samples to correct (FN={len(fn_idx)})")
            return []

        if self.debug:
            print(f"[FEEDBACK]   Attempting to mine corrective rules from {len(fn_idx)} FN samples")

        # Prepare data: positive = FN, negative = TN + FP
        neg_idx = list(feedback_stats['tn_idx']) + list(feedback_stats['fp_idx'])

        if len(neg_idx) == 0:
            return []

        X_fn = X.loc[fn_idx]
        X_neg = X.loc[neg_idx]

        y_residual = pd.Series([1] * len(X_fn) + [0] * len(X_neg))
        X_residual = pd.concat([X_fn, X_neg], ignore_index=True)

        # Encode categoricals
        X_residual_encoded = X_residual.copy()
        feature_mappings = {}

        for col in X_residual_encoded.columns:
            if not pd.api.types.is_numeric_dtype(X_residual_encoded[col]):
                unique_vals = X_residual_encoded[col].unique()
                mapping = {val: idx for idx, val in enumerate(sorted(unique_vals))}
                feature_mappings[col] = mapping
                X_residual_encoded[col] = X_residual_encoded[col].map(mapping)

        # Train tree with adaptive min_samples
        min_samples_leaf = max(2, min(5, len(fn_idx) // 3))
        min_samples_split = max(4, min_samples_leaf * 2)

        if self.debug:
            print(f"[FEEDBACK]   Training residual tree with min_samples_leaf={min_samples_leaf}")

        tree = DecisionTreeClassifier(
            max_depth=2,
            min_samples_leaf=min_samples_leaf,
            min_samples_split=min_samples_split,
            criterion='gini',
            random_state=42
        )
        tree.fit(X_residual_encoded, y_residual)

        residual_rules = self._extract_rules_from_tree(
            tree, X_residual, X_residual_encoded, feature_mappings
        )

        if self.debug:
            print(f"[FEEDBACK]   Extracted {len(residual_rules)} rules from residual tree")

        return residual_rules

    def _evaluate_rule_acceptance(
        self, candidate_rule: Rule, X: pd.DataFrame, y: pd.Series,
        current_f1: float, current_rule_set: List[Rule]
    ) -> Tuple[bool, float, Dict]:
        """
        Evaluate if candidate rule should be accepted.

        Enhanced to be more aggressive at reducing False Negatives.
        Accepts rules that improve Recall significantly, even if Precision drops.
        """
        test_rules = current_rule_set + [candidate_rule]
        new_f1, new_prec, new_rec, _ = self.evaluate_rule_set(test_rules, X, y)

        # Get metrics
        predictions = self.predict_with_rules(X, test_rules)
        new_tp = ((predictions == 1) & (y == 1)).sum()
        new_fp = ((predictions == 1) & (y == 0)).sum()

        current_predictions = self.predict_with_rules(X, current_rule_set)
        current_tp = ((current_predictions == 1) & (y == 1)).sum()

        # Calculate current recall
        current_total_positives = (y == 1).sum()
        current_recall = current_tp / current_total_positives if current_total_positives > 0 else 0.0
        new_recall = new_tp / current_total_positives if current_total_positives > 0 else 0.0

        # Enhanced acceptance criteria for reducing FN:
        # 1. Significant recall improvement (reduces FN)
        recall_improves = (new_recall - current_recall) >= 0.05  # 5% minimum recall improvement

        # 2. F1 improvement OR recall improvement with acceptable precision trade-off
        epsilon = 0.005  # Reduced from 0.01 to be more permissive
        f1_improves = (new_f1 - current_f1) >= epsilon

        # 3. Allow more FP when Recall improves significantly
        alpha = self.fp_penalty_ratio
        fp_penalty_ok = new_fp <= (alpha * new_tp) if new_tp > 0 else True

        # 4. Never reduce TP (don't lose already covered positives)
        tp_preserved = new_tp >= current_tp

        # Accept if:
        # - F1 improves AND all other criteria pass, OR
        # - Recall significantly improves AND precision doesn't drop too much
        precision_acceptable = new_prec >= 0.15  # Minimum precision threshold
        recall_boost = recall_improves and precision_acceptable and tp_preserved

        accepted = (f1_improves and fp_penalty_ok and tp_preserved) or recall_boost

        metrics = {
            'new_f1': new_f1,
            'new_tp': new_tp,
            'new_fp': new_fp,
            'new_recall': new_recall,
            'new_precision': new_prec,
            'current_recall': current_recall,
            'recall_improves': recall_improves,
            'f1_improves': f1_improves,
            'fp_penalty_ok': fp_penalty_ok,
            'tp_preserved': tp_preserved,
            'precision_acceptable': precision_acceptable,
            'recall_boost': recall_boost
        }

        return accepted, new_f1, metrics

    def _feedback_driven_refinement(
        self, X: pd.DataFrame, y: pd.Series,
        X_val: pd.DataFrame, y_val: pd.Series
    ) -> List[Rule]:
        """Iteratively refine rules using feedback from residuals."""
        if not self.enable_feedback_loop:
            return self.rules.copy()

        refined_rules = self.rules.copy()
        current_f1, _, _, _ = self.evaluate_rule_set(refined_rules, X_val, y_val)

        if self.debug:
            print(f"\n[FEEDBACK] Starting refinement. Initial F1: {current_f1:.4f}")
            print(f"[FEEDBACK] Max iterations: {self.max_feedback_iterations}")

        for iteration in range(self.max_feedback_iterations):
            if len(refined_rules) >= self.max_rules:
                if self.debug:
                    print(f"[FEEDBACK] Reached max rules ({self.max_rules})")
                break

            # Collect feedback and mine corrective rules
            feedback_stats = self._collect_feedback_statistics(X, y, refined_rules)

            if self.debug:
                print(f"\n[FEEDBACK] Iteration {iteration + 1}")
                print(f"[FEEDBACK]   TP: {feedback_stats['tp_count']}, "
                      f"FP: {feedback_stats['fp_count']}, "
                      f"FN: {feedback_stats['fn_count']}, "
                      f"TN: {feedback_stats['tn_count']}")
                print(f"[FEEDBACK]   Uncovered positives: {len(feedback_stats['uncovered_positives'])}")

            corrective_rules = self._mine_residual_rules(X, y, feedback_stats)

            if not corrective_rules:
                if self.debug:
                    print(f"[FEEDBACK] No corrective rules generated")
                break

            if self.debug:
                print(f"[FEEDBACK] Generated {len(corrective_rules)} corrective rules")
                # Show the corrective rules
                for i, rule in enumerate(corrective_rules, 1):
                    print(f"[FEEDBACK]   Candidate {i}: {rule}")

            # Try to find best corrective rule
            best_rule = None
            best_f1 = current_f1
            best_metrics = None

            for rule in corrective_rules:
                accepted, new_f1, metrics = self._evaluate_rule_acceptance(
                    rule, X_val, y_val, current_f1, refined_rules
                )

                if self.debug:
                    print(f"[FEEDBACK]   Evaluating: {rule}")
                    print(f"[FEEDBACK]     F1: {current_f1:.4f} -> {new_f1:.4f}")
                    print(f"[FEEDBACK]     Recall: {metrics['current_recall']:.4f} -> {metrics['new_recall']:.4f}")
                    print(f"[FEEDBACK]     Precision: {metrics['new_precision']:.4f}")
                    print(f"[FEEDBACK]     Accepted: {accepted}")
                    if not accepted:
                        reasons = []
                        if not metrics['f1_improves']:
                            reasons.append("F1 doesn't improve")
                        if not metrics.get('recall_boost'):
                            reasons.append("No recall boost")
                        if not metrics.get('tp_preserved'):
                            reasons.append("TP not preserved")
                        print(f"[FEEDBACK]     Rejected: {', '.join(reasons)}")

                if accepted and new_f1 > best_f1:
                    best_rule = rule
                    best_f1 = new_f1
                    best_metrics = metrics

                    if self.debug:
                        print(f"[FEEDBACK]   >>> Best rule so far!")
                        if metrics.get('recall_boost'):
                            print(f"[FEEDBACK]     Accepted via Recall Boost!")

            # Add rule if improvement (use lower threshold for feedback loop)
            feedback_threshold = self.min_f1_improvement / 2  # More permissive for refinement
            recall_boost = best_metrics.get('recall_boost') if best_metrics else False

            if best_rule is not None and (best_f1 > current_f1 + feedback_threshold or recall_boost):
                refined_rules.append(best_rule)
                current_f1 = best_f1

                if self.debug:
                    print(f"[FEEDBACK]   Added rule: {best_rule}")
                    if best_metrics:
                        print(f"[FEEDBACK]   New F1: {current_f1:.4f}, Recall: {best_metrics['new_recall']:.4f}, Total rules: {len(refined_rules)}")
            else:
                if self.debug:
                    print(f"[FEEDBACK] No rule improves F1 sufficiently. Stopping.")
                break

        return refined_rules

    # -------------------------------------------------------------------------
    # Training and Prediction
    # -------------------------------------------------------------------------

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Train the classifier using cross-validation."""
        # Generate candidate rules
        if self.use_tree_rules:
            if self.debug:
                print(f"[DEBUG] Using tree-based rule generation")
            candidate_rules = self.generate_tree_based_rules(X, y)
            # Fallback if tree produces no rules
            if len(candidate_rules) == 0:
                if self.debug:
                    print(f"[DEBUG] Tree generated no rules, falling back to exhaustive generation")
                candidate_rules = self.generate_candidate_rules(X)
        else:
            if self.debug:
                print(f"[DEBUG] Using exhaustive rule generation")
            candidate_rules = self.generate_candidate_rules(X)

        # Cross-validation to select rules
        from sklearn.model_selection import KFold

        kf = KFold(n_splits=self.cv_folds, shuffle=True, random_state=42)

        all_selected_rules = []
        fold_scores = []

        for train_idx, val_idx in kf.split(X):
            X_fold_train = X.iloc[train_idx]
            y_fold_train = y.iloc[train_idx]
            X_fold_val = X.iloc[val_idx]
            y_fold_val = y.iloc[val_idx]

            fold_rules, fold_f1 = self._select_rules_on_fold(
                X_fold_train, y_fold_train, X_fold_val, y_fold_val, candidate_rules
            )

            all_selected_rules.extend(fold_rules)
            fold_scores.append(fold_f1)

        # Select final rules by frequency
        from collections import Counter

        rule_counts = Counter(str(r) for r in all_selected_rules)

        unique_rules = {}
        for rule in all_selected_rules:
            rule_str = str(rule)
            if rule_str not in unique_rules:
                unique_rules[rule_str] = rule

        sorted_rules = sorted(
            unique_rules.values(),
            key=lambda r: rule_counts[str(r)],
            reverse=True
        )

        # Final selection on full training data
        top_k = min(len(sorted_rules), self.max_rules * 2)
        top_rules = sorted_rules[:top_k]

        self.rules = []
        best_f1 = 0.0
        remaining_rules = top_rules.copy()

        for _ in range(self.max_rules):
            best_rule = None
            best_rule_f1 = 0.0

            for rule in remaining_rules:
                test_rules = self.rules + [rule]
                f1, _, _, _ = self.evaluate_rule_set(test_rules, X, y)

                if f1 > best_rule_f1:
                    best_rule_f1 = f1
                    best_rule = rule

            if best_rule is not None and best_rule_f1 > best_f1 + self.min_f1_improvement:
                self.rules.append(best_rule)
                best_f1 = best_rule_f1
                remaining_rules.remove(best_rule)
            else:
                break

        self.best_cv_f1 = best_f1

        # Apply feedback-driven refinement
        if self.enable_feedback_loop and len(self.rules) > 0:
            if self.debug:
                print(f"\n[DEBUG] Applying feedback-driven refinement...")

            # Use full training data for both mining and evaluation
            # This is more robust for small datasets
            self.rules = self._feedback_driven_refinement(
                X, y, X, y
            )

            final_f1, _, _, _ = self.evaluate_rule_set(self.rules, X, y)
            self.best_cv_f1 = final_f1

            if self.debug:
                print(f"[DEBUG] Refinement complete. Final F1: {final_f1:.4f}, Total rules: {len(self.rules)}")

    def _select_rules_on_fold(
        self, X_train: pd.DataFrame, y_train: pd.Series,
        X_val: pd.DataFrame, y_val: pd.Series,
        candidate_rules: List[Rule]
    ) -> Tuple[List[Rule], float]:
        """Select best rules on a single fold."""
        # Filter rules by precision if constraint enabled
        if self.use_precision_constraint:
            filtered_rules = []
            for rule in candidate_rules:
                covered = rule.get_covered_indices(X_train)
                if covered.sum() == 0:
                    continue
                precision = y_train[covered].mean()
                if precision >= self.min_precision:
                    filtered_rules.append(rule)
            candidate_rules = filtered_rules

        if self.debug:
            print(f"[DEBUG] Filtered to {len(candidate_rules)} rules from candidates")

        # Greedy forward selection
        rule_set = []
        best_f1 = 0.0
        remaining_rules = candidate_rules.copy()

        for _ in range(self.max_rules):
            best_rule = None
            best_rule_f1 = 0.0

            for rule in remaining_rules:
                test_rules = rule_set + [rule]
                f1, _, _, _ = self.evaluate_rule_set(test_rules, X_val, y_val)

                if f1 > best_rule_f1:
                    best_rule_f1 = f1
                    best_rule = rule

            if best_rule is not None and best_rule_f1 > best_f1 + self.min_f1_improvement:
                rule_set.append(best_rule)
                best_f1 = best_rule_f1
                remaining_rules.remove(best_rule)
            else:
                break

        return rule_set, best_f1

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions on new data."""
        return self.predict_with_rules(X, self.rules)

    def score(self, X: pd.DataFrame, y: pd.Series) -> Tuple[float, float, float, float]:
        """Calculate F1, precision, recall, accuracy."""
        return self.evaluate_rule_set(self.rules, X, y)

    def print_rules(self):
        """Print learned rules."""
        print(f"\n=== Learned Rules ({len(self.rules)} rules) ===")
        for i, rule in enumerate(self.rules, 1):
            print(f"{i}. {rule}")

    def print_dnf(self):
        """Print rules in Disjunctive Normal Form (DNF)."""
        if not self.rules:
            print("\nDNF: FALSE (no rules)")
            return

        # Convert each rule (conjunction) to string
        conjunctions = []
        for rule in self.rules:
            if len(rule.conditions) == 1:
                conjunctions.append(f"({rule.conditions[0]})")
            else:
                conds = " AND ".join(f"({c})" for c in rule.conditions)
                conjunctions.append(f"({conds})")

        # Join with OR
        dnf = " OR ".join(conjunctions)
        print(f"\nDNF Rule:\n{dnf}")


# =============================================================================
# Convenience Function
# =============================================================================

def evaluate_classifier(
    X_train: pd.DataFrame, y_train: pd.Series,
    X_test: pd.DataFrame, y_test: pd.Series,
    max_rules: int = 10,
    min_precision: float = 0.5,
    cv_folds: int = 3,
    use_precision_constraint: bool = False,
    use_tree_rules: bool = False,
    enable_feedback_loop: bool = False,
    max_feedback_iterations: int = 3,
    fp_penalty_ratio: float = 0.5,
    debug: bool = False
) -> Dict:
    """
    Convenience function to train and evaluate the classifier.

    Returns dictionary with rules and performance metrics.
    """
    clf = RuleClassifier(
        max_rules=max_rules,
        min_precision=min_precision,
        cv_folds=cv_folds,
        use_precision_constraint=use_precision_constraint,
        use_tree_rules=use_tree_rules,
        enable_feedback_loop=enable_feedback_loop,
        max_feedback_iterations=max_feedback_iterations,
        fp_penalty_ratio=fp_penalty_ratio,
        debug=debug
    )

    clf.fit(X_train, y_train)

    train_f1, train_prec, train_rec, train_acc = clf.score(X_train, y_train)
    test_f1, test_prec, test_rec, test_acc = clf.score(X_test, y_test)

    # Combined dataset
    X_combined = pd.concat([X_train, X_test], ignore_index=True)
    y_combined = pd.concat([y_train, y_test], ignore_index=True)
    overall_f1, overall_prec, overall_rec, overall_acc = clf.score(X_combined, y_combined)

    if not debug:
        clf.print_rules()
        print("\n=== Training Performance ===")
        print(f"F1: {train_f1:.4f}, Precision: {train_prec:.4f}, Recall: {train_rec:.4f}, Accuracy: {train_acc:.4f}")

        print("\n=== Test Performance ===")
        print(f"F1: {test_f1:.4f}, Precision: {test_prec:.4f}, Recall: {test_rec:.4f}, Accuracy: {test_acc:.4f}")

        print("\n=== Overall Performance (Train + Test) ===")
        print(f"F1: {overall_f1:.4f}, Precision: {overall_prec:.4f}, Recall: {overall_rec:.4f}, Accuracy: {overall_acc:.4f}")

    return {
        'rules': clf.rules,
        'train_f1': train_f1,
        'test_f1': test_f1,
        'overall_f1': overall_f1,
        'test_precision': test_prec,
        'test_recall': test_rec
    }
