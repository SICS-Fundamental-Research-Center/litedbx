"""
Evaluation metrics for LDB Engine.

Provides functions to evaluate query results against ground truth.
"""
from typing import List

from data_structures import EvalResult
from logger_config import logger


def evaluate_set(retrieved_set: set, gt_set: set) -> EvalResult:
    """
    Evaluate retrieved set against ground truth set.

    Args:
        retrieved_set: Set of retrieved tuples
        gt_set: Set of ground truth tuples

    Returns:
        EvalResult with precision, recall, and F1 metrics
    """
    tp = len(retrieved_set & gt_set)
    fp = len(retrieved_set - gt_set)
    fn = len(gt_set - retrieved_set)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    logger.info(f"TP: {tp}, FP: {fp}, FN: {fn}")
    logger.info(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    return EvalResult(tp=tp, fp=fp, fn=fn, precision=precision, recall=recall, f1=f1)


def evaluate_list(pred: List[int], label: List[int]) -> EvalResult:
    """
    Evaluate predictions against labels.

    Args:
        pred: List of predicted labels (0 or 1)
        label: List of true labels (0 or 1)

    Returns:
        EvalResult with precision, recall, and F1 metrics
    """
    assert len(pred) == len(label), "Prediction and label lists must have the same length."

    tp = sum(1 for p, l in zip(pred, label) if p == 1 and l == 1)
    fp = sum(1 for p, l in zip(pred, label) if p == 1 and l == 0)
    fn = sum(1 for p, l in zip(pred, label) if p == 0 and l == 1)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    logger.info(f"TP: {tp}, FP: {fp}, FN: {fn}")
    logger.info(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    return EvalResult(tp=tp, fp=fp, fn=fn, precision=precision, recall=recall, f1=f1)
