"""Vendored SemBench evaluators -- verbatim copies with provenance.

Source checkout: ~/Workspace/SemBench @ c4b2bcff3a5907148f504cfb6faea4c6d3f90e8f
(2026-10-21). Copied 2026-11-19 by user order; unused metric families
(rank / clustering / limit-retrieval / pair-matching) intentionally NOT
copied.

Verify against the checkout (diffs each marked region, then runs the
17-case behavioural battery):

    python3 revision/exp/R1.D3.1/4-sembench-reuse/_check_verbatim.py

Vendored regions (source ranges inclusive, 1-indexed):
  [S1] src/evaluator/generic_evaluator.py:39-59
       -> QueryMetricAggregation + calculate_errors
  [S2] src/evaluator/generic_evaluator.py:259-303
       -> _generic_aggregation_evaluation (method)
  [S3] src/scenario/movie/evaluation/evaluate.py:233-325
       -> _evaluate_sentiment_counts (method)

Only textual adaptations (the region text between the >>>/<<< markers is
byte-identical to the source modulo trailing whitespace):
  - the two methods keep their ``self`` parameter and sit on thin shell
    classes (_SemBenchGeneric / _SemBenchMovie) instead of
    GenericEvaluator / MovieEvaluator, preserving the original
    indentation;
  - ``from dataclasses import dataclass`` is imported at module top.
"""

from dataclasses import dataclass

import pandas as pd

# >>> [S1] verbatim: src/evaluator/generic_evaluator.py:39-59
@dataclass
class QueryMetricAggregation:
    """Metrics for aggregation tasks (e.g., counting, summing)."""

    relative_error: float = 0.0
    absolute_error: float = 0.0
    mean_absolute_percentage_error: float = 0.0

    def calculate_errors(self, predicted: float, actual: float) -> None:
        """Populate the error fields based on *predicted* vs *actual*."""
        self.absolute_error = float(abs(predicted - actual))
        if actual != 0:
            self.relative_error = float(self.absolute_error / abs(actual))
            self.mean_absolute_percentage_error = float(
                self.relative_error * 100
            )
        else:
            self.relative_error = float("inf") if predicted != 0 else 0.0
            self.mean_absolute_percentage_error = float(
                self.relative_error * 100
            )
# <<< [S1] end


class _SemBenchGeneric:
    """Shell class carrying the vendored generic-evaluator method."""

    # >>> [S2] verbatim: src/evaluator/generic_evaluator.py:259-303
    def _generic_aggregation_evaluation(
        self, system_results: pd.DataFrame, ground_truth: pd.DataFrame
    ) -> QueryMetricAggregation:
        """
        Generic evaluation for aggregation queries.

        Assumes single value comparison. Handles string-to-numeric conversion.
        """

        m = QueryMetricAggregation()
        if len(system_results) != 1 or len(ground_truth) != 1:
            m.relative_error = 1.0
            m.absolute_error = float("inf")
            m.mean_absolute_percentage_error = 100.0
            return m

        def first_num(df):
            for c in df.columns:
                val = df[c].iloc[0]

                # Try to convert to numeric if it's a string
                if isinstance(val, str):
                    try:
                        val = float(val)
                    except (ValueError, TypeError):
                        continue

                # Check if it's numeric after potential conversion
                if pd.api.types.is_numeric_dtype(type(val)) or isinstance(
                    val, (int, float)
                ):
                    return val
            return None

        sys_val, gt_val = first_num(system_results), first_num(ground_truth)
        if sys_val is None or gt_val is None:
            (
                m.relative_error,
                m.absolute_error,
                m.mean_absolute_percentage_error,
            ) = (1.0, float("inf"), 100.0)
            return m

        m.calculate_errors(predicted=sys_val, actual=gt_val)
        return m
    # <<< [S2] end


class _SemBenchMovie:
    """Shell class carrying the vendored movie-evaluator method."""

    # >>> [S3] verbatim: src/scenario/movie/evaluation/evaluate.py:233-325
    def _evaluate_sentiment_counts(self, system_results: pd.DataFrame, ground_truth: pd.DataFrame) -> QueryMetricAggregation:
        """
        Evaluate Q7 sentiment count query (GROUP BY scoreSentiment with COUNT).

        Compares counts for each sentiment type and calculates aggregation errors.
        """
        if len(system_results) == 0 or len(ground_truth) == 0:
            return QueryMetricAggregation(
                relative_error=1.0,
                absolute_error=float("inf"),
                mean_absolute_percentage_error=100.0
            )

        # Use column positions: first column is sentiment, second is count
        if len(system_results.columns) < 2 or len(ground_truth.columns) < 2:
            return QueryMetricAggregation(
                relative_error=1.0,
                absolute_error=float("inf"),
                mean_absolute_percentage_error=100.0
            )

        sys_sentiment_col = system_results.columns[0]
        sys_count_col = system_results.columns[1]
        gt_sentiment_col = ground_truth.columns[0]
        gt_count_col = ground_truth.columns[1]

        # Create dictionaries for easy lookup
        sys_counts = {}
        for _, row in system_results.iterrows():
            sentiment = row[sys_sentiment_col]
            count = row[sys_count_col]
            if pd.notna(sentiment) and pd.notna(count):
                try:
                    sys_counts[str(sentiment).strip().upper()] = float(count)
                except (ValueError, TypeError):
                    continue

        gt_counts = {}
        for _, row in ground_truth.iterrows():
            sentiment = row[gt_sentiment_col]
            count = row[gt_count_col]
            if pd.notna(sentiment) and pd.notna(count):
                try:
                    gt_counts[str(sentiment).strip().upper()] = float(count)
                except (ValueError, TypeError):
                    continue

        if not sys_counts or not gt_counts:
            return QueryMetricAggregation(
                relative_error=1.0,
                absolute_error=float("inf"),
                mean_absolute_percentage_error=100.0
            )

        # Calculate errors for each sentiment type
        total_absolute_error = 0.0
        total_relative_error = 0.0
        valid_comparisons = 0

        # Get all sentiment types from both results
        all_sentiments = set(sys_counts.keys()) | set(gt_counts.keys())

        for sentiment in all_sentiments:
            sys_count = sys_counts.get(sentiment, 0.0)
            gt_count = gt_counts.get(sentiment, 0.0)

            # Calculate absolute error
            abs_error = abs(sys_count - gt_count)
            total_absolute_error += abs_error

            # Calculate relative error (avoid division by zero)
            if gt_count != 0:
                rel_error = abs_error / abs(gt_count)
                total_relative_error += rel_error
                valid_comparisons += 1
            elif sys_count != 0:
                # Ground truth is 0 but system predicted non-zero
                total_relative_error += 1.0
                valid_comparisons += 1
            # If both are 0, no error to add

        # Calculate final metrics
        result = QueryMetricAggregation()
        result.absolute_error = total_absolute_error

        if valid_comparisons > 0:
            result.relative_error = total_relative_error / valid_comparisons
            result.mean_absolute_percentage_error = result.relative_error * 100
        else:
            result.relative_error = 0.0
            result.mean_absolute_percentage_error = 0.0

        return result
    # <<< [S3] end


SEMBENCH_GENERIC = _SemBenchGeneric()
SEMBENCH_MOVIE = _SemBenchMovie()
