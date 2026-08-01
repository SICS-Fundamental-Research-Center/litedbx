# pylint: disable=broad-exception-caught,too-few-public-methods
# pylint: disable=missing-function-docstring,duplicate-code
# pylint: disable=logging-fstring-interpolation,too-many-locals
# pylint: disable=too-many-branches,too-many-statements
"""Reporting and usage-statistics helpers."""

import logging
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


class Reporting:
    """Methods for usage and result reporting."""

    def __init__(self, usage_statistics: list[dict[str, Any]]) -> None:
        self.usage_statistics = usage_statistics

    def report_usage_statistics(self) -> None:
        report_usage_statistics(self.usage_statistics[0])

    def report_evaluation_trace(self, execution_trace: dict) -> None:
        if execution_trace == {}:
            return
        report_evaluation_trace(execution_trace)

    def report_dynamic_results(self, eval_results: list) -> None:
        inc_rounds = [res["inc_round"] for res in eval_results]
        inc_ratios = [res["inc_ratio"] for res in eval_results]
        eval_results_single_step = [res["eval_results"] for res in eval_results]
        eval_time = [res["eval_time"] for res in eval_results]

        queries = eval_results_single_step[0].keys()

        try:
            for q in queries:
                eval_q_single_step = [
                    res[q] for res in eval_results_single_step
                ]

                trans_f1s = [
                    res["trans_eval"]["f1"] if res else None
                    for res in eval_q_single_step
                ]
                pred_f1s = [
                    res["pred_eval"]["f1"] if res else None
                    for res in eval_q_single_step
                ]
                error_certificates = [
                    res["error_certificate"] if res else None
                    for res in eval_q_single_step
                ]
                data_errs = [
                    res["data_err"] if res else None
                    for res in eval_q_single_step
                ]
                pred_errs = [
                    res["pred_err"] if res else None
                    for res in eval_q_single_step
                ]

                results = pd.DataFrame(
                    {
                        "inc_round": inc_rounds,
                        "inc_ratio": inc_ratios,
                        "trans_f1": trans_f1s,
                        "pred_f1": pred_f1s,
                        "error_certificate": error_certificates,
                        "data_err": data_errs,
                        "pred_err": pred_errs,
                        "eval_time": eval_time,
                    }
                )

                print(
                    "=" * 20
                    + f" Incremental Evaluation Results for Query {q}"
                    + "=" * 20
                )
                print(results)
        except Exception:
            print(eval_results)


def report_usage_statistics(usage_statistics: dict):
    """Report LLM usage statistics.

    Args:
        usage_statistics: Dictionary with usage stats for each phase
    """
    logger.info("=== LLM Usage Statistics ===")
    for item, stats in usage_statistics.items():
        logger.info(
            f"{item}: Prompt Tokens={stats['prompt_tokens']}, "
            f"Completion Tokens={stats['completion_tokens']}, "
            f"Total Tokens={stats['total_tokens']}, "
            f"Prompt Cost=${stats['prompt_cost']:.4f}, "
            f"Completion Cost=${stats['completion_cost']:.4f}, "
            f"Total Cost=${stats['total_cost']:.4f}"
        )

    total_cost = sum(stats["total_cost"] for stats in usage_statistics.values())
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

    assert execution_trace is not None, (
        "Execution trace is required for reporting."
    )

    # Metric maps are authoritative; top-level keys also include metadata.
    all_query_names = {
        q_name
        for results in execution_trace.values()
        for q_name in results.get("L_static", {})
    }
    selected_iter = max(execution_trace)

    # ========== SECTION 1: OVERVIEW TABLE ==========
    overview_data = []
    for iter_idx, results in execution_trace.items():
        for q_name in all_query_names:
            candidate = results.get("candidate", {}).get(q_name, "")
            if iter_idx == selected_iter:
                candidate = f"selected:{candidate}"
            row = {
                "Iter": iter_idx,
                "Candidate": candidate,
                "NFeat": len(results.get("features", {}).get(q_name, [])),
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
            row["penalty_rew"] = (
                f"{results.get('penalty_rew', {}).get(q_name, 0):.2f}"
            )
            row["L_LOO"] = f"{results.get('L_LOO', {}).get(q_name, 0):.2f}"
            row["penalty_LOO"] = (
                f"{results.get('penalty_LOO', {}).get(q_name, 0):.2f}"
            )
            row["L_obj"] = f"{results.get('L_obj', {}).get(q_name, 0):.2f}"
            row["L_subj"] = f"{results.get('L_subj', {}).get(q_name, 0):.2f}"
            row["L_static"] = (
                f"{results.get('L_static', {}).get(q_name, 0):.2f}"
            )
            row["memory_cost"] = (
                f"{results.get('memory_cost', {}).get(q_name, 0):.2f}"
            )
            row["overall_selectivity"] = (
                f"{results.get('overall_selectivity', {}).get(q_name, 0):.4f}"
            )
            row["total_size"] = results.get("total_size", {}).get(q_name, 0)

            overview_data.append(row)

    df_overview = pd.DataFrame(overview_data)
    col_order = [
        "Iter",
        "Candidate",
        "NFeat",
        "Query",
        "pred_f1",
        "pred_p",
        "pred_r",
        "trans_f1",
        "trans_p",
        "trans_r",
        "L_rew",
        "penalty_rew",
        "L_obj",
        "L_LOO",
        "penalty_LOO",
        "L_subj",
        "L_static",
        "memory_cost",
        "overall_selectivity",
        "total_size",
    ]
    col_order = [c for c in col_order if c in df_overview.columns]
    df_overview = df_overview[col_order]

    print("\n" + "=" * 150)
    print("OVERVIEW - Rewrite Candidate Evaluations")
    print("=" * 150)
    print(df_overview.to_string(index=False))
    print("=" * 150)

    # ========== SECTION 2: AVERAGE ERROR ==========
    avg_errors = []
    for iter_idx, results in execution_trace.items():
        candidates = sorted(set(results.get("candidate", {}).values()))
        candidate = ",".join(candidates)
        if iter_idx == selected_iter:
            candidate = f"selected:{candidate}"
        avg_errors.append(
            {
                "Iter": iter_idx,
                "Candidate": candidate,
                "L_avg": f"{results.get('L_avg', 0):.2f}",
            }
        )

    print("\nAverage Error per Candidate Evaluation:")
    print("-" * 40)
    print(pd.DataFrame(avg_errors).to_string(index=False))
    print("-" * 40)

    # ========== SECTION 3: SELECTED RULES PER QUERY ==========
    print("\n" + "=" * 100)
    print("SELECTED RULES PER QUERY")
    print("=" * 100)

    selected_results = execution_trace[selected_iter]
    for q_name in sorted(all_query_names):
        print(f"\n{'=' * 80}")
        print(f"Query: {q_name}")
        print("=" * 80)

        n_features = len(selected_results.get("features", {}).get(q_name, []))
        candidate = selected_results.get("candidate", {}).get(q_name, "")
        print(
            f"\n[Selected] @ Iter {selected_iter} "
            f"({candidate}, NFeat={n_features})"
        )

        if q_name in selected_results.get("features", {}):
            print(f"Features: {selected_results['features'][q_name]}")

        if q_name in selected_results.get("rules", {}):
            print("Rules:")
            print(_format_rules(selected_results["rules"][q_name]))

        if q_name in selected_results.get("trans_eval", {}):
            trans_eval = selected_results["trans_eval"][q_name]
            print(
                f"Metrics: trans_f1={trans_eval.get('f1', 0):.2f}, "
                f"L_static="
                f"{selected_results.get('L_static', {}).get(q_name, 0):.2f}"
            )

    print("\n" + "=" * 100 + "\n")
