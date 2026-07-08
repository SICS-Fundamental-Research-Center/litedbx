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

    def update_statistics(self, key: str, value: dict[str, Any]) -> None:
        assert key in self.usage_statistics[0], f"Invalid statistics key: {key}"
        for k, v in value.items():
            self.usage_statistics[0][key][k] += v

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

                results = pd.DataFrame(
                    {
                        "inc_round": inc_rounds,
                        "inc_ratio": inc_ratios,
                        "trans_f1": trans_f1s,
                        "pred_f1": pred_f1s,
                        "error_certificate": error_certificates,
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

    # Get all query names
    all_query_names = set()
    for results in execution_trace.values():
        for key in results.keys():
            if key not in [
                "rules",
                "features",
                "pred_eval",
                "trans_eval",
                "L_rew",
                "penalty_rew",
                "L_LOO",
                "penalty_LOO",
                "L_obj",
                "L_subj",
                "L_static",
                "L_avg",
                "memory_cost",
                "selectivity",
            ]:
                all_query_names.add(key)

    # Find best per-query trans_f1 and global lowest L_avg.
    best_trans_f1_iters = {}  # query_name -> (iter_idx, trans_f1)
    global_best_iter = min(
        execution_trace.keys(),
        key=lambda i: execution_trace[i].get("L_avg", float("inf")),
    )

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
            selectivity = results.get("selectivity", {}).get(q_name, {})
            row["overall_selectivity"] = (
                f"{selectivity.get('overall_selectivity', 0):.4f}"
                if isinstance(selectivity, dict)
                else "0.0000"
            )

            overview_data.append(row)

    df_overview = pd.DataFrame(overview_data)
    col_order = [
        "Iter",
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
    ]
    col_order = [c for c in col_order if c in df_overview.columns]
    df_overview = df_overview[col_order]

    print("\n" + "=" * 150)
    print("OVERVIEW - Evaluation Metrics per Iteration")
    print("=" * 150)
    print(df_overview.to_string(index=False))
    print("=" * 150)

    # ========== SECTION 2: AVERAGE ERROR ==========
    avg_errors = [
        {"Iter": i, "NFeat": i, "L_avg": f"{results.get('L_avg', 0):.2f}"}
        for i, results in execution_trace.items()
    ]

    print("\nAverage Error per Iteration:")
    print("-" * 40)
    print(pd.DataFrame(avg_errors).to_string(index=False))
    print("-" * 40)

    # ========== SECTION 3: BEST RULES PER QUERY ==========
    print("\n" + "=" * 100)
    print("BEST RULES PER QUERY")
    print("=" * 100)

    for q_name in sorted(all_query_names):
        print(f"\n{'=' * 80}")
        print(f"Query: {q_name}")
        print("=" * 80)

        # Show rules from iteration with highest trans_f1
        if q_name in best_trans_f1_iters:
            iter_idx, trans_f1 = best_trans_f1_iters[q_name]
            results = execution_trace[iter_idx]

            print(
                f"\n[Highest trans_f1={trans_f1:.2f}] @ Iter "
                f"{iter_idx} (NFeat={iter_idx + 1})"
            )

            if "features" in results and q_name in results["features"]:
                print(f"Features: {results['features'][q_name]}")

            if "rules" in results and q_name in results["rules"]:
                rules = results["rules"][q_name]
                print("Rules:")
                print(_format_rules(rules))

            if "trans_eval" in results and q_name in results["trans_eval"]:
                te = results["trans_eval"][q_name]
                print(
                    f"Metrics: trans_f1={te.get('f1', 0):.2f}, "
                    f"L_static={results.get('L_static', {}).get(q_name, 0):.2f}"
                )

        # Show rules from iteration with lowest L_avg (global best)
        results = execution_trace[global_best_iter]
        l_avg = results.get("L_avg", 0)

        print(
            f"\n[Lowest L_avg={l_avg:.2f}] @ Iter "
            f"{global_best_iter} (NFeat={global_best_iter + 1})"
        )

        if "features" in results and q_name in results["features"]:
            print(f"Features: {results['features'][q_name]}")

        if "rules" in results and q_name in results["rules"]:
            rules = results["rules"][q_name]
            print("Rules:")
            print(_format_rules(rules))

        if "trans_eval" in results and q_name in results["trans_eval"]:
            te = results["trans_eval"][q_name]
            print(
                f"Metrics: trans_f1={te.get('f1', 0):.2f}, "
                f"L_static={results.get('L_static', {}).get(q_name, 0):.2f}"
            )

    print("\n" + "=" * 100 + "\n")
