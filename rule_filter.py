"""
Rule filtering logic for LDB Engine.

Handles static and semantic rule filtering for dataframes.
"""
import asyncio
from pathlib import Path
from typing import List, Tuple, Union, Literal, Dict

import pandas as pd

from data_structures import UCQ
from semantic_ops import sem_coloring
import logging
logger = logging.getLogger(__name__)


def prefilter_by_static_rules(
    df: pd.DataFrame,
    workloads: Dict[str, UCQ],
    query_name: str,
    ckpt_home: Path,
    enable_cache: bool = True
) -> pd.DataFrame:
    """
    Filter data based on static rules.

    Args:
        df: Input dataframe
        workloads: Dictionary of query workloads
        query_name: Name of the query to apply
        ckpt_home: Checkpoint directory for caching
        enable_cache: Whether to use cached results

    Returns:
        Filtered dataframe
    """
    # Check whether the base result already exists.
    if enable_cache and (ckpt_home / f"{query_name}_base.csv").exists():
        logger.debug(f"Loading cached base result for query {query_name}...")
        return pd.read_csv(ckpt_home / f"{query_name}_base.csv").reset_index(drop=True)

    query = workloads[query_name]
    result = pd.DataFrame()

    # Make sure all selected columns cannot have null values.
    involved_cols = set()
    for cq in query.rules:
        for col, _, _ in cq.static_rules:
            involved_cols.add(col)
        for col, _ in cq.sem_rules:
            involved_cols.add(col)
    df = df.dropna(subset=list(involved_cols))

    # Apply static rules.
    for cq in query.rules:
        df_cp = df.copy()
        for col, op, val in cq.static_rules:
            df_cp = _apply_rule_operator(df_cp, col, op, val)
        result = pd.concat([result, df_cp], ignore_index=True)
    result = result.drop_duplicates()

    if enable_cache:
        result.to_csv(ckpt_home / f"{query_name}_base.csv", index=False)
        logger.debug(f"Stored base result of {query_name} to checkpoint.")

    return result


async def prefilter_by_semantic_rules(
    df: pd.DataFrame,
    workloads: Dict[str, UCQ],
    query_name: str,
    llm_client,
    ckpt_home: Path,
    early_positive: bool = True,
    drop_neg: bool = True,
    enable_cache: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Filter data based on semantic rules using LLM.

    Args:
        df: Input dataframe
        workloads: Dictionary of query workloads
        query_name: Name of the query to apply
        llm_client: LLM client
        ckpt_home: Checkpoint directory for caching
        early_positive: Whether to return early positive samples
        drop_neg: Whether to drop negative samples
        enable_cache: Whether to use cached results

    Returns:
        Tuple of (early_positive_df, result_df)
    """
    if enable_cache and \
        (ckpt_home / f"{query_name}_sem_early_positive.csv").exists() and \
                (ckpt_home / f"{query_name}_sem_prefilter_result.csv").exists():
        logger.debug(f"Loading cached semantic prefilter results for query {query_name}...")
        early_positive_df = pd.read_csv(ckpt_home / f"{query_name}_sem_early_positive.csv").reset_index(drop=True)
        result_df = pd.read_csv(ckpt_home / f"{query_name}_sem_prefilter_result.csv").reset_index(drop=True)
        return early_positive_df, result_df

    query = workloads[query_name]
    fired_df = df.copy()
    sem_flags = []
    for cq in query.rules:
        # Reset the sem_flag column.
        fired_df["sem_flag"] = 0

        # Color using one collection of conjunctive rules.
        colored_df = await sem_coloring(fired_df, cq.sem_rules, llm_client)

        # Collect the sem_flags.
        sem_flags.append(colored_df["sem_flag"].tolist())

    # Aggregate the sem_flags to sem_flag.
    aggregated_flags = []
    for i in range(len(sem_flags[0])):
        values_at_index = [flags[i] for flags in sem_flags]
        if 1 in values_at_index:
            aggregated_flags.append(1)
        elif all(v == 0 for v in values_at_index):
            aggregated_flags.append(0)
        else:
            aggregated_flags.append(-1)

    fired_df["sem_flag"] = aggregated_flags

    early_positive_df, result_df = pd.DataFrame(), pd.DataFrame()

    if early_positive:
        early_positive_df = fired_df[fired_df["sem_flag"] == 1].drop(columns=["sem_flag"])
    else:
        early_positive_df = pd.DataFrame(columns=fired_df.columns).drop(columns=["sem_flag"])

    if drop_neg and early_positive:
        early_positive_df = fired_df[fired_df["sem_flag"] == 1].drop(columns=["sem_flag"])
        result_df = fired_df[fired_df["sem_flag"] == 0].drop(columns=["sem_flag"])
    elif drop_neg and not early_positive:
        early_positive_df = pd.DataFrame(columns=fired_df.columns).drop(columns=["sem_flag"])
        result_df = fired_df[(fired_df["sem_flag"] == 0) | (fired_df["sem_flag"] == 1)].drop(columns=["sem_flag"])
    elif early_positive and not drop_neg:
        early_positive_df = fired_df[fired_df["sem_flag"] == 1].drop(columns=["sem_flag"])
        result_df = fired_df[fired_df["sem_flag"] == 0 | (fired_df["sem_flag"] == -1)].drop(columns=["sem_flag"])
    else:
        early_positive_df = pd.DataFrame(columns=fired_df.columns).drop(columns=["sem_flag"])
        result_df = fired_df.drop(columns=["sem_flag"])

    if enable_cache:
        early_positive_df.to_csv(ckpt_home / f"{query_name}_sem_early_positive.csv", index=False)
        result_df.to_csv(ckpt_home / f"{query_name}_sem_prefilter_result.csv", index=False)
        logger.debug(f"Stored semantic prefilter results of {query_name} to checkpoint.")

    return early_positive_df, result_df


def filter_by_rewritten_rules(df: pd.DataFrame, query: UCQ) -> pd.DataFrame:
    """
    Filter data based on rewritten rules (backup + positive + negative).

    Args:
        df: Input dataframe
        query: UCQ query with rewritten rules

    Returns:
        Filtered dataframe
    """
    # Step 1: Use LLM-guided backup rules to recall relatively high-confidence samples.
    backup_result = pd.DataFrame()
    for cq in query.rules:
        df_cp = df.copy()
        for col, op, val in cq.backup_rules:
            logger.debug(f"Applying backup rule: {col} {op} {val}")
            df_cp = _apply_rule_operator(df_cp, col, op, val)
        backup_result = pd.concat([backup_result, df_cp], ignore_index=True)
    backup_result = backup_result.drop_duplicates()

    # Step 2: Use rewritten neg rules to reduce false positives.
    # For each CQ, apply neg rules as POSITIVE conjunction to get samples to EXCLUDE,
    # then remove them from backup_result (De Morgan's law)
    for cq in query.rules:
        if not cq.rewritten_neg_rules:
            continue

        # Apply all neg rules as positive conjunction to get the set to exclude
        df_exclude = backup_result.copy()
        for col, op, val in cq.rewritten_neg_rules:
            logger.debug(f"Applying rewritten neg rule (to get exclusion set): {col} {op} {val}")
            df_exclude = _apply_rule_operator(df_exclude, col, op, val)

        # Remove the excluded set from backup_result
        exclude_indices = set(df_exclude.index)
        backup_result = backup_result[~backup_result.index.isin(exclude_indices)]

    # Combine backup_result (after negation) with pos_result and deduplicate
    final_result = backup_result.drop_duplicates()

    return final_result


def _apply_rule_operator(
    df: pd.DataFrame,
    col: str,
    op: Literal["Eq", "Gt", "Lt", "Ge", "Le", "In"],
    val: Union[str, int, float, bool, List[Union[str, int, float, bool]]]
) -> pd.DataFrame:
    """
    Apply a single rule operator to a dataframe.

    Args:
        df: Input dataframe
        col: Column name
        op: Operator (Eq, Gt, Lt, Ge, Le, In)
        val: Value(s) to compare against

    Returns:
        Filtered dataframe
    """
    if op == "Eq":
        return df[df[col] == val]
    elif op == "Gt":
        return df[df[col] > val]
    elif op == "Lt":
        return df[df[col] < val]
    elif op == "Ge":
        return df[df[col] >= val]
    elif op == "Le":
        return df[df[col] <= val]
    elif op == "In":
        return df[df[col].isin(val)]  # type: ignore
    else:
        raise ValueError(f"Unsupported operation: {op}")
