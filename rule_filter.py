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


async def prefilter_by_proxies(
    df: pd.DataFrame,
    workloads: Dict[str, UCQ],
    query_name: str,
    llm_client,
    ckpt_home: Path,
    ckpt_prefix: str = "",
    enable_cache: bool = True,
) -> pd.DataFrame:
    """
    Filter data based on semantic rules using proxy models.

    Args:
        df: Input dataframe
        workloads: Dictionary of query workloads
        query_name: Name of the query to apply
        llm_client: LLM client
        ckpt_home: Checkpoint directory for caching
        enable_cache: Whether to use cached results

    Returns:
        Filtered dataframe
    """
    if enable_cache and \
        (ckpt_home / f"{ckpt_prefix}_{query_name}_sem_prefilter_result.csv").exists():
        logger.debug(f"Loading cached semantic prefilter results for query {query_name}...")
        result_df = pd.read_csv(
            ckpt_home / f"{ckpt_prefix}_{query_name}_sem_prefilter_result.csv").reset_index(drop=True)
        return result_df

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
            aggregated_flags.append(1)  # At least one rule matched.
        elif all(v == 0 for v in values_at_index):
            aggregated_flags.append(0)  # No rule matched.
        else:
            aggregated_flags.append(-1)  # All matched rules failed.

    fired_df["sem_flag"] = aggregated_flags

    result_df = fired_df[(fired_df["sem_flag"] != -1)].drop(columns=["sem_flag"])

    if enable_cache:
        result_df.to_csv(
            ckpt_home / f"{ckpt_prefix}_{query_name}_sem_prefilter_result.csv", index=False)
        logger.debug(f"Stored semantic prefilter results of {query_name} to checkpoint.")

    return result_df

def filter_by_rewritten_rules(
    df: pd.DataFrame,
    query: UCQ,
    backup_weight: float = 1.0,
    pos_weight: float = 1.0,
    neg_weight: float = -1.0
) -> pd.DataFrame:
    """
    Filter data based on rewritten rules using a scoring system.

    Each instance starts with a score of 0. Rules are applied and matching instances
    have their score updated by the corresponding weight. Final result includes
    instances with score > 0.

    Args:
        df: Input dataframe
        query: UCQ query with rewritten rules
        backup_weight: Weight to add when backup rule matches (default: 1.0)
        pos_weight: Weight to add when positive rule matches (default: 1.0)
        neg_weight: Weight to add when negative rule matches (default: -1.0)

    Returns:
        Filtered dataframe with score > 0
    """
    # Initialize score column with 0
    df = df.copy()
    df["score"] = 0

    # Apply backup rules - add weight for each matching rule collection
    for cq in query.rules:
        if not cq.backup_rules:
            continue

        # Create a mask for this CQ's backup rules (conjunction)
        mask = pd.Series([True] * len(df), index=df.index)
        for col, op, val in cq.backup_rules:
            logger.debug(f"Applying backup rule: {col} {op} {val}")
            rule_mask = _get_rule_mask(df, col, op, val)
            mask &= rule_mask

        # Add backup_weight to score for matching instances
        df.loc[mask, "score"] += backup_weight

    # Apply positive (pos) rules - add weight for each matching rule collection
    for cq in query.rules:
        if not cq.rewritten_pos_rules:
            continue

        # Create a mask for this CQ's pos rules (conjunction)
        mask = pd.Series([True] * len(df), index=df.index)
        for col, op, val in cq.rewritten_pos_rules:
            logger.debug(f"Applying rewritten pos rule: {col} {op} {val}")
            rule_mask = _get_rule_mask(df, col, op, val)
            mask &= rule_mask

        # Add pos_weight to score for matching instances
        df.loc[mask, "score"] += pos_weight

    # Apply negative (neg) rules - add weight for each matching rule collection
    for cq in query.rules:
        if not cq.rewritten_neg_rules:
            continue

        # Create a mask for this CQ's neg rules (conjunction)
        mask = pd.Series([True] * len(df), index=df.index)
        for col, op, val in cq.rewritten_neg_rules:
            logger.debug(f"Applying rewritten neg rule: {col} {op} {val}")
            rule_mask = _get_rule_mask(df, col, op, val)
            mask &= rule_mask

        # Add neg_weight to score for matching instances
        df.loc[mask, "score"] += neg_weight

    # Filter instances with score > 0
    result_df = df[df["score"] > 0].drop(columns=["score"])

    return result_df


def _get_rule_mask(
    df: pd.DataFrame,
    col: str,
    op: Literal["Eq", "Gt", "Lt", "Ge", "Le", "In"],
    val: Union[str, int, float, bool, List[Union[str, int, float, bool]]]
) -> pd.Series:
    """
    Get a boolean mask for a single rule operator.

    Args:
        df: Input dataframe
        col: Column name
        op: Operator (Eq, Gt, Lt, Ge, Le, In)
        val: Value(s) to compare against

    Returns:
        Boolean Series indicating which rows match the rule
    """
    if op == "Eq":
        return df[col] == val
    elif op == "Gt":
        return df[col] > val
    elif op == "Lt":
        return df[col] < val
    elif op == "Ge":
        return df[col] >= val
    elif op == "Le":
        return df[col] <= val
    elif op == "In":
        return df[col].isin(val)  # type: ignore
    else:
        raise ValueError(f"Unsupported operation: {op}")


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
    mask = _get_rule_mask(df, col, op, val)
    return df[mask]
