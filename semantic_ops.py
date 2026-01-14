"""
Semantic operations for LDB Engine.

Handles LLM-based semantic feature extraction and mapping.
"""
from pathlib import Path
from typing import List, Tuple, Type
import pandas as pd
from pydantic import BaseModel

from llm_client import LiteLLMWrapper
from data_structures import (
    PopulationSpecs, 
    BooleanFeatureResponse, 
    IntFeatureResponse, 
    FloatFeatureResponse
)
import logging
logger = logging.getLogger(__name__)


def detect_modality(data_item: str) -> str:
    """
    Detect the modality of a data item.

    Args:
        data_item: String representation of the data item

    Returns:
        "IMAGE" if the data item is an image path, "TEXT" otherwise
    """
    if any(
        data_item.endswith(extension) for extension in [".png", ".jpg", ".jpeg"]
    ):
        return "IMAGE"
    else:
        return "TEXT"


async def sem_coloring(
    df: pd.DataFrame,
    sem_rules: List[Tuple[str, str, str]],
    llm_client: LiteLLMWrapper
) -> pd.DataFrame:
    """
    Filter high-confidence negative samples or low-confidence positive samples.

    Args:
        df: Input dataframe
        sem_rules: List of (column_name, semantic_description) tuples
        llm_client: LLM client for semantic evaluation

    Returns:
        Dataframe with sem_flag column added (1=positive, -1=negative, 0=unknown)
    """
    df_cp = df.copy()
    if "sem_flag" not in df_cp.columns:
        df_cp["sem_flag"] = 0

    for col_name, condition, prompt_ in sem_rules:
        modality = detect_modality(df_cp[col_name].iloc[0])
        semantic_desc = prompt_.format(COL=col_name, CONDITION=condition)
        consensus_results = await llm_client.invoke_parallel_consensus(
            modality=modality,
            prompt=semantic_desc,
            data_items=df_cp[col_name].astype(str).tolist(),
            response_model=BooleanFeatureResponse,
        )
        for result in consensus_results:
            pos, is_match = result
            col_idx = df_cp.columns.get_loc("sem_flag")  # type: ignore
            if df_cp.iat[pos, col_idx] == -1:  # type: ignore
                # Already failed by the consensus test for at least one rule.
                continue
            df_cp.iat[pos, col_idx] = 1 if is_match else -1  # type: ignore

    return df_cp


async def sem_mapping(
    df: pd.DataFrame,
    col_name: str,
    new_col_name: str,
    prompt: str,
    response_model: Type[BaseModel],
    llm_client: LiteLLMWrapper,
    ckpt_home: Path,
    ckpt_prefix: str = "",
    enable_cache: bool = True
) -> pd.DataFrame:
    """
    Apply semantic mapping to create a new column.

    Args:
        df: Input dataframe
        col_name: Source column name
        new_col_name: Target column name
        prompt: Prompt for LLM
        response_model: Response model for LLM
        llm_client: LLM client
        ckpt_home: Checkpoint directory for caching
        enable_cache: Whether to use cached results

    Returns:
        Dataframe with new column added
    """
    ckpt_path = ckpt_home / f"{ckpt_prefix}_SEMMAP_{new_col_name}.csv"
    if enable_cache and ckpt_path.exists():
        logger.debug(f"Loading cached semantic mapping for column '{new_col_name}'...")
        cached_df = pd.read_csv(ckpt_path).reset_index(drop=True)
        assert len(cached_df) == len(df), \
            "Cached semantic mapping length does not match current dataframe length."
        df[new_col_name] = cached_df[new_col_name]
        return df

    data_items = df[col_name].astype(str).tolist()
    modality = detect_modality(data_items[0] if data_items else "")
    llm_labels = await llm_client.invoke_parallel(
        modality=modality,
        is_remote=True,
        prompt=prompt,
        data_items=data_items,
        response_model=response_model,
    )
    df[new_col_name] = llm_labels

    if enable_cache:
        feature_col = df[[new_col_name]]
        feature_col.reset_index(drop=True, inplace=True)
        feature_col.to_csv(ckpt_path, index=False)
        logger.debug(f"Stored semantic mapping for column '{new_col_name}' to checkpoint.")

    return df


async def sem_multi_mapping(
    df: pd.DataFrame,
    mapping_specs: PopulationSpecs,
    llm_client: LiteLLMWrapper,
    ckpt_home: Path,
    ckpt_prefix: str = "",
    enable_cache: bool = True
) -> pd.DataFrame:
    """
    Apply multiple semantic mappings in batch.

    Args:
        df: Input dataframe
        mapping_specs: Specifications for mappings
        llm_client: LLM client
        ckpt_home: Checkpoint directory for caching
        enable_cache: Whether to use cached results

    Returns:
        Dataframe with new columns added
    """
    for spec in mapping_specs.value:
        cache_path = ckpt_home / f"{ckpt_prefix}_SEMMAP_{spec.target_col}.csv"
        if enable_cache and cache_path.exists():
            logger.debug(f"Loading cached semantic mapping for feature: {spec.target_col}...")
            cached_df = pd.read_csv(cache_path).reset_index(drop=True)
            assert len(cached_df) == len(df), (
                f"Cached semantic mapping length does not match current dataframe length. "
                f"Expected {len(df)}, got {len(cached_df)}."
            )
            df[spec.target_col] = cached_df[spec.target_col]
            continue
        df = await sem_mapping(
            df,
            spec.source_col,
            spec.target_col,
            spec.prompt,
            BooleanFeatureResponse if spec.feature_type == "bool" else
            IntFeatureResponse if spec.feature_type == "int" else
            FloatFeatureResponse,
            llm_client,
            ckpt_home,
            enable_cache=False
        )
        if enable_cache:
            feature_col = df[[spec.target_col]]
            feature_col.to_csv(cache_path, index=False)
            logger.debug(f"Stored semantic mapping for feature: {spec.target_col} to checkpoint.")
    return df
