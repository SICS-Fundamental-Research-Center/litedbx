"""
Feature space generation for LDB Engine.

Handles LLM-based feature candidate generation.
"""
import json
from pathlib import Path
from typing import Dict, Type

import pandas as pd

from data_structures import PopulationSpecs, UCQ
from llm_client import LiteLLMWrapper
from prompts import PROMPTS
from semantic_ops import detect_modality
from logger_config import logger


async def generate_feature_space(
    workloads: Dict[str, UCQ],
    data_views: Dict[str, pd.DataFrame],
    llm_client: LiteLLMWrapper,
    response_model: Type[PopulationSpecs],
    ckpt_home: Path,
    enable_cache: bool = True
) -> Dict[str, PopulationSpecs]:
    """
    Generate feature space for each query using LLM.

    Args:
        workloads: Dictionary of query workloads
        data_views: Dictionary of data views for each query
        llm_client: LLM client
        response_model: Response model for LLM
        ckpt_home: Checkpoint directory for caching
        enable_cache: Whether to use cached results

    Returns:
        Dictionary mapping query names to population specifications
    """
    feature_space_signature = "_".join(workloads.keys())
    cache_path = ckpt_home / f"feature_space_{feature_space_signature}.json"

    if enable_cache and cache_path.exists():
        logger.debug("Loading cached feature space...")
        with open(cache_path, 'r') as f:
            cached_data = json.load(f)
        # Deserialize back to Dict[str, List[PopulationSpec]]
        from data_structures import PopulationSpec
        population_specs = {
            query_name: PopulationSpecs(value=[PopulationSpec(**spec) for spec in specs])
            for query_name, specs in cached_data.items()
        }
        return population_specs

    population_specs = {}
    for query_name, workload in workloads.items():
        data_view = data_views[query_name]
        for rule in workload.rules:
            for col_name, semantic_desc in rule.sem_rules:
                # Determine the modality of the source column.
                data_modality = detect_modality(
                    data_view.iloc[0][col_name]
                ) if len(data_view) > 0 else "TEXT"

                # Sample data from the source column.
                sample_data = data_view[col_name].astype(str).dropna().\
                    sample(n=min(10, len(data_view)), random_state=42).tolist()

                prompt = PROMPTS["GEN_FEAT_CANDIDATE_PROMPT"].format(
                    MODALITY=data_modality,
                    DESC=semantic_desc,
                    SAMPLE_DATA="\n".join(sample_data),
                    SOURCE_COL=col_name,
                )

                llm_response = llm_client.invoke(
                    modality="TEXT",
                    is_remote=True,
                    prompt=prompt,
                    response_model=response_model,
                )
                if query_name not in population_specs:
                    population_specs[query_name] = []
                population_specs[query_name].extend(llm_response)

    # Serialize PopulationSpec objects to dicts
    if enable_cache:
        cache_data = {
            query_name: [spec.model_dump() for spec in specs]
            for query_name, specs in population_specs.items()
        }
        with open(cache_path, 'w') as f:
            json.dump(cache_data, f, indent=2)
        logger.debug(f"Stored feature space to checkpoint: {cache_path}")

    return population_specs
