import pandas as pd
import logging
import yaml
from typing import Optional, Tuple
from llm import LdbLLMClient
from .sem_query import Predicate
from .llm_resp_templates import (
    PopulationSpec, 
    IntFeatureResponse, 
    FloatFeatureResponse, 
    BooleanFeatureResponse
)

logger = logging.getLogger(__name__)

class LdbData:
    def __init__(
            self, 
            data_dir: Optional[str] = None, 
            df: Optional[pd.DataFrame] = None, 
            config: Optional[dict] = None):

        if df is None:
            assert data_dir is not None, \
                "Data directory must be provided when DataFrame is not directly passed."
            self.df = pd.read_csv(f"{data_dir}/data_full.csv")
            with open(f"{data_dir}/config.yaml", "r") as f:
                self.config = yaml.safe_load(f)
            self.base_features = self.config["base_features"]
            self.id_features = self.config["id_features"]
            self.foreign_keys = self.config["foreign_keys"]
        else:
            assert config is not None, \
                "Config must be provided when DataFrame is directly passed."
            self.df = df
            self.config = config
            self.base_features = self.config["base_features"]
            self.id_features = self.config["id_features"]
            self.foreign_keys = self.config["foreign_keys"]

        # Preprocessing
        self.df = self.df.fillna("")


    def exclude_fk_and_id(self):
        return self.df.drop(columns=self.id_features + self.foreign_keys)


    def select_active_features(self, active_external_features: list[str])  -> pd.DataFrame:
        selected_features = self.base_features + active_external_features
        ret = self.df[selected_features]
        return ret


    def sigma_retrieve_ucq(
            self, Sigma: list[list[Predicate]]) -> pd.Index:
        # If Sigma is empty or contains empty groups, return all data
        if not Sigma or all(not group for group in Sigma):
            logger.info("UCQ is empty - returning all data")
            return self.df.index

        # Start with all rows as False (no match)
        mask = pd.Series([False] * len(self.df), index=self.df.index)

        # Process each conjunctive group (OR logic between groups)
        for group_idx, conjunctive_group in enumerate(Sigma):
            if not conjunctive_group:
                # Empty group means no filter for this group
                logger.warning(f"Empty conjunctive group at index {group_idx} - skipping")
                continue

            logger.info(f"Processing conjunctive group {group_idx}: {conjunctive_group}")

            # Start with all rows as True for this group (AND logic within group)
            group_mask = pd.Series([True] * len(self.df), index=self.df.index)

            # Apply all predicates in this conjunctive group with AND logic
            for predicate in conjunctive_group:
                group_mask &= self._cq_map(predicate)

            # Combine with overall mask using OR logic
            mask |= group_mask

        return self.df[mask].index


    async def sync_with_enriched_features(
            self, 
            enriched_features: list[PopulationSpec], 
            llm_client: LdbLLMClient,
            is_remote: bool = False) -> None:
        current_external_features = set(self.df.columns) - set(self.base_features) \
            - set(self.id_features) - set(self.foreign_keys)
        
        features_to_remove = [col for col in current_external_features
                              if col not in {spec.target_col for spec in enriched_features}]
        features_to_add = [spec for spec in enriched_features 
                           if spec.target_col not in current_external_features]
        
        # Remove features that are no longer in the enriched feature space
        self.df.drop(columns=features_to_remove, inplace=True)

        # Add new features to the DataFrame with default values
        for spec in features_to_add:
            assert spec.target_col not in self.df.columns, (
                f"Target column '{spec.target_col}' already exists in DataFrame. "
            )
            self.df[spec.target_col] = await self._sem_map(
                spec=spec, llm_client=llm_client, is_remote=is_remote
            )


    def _cq_map(self, predicate: Predicate) -> pd.Series:
        # Check whether field is within the DataFrame columns
        if predicate.field not in self.df.columns:
            raise ValueError(
                f"Field '{predicate.field}' not found in DataFrame columns. "
                f"Available columns: {list(self.df.columns)}"
            )

        # Apply the predicate to generate the mask
        if predicate.op == ">":
            return self.df[predicate.field] > predicate.value
        elif predicate.op == ">=":
            return self.df[predicate.field] >= predicate.value
        elif predicate.op == "<":
            return self.df[predicate.field] < predicate.value
        elif predicate.op == "<=":
            return self.df[predicate.field] <= predicate.value
        elif predicate.op == "==":
            return self.df[predicate.field] == predicate.value
        elif predicate.op == "!=":
            return self.df[predicate.field] != predicate.value
        else:
            raise ValueError(f"Unsupported operator: {predicate.op}")


    async def _sem_map(
            self, spec: PopulationSpec, llm_client: LdbLLMClient, 
            is_remote: bool=True) -> pd.Series:
        response_model = None
        if spec.feature_type == "int":
            response_model = IntFeatureResponse
        elif spec.feature_type == "float":
            response_model = FloatFeatureResponse
        elif spec.feature_type == "bool":
            response_model = BooleanFeatureResponse
        else:
            raise ValueError(f"Unsupported feature type: {spec.feature_type}")
        assert response_model is not None, \
            f"Fail to set resp. model with feature type: {spec.feature_type}."

        data_items = self.df[spec.source_col].tolist()
        data_items_wrapped = [[item] for item in data_items]  # Wrap each item in a list

        resp = await llm_client.invoke_parallel(
            is_remote=is_remote,
            modality=spec.source_modality,
            prompt=spec.prompt,
            data_items=data_items_wrapped,
            response_model=response_model,
        )

        return pd.Series([r.value for r in resp], index=self.df.index)  # type: ignore

