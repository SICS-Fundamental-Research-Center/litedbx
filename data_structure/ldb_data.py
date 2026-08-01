"""DataFrame-backed LiteDBX data operations."""

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from llm import LdbLLMClient

from .llm_resp_templates import (
    BooleanFeatureResponse,
    FloatFeatureResponse,
    IntFeatureResponse,
    PopulationSpec,
)
from .sem_query import Predicate

logger = logging.getLogger(__name__)

FEATURE_MATERIALIZATION_SCHEMA = 1


class LdbData:
    """Wrap a DataFrame with LiteDBX schema and feature operations."""

    def __init__(
        self,
        data_dir: str | None = None,
        df: pd.DataFrame | None = None,
        config: dict | None = None,
    ):

        if df is None:
            if data_dir is None:
                raise ValueError("data_dir is required when df is not provided")
            self.df, self.config = self._load_from_dir(data_dir)
        else:
            if config is None:
                raise ValueError("config is required when df is provided")
            self.df = df.copy()
            self.config = config

        self.base_features = self.config["base_features"]
        self.id_features = self.config["id_features"]
        self.foreign_keys = self.config["foreign_keys"]
        self.df = self.df.fillna("")

    @staticmethod
    def _load_from_dir(data_dir: str) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Load the full data frame and schema config from a data directory."""
        data_path = Path(data_dir)
        df = pd.read_csv(data_path / "data_full.csv")
        with (data_path / "config.yaml").open(encoding="utf-8") as config_file:
            config = yaml.safe_load(config_file)
        return df, config

    def exclude_fk_and_id(self) -> pd.DataFrame:
        """Return data without ID and foreign-key columns."""
        return self.df.drop(columns=self.id_features + self.foreign_keys)

    def select_active_features(
        self, active_external_features: list[str]
    ) -> pd.DataFrame:
        """Return base features plus selected active external features."""
        selected_features = self.base_features + active_external_features
        missing_features = [
            feature for feature in selected_features if feature not in self.df
        ]
        if missing_features:
            raise ValueError(f"Missing active features: {missing_features}")
        return self.df[selected_features]

    def sigma_retrieve_ucq(self, sigma: list[list[Predicate]]) -> pd.Index:
        """Return rows satisfying a union of conjunctive predicates."""
        # If Sigma is empty or contains empty groups, return all data
        if not sigma or all(not group for group in sigma):
            logger.debug("UCQ is empty - returning all data")
            return self.df.index

        # Start with all rows as False (no match)
        mask = pd.Series([False] * len(self.df), index=self.df.index)

        # Process each conjunctive group (OR logic between groups)
        for group_idx, conjunctive_group in enumerate(sigma):
            if not conjunctive_group:
                logger.warning(
                    "Empty conjunctive group at index %s matches all rows",
                    group_idx,
                )
                return self.df.index

            logger.debug(
                "Processing conjunctive group %s: %s",
                group_idx,
                conjunctive_group,
            )

            # Start with all rows as True for this group.
            # Predicates within the group use AND logic.
            group_mask = pd.Series([True] * len(self.df), index=self.df.index)

            # Apply all predicates in this conjunctive group with AND logic
            for predicate in conjunctive_group:
                group_mask &= self._cq_map(predicate)

            # Combine with overall mask using OR logic
            mask |= group_mask

        return self.df[mask].index

    def expected_enriched_columns(
        self, enriched_features: list[PopulationSpec]
    ) -> set[str]:
        """Return the complete schema expected after feature enrichment."""
        return set(
            self.base_features
            + self.id_features
            + self.foreign_keys
            + [spec.target_col for spec in enriched_features]
        )

    @staticmethod
    def feature_materialization_context_key(
        enriched_features: list[PopulationSpec],
        llm_client: LdbLLMClient,
        is_remote: bool,
    ) -> str:
        """Fingerprint feature semantics and the inference implementation."""
        model_key = "REMOTE_MODELS" if is_remote else "LOCAL_MODELS"
        payload = {
            "schema": FEATURE_MATERIALIZATION_SCHEMA,
            "features": [spec.model_dump() for spec in enriched_features],
            "models": llm_client.config.get(model_key, {}),
            "inference": {
                key: llm_client.config.get(key)
                for key in (
                    "max_tokens",
                    "top_p",
                    "temperature",
                    "random_seed",
                )
            },
        }
        serialized = json.dumps(payload, sort_keys=True, default=str)
        return hashlib.sha1(serialized.encode("utf-8")).hexdigest()

    def reuse_cached_features(self, cached_df: pd.DataFrame) -> bool:
        """Reuse cached feature columns when base rows match exactly."""
        base_columns = self.base_features + self.id_features + self.foreign_keys
        if len(cached_df) != len(self.df) or any(
            column not in cached_df for column in base_columns
        ):
            return False
        current_base = self.df[base_columns].reset_index(drop=True)
        cached_base = cached_df[base_columns].reset_index(drop=True)
        current_base = current_base.where(current_base.notna(), "")
        cached_base = cached_base.where(cached_base.notna(), "")
        try:
            pd.testing.assert_frame_equal(
                current_base, cached_base, check_dtype=False
            )
        except AssertionError:
            return False

        external_columns = [
            column for column in cached_df.columns if column not in base_columns
        ]
        for column in external_columns:
            if column not in self.df:
                self.df[column] = cached_df[column].to_numpy()
        return True

    async def sync_with_enriched_features(
        self,
        enriched_features: list[PopulationSpec],
        llm_client: LdbLLMClient,
        is_remote: bool = False,
    ) -> None:
        """Synchronize DataFrame columns with enriched feature specs."""
        current_external_features = (
            set(self.df.columns)
            - set(self.base_features)
            - set(self.id_features)
            - set(self.foreign_keys)
        )

        features_to_remove = [
            col
            for col in current_external_features
            if col not in {spec.target_col for spec in enriched_features}
        ]
        features_to_add = [
            spec
            for spec in enriched_features
            if spec.target_col not in current_external_features
        ]

        # Remove features that are no longer in the enriched feature space
        self.df.drop(columns=features_to_remove, inplace=True)

        # Add new features to the DataFrame with default values
        for spec in features_to_add:
            if spec.target_col in self.df.columns:
                raise ValueError(
                    f"Target column '{spec.target_col}' already exists "
                    "in DataFrame."
                )
            self.df[spec.target_col] = await self.sem_map(
                spec=spec, llm_client=llm_client, is_remote=is_remote
            )

    async def sem_map(
        self,
        spec: PopulationSpec,
        llm_client: LdbLLMClient,
        is_remote: bool = True,
    ) -> pd.Series:
        """Map one semantic feature spec into a pandas Series."""
        response_model = None
        if spec.feature_type == "int":
            response_model = IntFeatureResponse
        elif spec.feature_type == "float":
            response_model = FloatFeatureResponse
        elif spec.feature_type == "bool":
            response_model = BooleanFeatureResponse
        else:
            raise ValueError(f"Unsupported feature type: {spec.feature_type}")
        if spec.source_col not in self.df:
            raise ValueError(f"Source column not found: {spec.source_col}")

        data_items = self.df[spec.source_col].tolist()
        data_items_wrapped = [
            [item] for item in data_items
        ]  # Wrap each item in a list

        resp = await llm_client.invoke_parallel(
            is_remote=is_remote,
            modality=spec.source_modality,
            prompt=spec.prompt,
            data_items=data_items_wrapped,
            response_model=response_model,
        )

        values = [r.value for r in resp]  # type: ignore[attr-defined]
        return pd.Series(values, index=self.df.index)

    def _cq_map(self, predicate: Predicate) -> pd.Series:
        """Evaluate one conjunctive-query predicate over the DataFrame."""
        # Check whether field is within the DataFrame columns
        if predicate.field not in self.df.columns:
            raise ValueError(
                f"Field '{predicate.field}' not found in DataFrame columns. "
                f"Available columns: {list(self.df.columns)}"
            )

        # Apply the predicate to generate the mask
        if predicate.op == ">":
            return self.df[predicate.field] > predicate.value
        if predicate.op == ">=":
            return self.df[predicate.field] >= predicate.value
        if predicate.op == "<":
            return self.df[predicate.field] < predicate.value
        if predicate.op == "<=":
            return self.df[predicate.field] <= predicate.value
        if predicate.op == "==":
            return self.df[predicate.field] == predicate.value
        if predicate.op == "!=":
            return self.df[predicate.field] != predicate.value
        raise ValueError(f"Unsupported operator: {predicate.op}")
