import pandas as pd
import logging
import yaml
from typing import Optional
from llm import LdbLLMClient
from .sem_query import Predicate
from .llm_resp_templates import PopulationSpec, IntFeatureResponse, FloatFeatureResponse, BooleanFeatureResponse

logger = logging.getLogger(__name__)

class LdbData:
    def __init__(self, data_dir: Optional[str] = None, df: Optional[pd.DataFrame] = None, config: Optional[dict] = None):
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


    def select_active_features(self, active_external_features: list[str]) -> pd.DataFrame:
        selected_features = self.base_features + active_external_features
        return self.df[selected_features]


    def sigma_retrieve(
            self, Sigma: list[Predicate], reset_index: bool = False) -> 'LdbData':
        # Start with all rows as True (no filter)
        mask = pd.Series([True] * len(self.df), index=self.df.index)

        # Combine masks for all predicates with AND logic
        for predicate in Sigma:
            logger.info(f"Applying predicate: {predicate}")
            mask &= self._cq_map(predicate)

        result = self.df[mask].copy()
        if reset_index:
            result = result.reset_index(drop=True)
        return LdbData(df=result, config=self.config)

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

