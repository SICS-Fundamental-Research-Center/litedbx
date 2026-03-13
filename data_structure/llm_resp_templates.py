from pydantic import BaseModel
from typing import Literal


class PopulationSpec(BaseModel):
    source_col: str
    source_modality: Literal["Text", "Image"]
    target_col: str
    prompt: str
    feature_type: Literal["bool", "float", "int", "undefined"]


class PopulationSpecs(BaseModel):
    value: list[PopulationSpec]


class BooleanFeatureResponse(BaseModel):
    """Response model for boolean feature extraction."""
    value: bool


class IntFeatureResponse(BaseModel):
    """Response model for int feature extraction."""
    value: int


class FloatFeatureResponse(BaseModel):
    """Response model for float feature extraction."""
    value: float

class StringListFeatureResponse(BaseModel):
    """Response model for list of strings feature extraction."""
    value: list[str]


class FeatureRefinementResponse(BaseModel):
    """Response model for feature space refinement."""
    to_add: list[PopulationSpec]
    to_remove: list[str]


class PredicateResponse(BaseModel):
    field: str
    op: Literal[">", ">=", "<", "<=", "==", "!="]
    value: str | int | float | bool

class PredicateResponses(BaseModel):
    value: list[PredicateResponse]
    can_exact_match: bool = False
