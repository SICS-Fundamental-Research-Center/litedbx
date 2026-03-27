from pydantic import BaseModel, field_validator
from typing import Literal


class PopulationSpec(BaseModel):
    source_col: str
    source_modality: Literal["Text", "Image", "VectorText"]
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
    field: list[str]  # Merged/grouped fields from semantic grouping
    op: Literal[">", ">=", "<", "<=", "==", "!="]
    value: list[str | int | float | bool]

    @field_validator('field')
    @classmethod
    def validate_field_not_empty(cls, v):
        """Validate that the field list is not empty."""
        if len(v) == 0:
            raise ValueError("Field list cannot be empty")
        return v

    @field_validator('value')
    @classmethod
    def validate_value_length(cls, v, info):
        """
        Validate that the value list length matches the operator.

        - For == and != operators: value list can have length >= 1 (disjunctive equality)
        - For all other operators (>, >=, <, <=): value list must have length == 1
        """
        op = info.data.get('op')

        if op in ['==', '!=']:
            # For equality operators, allow multiple values (disjunctive case)
            if len(v) == 0:
                raise ValueError(f"Value list cannot be empty for operator '{op}'")
        else:
            # For comparison operators, only allow single value
            if len(v) != 1:
                raise ValueError(
                    f"Operator '{op}' requires exactly one value, but got {len(v)} values. "
                    f"Multiple values are only supported for '==' and '!=' operators."
                )
        return v


class PredicateResponses(BaseModel):
    value: list[list[PredicateResponse]]
    can_exact_match: bool = False


class RelevantFieldsResponse(BaseModel):
    """Response model for identifying query-relevant fields with categories."""
    value: dict[str, list[str]]
