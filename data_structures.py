"""
Data structures for LDB Engine.

Defines the core data models used throughout the system.
"""
from dataclasses import dataclass, field
from typing import List, Tuple, Union, Literal
from pydantic import BaseModel


@dataclass
class CQ:
    """Conjunctive Query - represents a single collection of conjunctive rules."""
    static_rules: List[
        Tuple[
            str,
            Literal["Eq", "Gt", "Lt", "Ge", "Le", "In"],
            Union[str, int, float, List[Union[str, int, float]]],
        ]
    ] = field(default_factory=list)
    sem_rules: List[Tuple[str, str]] = field(default_factory=list)
    backup_rules: List[
        Tuple[
            str,
            Literal["Eq", "Gt", "Lt", "Ge", "Le", "In"],
            Union[str, int, float, bool, List[Union[str, int, float, bool]]],
        ]
    ] = field(default_factory=list)
    rewritten_pos_rules: List[
        Tuple[
            str,
            Literal["Eq", "Gt", "Lt", "Ge", "Le", "In"],
            Union[str, int, float, bool, List[Union[str, int, float, bool]]],
        ]
    ] = field(default_factory=list)
    rewritten_neg_rules: List[
        Tuple[
            str,
            Literal["Eq", "Gt", "Lt", "Ge", "Le", "In"],
            Union[str, int, float, bool, List[Union[str, int, float, bool]]],
        ]
    ] = field(default_factory=list)


@dataclass
class UCQ:
    """Union of Conjunctive Queries - represents a disjunction of CQs."""
    select_cols: List[str]
    rules: List[CQ]


@dataclass
class EvalResult:
    """Evaluation result with TP, FP, FN and derived metrics."""
    tp: int
    fp: int
    fn: int
    precision: float
    recall: float
    f1: float


class PopulationSpec(BaseModel):
    """Specification for populating a feature column."""
    source_col: str
    target_col: str
    prompt: str
    feature_type: Literal["bool", "float", "int"]


class PopulationSpecs(BaseModel):
    """Collection of population specifications."""
    value: List[PopulationSpec]


class BooleanFeatureResponse(BaseModel):
    """Response model for boolean feature extraction."""
    value: bool


class IntFeatureResponse(BaseModel):
    """Response model for int feature extraction."""
    value: int


class FloatFeatureResponse(BaseModel):
    """Response model for float feature extraction."""
    value: float