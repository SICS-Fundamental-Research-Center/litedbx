import logging
from typing import Literal

logger = logging.getLogger(__name__)

class Predicate:
    VALID_OPS = {">", ">=", "<", "<=", "==", "!="}

    def __init__(self, field: str, op: str, value):
        if op not in self.VALID_OPS:
            raise ValueError(f"Invalid operator '{op}'. Must be one of: {self.VALID_OPS}")

        if not isinstance(value, (int, float, str, bool)):
            raise ValueError(
                f"Invalid value type '{type(value).__name__}'. Must be int, float, str, or bool"
            )

        self.field = field
        self.op = op
        self.value = value

    def __repr__(self) -> str:
        return f"Predicate(field='{self.field}', op='{self.op}', value={repr(self.value)})"


class SemPredicate:
    def __init__(self, field: str, modality: Literal["Text", "Image", "VectorText", "VectorImage"], succ_cond: str, prompt: str):
        self.field = field
        self.modality = modality
        self.succ_cond = succ_cond
        self.prompt = prompt

    def __repr__(self) -> str:
        return f"SemPredicate(field='{self.field}', modality='{self.modality}', succ_cond='{self.succ_cond}', prompt={repr(self.prompt)})"


class SemCQ:
    def __init__(self, selected: list[str], Sigma: list[Predicate], Ps: list[SemPredicate]):
        self.selected = selected
        self.Sigma = Sigma
        self.Ps = Ps
        self.Ps_translated = []

