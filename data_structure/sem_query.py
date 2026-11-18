import logging
from pathlib import Path
from typing import Callable, Literal

import pandas as pd

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
    def __init__(self, field: str, modality: Literal["Text", "Image", "Audio", "VectorText", "VectorImage", "VectorAudio"], succ_cond: str, prompt: str):
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


class SemUCQ:
    """Composed semantic query: declarative assembly + evaluation.

    ``queries`` maps ENGINE query names to branch SemCQs -- the dict keys
    are the names the engine runs the branches under (owning UCQs name
    their branches ``{ucq}__{branch}``; composition UCQs reuse the engine
    names of already-defined branches so shared SemCQs are annotated
    once). Every branch executes as an independent filter query with
    tuple-level supervision, exactly like a standalone SemCQ.

    ``assembler`` composes the branch result tables (full-row,
    universe-joined DataFrames keyed by engine name) into the answer
    table -- aggregate answers deliver a one-row frame. ``evaluator``
    scores the answer table against the ground-truth file at ``gt_path``;
    the eval type follows the GT format (rowset -> set F1, scalar ->
    relative error, key/count table -> per-key relative error).
    """

    def __init__(
        self,
        queries: dict[str, SemCQ],
        gt_path: Path | str,
        assembler: Callable[[dict[str, pd.DataFrame]], pd.DataFrame],
        evaluator: Callable[[pd.DataFrame, Path], dict],
    ):
        if not queries:
            raise ValueError("SemUCQ requires at least one branch query.")
        selected = {tuple(q.selected) for q in queries.values()}
        if len(selected) != 1:
            raise ValueError(
                "SemUCQ branches must share the same selected columns; got "
                f"{[list(s) for s in selected]}"
            )
        if len(set(queries)) != len(queries):
            raise ValueError(f"Duplicate branch names: {list(queries)}")
        self.queries = queries
        self.gt_path = Path(gt_path)
        self.assembler = assembler
        self.evaluator = evaluator

    def __repr__(self) -> str:
        return f"SemUCQ(branches={list(self.queries)})"

