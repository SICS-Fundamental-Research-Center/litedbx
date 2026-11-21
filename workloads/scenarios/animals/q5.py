# pylint: disable=missing-function-docstring,invalid-name
"""animals Q5 (paper query, animals scenario): distinct cities where
elephants appear (either modality) -- reuses the elephant_any branches.

Official NL query (files/animals/query/natural_language/Q5.txt):
    What is the list of cities for which we have either images or
    audio recordings of elephants?
"""
from data_structure import SemUCQ
from workloads.core.answer_eval import eval_filter
from workloads.core.ucq_exec import distinct_values, union_frames
from workloads.scenarios.animals._common import (
    ELEPHANT_ANY_QUERIES,
    GT_DIR,
)

Q5 = SemUCQ(
    queries=dict(ELEPHANT_ANY_QUERIES),
    gt_path=GT_DIR / "Q5.csv",
    assembler=lambda t: distinct_values(
        union_frames(list(t.values()), ["media_id"]), ["City"]),
    evaluator=lambda a, p: eval_filter(a, p, "City"),
)
