# pylint: disable=missing-function-docstring,invalid-name
"""animals Q8 (paper query, animals scenario): cities with both
elephant and monkey activity (either modality) -- intersection of the two
any-unions.

Official NL query (files/animals/query/natural_language/Q8.txt):
    What is the list of cities for which we found at least one image
    or audio recording of elephants and at least one image or audio
    recording of monkeys?
"""
from data_structure import SemUCQ
from workloads.core.answer_eval import eval_filter
from workloads.core.ucq_exec import (
    distinct_values,
    intersect_frames,
    union_frames,
)
from workloads.scenarios.animals._common import (
    ELEPHANT_ANY_QUERIES,
    GT_DIR,
    MONKEY_ANY_QUERIES,
)

Q8 = SemUCQ(
    queries={**ELEPHANT_ANY_QUERIES, **MONKEY_ANY_QUERIES},
    gt_path=GT_DIR / "Q8.csv",
    assembler=lambda t: intersect_frames(
        [distinct_values(union_frames(
            [t[k] for k in ELEPHANT_ANY_QUERIES], ["media_id"]), ["City"]),
         distinct_values(union_frames(
            [t[k] for k in MONKEY_ANY_QUERIES], ["media_id"]), ["City"])],
        ["City"]),
    evaluator=lambda a, p: eval_filter(a, p, "City"),
)
