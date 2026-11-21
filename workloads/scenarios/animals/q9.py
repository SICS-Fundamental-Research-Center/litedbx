# pylint: disable=missing-function-docstring,invalid-name
"""animals Q9 (paper query, animals scenario): cities with monkey
images AND monkey audio -- city-set intersection over the two monkey
rowsets.

Official NL query (files/animals/query/natural_language/Q9.txt):
    What is the list of cities for which we have both images and
    audio recordings of monkeys?
"""
from data_structure import SemUCQ
from workloads.core.answer_eval import eval_filter
from workloads.core.ucq_exec import distinct_values, intersect_frames
from workloads.scenarios.animals._common import (
    GT_DIR,
    MONKEY_AUD_SEMCQ,
    MONKEY_IMG_SEMCQ,
)

Q9 = SemUCQ(
    queries={"monkey_img": MONKEY_IMG_SEMCQ, "monkey_aud": MONKEY_AUD_SEMCQ},
    gt_path=GT_DIR / "Q9.csv",
    assembler=lambda t: intersect_frames(
        [distinct_values(t["monkey_img"], ["City"]),
         distinct_values(t["monkey_aud"], ["City"])],
        ["City"]),
    evaluator=lambda a, p: eval_filter(a, p, "City"),
)
