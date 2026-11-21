# pylint: disable=missing-function-docstring,invalid-name
"""animals Q6 (paper query, animals scenario): cities with monkey
images but no monkey audio -- city-set difference over the two monkey
rowsets.

Official NL query (files/animals/query/natural_language/Q6.txt):
    For which cities do we have images of monkeys but no audio
    recordings of monkeys?
"""
from data_structure import SemUCQ
from workloads.core.answer_eval import eval_filter
from workloads.core.ucq_exec import difference_frames, distinct_values
from workloads.scenarios.animals._common import (
    GT_DIR,
    MONKEY_AUD_SEMCQ,
    MONKEY_IMG_SEMCQ,
)

Q6 = SemUCQ(
    queries={"monkey_img": MONKEY_IMG_SEMCQ, "monkey_aud": MONKEY_AUD_SEMCQ},
    gt_path=GT_DIR / "Q6.csv",
    assembler=lambda t: difference_frames(
        distinct_values(t["monkey_img"], ["City"]),
        distinct_values(t["monkey_aud"], ["City"]),
        ["City"]),
    evaluator=lambda a, p: eval_filter(a, p, "City"),
)
