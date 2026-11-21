# pylint: disable=missing-function-docstring,invalid-name
"""animals Q4 (paper query, animals scenario): the city with the most
elephant recordings -- argmax over the elephant-audio rowset via the
standardized ``group_extreme``, scored by the SemBench binary caliber
(same convention as Q3).

Official NL query (files/animals/query/natural_language/Q4.txt):
    What is the city for which we have most recordings of elephants
    (ties are broken arbitrarily, consider only cities with at least
    one elephant recording)?
"""
from data_structure import SemUCQ
from workloads.core.answer_eval import eval_argmax_choice
from workloads.core.ucq_exec import group_extreme
from workloads.scenarios.animals._common import (
    ELEPHANT_AUD_SEMCQ,
    GT_DIR,
)

Q4 = SemUCQ(
    queries={"elephant_aud": ELEPHANT_AUD_SEMCQ},
    gt_path=GT_DIR / "Q4.csv",
    assembler=lambda t: group_extreme(t["elephant_aud"], ["City"]),
    evaluator=lambda a, p: eval_argmax_choice(a, p, ["City"]),
)
