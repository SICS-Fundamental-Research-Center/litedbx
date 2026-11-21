# pylint: disable=missing-function-docstring,invalid-name
"""animals Q3 (paper query, animals scenario): the city with the most
zebra images -- argmax over the zebra rowset via the standardized
``group_extreme`` (lexicographically first among tied cities) scored by
the SemBench binary caliber (one-row answer counts fully iff it is one
of the GT's tied cities).

Official NL query (files/animals/query/natural_language/Q3.txt):
    What is the city where we captured most pictures of zebras (ties
    are broken arbitrarily, consider only cities with at least one
    zebra picture)?
"""
from data_structure import SemUCQ
from workloads.core.answer_eval import eval_argmax_choice
from workloads.core.ucq_exec import group_extreme
from workloads.scenarios.animals._common import (
    GT_DIR,
    ZEBRA_IMG_SEMCQ,
)

Q3 = SemUCQ(
    queries={"zebra_img": ZEBRA_IMG_SEMCQ},
    gt_path=GT_DIR / "Q3.csv",
    assembler=lambda t: group_extreme(t["zebra_img"], ["City"]),
    evaluator=lambda a, p: eval_argmax_choice(a, p, ["City"]),
)
