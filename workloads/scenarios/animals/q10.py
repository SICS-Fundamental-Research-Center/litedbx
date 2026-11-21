# pylint: disable=missing-function-docstring,invalid-name
"""animals Q10 (paper query, animals scenario): the city and station
with the most zebra pictures -- argmax over (City, StationID) pairs of
the zebra rowset via the standardized ``group_extreme`` (lexicographically
first among tied pairs), scored by the SemBench binary pair caliber.

Official NL query (files/animals/query/natural_language/Q10.txt):
    What is the city and station with most associated pictures
    showing zebras (ties are broken arbitrarily, only consider
    cities with at least one zebra picture)?
"""
from data_structure import SemUCQ
from workloads.core.answer_eval import eval_argmax_choice
from workloads.core.ucq_exec import group_extreme
from workloads.scenarios.animals._common import (
    GT_DIR,
    ZEBRA_IMG_SEMCQ,
)

Q10 = SemUCQ(
    queries={"zebra_img": ZEBRA_IMG_SEMCQ},
    gt_path=GT_DIR / "Q10.csv",
    assembler=lambda t: group_extreme(t["zebra_img"], ["City", "StationID"]),
    evaluator=lambda a, p: eval_argmax_choice(
        a, p, ["City", "StationID"]),
)
