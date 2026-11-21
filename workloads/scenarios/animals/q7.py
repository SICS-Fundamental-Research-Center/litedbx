# pylint: disable=missing-function-docstring,invalid-name
"""animals Q7 (paper query, animals scenario): cities where zebras
and impalas co-occur -- intersection of the zebra and impala image
rowsets' city sets. Impala is a new image predicate (11-18); its census
GT (impala_img.csv, 136 rows) closes exactly onto official Q7.

Official NL query (files/animals/query/natural_language/Q7.txt):
    What are the cities for which zebras and impala co-occur, i.e.,
    we have at least one image of a zebra and at least one image of
    an impala?
"""
from data_structure import SemUCQ
from workloads.core.answer_eval import eval_filter
from workloads.core.ucq_exec import distinct_values, intersect_frames
from workloads.scenarios.animals._common import (
    GT_DIR,
    IMPALA_IMG_SEMCQ,
    ZEBRA_IMG_SEMCQ,
)

Q7 = SemUCQ(
    queries={"zebra_img": ZEBRA_IMG_SEMCQ, "impala_img": IMPALA_IMG_SEMCQ},
    gt_path=GT_DIR / "Q7.csv",
    assembler=lambda t: intersect_frames(
        [distinct_values(t["zebra_img"], ["City"]),
         distinct_values(t["impala_img"], ["City"])],
        ["City"]),
    evaluator=lambda a, p: eval_filter(a, p, "City"),
)
