# pylint: disable=missing-function-docstring,invalid-name
"""animals Q1 (paper query, animals scenario): how many pictures of
zebras -- COUNT over the zebra rowset (SemBench scalar rel-err eval).

Official NL query (files/animals/query/natural_language/Q1.txt):
    How many pictures of zebras do we have in our database?
"""
import pandas as pd
from data_structure import SemUCQ
from workloads.core.answer_eval import eval_rel_err
from workloads.scenarios.animals._common import (
    ZEBRA_IMG_SEMCQ,
    GT_DIR,
)

Q1 = SemUCQ(
    queries={"zebra_img": ZEBRA_IMG_SEMCQ},
    gt_path=GT_DIR / "Q1.csv",
    assembler=lambda t: pd.DataFrame({"count": [len(t["zebra_img"])]}),
    evaluator=eval_rel_err,
)
