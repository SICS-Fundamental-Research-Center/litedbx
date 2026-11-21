# pylint: disable=missing-function-docstring,invalid-name
"""movie Q3 (aggregate): count of clearly-positive taken_3 reviews.

Official NL query (files/movie/query/natural_language/Q3.txt):
    "Count of positive reviews for movie \\"taken_3\\". Return the count
    as positive_review_cnt."

R1.D3.1 domain expansion (11-16, SemUCQ eval 11-18): shares Q2's semantic
predicate verbatim -- the count is deterministic postprocessing over the
same certified row set, carried as Q2's SemCQ under its own engine name.
Engine rowset GT (data/movie/ground_truth/Q3.csv) == official Q2 rowset,
anchor-checked by data/movie/build_gt.py.
"""
import pandas as pd

from data_structure import SemUCQ
from workloads.core.answer_eval import eval_rel_err
from workloads.scenarios.movie._common import OFFICIAL_GT
from workloads.scenarios.movie.q2 import Q2_SEMCQ

Q3 = SemUCQ(
    queries={"Q3": Q2_SEMCQ},
    gt_path=OFFICIAL_GT / "Q3.csv",
    assembler=lambda t: pd.DataFrame(
        {"positive_review_cnt": [len(t["Q3"])]}
    ),
    evaluator=eval_rel_err,
)
