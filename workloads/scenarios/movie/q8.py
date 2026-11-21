# pylint: disable=missing-function-docstring,invalid-name
"""movie Q8 (aggregate): NEGATIVE/POSITIVE group counts over taken_3.

Official NL query (files/movie/query/natural_language/Q8.txt):
    "Calculate the number of positive and negative reviews for movie
    \\"taken_3\\". Return scoreSentiment and count."

R1.D3.1 domain expansion (11-16, SemUCQ eval 11-18): the positive count is
Q2's row count; the negative count is MOVIE_N minus it.
"""
import pandas as pd

from data_structure import SemUCQ
from workloads.core.answer_eval import eval_group_rel_err
from workloads.scenarios.movie._common import MOVIE_N, OFFICIAL_GT
from workloads.scenarios.movie.q2 import Q2_SEMCQ

Q8 = SemUCQ(
    queries={"Q8": Q2_SEMCQ},
    gt_path=OFFICIAL_GT / "Q8.csv",
    assembler=lambda t: pd.DataFrame(
        {
            "scoreSentiment": ["NEGATIVE", "POSITIVE"],
            "count": [MOVIE_N - len(t["Q8"]), len(t["Q8"])],
        }
    ),
    evaluator=lambda a, p: eval_group_rel_err(
        a, p, "scoreSentiment", "count"
    ),
)
