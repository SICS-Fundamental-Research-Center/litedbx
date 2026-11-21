# pylint: disable=missing-function-docstring,invalid-name
"""movie Q4 (aggregate): positivity ratio over the taken_3 sample.

Official NL query (files/movie/query/natural_language/Q4.txt):
    "Positivity ratio of reviews for movie \\"taken_3\\". Return the
    ratio as positivity_ratio."

R1.D3.1 domain expansion (11-16, SemUCQ eval 11-18): Q2's row count over
MOVIE_N (the taken_3 sample size).
"""
import pandas as pd

from data_structure import SemUCQ
from workloads.core.answer_eval import eval_rel_err
from workloads.scenarios.movie._common import MOVIE_N, OFFICIAL_GT
from workloads.scenarios.movie.q2 import Q2_SEMCQ

Q4 = SemUCQ(
    queries={"Q4": Q2_SEMCQ},
    gt_path=OFFICIAL_GT / "Q4.csv",
    assembler=lambda t: pd.DataFrame(
        {"positivity_ratio": [len(t["Q4"]) / MOVIE_N]}
    ),
    evaluator=eval_rel_err,
)
