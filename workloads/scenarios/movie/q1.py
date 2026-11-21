# pylint: disable=missing-function-docstring,invalid-name
"""movie Q1: reviews (any title) whose review text is clearly positive.

Official NL query (files/movie/query/natural_language/Q1.txt):
    "Five clearly positive reviews (any movie). Return reviewId."
"""
from data_structure import Predicate, SemCQ, SemPredicate
from workloads.scenarios.movie._common import ENGINE_GT, self_ucq

Q1_SEMCQ = SemCQ(
    selected=["reviewId"],
    Sigma=[Predicate("reviewText", "!=", "")],
    Ps=[
        SemPredicate(
            field="reviewText",
            modality="Text",
            succ_cond="The movie review is clearly positive",
            prompt=(
                "You are a movie reviewer. "
                "Please determine if the given movie review "
                "is clearly positive. "
                'Please JUST answer "True" if they do, and "False" otherwise. '
                "Do NOT provide any explanations."
            ),
        )
    ],
)

Q1 = self_ucq("Q1", Q1_SEMCQ, ENGINE_GT / "Q1.csv")
