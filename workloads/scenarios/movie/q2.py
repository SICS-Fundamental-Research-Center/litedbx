# pylint: disable=missing-function-docstring,invalid-name
"""movie Q2: clearly-positive reviews among the taken_3 sample.

Official NL query (files/movie/query/natural_language/Q2.txt):
    "Five positive reviews for movie \\"taken_3\\". Return reviewId."
"""
from data_structure import Predicate, SemCQ, SemPredicate
from workloads.scenarios.movie._common import ENGINE_GT, self_ucq

Q2_SEMCQ = SemCQ(
    selected=["reviewId"],
    Sigma=[Predicate("id", "==", "taken_3"), Predicate("reviewText", "!=", "")],
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

Q2 = self_ucq("Q2", Q2_SEMCQ, ENGINE_GT / "Q2.csv")
