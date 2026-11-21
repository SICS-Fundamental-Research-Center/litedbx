# pylint: disable=missing-function-docstring,invalid-name
"""mmqa Q3g (movie): movies whose text indicates a biographical comedy.

Official NL query (files/mmqa/query/natural_language/q3g.json "nl_question"):
    "Which movies are biographical comedies?"

The json "ground_truth" array is the official answer
key, set-verified by data/mmqa/build_gt.py.
"""
from data_structure import Predicate, SemCQ, SemPredicate
from workloads.scenarios.mmqa._common import gt_of, self_ucq

Q3G_SEMCQ = SemCQ(
    selected=["title"],
    Sigma=[Predicate("text", "!=", "")],
    Ps=[
        SemPredicate(
            field="text",
            modality="Text",
            succ_cond="The movie is a biographical comedy",
            prompt=(
                "Please determine if the given text indicate "
                "that the movie is a biographical comedy. "
                'Please JUST answer "True" if they do, and "False" otherwise. '
                "Do NOT provide any explanations."
            ),
        )
    ],
)

Q3G = self_ucq("Q3g", Q3G_SEMCQ, gt_of("Q3g"))
