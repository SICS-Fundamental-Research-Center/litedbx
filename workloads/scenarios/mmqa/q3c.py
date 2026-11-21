# pylint: disable=missing-function-docstring,invalid-name
"""mmqa Q3c (movie): movies whose text indicates romance.

Official NL query (files/mmqa/query/natural_language/q3c.json "nl_question"):
    "Which movies are romances?"

The json "ground_truth" array is the official answer
key, set-verified by data/mmqa/build_gt.py.
"""
from data_structure import Predicate, SemCQ, SemPredicate
from workloads.scenarios.mmqa._common import gt_of, self_ucq

Q3C_SEMCQ = SemCQ(
    selected=["title"],
    Sigma=[Predicate("text", "!=", "")],
    Ps=[
        SemPredicate(
            field="text",
            modality="Text",
            succ_cond="The movie is romance",
            prompt=(
                "Please determine if the given text indicate "
                "that the movie is romance. "
                'Please JUST answer "True" if they do, and "False" otherwise. '
                "Do NOT provide any explanations."
            ),
        )
    ],
)

Q3C = self_ucq("Q3c", Q3C_SEMCQ, gt_of("Q3c"))
