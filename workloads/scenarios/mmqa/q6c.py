# pylint: disable=missing-function-docstring,invalid-name
"""mmqa Q6c (airport): airlines with destinations in Europe.

Official NL query (files/mmqa/query/natural_language/q6c.json "nl_question"):
    "Which airlines have destinations in Europe?"

The json "ground_truth" array is the official answer
key, set-verified by data/mmqa/build_gt.py.
"""
from data_structure import Predicate, SemCQ, SemPredicate
from workloads.scenarios.mmqa._common import gt_of, self_ucq

Q6C_SEMCQ = SemCQ(
    selected=["Airlines"],
    Sigma=[Predicate("Destinations", "!=", "")],
    Ps=[
        SemPredicate(
            field="Destinations",
            modality="VectorText",
            succ_cond="The airline has destinations in Europe",
            prompt=(
                "Please determine if the given text indicate "
                "that the airline has destinations in Europe. "
                'Please JUST answer "True" if they do, and "False" otherwise. '
                "Do NOT provide any explanations."
            ),
        )
    ],
)

Q6C = self_ucq("Q6c", Q6C_SEMCQ, gt_of("Q6c"))
