# pylint: disable=missing-function-docstring,invalid-name
"""mmqa Q6b (airport): airlines with destinations in Germany.

Official NL query (files/mmqa/query/natural_language/q6b.json "nl_question"):
    "Which airlines have destinations in Germany?"

The json "ground_truth" array is the official answer
key, set-verified by data/mmqa/build_gt.py.
"""
from data_structure import Predicate, SemCQ, SemPredicate
from workloads.scenarios.mmqa._common import gt_of, self_ucq

Q6B_SEMCQ = SemCQ(
    selected=["Airlines"],
    Sigma=[Predicate("Destinations", "!=", "")],
    Ps=[
        SemPredicate(
            field="Destinations",
            modality="Text",
            succ_cond="The airline has destinations in Germany",
            prompt=(
                "Please determine if the given text indicate "
                "that the airline has destinations in Germany. "
                'Please JUST answer "True" if they do, and "False" otherwise. '
                "Do NOT provide any explanations."
            ),
        )
    ],
)

Q6B = self_ucq("Q6b", Q6B_SEMCQ, gt_of("Q6b"))
