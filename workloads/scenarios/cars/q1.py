# pylint: disable=missing-function-docstring,invalid-name
"""cars Q1: complaints entailing the car was in a crash/accident/collision.

Official NL query (files/cars/query/natural_language/Q1.txt):
    "Find cars that were in a crash/accident/collision."
"""
from data_structure import Predicate, SemCQ, SemPredicate
from workloads.scenarios.cars._common import GT_DIR, self_ucq

Q1_SEMCQ = SemCQ(
    selected=["car_id"],
    Sigma=[Predicate("summary", "!=", "")],
    Ps=[
        SemPredicate(
            field="summary",
            modality="Text",
            succ_cond="The complaint entails the car was in a crash/accident/collision",
            prompt=(
                "You are given a textual complaint entailing that the car was "
                "in a crash/accident/collision. "
                "Please determine if the given complaint indicates that the car "
                "was in a crash/accident/collision. "
                'Please JUST answer "True" if it does, and "False" otherwise. '
                "Do NOT provide any explanations."
            ),
        )
    ],
)

Q1 = self_ucq("Q1", Q1_SEMCQ, GT_DIR / "Q1.csv")
