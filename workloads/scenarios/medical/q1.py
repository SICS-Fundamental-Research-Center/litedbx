# pylint: disable=missing-function-docstring,invalid-name
"""medical Q1: patients whose symptoms indicate an allergy."""
from data_structure import Predicate, SemCQ, SemPredicate
from workloads.scenarios.medical._common import GT_DIR, self_ucq

Q1_SEMCQ = SemCQ(
    selected=["patient_id"],
    Sigma=[Predicate("symptoms", "!=", "")],
    Ps=[
        SemPredicate(
            field="symptoms",
            modality="Text",
            succ_cond="The patient has an allergy",
            prompt=(
                "You are a medical expert. "
                "Please determine if the given symptom (eg, angioedema) indicate "
                "that the patient has an allergy. "
                'Please JUST answer "True" if they do, and "False" otherwise. '
                "Do NOT provide any explanations."
            ),
        )
    ],
)

Q1 = self_ucq("Q1", Q1_SEMCQ, GT_DIR / "Q1.csv")
