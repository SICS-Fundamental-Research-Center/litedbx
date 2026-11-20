# pylint: disable=missing-function-docstring,invalid-name
"""medical Q4 (aggregate): average age over acne patients.

R1.D3.1 domain expansion (11-16), re-encoded 11-17; SemUCQ answer eval
11-18 (official GT is a scalar AVG -> relative error). Runs under the
medical scenario (native ``age`` column)."""
import pandas as pd

from data_structure import Predicate, SemCQ, SemPredicate, SemUCQ
from workloads.core.answer_eval import eval_rel_err
from workloads.scenarios.medical._common import GT_DIR

Q4_SEMCQ = SemCQ(
    selected=["patient_id"],
    Sigma=[Predicate("symptoms", "!=", "")],
    Ps=[
        SemPredicate(
            field="symptoms",
            modality="Text",
            succ_cond="The patient has acne",
            prompt=(
                "You are a medical expert. "
                "Please determine if the given symptoms indicate "
                "that the patient has acne. "
                'Please JUST answer "True" if they do, and "False" otherwise. '
                "Do NOT provide any explanations."
            ),
        )
    ],
)

# Branch key Q4__tuple (not "Q4"): the engine reads
# ground_truth/{branch}.csv for labels, and the official scalar answer
# lives in Q4.csv -- a self-named branch would collide with it.
Q4 = SemUCQ(
    queries={"Q4__tuple": Q4_SEMCQ},
    gt_path=GT_DIR / "Q4.csv",
    assembler=lambda t: pd.DataFrame(
        {"average_acne_age": [t["Q4__tuple"]["age"].mean()]}
    ),
    evaluator=eval_rel_err,
)
