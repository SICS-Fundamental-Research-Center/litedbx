# pylint: disable=missing-function-docstring,invalid-name
"""cars Q4 (aggregate): average car age over engine-problem complaints.

Official NL query (files/cars/query/natural_language/Q4.txt):
    "What is the average age of cars with engine problems?"

R1.D3.1 domain expansion (11-16): references the native ``year`` column
(sf_19672 universe).
"""
import pandas as pd

from data_structure import Predicate, SemCQ, SemPredicate, SemUCQ
from workloads.core.answer_eval import eval_rel_err
from workloads.scenarios.cars._common import GT_DIR

Q4_SEMCQ = SemCQ(
    selected=["car_id"],
    Sigma=[Predicate("summary", "!=", "")],
    Ps=[
        SemPredicate(
            field="summary",
            modality="Text",
            succ_cond="The complaint indicates the car has engine problems",
            prompt=(
                "You are given a textual complaint about a car. "
                "Please determine if the complaint indicates that the car "
                "has engine problems. "
                'Please JUST answer "True" if it does, and "False" otherwise. '
                "Do NOT provide any explanations."
            ),
        )
    ],
)

# Branch key Q4__tuple (not "Q4"): the engine reads
# ground_truth/{branch}.csv for labels, and the official scalar answer
# lives in Q4.csv -- a self-named branch would collide with it
# (rowset built by data/cars/build_gt.py).
Q4 = SemUCQ(
    queries={"Q4__tuple": Q4_SEMCQ},
    gt_path=GT_DIR / "Q4.csv",
    assembler=lambda t: pd.DataFrame(
        {"average_age": [(2026 - t["Q4__tuple"]["year"].astype(float)).mean()]}
    ),
    evaluator=eval_rel_err,
)
