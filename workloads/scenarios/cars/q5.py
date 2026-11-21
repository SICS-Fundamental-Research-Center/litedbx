# pylint: disable=missing-function-docstring,invalid-name
"""cars Q5 (aggregate): count of damaged automatic cars (image AND audio).

Official NL query (files/cars/query/natural_language/Q5.txt):
    "How many automatic cars are damaged according to both audio and
    images?"

R1.D3.1 domain expansion (11-16)."""
import pandas as pd

from data_structure import Predicate, SemCQ, SemPredicate, SemUCQ
from workloads.core.answer_eval import eval_group_rel_err
from workloads.scenarios.cars._common import GT_DIR

Q5_SEMCQ = SemCQ(
    selected=["car_id"],
    Sigma=[
        Predicate("transmission", "==", "Automatic"),
        Predicate("image_path", "!=", ""),
        Predicate("audio_path", "!=", ""),
    ],
    Ps=[
        SemPredicate(
            field="image_path",
            modality="VectorImage",
            succ_cond="The car is damaged according to the image",
            prompt=(
                "You are given an image of a vehicle or its parts. "
                "Please determine if the car is damaged according to the "
                "image. "
                'Please JUST answer "True" if it is damaged, and "False" '
                "otherwise. Do NOT provide any explanations."
            ),
        ),
        SemPredicate(
            field="audio_path",
            modality="VectorAudio",
            succ_cond="The car is damaged according to the audio recording",
            prompt=(
                "You are given an audio recording of car diagnostics. "
                "Please determine if the car is damaged according to the "
                "recording. "
                'Please JUST answer "True" if it is damaged, and "False" '
                "otherwise. Do NOT provide any explanations."
            ),
        ),
    ],
)

# Branch key Q5__tuple (not "Q5"): same collision rule as Q4 -- the
# official group-answer GT occupies the bare name (rowset built by
# data/cars/build_gt.py).
Q5 = SemUCQ(
    queries={"Q5__tuple": Q5_SEMCQ},
    gt_path=GT_DIR / "Q5.csv",
    assembler=lambda t: pd.DataFrame(
        {"transmission": ["Automatic"], "count": [len(t["Q5__tuple"])]}
    ),
    evaluator=lambda a, p: eval_group_rel_err(a, p, "transmission", "count"),
)
