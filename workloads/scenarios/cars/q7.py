# pylint: disable=missing-function-docstring,invalid-name
"""cars Q7: at least one of dented (image) / worn-out brakes (audio) /
electrical-system problems (text) -- a UNION of three owned branches.

Official NL query (files/cars/query/natural_language/Q7.txt):
    "Find cars that are either dented (image), have worn out brakes
    (audio), or have electrical system problems (text), i.e., damaged at
    least according to a single modality."

Official SemBench Q7 semantics; the 11-16 encoding (Sigma=[] + three
AND-composed Ps) was logically vacuous, so each branch here carries its
own non-empty Sigma constraint and its own full labeling budget (11-18).
Official GT formula per branch: image damage_status contains "dented" /
audio detailed_problem == "worn_out_brakes" / text component_class ==
"ELECTRICAL SYSTEM". Branch rowsets are built by data/cars/build_gt.py.
"""
from data_structure import Predicate, SemCQ, SemPredicate, SemUCQ
from workloads.core.answer_eval import eval_filter
from workloads.core.ucq_exec import union_frames
from workloads.scenarios.cars._common import GT_DIR

Q7 = SemUCQ(
    queries={
        "Q7__img": SemCQ(
            selected=["car_id"],
            Sigma=[Predicate("image_path", "!=", "")],
            Ps=[
                SemPredicate(
                    field="image_path",
                    modality="VectorImage",
                    succ_cond="The car is dented according to the image",
                    prompt=(
                        "You are given an image of a vehicle or its parts. "
                        "Please determine if the car has dents (is dented) "
                        "according to the image. "
                        'Please JUST answer "True" if it does, and "False" '
                        "otherwise. Do NOT provide any explanations."
                    ),
                )
            ],
        ),
        "Q7__aud": SemCQ(
            selected=["car_id"],
            Sigma=[Predicate("audio_path", "!=", "")],
            Ps=[
                SemPredicate(
                    field="audio_path",
                    modality="VectorAudio",
                    succ_cond="The car has worn out brakes according to the audio",
                    prompt=(
                        "You are given an audio recording of car diagnostics. "
                        "Please determine if the car has worn out brakes "
                        "according to the recording. "
                        'Please JUST answer "True" if it does, and "False" '
                        "otherwise. Do NOT provide any explanations."
                    ),
                )
            ],
        ),
        "Q7__txt": SemCQ(
            selected=["car_id"],
            Sigma=[Predicate("summary", "!=", "")],
            Ps=[
                SemPredicate(
                    field="summary",
                    modality="Text",
                    succ_cond="The complaint indicates electrical system problems",
                    prompt=(
                        "You are given a textual complaint about a car. "
                        "Please determine if the complaint indicates that "
                        "the car has electrical system problems. "
                        'Please JUST answer "True" if it does, and "False" '
                        "otherwise. Do NOT provide any explanations."
                    ),
                )
            ],
        ),
    },
    gt_path=GT_DIR / "Q7.csv",
    assembler=lambda t: union_frames(list(t.values()), ["car_id"]),
    evaluator=lambda a, p: eval_filter(a, p, "car_id"),
)
