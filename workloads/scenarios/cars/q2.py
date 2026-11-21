# pylint: disable=missing-function-docstring,invalid-name
"""cars Q2: electric cars whose diagnostic audio shows a dead battery.

Official NL query (files/cars/query/natural_language/Q2.txt):
    "Find electric cars with available audio recordings that show a dead
    battery."
"""
from data_structure import Predicate, SemCQ, SemPredicate
from workloads.scenarios.cars._common import GT_DIR, self_ucq

Q2_SEMCQ = SemCQ(
    selected=["car_id"],
    Sigma=[
        Predicate("fuel_type", "==", "Electric"),
        Predicate("audio_path", "!=", ""),
    ],
    Ps=[
        SemPredicate(
            field="audio_path",
            modality="VectorAudio",
            succ_cond="The car from the audio recording has a dead battery",
            prompt=(
                "You are given an audio recording of car diagnostics. "
                "Please determine if the car from the recording has a dead "
                "battery. "
                'Please JUST answer "True" if it does, and "False" otherwise. '
                "Do NOT provide any explanations."
            ),
        )
    ],
)

Q2 = self_ucq("Q2", Q2_SEMCQ, GT_DIR / "Q2.csv")
