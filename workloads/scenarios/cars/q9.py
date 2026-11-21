# pylint: disable=missing-function-docstring,invalid-name
"""cars Q9: torn-on-image AND bad-ignition-on-audio cars.

Official NL query (files/cars/query/natural_language/Q9.txt):
    "Find cars that are torn according to images and have bad ignition
    according to audio."

NOTE: SemBench issues ONE joint call (image+audio in a single prompt); our
framework has one modality per predicate, so Q9 is split into two
single-modality predicates AND-composed (deviation, disclosed in
_spec.md)."""
from data_structure import Predicate, SemCQ, SemPredicate
from workloads.scenarios.cars._common import GT_DIR, self_ucq

Q9_SEMCQ = SemCQ(
    selected=["car_id"],
    Sigma=[
        Predicate("image_path", "!=", ""),
        Predicate("audio_path", "!=", ""),
    ],
    Ps=[
        SemPredicate(
            field="image_path",
            modality="VectorImage",
            succ_cond="The car is torn according to the image",
            prompt=(
                "You are given an image of a vehicle or its parts. "
                "Please determine if the car is torn. "
                'Please JUST answer "True" if it is torn, and "False" otherwise. '
                "Do NOT provide any explanations."
            ),
        ),
        SemPredicate(
            field="audio_path",
            modality="VectorAudio",
            succ_cond="The car has bad ignition according to the audio",
            prompt=(
                "You are given an audio recording of car diagnostics. "
                "Please determine if the car from the recording has bad "
                "ignition. "
                'Please JUST answer "True" if it does, and "False" otherwise. '
                "Do NOT provide any explanations."
            ),
        ),
    ],
)

Q9 = self_ucq("Q9", Q9_SEMCQ, GT_DIR / "Q9.csv")
