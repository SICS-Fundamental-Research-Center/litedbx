# pylint: disable=missing-function-docstring,invalid-name
"""cars Q3: manual-transmission cars whose image shows no damage.

Official NL query (files/cars/query/natural_language/Q3.txt):
    "Find ten cars with manual transmission that are not damaged
    according to images. Return the VIN."
"""
from data_structure import Predicate, SemCQ, SemPredicate
from workloads.scenarios.cars._common import GT_DIR, self_ucq

Q3_SEMCQ = SemCQ(
    selected=["vin"],
    Sigma=[
        Predicate("transmission", "==", "Manual"),
        Predicate("image_path", "!=", ""),
    ],
    Ps=[
        SemPredicate(
            field="image_path",
            modality="VectorImage",
            succ_cond="The car is not damaged according to the image",
            prompt=(
                "You are given an image of a vehicle or its parts. "
                "Please determine if the car is not damaged. "
                'Please JUST answer "True" if it is not damaged, and "False" '
                "otherwise. Do NOT provide any explanations."
            ),
        )
    ],
)

Q3 = self_ucq("Q3", Q3_SEMCQ, GT_DIR / "Q3.csv")
