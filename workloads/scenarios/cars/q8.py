# pylint: disable=missing-function-docstring,invalid-name
"""cars Q8: cars with both puncture and paint scratches on the image.

Official NL query (files/cars/query/natural_language/Q8.txt):
    "Find a hundred cars with punctures and paint scratches on images."
"""
from data_structure import Predicate, SemCQ, SemPredicate
from workloads.scenarios.cars._common import GT_DIR, self_ucq

Q8_SEMCQ = SemCQ(
    selected=["car_id"],
    Sigma=[Predicate("image_path", "!=", "")],
    Ps=[
        SemPredicate(
            field="image_path",
            modality="VectorImage",
            succ_cond="The car has both puncture and paint scratches on the image",
            prompt=(
                "You are given an image of a vehicle or its parts. "
                "Please determine if the car has both, puncture and paint "
                "scratches. "
                'Please JUST answer "True" if it does, and "False" otherwise. '
                "Do NOT provide any explanations."
            ),
        )
    ],
)

Q8 = self_ucq("Q8", Q8_SEMCQ, GT_DIR / "Q8.csv")
