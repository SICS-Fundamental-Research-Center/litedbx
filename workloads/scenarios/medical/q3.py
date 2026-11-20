# pylint: disable=missing-function-docstring,invalid-name
"""medical Q3: family-cancer-history patients with sick-lung X-rays."""
from data_structure import Predicate, SemCQ, SemPredicate
from workloads.scenarios.medical._common import GT_DIR, self_ucq

Q3_SEMCQ = SemCQ(
    selected=["patient_id"],
    Sigma=[
        Predicate("did_family_have_cancer", "==", 1),
        Predicate("image_path_xray", "!=", ""),
    ],
    Ps=[
        SemPredicate(
            field="image_path_xray",
            modality="Image",
            succ_cond=(
                "This X-ray image of human lungs shows that there are "
                "lung problems (considered sick/disease) according to "
                "the X-ray image"
            ),
            prompt=(
                "You are a radiology expert. "
                "Please determine if the given X-ray image indicate "
                "that there are lung problems (considered sick/disease) "
                "according to the X-ray image. "
                'Please JUST answer "True" if they do, and "False" otherwise. '
                "Do NOT provide any explanations."
            ),
        )
    ],
)

Q3 = self_ucq("Q3", Q3_SEMCQ, GT_DIR / "Q3.csv")
