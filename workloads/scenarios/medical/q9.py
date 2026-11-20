# pylint: disable=missing-function-docstring,invalid-name
"""medical Q9: patients with BOTH a malignant skin mole AND a sick-lung
X-ray (AND-composed pair)."""
from data_structure import Predicate, SemCQ, SemPredicate
from workloads.scenarios.medical._common import GT_DIR, self_ucq

Q9_SEMCQ = SemCQ(
    selected=["patient_id"],
    Sigma=[
        Predicate("image_path", "!=", ""),
        Predicate("image_path_xray", "!=", ""),
    ],
    Ps=[
        SemPredicate(
            field="image_path",
            modality="Image",
            succ_cond=(
                "This image shows a malignant human skin mole "
                "(considered abnormal/cancerous/sick) according to "
                "the image"
            ),
            prompt=(
                "You are a dermatology expert. "
                "Please determine if the given skin image indicate "
                "that it shows a malignant human skin mole (considered "
                "abnormal/cancerous/sick) according to the image. "
                'Please JUST answer "True" if they do, and "False" otherwise. '
                "Do NOT provide any explanations."
            ),
        ),
        SemPredicate(
            field="image_path_xray",
            modality="Image",
            succ_cond=(
                "This X-ray image shows a sick human lung according "
                "to the X-ray image"
            ),
            prompt=(
                "You are a radiology expert. "
                "Please determine if the given X-ray image indicate "
                "that there are lung problems (considered sick/disease) "
                "according to the X-ray image. "
                'Please JUST answer "True" if they do, and "False" otherwise. '
                "Do NOT provide any explanations."
            ),
        ),
    ],
)

Q9 = self_ucq("Q9", Q9_SEMCQ, GT_DIR / "Q9.csv")
