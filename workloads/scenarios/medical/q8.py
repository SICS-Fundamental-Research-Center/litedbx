# pylint: disable=missing-function-docstring,invalid-name
"""medical Q8: family-cancer-history patients with malignant skin moles."""
from data_structure import Predicate, SemCQ, SemPredicate
from workloads.scenarios.medical._common import GT_DIR, self_ucq

Q8_SEMCQ = SemCQ(
    selected=["patient_id"],
    Sigma=[
        Predicate("did_family_have_cancer", "==", 1),
        Predicate("image_path", "!=", ""),
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
        )
    ],
)

Q8 = self_ucq("Q8", Q8_SEMCQ, GT_DIR / "Q8.csv")
