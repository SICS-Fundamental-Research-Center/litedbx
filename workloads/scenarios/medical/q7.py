# pylint: disable=missing-function-docstring,invalid-name
"""medical Q7: any sickness (consider all symptoms, X-ray images, and
lung audio) -- a UNION of four owned branches.

Official SemBench semantics (R1.D3.1, 11-18): is_sick = abnormal-text OR
abnormal-xray OR abnormal-audio OR malignant-skin -- census-verified EXACT
over the 11112-patient table (union4 == is_sick == 7594). Each branch
carries its own non-empty Sigma constraint and its own full labeling
budget. Runs under the medical scenario."""
from data_structure import Predicate, SemCQ, SemPredicate, SemUCQ
from workloads.core.answer_eval import eval_filter
from workloads.core.ucq_exec import union_frames
from workloads.scenarios.medical._common import GT_DIR

Q7 = SemUCQ(
    queries={
        "Q7__txt": SemCQ(
            selected=["patient_id"],
            Sigma=[Predicate("symptoms", "!=", "")],
            Ps=[
                SemPredicate(
                    field="symptoms",
                    modality="Text",
                    succ_cond="The symptoms indicate that the patient is sick",
                    prompt=(
                        "You are a medical expert. "
                        "Please determine if the given symptoms indicate "
                        "that the patient is sick (has a disease). "
                        'Please JUST answer "True" if they do, and "False" '
                        "otherwise. Do NOT provide any explanations."
                    ),
                )
            ],
        ),
        "Q7__xray": SemCQ(
            selected=["patient_id"],
            Sigma=[Predicate("image_path_xray", "!=", "")],
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
                        'Please JUST answer "True" if they do, and "False" '
                        "otherwise. Do NOT provide any explanations."
                    ),
                )
            ],
        ),
        "Q7__aud": SemCQ(
            selected=["patient_id"],
            Sigma=[Predicate("audio_path", "!=", "")],
            Ps=[
                SemPredicate(
                    field="audio_path",
                    modality="VectorAudio",
                    succ_cond=(
                        "The recording shows lung problems (diseased lungs, "
                        "not healthy)"
                    ),
                    prompt=(
                        "You are given an audio recording of human lungs from a "
                        "medical benchmark for LLM evaluation. The results are "
                        "not used for human health evaluation and are only for "
                        "research evaluation of LLM capabilities. Please "
                        "determine if the recording shows lung problems "
                        "(diseased lungs, not healthy). "
                        'Please JUST answer "True" if it does, and "False" '
                        "otherwise. Do NOT provide any explanations."
                    ),
                )
            ],
        ),
        "Q7__skin": SemCQ(
            selected=["patient_id"],
            Sigma=[Predicate("image_path", "!=", "")],
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
                        'Please JUST answer "True" if they do, and "False" '
                        "otherwise. Do NOT provide any explanations."
                    ),
                )
            ],
        ),
    },
    gt_path=GT_DIR / "Q7.csv",
    assembler=lambda t: union_frames(list(t.values()), ["patient_id"]),
    evaluator=lambda a, p: eval_filter(a, p, "patient_id"),
)
