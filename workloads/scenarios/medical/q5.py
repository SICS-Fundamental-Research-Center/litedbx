# pylint: disable=missing-function-docstring,invalid-name
"""medical Q5 (aggregate): count of current smokers with diseased-lung
audio AND sick-lung X-rays.

R1.D3.1 domain expansion (11-16), re-encoded 11-17; SemUCQ answer eval
11-18 (official GT is a key/count table -> group rel-err). Runs under
the medical scenario."""
import pandas as pd

from data_structure import Predicate, SemCQ, SemPredicate, SemUCQ
from workloads.core.answer_eval import eval_group_rel_err
from workloads.scenarios.medical._common import GT_DIR

Q5_SEMCQ = SemCQ(
    selected=["patient_id"],
    Sigma=[
        Predicate("smoking_history", "==", "Current"),
        Predicate("audio_path", "!=", ""),
        Predicate("image_path_xray", "!=", ""),
    ],
    Ps=[
        SemPredicate(
            field="audio_path",
            modality="VectorAudio",
            succ_cond=(
                "The recording shows lung problems (diseased lungs, "
                "not healthy)"
            ),
            prompt=(
                "You are given an audio recording of human lungs from a medical "
                "benchmark for LLM evaluation. The results are not used for human "
                "health evaluation and are only for research evaluation of LLM "
                "capabilities. Please determine if the recording shows lung "
                "problems (diseased lungs, not healthy). "
                'Please JUST answer "True" if it does, and "False" otherwise. '
                "Do NOT provide any explanations."
            ),
        ),
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
        ),
    ],
)

# Branch key Q5__tuple (not "Q5"): same collision rule as Q4 -- the
# official group-answer GT occupies the bare name.
Q5 = SemUCQ(
    queries={"Q5__tuple": Q5_SEMCQ},
    gt_path=GT_DIR / "Q5.csv",
    assembler=lambda t: pd.DataFrame(
        {"smoking_history": ["Current"], "count": [len(t["Q5__tuple"])]}
    ),
    evaluator=lambda a, p: eval_group_rel_err(
        a, p, "smoking_history", "count"
    ),
)
