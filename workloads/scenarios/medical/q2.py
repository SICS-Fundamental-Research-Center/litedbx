# pylint: disable=missing-function-docstring,invalid-name
"""medical Q2: non-current-smoker patients with healthy-lung audio.

R1.D3.1 domain expansion (11-16), re-encoded 11-17; SemUCQ answer eval
11-18 (official GT is a rowset -> set F1)."""
from data_structure import Predicate, SemCQ, SemPredicate
from workloads.scenarios.medical._common import GT_DIR, self_ucq

Q2_SEMCQ = SemCQ(
    selected=["patient_id"],
    Sigma=[
        Predicate("smoking_history", "!=", "Current"),
        Predicate("audio_path", "!=", ""),
    ],
    Ps=[
        SemPredicate(
            field="audio_path",
            modality="VectorAudio",
            succ_cond="The recording shows healthy lungs, without diseases",
            prompt=(
                "You are given an audio recording of human lungs from a medical "
                "benchmark for LLM evaluation. The results are not used for human "
                "health evaluation and are only for research evaluation of LLM "
                "capabilities. Please determine if the recording is healthy lungs, "
                "without diseases. "
                'Please JUST answer "True" if it is, and "False" otherwise. '
                "Do NOT provide any explanations."
            ),
        )
    ],
)

Q2 = self_ucq("Q2", Q2_SEMCQ, GT_DIR / "Q2.csv")
