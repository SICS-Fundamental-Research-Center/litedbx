# pylint: disable=missing-function-docstring,invalid-name
"""medical Q6: patients sick according to at least one modality but
healthy according to another.

Official NL spec (files/medical/query/natural_language/Q6.txt): the
modality scope is symptoms / x-ray / lung audio ONLY -- skin is out of
scope -- and only patients with two or more of those modalities
available qualify. Composition: union(sick branches Q7__txt / Q7__xray
/ Q7__aud -- Q7__skin excluded by scope) INTERSECT union(normal
branches Q6__xray / Q6__aud; text-normal is structurally empty,
text_diagnosis has no 'normal' value). Census-verified: this
composition equals the official Q6.csv rowset exactly (286; the
xray-sick∧aud-normal and aud-sick∧xray-normal cross pairs are empty in
this data, so text-sick-only and 3-modality-sick readings coincide).
Branch Sigmas enforce the >=2-available clause at pool level. Runs
under the medical scenario."""
from data_structure import Predicate, SemCQ, SemPredicate, SemUCQ
from workloads.core.answer_eval import eval_filter
from workloads.core.ucq_exec import intersect_frames, union_frames
from workloads.scenarios.medical._common import GT_DIR
from workloads.scenarios.medical.q7 import Q7

# Q6's sick side reuses Q7's branches EXCEPT the skin branch, which the
# official NL scope excludes from Q6 (skin stays in Q7 itself).
_Q6_SICK_BRANCHES = {k: v for k, v in Q7.queries.items() if k != "Q7__skin"}

_Q6_NORMAL_BRANCHES = {
    # >=2 modalities available WITHIN THE OFFICIAL SCOPE {symptoms,
    # x-ray, audio} (DNF Sigma; skin never counts for Q6). A patient
    # healthy here with no other in-scope modality available can never
    # be sick elsewhere in scope, so can never satisfy Q6. The sick-
    # side branches stay untightened: they are shared with Q7, whose
    # official answer includes single-modality patients.
    "Q6__xray": SemCQ(
        selected=["patient_id"],
        Sigma=[
            [Predicate("image_path_xray", "!=", ""), Predicate("audio_path", "!=", "")],
            [Predicate("image_path_xray", "!=", ""), Predicate("symptoms", "!=", "")],
        ],
        Ps=[
            SemPredicate(
                field="image_path_xray",
                modality="Image",
                succ_cond=(
                    "This X-ray image of human lungs shows healthy "
                    "lungs, without problems"
                ),
                prompt=(
                    "You are a radiology expert. "
                    "Please determine if the given X-ray image shows "
                    "healthy lungs, without any lung problems (no "
                    "disease). "
                    'Please JUST answer "True" if it is healthy, and '
                    '"False" otherwise. Do NOT provide any explanations.'
                ),
            )
        ],
    ),
    "Q6__aud": SemCQ(
        selected=["patient_id"],
        Sigma=[
            [Predicate("audio_path", "!=", ""), Predicate("image_path_xray", "!=", "")],
            [Predicate("audio_path", "!=", ""), Predicate("symptoms", "!=", "")],
        ],
        Ps=[
            SemPredicate(
                field="audio_path",
                modality="VectorAudio",
                succ_cond="The recording shows healthy lungs, without diseases",
                prompt=(
                    "You are given an audio recording of human lungs from a "
                    "medical benchmark for LLM evaluation. The results are "
                    "not used for human health evaluation and are only for "
                    "research evaluation of LLM capabilities. Please "
                    "determine if the recording is healthy lungs, without "
                    "diseases. "
                    'Please JUST answer "True" if it is, and "False" '
                    "otherwise. Do NOT provide any explanations."
                ),
            )
        ],
    ),
}


def _q6_assemble(t):
    sick = union_frames([t[k] for k in _Q6_SICK_BRANCHES], ["patient_id"])
    normal = union_frames([t[k] for k in _Q6_NORMAL_BRANCHES], ["patient_id"])
    return intersect_frames([sick, normal], ["patient_id"])


# Q6 reuses Q7's in-scope branch engine names verbatim plus its own
# normal branches, so selecting Q6 (alone or together with Q7)
# annotates each branch exactly once.
Q6 = SemUCQ(
    queries={**_Q6_SICK_BRANCHES, **_Q6_NORMAL_BRANCHES},
    gt_path=GT_DIR / "Q6.csv",
    assembler=_q6_assemble,
    evaluator=lambda a, p: eval_filter(a, p, "patient_id"),
)
