# pylint: disable=missing-function-docstring,invalid-name
"""cars Q6: cars damaged according to one modality but not another --
cars where the modalities DISAGREE (at least one judges damaged while
at least one other judges normal).

Official NL query (files/cars/query/natural_language/Q6.txt):
    "Find cars that are damaged according to one modality but not the
    other. For this query, for complaints, check if the car was on
    fire."

The second sentence is NOT an extra AND condition -- it DEFINES the
text modality's "damaged" verdict: for complaints, damaged == the car
was on fire (so text-normal == not on fire). Census refutation of the
AND reading: only 13 of the 1906 official-GT cars have fire==True (237
fire cars exist in total), so official GT cannot be a subset of fire
cars. The official census formula and this encoding agree set-exactly:

    union(image-damaged, audio-damaged, on-fire)
    INTERSECT union(image-ok, audio-ok, not-on-fire)
    restricted to cars with >=2 modalities available   == Q6.csv (1906)

Encoding mirrors the official gold's shape (Q6.sql): THREE AI.IF
judgments only, with the normal side DERIVED as available-and-not-sick
-- the official ok verdicts are complements, never a second inverted
LLM campaign. Per user order 2026-11-21 this replaces the 11-18
six-branch double-polarity form (whose ok branches re-labeled the same
media with inverted prompts at double budget and with True/True label
inconsistency risk):

    3 face branches   Sigma = own modality available AND 2-of-3
                      available (DNF, Plan-E shape as validated on
                      medical Q6), Ps = [] -- pure structural
                      retrieval, zero LLM cost
    3 dmg branches    img_dmg / aud_dmg / fire (prompts unchanged)
    assembler         ok_i = difference_frames(face_i, dmg_i)
                      answer = (union dmg) INTERSECT (union ok)

Runs under the cars scenario; per-branch rowsets are census-built by
data/cars/build_gt.py (branch composition anchors onto official Q6).
The ok rowsets are no longer materialized GT files -- derived at
assembly, complement-consistent by construction.
"""
from data_structure import Predicate, SemCQ, SemPredicate, SemUCQ
from workloads.core.answer_eval import eval_filter
from workloads.core.ucq_exec import (
    difference_frames,
    intersect_frames,
    union_frames,
)
from workloads.scenarios.cars._common import GT_DIR

_IMG_TWO_OF_THREE = [
    [Predicate("image_path", "!=", ""), Predicate("audio_path", "!=", "")],
    [Predicate("image_path", "!=", ""), Predicate("summary", "!=", "")],
]
_AUD_TWO_OF_THREE = [
    [Predicate("audio_path", "!=", ""), Predicate("image_path", "!=", "")],
    [Predicate("audio_path", "!=", ""), Predicate("summary", "!=", "")],
]
_TXT_TWO_OF_THREE = [
    [Predicate("summary", "!=", ""), Predicate("image_path", "!=", "")],
    [Predicate("summary", "!=", ""), Predicate("audio_path", "!=", "")],
]

_DMG_BRANCHES = {
    "Q6__img_dmg": SemCQ(
        selected=["car_id"],
        Sigma=_IMG_TWO_OF_THREE,
        Ps=[
            SemPredicate(
                field="image_path",
                modality="VectorImage",
                succ_cond="The car is damaged according to the image",
                prompt=(
                    "You are given an image of a vehicle or its parts. "
                    "Please determine if the car is damaged according to the "
                    "image. "
                    'Please JUST answer "True" if it is damaged, and "False" '
                    "otherwise. Do NOT provide any explanations."
                ),
            )
        ],
    ),
    "Q6__aud_dmg": SemCQ(
        selected=["car_id"],
        Sigma=_AUD_TWO_OF_THREE,
        Ps=[
            SemPredicate(
                field="audio_path",
                modality="VectorAudio",
                succ_cond="The car is damaged according to the audio recording",
                prompt=(
                    "You are given an audio recording of car diagnostics. "
                    "Please determine if the car is damaged according to the "
                    "recording. "
                    'Please JUST answer "True" if it is damaged, and "False" '
                    "otherwise. Do NOT provide any explanations."
                ),
            )
        ],
    ),
    "Q6__fire": SemCQ(
        selected=["car_id"],
        Sigma=_TXT_TWO_OF_THREE,
        Ps=[
            SemPredicate(
                field="summary",
                modality="Text",
                succ_cond="The complaint entails that the car was on fire",
                prompt=(
                    "You are given a textual complaint about a car. "
                    "Please determine if the complaint entails that the car was "
                    "on fire. "
                    'Please JUST answer "True" if it does, and "False" otherwise. '
                    "Do NOT provide any explanations."
                ),
            )
        ],
    ),
}

# Structural faces the ok side is derived FROM (face - dmg); Ps=[] --
# a pure Sigma retrieval carries no labeling target, so these branches
# never call the LLM.
_FACE_BRANCHES = {
    "Q6__img_face": SemCQ(
        selected=["car_id"], Sigma=_IMG_TWO_OF_THREE, Ps=[]
    ),
    "Q6__aud_face": SemCQ(
        selected=["car_id"], Sigma=_AUD_TWO_OF_THREE, Ps=[]
    ),
    "Q6__txt_face": SemCQ(
        selected=["car_id"], Sigma=_TXT_TWO_OF_THREE, Ps=[]
    ),
}


def _q6_assemble(t):
    damaged = union_frames(
        [t[k] for k in _DMG_BRANCHES], ["car_id"])
    normal = union_frames(
        [
            difference_frames(t["Q6__img_face"], t["Q6__img_dmg"], ["car_id"]),
            difference_frames(t["Q6__aud_face"], t["Q6__aud_dmg"], ["car_id"]),
            difference_frames(t["Q6__txt_face"], t["Q6__fire"], ["car_id"]),
        ],
        ["car_id"])
    return intersect_frames([damaged, normal], ["car_id"])


Q6 = SemUCQ(
    queries={**_DMG_BRANCHES, **_FACE_BRANCHES},
    gt_path=GT_DIR / "Q6.csv",
    assembler=_q6_assemble,
    evaluator=lambda a, p: eval_filter(a, p, "car_id"),
)
