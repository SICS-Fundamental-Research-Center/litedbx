"""Shared paths, predicates, and engine-filter declarations for the
animals scenario (wildlife full-media universe: one row per
image/recording, 200 img + 66 aud from sf_200, split ``image_path``/
``audio_path`` columns). Assembler/evaluator operators are standardized
(workloads/core/ucq_exec.py + answer_eval.py, 2026-11-21) -- no
scenario-local eval helpers."""
from pathlib import Path

from data_structure import Predicate, SemCQ, SemPredicate, SemUCQ
from workloads.core.answer_eval import eval_filter
from workloads.core.ucq_exec import single, union_frames

REPO = Path(__file__).resolve().parents[3]
CURRENT_DIR = REPO / "workloads"  # config.yaml lives here
DATASET_PATH = REPO / "data/animals"
GT_DIR = DATASET_PATH / "ground_truth"

_FULL_IMG_SIGMA = [Predicate("image_path", "!=", "")]
_FULL_AUD_SIGMA = [Predicate("audio_path", "!=", "")]


def self_ucq(name: str, cq: SemCQ, gt: Path) -> SemUCQ:
    """Uniform UCQ form of a plain filter query: the branch runs under
    its own engine name and the answer is the retrieved rowset itself."""
    return SemUCQ(
        queries={name: cq},
        gt_path=gt,
        assembler=single(name),
        evaluator=lambda a, p: eval_filter(a, p, cq.selected[0]),
    )


def _full_img_predicate(species: str, plural: str, article: str = "a") -> SemPredicate:
    return SemPredicate(
        field="image_path",
        modality="Image",
        succ_cond=f"The image shows {article} {species} ({plural})",
        prompt=(
            "You are given an image from a wildlife camera trap. "
            f"Please determine if the image shows {article} {species} "
            f"({plural}). "
            'Please JUST answer "True" if it does, and "False" otherwise. '
            "Do NOT provide any explanations."
        ),
    )


def _full_aud_predicate(species: str) -> SemPredicate:
    return SemPredicate(
        field="audio_path",
        modality="VectorAudio",
        succ_cond=f"The recording is the call of a {species}",
        prompt=(
            "You are given an audio recording of an animal voice. "
            f"Please determine if the recording is the call of a {species}. "
            'Please JUST answer "True" if it is, and "False" otherwise. '
            "Do NOT provide any explanations."
        ),
    )


# --- Engine-level filters (NOT paper queries) -----------------------------
# Single definition site for the six sub-question filters and their
# standalone UCQ wrappers. They are kept selectable because arm targeting
# needs filter-level task keys (night arm B = image-only filters; audio
# arms = *_aud / *_any) and the no-cache reruns target single filters
# (e.g. elephant). The paper queries (q1-q10.py) compose these same SemCQ
# objects, so co-selection never double-annotates a branch (flatten
# identity check).

ZEBRA_IMG_SEMCQ = SemCQ(
    selected=["media_id"], Sigma=_FULL_IMG_SIGMA,
    Ps=[_full_img_predicate("zebra", "zebras")],
)
ELEPHANT_AUD_SEMCQ = SemCQ(
    selected=["media_id"], Sigma=_FULL_AUD_SIGMA,
    Ps=[_full_aud_predicate("elephant")],
)
MONKEY_IMG_SEMCQ = SemCQ(
    selected=["media_id"], Sigma=_FULL_IMG_SIGMA,
    Ps=[_full_img_predicate("monkey", "monkeys")],
)
MONKEY_AUD_SEMCQ = SemCQ(
    selected=["media_id"], Sigma=_FULL_AUD_SIGMA,
    Ps=[_full_aud_predicate("monkey")],
)
# Q7's second branch: impala images (new predicate 11-18; census GT
# impala_img.csv = 136 rows, token-match over sf_200 native labels).
IMPALA_IMG_SEMCQ = SemCQ(
    selected=["media_id"], Sigma=_FULL_IMG_SIGMA,
    Ps=[_full_img_predicate("impala", "impalas", "an")],
)

# Either-modality sightings: a UNION of two owned branches per species
# ("any"-variant semantics; the 11-16 AND-composition was vacuous on the
# split-column universe -- each row fails the other modality's predicate).
ELEPHANT_ANY_QUERIES = {
    "elephant_any__img": SemCQ(
        selected=["media_id"], Sigma=_FULL_IMG_SIGMA,
        Ps=[_full_img_predicate("elephant", "elephants")],
    ),
    "elephant_any__aud": SemCQ(
        selected=["media_id"], Sigma=_FULL_AUD_SIGMA,
        Ps=[_full_aud_predicate("elephant")],
    ),
}
MONKEY_ANY_QUERIES = {
    "monkey_any__img": SemCQ(
        selected=["media_id"], Sigma=_FULL_IMG_SIGMA,
        Ps=[_full_img_predicate("monkey", "monkeys")],
    ),
    "monkey_any__aud": SemCQ(
        selected=["media_id"], Sigma=_FULL_AUD_SIGMA,
        Ps=[_full_aud_predicate("monkey")],
    ),
}

# Standalone filter UCQs (diagnostics; keys == their engine names).
zebra_img = self_ucq(
    "zebra_img", ZEBRA_IMG_SEMCQ, GT_DIR / "zebra_img.csv")
elephant_aud = self_ucq(
    "elephant_aud", ELEPHANT_AUD_SEMCQ, GT_DIR / "elephant_aud.csv")
monkey_img = self_ucq(
    "monkey_img", MONKEY_IMG_SEMCQ, GT_DIR / "monkey_img.csv")
monkey_aud = self_ucq(
    "monkey_aud", MONKEY_AUD_SEMCQ, GT_DIR / "monkey_aud.csv")

ELEPHANT_ANY = SemUCQ(
    queries=ELEPHANT_ANY_QUERIES,
    gt_path=GT_DIR / "elephant_any.csv",
    assembler=lambda t: union_frames(list(t.values()), ["media_id"]),
    evaluator=lambda a, p: eval_filter(a, p, "media_id"),
)
MONKEY_ANY = SemUCQ(
    queries=MONKEY_ANY_QUERIES,
    gt_path=GT_DIR / "monkey_any.csv",
    assembler=lambda t: union_frames(list(t.values()), ["media_id"]),
    evaluator=lambda a, p: eval_filter(a, p, "media_id"),
)
