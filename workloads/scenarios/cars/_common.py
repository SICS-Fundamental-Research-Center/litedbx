"""Shared paths and helpers for the cars scenario queries."""
from pathlib import Path

from data_structure import SemCQ, SemUCQ
from workloads.core.answer_eval import eval_filter
from workloads.core.ucq_exec import single

REPO = Path(__file__).resolve().parents[3]
CURRENT_DIR = REPO / "workloads"  # config.yaml lives here
DATASET_PATH = REPO / "data/cars"
# Ground truth, two layers side by side (11-20 build_gt.py layout, same
# scheme as medical): bare-name files are verbatim copies of the official
# no-suffix GTs (official scale IS sf_19672, seed 42; census-verified
# 9/9 + Q4 scalar exact + Q2 disclosed official-empty-vs-1 label-noise
# car); per-branch/per-tuple rowsets the ENGINE labels against
# (Q6__img_dmg/__aud_dmg/__fire/__img_ok/__aud_ok/__nofire,
# Q7__img/__aud/__txt, Q4__tuple, Q5__tuple) are census-built by
# data/cars/build_gt.py. Displaced 11-18 rowsets + _answer copies are
# archived under data/cars/archived/.
GT_DIR = DATASET_PATH / "ground_truth"


def self_ucq(name: str, cq: SemCQ, gt: Path) -> SemUCQ:
    """Uniform UCQ form of a plain filter query: the branch runs under
    its own engine name and the answer is the retrieved rowset itself."""
    return SemUCQ(
        queries={name: cq},
        gt_path=gt,
        assembler=single(name),
        evaluator=lambda a, p: eval_filter(a, p, cq.selected[0]),
    )
