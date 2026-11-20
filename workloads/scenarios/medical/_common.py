"""Shared paths and helpers for the medical scenario queries."""
from pathlib import Path

from data_structure import SemCQ, SemUCQ
from workloads.core.answer_eval import eval_filter
from workloads.core.ucq_exec import single

REPO = Path(__file__).resolve().parents[3]
CURRENT_DIR = REPO / "workloads"  # config.yaml lives here
DATASET_PATH = REPO / "data/medical"
# Ground truth, two layers side by side (11-18 consolidation): bare-name
# files are engine-layer rowsets (per-branch / composed); ``Q*_answer.csv``
# are the aggregate answer GTs used by the official-answer evaluators.
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
