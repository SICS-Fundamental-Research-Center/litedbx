"""Shared paths and helpers for the mmqa scenario queries."""
from pathlib import Path

from data_structure import SemCQ, SemUCQ
from workloads.core.answer_eval import eval_filter
from workloads.core.ucq_exec import single

REPO = Path(__file__).resolve().parents[3]
CURRENT_DIR = REPO / "workloads"  # config.yaml lives here
DATASET_PATH = REPO / "data/mmqa"
# mmqa splits per query across two sub-datasets (movie / airport).
DATA_MAP = {
    "Q3a": DATASET_PATH / "movie",
    "Q3c": DATASET_PATH / "movie",
    "Q3f": DATASET_PATH / "movie",
    "Q3g": DATASET_PATH / "movie",
    "Q6a": DATASET_PATH / "airport",
    "Q6b": DATASET_PATH / "airport",
    "Q6c": DATASET_PATH / "airport",
}


def self_ucq(name: str, cq: SemCQ, gt: Path) -> SemUCQ:
    """Uniform UCQ form of a plain filter query: the branch runs under
    its own engine name and the answer is the retrieved rowset itself."""
    return SemUCQ(
        queries={name: cq},
        gt_path=gt,
        assembler=single(name),
        evaluator=lambda a, p: eval_filter(a, p, cq.selected[0]),
    )


def gt_of(name: str) -> Path:
    """Answer GT of a (self-named) mmqa query, following DATA_MAP."""
    return DATA_MAP[name] / "ground_truth" / f"{name}.csv"
