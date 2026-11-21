"""Shared paths and helpers for the movie scenario queries."""
from pathlib import Path

from data_structure import SemCQ, SemUCQ
from workloads.core.answer_eval import eval_filter
from workloads.core.ucq_exec import single

REPO = Path(__file__).resolve().parents[3]
CURRENT_DIR = REPO / "workloads"  # config.yaml lives here
DATASET_PATH = REPO / "data/movie"
# Engine / rowset GT for the plain filter queries (Q1/Q2) and the
# aggregate branches (Q3/Q4/Q8 = official-Q2-rowset copies); set-verified
# against the officials by data/movie/build_gt.py (11-20; Q2 byte diff =
# trailing newline only, disclosed). Unwired Q5/Q6/Q7/Q9/Q10 untouched.
ENGINE_GT = DATASET_PATH / "ground_truth"
# Official SemBench answer GT (aggregate queries Q3/Q4/Q8).
OFFICIAL_GT = REPO / "files/movie/raw_results/ground_truth"
# taken_3 sample size: denominator of the Q4 ratio and the NEGATIVE base
# of the Q8 group counts.
MOVIE_N = 120


def self_ucq(name: str, cq: SemCQ, gt: Path) -> SemUCQ:
    """Uniform UCQ form of a plain filter query: the branch runs under
    its own engine name and the answer is the retrieved rowset itself."""
    return SemUCQ(
        queries={name: cq},
        gt_path=gt,
        assembler=single(name),
        evaluator=lambda a, p: eval_filter(a, p, cq.selected[0]),
    )
