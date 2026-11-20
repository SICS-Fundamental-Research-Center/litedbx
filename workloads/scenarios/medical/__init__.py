# pylint: disable=missing-function-docstring,invalid-name
"""medical scenario: SemUCQ queries over the medical-patient dataset
(full-modality engine universe: audio_path for the 112 recorded
patients; R1.D3.1, 11-16)."""
import yaml

from workloads.ldb_workload import LdbWorkload
from workloads.scenarios.medical._common import CURRENT_DIR, DATASET_PATH
from workloads.scenarios.medical.q1 import Q1
from workloads.scenarios.medical.q2 import Q2
from workloads.scenarios.medical.q3 import Q3
from workloads.scenarios.medical.q4 import Q4
from workloads.scenarios.medical.q5 import Q5
from workloads.scenarios.medical.q6 import Q6
from workloads.scenarios.medical.q7 import Q7
from workloads.scenarios.medical.q8 import Q8
from workloads.scenarios.medical.q9 import Q9

SEM_QUERIES = {
    "Q1": Q1,
    "Q2": Q2,
    "Q3": Q3,
    "Q4": Q4,
    "Q5": Q5,
    "Q7": Q7,
    "Q6": Q6,
    "Q8": Q8,
    "Q9": Q9,
}


def get_workload(queries: list[str], config: dict | None = None) -> LdbWorkload:
    selected_queries = {}
    for q in queries:
        assert q in SEM_QUERIES, f"Invalid query {q} in medical dataset."
        selected_queries[q] = SEM_QUERIES[q]

    if config is None:
        with open(CURRENT_DIR / "config.yaml") as f:
            config = yaml.safe_load(f)
    assert config is not None, "Fail to load the workload config."

    return LdbWorkload(
        data_dir=str(DATASET_PATH),
        scenario="medical",
        queries=selected_queries,
        config=config,
    )

