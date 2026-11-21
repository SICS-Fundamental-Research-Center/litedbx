# pylint: disable=missing-function-docstring,invalid-name
"""cars scenario: SemUCQ queries over the car-complaints dataset
(seed-42 sf_19672 engine universe at the OFFICIAL SemBench scale;
native columns only, label columns GT-side only -- the derived n_avail
column was removed from the base table 11-20 per the iron rule).

AND-only, non-aggregation subset of SemBench cars (prompts mirror the
original BigQuery AI.IF texts), plus the R1.D3.1 domain expansion
(aggregates + multi-modality, 11-16)."""
import yaml

from workloads.ldb_workload import LdbWorkload
from workloads.scenarios.cars._common import CURRENT_DIR, DATASET_PATH
from workloads.scenarios.cars.q1 import Q1
from workloads.scenarios.cars.q2 import Q2
from workloads.scenarios.cars.q3 import Q3
from workloads.scenarios.cars.q4 import Q4
from workloads.scenarios.cars.q5 import Q5
from workloads.scenarios.cars.q6 import Q6
from workloads.scenarios.cars.q7 import Q7
from workloads.scenarios.cars.q8 import Q8
from workloads.scenarios.cars.q9 import Q9

SEM_QUERIES = {
    "Q1": Q1,
    "Q2": Q2,
    "Q3": Q3,
    "Q4": Q4,
    "Q5": Q5,
    "Q6": Q6,
    "Q7": Q7,
    "Q8": Q8,
    "Q9": Q9,
}


def get_workload(queries: list[str], config: dict | None = None) -> LdbWorkload:
    selected_queries = {}
    for q in queries:
        assert q in SEM_QUERIES, f"Invalid query {q} in cars dataset."
        selected_queries[q] = SEM_QUERIES[q]

    if config is None:
        with open(CURRENT_DIR / "config.yaml") as f:
            config = yaml.safe_load(f)
    assert config is not None, "Fail to load the workload config."

    return LdbWorkload(
        data_dir=str(DATASET_PATH),
        scenario="cars",
        queries=selected_queries,
        config=config,
    )

