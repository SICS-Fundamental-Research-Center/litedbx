# pylint: disable=missing-function-docstring,invalid-name
"""ecomm scenario: SemUCQ queries over the e-commerce product dataset."""
import yaml

from workloads.ldb_workload import LdbWorkload
from workloads.scenarios.ecomm._common import CURRENT_DIR, DATASET_PATH
from workloads.scenarios.ecomm.q1 import Q1
from workloads.scenarios.ecomm.q13 import Q13
from workloads.scenarios.ecomm.q2 import Q2

SEM_QUERIES = {
    "Q1": Q1,
    "Q2": Q2,
    "Q13": Q13,
}


def get_workload(queries: list[str], config: dict | None = None) -> LdbWorkload:
    sem_queries = {}
    for q in queries:
        assert q in SEM_QUERIES, f"Invalid query {q} in ecomm dataset."
        sem_queries[q] = SEM_QUERIES[q]

    if config is None:
        with open(CURRENT_DIR / "config.yaml") as f:
            config = yaml.safe_load(f)
    assert config is not None, "Fail to load the workload config."

    return LdbWorkload(
        data_dir=str(DATASET_PATH),
        scenario="ecomm",
        queries=sem_queries,
        config=config,
    )
