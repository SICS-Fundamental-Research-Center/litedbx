# pylint: disable=missing-function-docstring,invalid-name
"""movie scenario: SemUCQ queries over the movie-review dataset."""
import yaml

from workloads.ldb_workload import LdbWorkload
from workloads.scenarios.movie._common import CURRENT_DIR, DATASET_PATH
from workloads.scenarios.movie.q1 import Q1
from workloads.scenarios.movie.q2 import Q2
from workloads.scenarios.movie.q3 import Q3
from workloads.scenarios.movie.q4 import Q4
from workloads.scenarios.movie.q8 import Q8

SEM_QUERIES = {
    "Q1": Q1,
    "Q2": Q2,
    "Q3": Q3,
    "Q4": Q4,
    "Q8": Q8,
}


def get_workload(queries: list[str], config: dict | None = None) -> LdbWorkload:
    sem_queries = {}
    for q in queries:
        assert q in SEM_QUERIES, f"Invalid query {q} in movie dataset."
        sem_queries[q] = SEM_QUERIES[q]

    if config is None:
        with open(CURRENT_DIR / "config.yaml") as f:
            config = yaml.safe_load(f)
    assert config is not None, "Fail to load the workload config."

    return LdbWorkload(
        data_dir=str(DATASET_PATH),
        scenario="movie",
        queries=sem_queries,
        config=config,
    )

