# pylint: disable=missing-function-docstring,invalid-name
"""mmqa scenario: SemUCQ queries over the movie/airport sub-datasets."""
import yaml

from data_structure import SemCQ
from workloads.ldb_workload import LdbWorkload
from workloads.scenarios.mmqa._common import (
    CURRENT_DIR,
    DATA_MAP,
    DATASET_PATH,
    gt_of,
    self_ucq,
)
from workloads.scenarios.mmqa.q3a import Q3A
from workloads.scenarios.mmqa.q3c import Q3C
from workloads.scenarios.mmqa.q3f import Q3F
from workloads.scenarios.mmqa.q3g import Q3G
from workloads.scenarios.mmqa.q6a import Q6A
from workloads.scenarios.mmqa.q6b import Q6B
from workloads.scenarios.mmqa.q6c import Q6C

SEM_QUERIES = {
    "Q3a": Q3A,
    "Q3c": Q3C,
    "Q3f": Q3F,
    "Q3g": Q3G,
    "Q6a": Q6A,
    "Q6b": Q6B,
    "Q6c": Q6C,
}


def branch(name: str) -> SemCQ:
    """Engine branch of a self-named single-branch mmqa UCQ."""
    return SEM_QUERIES[name].queries[name]


def get_workload(queries: list[str], config: dict | None = None) -> LdbWorkload:
    sem_queries = {}
    dataset_path = None
    for q in queries:
        assert q in SEM_QUERIES, f"Invalid query {q} in mmqa dataset."
        if dataset_path is not None:
            assert dataset_path == DATA_MAP[q], (
                "Queries from different datasets: "
                f"{dataset_path} and {DATA_MAP[q]}"
            )
        sem_queries[q] = SEM_QUERIES[q]
        dataset_path = DATA_MAP[q]

    if config is None:
        with open(CURRENT_DIR / "config.yaml") as f:
            config = yaml.safe_load(f)
    assert config is not None, "Fail to load the workload config."

    return LdbWorkload(
        data_dir=str(dataset_path),
        scenario="mmqa",
        queries=sem_queries,
        config=config,
    )
