# pylint: disable=missing-module-docstring,missing-function-docstring
# pylint: disable=unspecified-encoding,duplicate-code,invalid-name
from pathlib import Path

import yaml

from data_structure import Predicate, SemCQ, SemPredicate
from workloads.ldb_workload import LdbWorkload

DATASET_PATH = Path(__file__).parent.parent.parent / "data/animals"
CURRENT_DIR = Path(__file__).parent.parent

Q7 = SemCQ(
    selected=["City"],
    Sigma=[Predicate("ImagePath", "!=", "")],
    Ps=[
        SemPredicate(
            field="ImagePath",
            modality="VectorImage",
            succ_cond="The images belong to this city contain zebra(s)",
            prompt=(
                "You are a zoology expert. "
                "Please determine if the given images contain zebra(s). "
                'Please JUST answer "True" if they do, and "False" otherwise. '
                "Do NOT provide any explanations."
            ),
        ),
        SemPredicate(
            field="ImagePath",
            modality="VectorImage",
            succ_cond="The images belong to this city contain impala(s)",
            prompt=(
                "You are a zoology expert. "
                "Please determine if the given images contain impala(s). "
                'Please JUST answer "True" if they do, and "False" otherwise. '
                "Do NOT provide any explanations."
            ),
        ),
    ],
)


SEM_QUERIES = {
    "Q7": Q7,
}


def get_workload(queries: list[str], config: dict | None = None) -> LdbWorkload:
    sem_queries = {}
    for q in queries:
        assert q in SEM_QUERIES, f"Invalid query {q} in animals dataset."
        sem_queries[q] = SEM_QUERIES[q]

    if config is None:
        with open(CURRENT_DIR / "config.yaml") as f:
            config = yaml.safe_load(f)
    assert config is not None, "Fail to load the workload config."

    return LdbWorkload(
        data_dir=str(DATASET_PATH),
        scenario="animals",
        queries=sem_queries,
        config=config,
    )
