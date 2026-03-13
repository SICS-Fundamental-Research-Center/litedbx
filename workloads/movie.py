from pathlib import Path
from typing import Optional
import yaml
from data_structure import Predicate, SemPredicate, SemCQ
from .ldb_workload import LdbWorkload

DATASET_PATH = Path(__file__).parent.parent / "data/movie_sf_2000"
CURRENT_DIR = Path(__file__).parent

Q1 = SemCQ(
    selected=["reviewId"],
    Sigma=[Predicate("reviewText", "!=", "")],
    Ps=[
        SemPredicate(
            field="reviewText",
            modality="Text",
            succ_cond="The movie review is clearly positive",
            prompt=(
                "You are a movie reviewer. " 
                "Please determine if the given movie review "
                "is clearly positive. "
                "Please JUST answer \"True\" if they do, and \"False\" otherwise. "
                "Do NOT provide any explanations."
        ))
])

Q2 = SemCQ(
    selected=["reviewId"],
    Sigma=[
        Predicate("id", "==", "taken_3"),
        Predicate("reviewText", "!=", "")
    ],
    Ps=[
        SemPredicate(
            field="reviewText",
            modality="Text",
            succ_cond="The movie review is clearly positive",
            prompt=(
                "You are a movie reviewer. " 
                "Please determine if the given movie review "
                "is clearly positive. "
                "Please JUST answer \"True\" if they do, and \"False\" otherwise. "
                "Do NOT provide any explanations."
        ))
])

SEM_QUERIES = {
    "Q1": Q1,
    "Q2": Q2,
}


def get_workload(queries: list[str], config: Optional[dict] = None) -> LdbWorkload:
    sem_queries = {}
    for q in queries:
        assert q in SEM_QUERIES, f"Invalid query {q} in movie dataset."
        sem_queries[q] = SEM_QUERIES[q]
    
    if config is None:
        with open(CURRENT_DIR / "config.yaml") as f:
            config = yaml.safe_load(f)
    assert config is not None, "Fail to load the workload config."

    return LdbWorkload(data_dir=str(DATASET_PATH), scenario="movie", queries=sem_queries, config=config)

