from pathlib import Path
from typing import Optional
import yaml
from data_structure import Predicate, SemPredicate, SemCQ
from .ldb_workload import LdbWorkload

DATASET_PATH = Path(__file__).parent.parent / "data/mmqa"
CURRENT_DIR = Path(__file__).parent

Q3a = SemCQ(
    selected=["title"],
    Sigma=[Predicate("text", "!=", "")],
    Ps=[
        SemPredicate(
            field="text",
            modality="Text",
            succ_cond="The movie is a comedy",
            prompt=(
                "Please determine if the given text indicate "
                "that the movie is a comedy. "
                "Please JUST answer \"True\" if they do, and \"False\" otherwise. "
                "Do NOT provide any explanations."
        ))
])

Q3f = SemCQ(
    selected=["title"],
    Sigma=[Predicate("text", "!=", "")],
    Ps=[
        SemPredicate(
            field="text",
            modality="Text",
            succ_cond="The movie is a romantic comedy",
            prompt=(
                "Please determine if the given text indicate "
                "that the movie is a romantic comedy. "
                "Please JUST answer \"True\" if they do, and \"False\" otherwise. "
                "Do NOT provide any explanations."
        ))
])


Q6a = SemCQ(
    selected=["Airlines"],
    Sigma=[Predicate("Destinations", "!=", "")],
    Ps=[
        SemPredicate(
            field="Destinations",
            modality="Text",
            succ_cond="The airline has destinations in Frankfurt",
            prompt=(
                "Please determine if the given text indicate "
                "that the airline has destinations in Frankfurt. "
                "Please JUST answer \"True\" if they do, and \"False\" otherwise. "
                "Do NOT provide any explanations."
        ))
])


Q6b = SemCQ(
    selected=["Airlines"],
    Sigma=[Predicate("Destinations", "!=", "")],
    Ps=[
        SemPredicate(
            field="Destinations",
            modality="Text",
            succ_cond="The airline has destinations in Germany",
            prompt=(
                "Please determine if the given text indicate "
                "that the airline has destinations in Germany. "
                "Please JUST answer \"True\" if they do, and \"False\" otherwise. "
                "Do NOT provide any explanations."
        ))
])


Q6c = SemCQ(
    selected=["Airlines"],
    Sigma=[Predicate("Destinations", "!=", "")],
    Ps=[
        SemPredicate(
            field="Destinations",
            modality="Text",
            succ_cond="The airline has destinations in Europe",
            prompt=(
                "Please determine if the given text indicate "
                "that the airline has destinations in Europe. "
                "Please JUST answer \"True\" if they do, and \"False\" otherwise. "
                "Do NOT provide any explanations."
        ))
])


SEM_QUERIES = {
    "Q3a": Q3a,
    "Q3f": Q3f,
    "Q6a": Q6a,
    "Q6b": Q6b,
    "Q6c": Q6c,
}

DATA_MAP = {
    "Q3a": DATASET_PATH / "movie",
    "Q3f": DATASET_PATH / "movie",
    "Q6a": DATASET_PATH / "airport",
    "Q6b": DATASET_PATH / "airport",
    "Q6c": DATASET_PATH / "airport",
}


def get_workload(queries: list[str], config: Optional[dict] = None) -> LdbWorkload:
    sem_queries = {}
    dataset_path = None
    for q in queries:
        assert q in SEM_QUERIES, f"Invalid query {q} in mmqa dataset."
        if dataset_path is not None:
            assert dataset_path == DATA_MAP[q], \
                f"Queries from different datasets: {dataset_path} and {DATA_MAP[q]}"
        sem_queries[q] = SEM_QUERIES[q]
        dataset_path = DATA_MAP[q]
    
    if config is None:
        with open(CURRENT_DIR / "config.yaml") as f:
            config = yaml.safe_load(f)
    assert config is not None, "Fail to load the workload config."

    return LdbWorkload(data_dir=str(dataset_path), scenario="mmqa", queries=sem_queries, config=config)
