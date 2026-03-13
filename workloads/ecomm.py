from pathlib import Path
from typing import Optional
import yaml
from data_structure import Predicate, SemPredicate, SemCQ
from .ldb_workload import LdbWorkload

DATASET_PATH = Path(__file__).parent.parent / "data/ecomm_sf_2000"
CURRENT_DIR = Path(__file__).parent

Q1 = SemCQ(
    selected=["id"],
    Sigma=[Predicate("productNameAndDesc", "!=", "")],
    Ps=[
        SemPredicate(
            field="productNameAndDesc",
            modality="Text",
            succ_cond="The product is a backpack from Reebok",
            prompt=(
                "Please determine if the given product is a backpack from Reebok. "
                "Please JUST answer \"True\" if they do, and \"False\" otherwise. "
                "Do NOT provide any explanations."
        ))
])


Q2 = SemCQ(
    selected=["id"],
    Sigma=[Predicate("image_path", "!=", "")],
    Ps=[
        SemPredicate(
            field="image_path",
            modality="Image",
            succ_cond="The displayed product shows a (pair of) sports shoe(s) and the shoe(s) have the colors yellow and silver",
            prompt=(
                "Please determine if the given product shows a (pair of) sports shoe(s) and the shoe(s) have the colors yellow and silver. "
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
        assert q in SEM_QUERIES, f"Invalid query {q} in ecomm dataset."
        sem_queries[q] = SEM_QUERIES[q]
    
    if config is None:
        with open(CURRENT_DIR / "config.yaml") as f:
            config = yaml.safe_load(f)
    assert config is not None, "Fail to load the workload config."

    return LdbWorkload(data_dir=str(DATASET_PATH), scenario="ecomm", queries=sem_queries, config=config)

