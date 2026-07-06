# pylint: disable=missing-module-docstring,missing-function-docstring
# pylint: disable=unspecified-encoding,duplicate-code,invalid-name
from pathlib import Path

import yaml

from data_structure import Predicate, SemCQ, SemPredicate
from workloads.ldb_workload import LdbWorkload

DATASET_PATH = Path(__file__).parent.parent.parent / "data/ecomm_sf_2000"
CURRENT_DIR = Path(__file__).parent.parent

Q1 = SemCQ(
    selected=["id"],
    Sigma=[Predicate("productNameAndDesc", "!=", "")],
    Ps=[
        SemPredicate(
            field="productNameAndDesc",
            modality="Text",
            succ_cond="The product is a backpack from Reebok",
            prompt=(
                "Please determine if the given product is a backpack "
                "from Reebok. "
                'Please JUST answer "True" if they do, and "False" otherwise. '
                "Do NOT provide any explanations."
            ),
        )
    ],
)


Q2 = SemCQ(
    selected=["id"],
    Sigma=[Predicate("image_path", "!=", "")],
    Ps=[
        SemPredicate(
            field="image_path",
            modality="Image",
            succ_cond=(
                "The displayed product shows a (pair of) sports shoe(s) "
                "and the shoe(s) have the colors yellow and silver"
            ),
            prompt=(
                "Please determine if the given product shows a (pair of) "
                "sports shoe(s) and the shoe(s) have the colors yellow "
                "and silver. "
                'Please JUST answer "True" if they do, and "False" otherwise. '
                "Do NOT provide any explanations."
            ),
        )
    ],
)


Q13 = SemCQ(
    selected=["id"],
    Sigma=[
        Predicate("productNameAndDesc", "!=", ""),
        Predicate("image_path", "!=", ""),
    ],
    Ps=[
        SemPredicate(
            field="productNameAndDesc",
            modality="Text",
            succ_cond=(
                "The product is a running t-shirt for men "
                "with a round neck and short sleeves, \n"
                "preferably in blue or black, "
                "but not bright colors like white. "
                "Also definitely not green. \n"
                "It should be suitable for outdoor running in warm weather. \n"
                "If the t-shirt is not green, it should at least "
                "feature a striped design."
            ),
            prompt=(
                "You will receive a textural description of the product. "
                "Please determine if this description matches "
                "the following description: \n"
                "The product is a running t-shirt for men "
                "with a round neck and short sleeves, \n"
                "preferably in blue or black, "
                "but not bright colors like white. "
                "Also definitely not green. \n"
                "It should be suitable for outdoor running in warm weather. \n"
                "If the t-shirt is not green, it should at least "
                "feature a striped design. "
                'Please JUST answer "True" if the product matches the '
                'description, and "False" otherwise. '
                "Do NOT provide any explanations."
            ),
        ),
        SemPredicate(
            field="image_path",
            modality="Image",
            succ_cond=(
                "The displayed product shows a running shirt for "
                "men with a round neck and short sleeves, \n"
                "preferably in blue or black, "
                "but not bright colors like white. "
                "Also definitely not green. \n"
                "It should be suitable for outdoor running in warm weather. \n"
                "If the t-shirt is not green, it should at least "
                "feature a striped design."
            ),
            prompt=(
                "You will receive an image of the product. "
                "Please determine if the given product matches "
                "the following description: \n"
                "The product is a running t-shirt for men "
                "with a round neck and short sleeves, \n"
                "preferably in blue or black, "
                "but not bright colors like white. "
                "Also definitely not green. \n"
                "It should be suitable for outdoor running in a warm "
                "weather. \n"
                "If the t-shirt is not green, it should at least "
                "feature a striped design. "
                'Please JUST answer "True" if the product matches the '
                'description, and "False" otherwise. '
                "Do NOT provide any explanations."
            ),
        ),
    ],
)


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
