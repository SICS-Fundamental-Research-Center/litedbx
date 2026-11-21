# pylint: disable=missing-function-docstring,invalid-name
"""ecomm Q1: products that are a backpack from Reebok.

Official NL query (files/ecomm/queries/q1.toml [definition]):
    "Based on the textual description and the title of the product,
    find the product ids of products that are backpacks from Reebok."
"""
from data_structure import Predicate, SemCQ, SemPredicate
from workloads.scenarios.ecomm._common import GT_DIR, self_ucq

Q1_SEMCQ = SemCQ(
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

Q1 = self_ucq("Q1", Q1_SEMCQ, GT_DIR / "Q1.csv")
