# pylint: disable=missing-function-docstring,invalid-name
"""ecomm Q2: product images showing yellow-and-silver sports shoes.

Official NL query (files/ecomm/queries/q2.toml [definition]):
    "Based on the image representation of the product, find the product
    ids of products where the image shows a pair of sports shoes that
    are predominantly yellow and silver."
"""
from data_structure import Predicate, SemCQ, SemPredicate
from workloads.scenarios.ecomm._common import GT_DIR, self_ucq

Q2_SEMCQ = SemCQ(
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

Q2 = self_ucq("Q2", Q2_SEMCQ, GT_DIR / "Q2.csv")
