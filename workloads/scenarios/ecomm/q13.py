# pylint: disable=missing-function-docstring,invalid-name
"""ecomm Q13: running t-shirts matching a long textual description,
AND-composed over the product text and the product image.

Official spec (files/ecomm/queries/q13.toml): natural_language is EMPTY
by design; the official definition is the ground_truth SQL -- Men/Sports
Tshirts, baseColour Blue or Black, 'Short Sleeves' in Sleeve Length,
'Round Neck' in Neck, 'Striped' in Pattern, season <> Winter (12 ids at
sf_2000). The description below is our encoding of that SQL as a
semantic predicate:

    "The product is a running t-shirt for men with a round neck and
    short sleeves, preferably in blue or black, but not bright colors
    like white. Also definitely not green. It should be suitable for
    outdoor running in warm weather. If the t-shirt is not green, it
    should at least feature a striped design."
"""
from data_structure import Predicate, SemCQ, SemPredicate
from workloads.scenarios.ecomm._common import GT_DIR, self_ucq

Q13_SEMCQ = SemCQ(
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

Q13 = self_ucq("Q13", Q13_SEMCQ, GT_DIR / "Q13.csv")
