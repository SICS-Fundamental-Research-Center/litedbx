# pylint: disable=missing-function-docstring,invalid-name
"""animals Q2 (paper query, animals scenario): how many elephant
sound recordings -- COUNT over the elephant-audio rowset (scalar
rel-err eval).

Official NL query (files/animals/query/natural_language/Q2.txt):
    How many sound recordings of elephants do we have in our
    database?
"""
import pandas as pd
from data_structure import SemUCQ
from workloads.core.answer_eval import eval_rel_err
from workloads.scenarios.animals._common import (
    ELEPHANT_AUD_SEMCQ,
    GT_DIR,
)

Q2 = SemUCQ(
    queries={"elephant_aud": ELEPHANT_AUD_SEMCQ},
    gt_path=GT_DIR / "Q2.csv",
    assembler=lambda t: pd.DataFrame({"count": [len(t["elephant_aud"])]}),
    evaluator=eval_rel_err,
)
