# pylint: disable=missing-function-docstring,invalid-name
"""animals scenario: wildlife full-media universe (one row per image/
recording, 200 img + 66 aud from sf_200). All 10 paper queries (Q1-Q10,
files q1-q10.py) are SemUCQs composing the engine filters declared in
_common.py (union / argmax / city-set algebra) with official GT +
evaluator declarations (11-16 universe, 11-18 SemUCQ eval + paper-ID
layout)."""
import yaml

from workloads.ldb_workload import LdbWorkload
from workloads.scenarios.animals._common import (
    CURRENT_DIR,
    DATASET_PATH,
)
from workloads.scenarios.animals._common import (
    ELEPHANT_ANY,
    MONKEY_ANY,
    elephant_aud,
    monkey_aud,
    monkey_img,
    zebra_img,
)
from workloads.scenarios.animals.q1 import Q1
from workloads.scenarios.animals.q2 import Q2
from workloads.scenarios.animals.q3 import Q3
from workloads.scenarios.animals.q4 import Q4
from workloads.scenarios.animals.q5 import Q5
from workloads.scenarios.animals.q6 import Q6
from workloads.scenarios.animals.q7 import Q7
from workloads.scenarios.animals.q8 import Q8
from workloads.scenarios.animals.q9 import Q9
from workloads.scenarios.animals.q10 import Q10

SEM_QUERIES = {
    # Paper queries: all 10 SemBench animals queries (Q1-Q10). Q1/Q2/Q4/
    # Q10 compose existing rowsets (count / argmax); Q7 adds an impala
    # image predicate (labeled only when Q7 is selected, 11-18).
    "Q1": Q1,
    "Q2": Q2,
    "Q3": Q3,
    "Q4": Q4,
    "Q5": Q5,
    "Q6": Q6,
    "Q7": Q7,
    "Q8": Q8,
    "Q9": Q9,
    "Q10": Q10,
    # Engine-level filter diagnostics (NOT paper queries): kept selectable
    # for arm targeting (night arm B = image-only; audio arms = *_aud/*_any)
    # and single-filter no-cache reruns. Definitions live in _common.py.
    "zebra_img": zebra_img,
    "elephant_aud": elephant_aud,
    "elephant_any": ELEPHANT_ANY,
    "monkey_img": monkey_img,
    "monkey_aud": monkey_aud,
    "monkey_any": MONKEY_ANY,
}


def get_workload(queries: list[str], config: dict | None = None) -> LdbWorkload:
    selected_queries = {}
    for q in queries:
        assert q in SEM_QUERIES, f"Invalid query {q} in animals dataset."
        selected_queries[q] = SEM_QUERIES[q]

    if config is None:
        with open(CURRENT_DIR / "config.yaml") as f:
            config = yaml.safe_load(f)
    assert config is not None, "Fail to load the workload config."

    return LdbWorkload(
        data_dir=str(DATASET_PATH),
        scenario="animals",
        queries=selected_queries,
        config=config,
    )
