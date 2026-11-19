"""Engine-level flattening of all-UCQ workloads + table algebra.

- ``flatten_sem_queries``: map an all-``{name: SemUCQ}`` workload dict
  onto engine-level SemCQs (plain filter queries are wrapped in
  self-named single-branch UCQs at the scenario layer, so the workload
  contract is uniformly UCQ). A SemUCQ's ``queries`` keys ARE engine
  query names: owning UCQs name their branches ``{ucq}__{branch}``;
  composition UCQs reuse the engine names of already-defined branches so
  shared SemCQs are annotated once (medical Q6 = Q7's sick branches +
  Q6's own normal branches). The same SemCQ object may live under several
  engine names (movie Q3/Q4/Q8 clone Q2); one engine name must not map
  to two different objects.
- ``union_frames`` / ``intersect_frames`` / ``single``: assembler-side
  table algebra over the shared ``selected`` columns.
"""

from __future__ import annotations

import pandas as pd

from data_structure import SemCQ, SemUCQ


def flatten_sem_queries(
    sem_queries: dict[str, SemUCQ],
) -> tuple[dict[str, SemCQ], dict[str, list[str]]]:
    """Expand SemUCQ entries into engine-level branch SemCQs.

    Returns ``(flat_queries, ucq_map)`` where ``flat_queries`` maps every
    engine-level query name to a plain SemCQ and ``ucq_map`` maps each
    SemUCQ name to its branch engine query names (dict order).
    """
    flat_queries: dict[str, SemCQ] = {}
    ucq_map: dict[str, list[str]] = {}
    for name, query in sem_queries.items():
        if not isinstance(query, SemUCQ):
            raise TypeError(
                f"Query {name!r} must be a SemUCQ -- wrap plain SemCQs in "
                "a self-named single-branch UCQ (scenario _common.self_ucq); "
                f"got {type(query).__name__}."
            )
        for engine_name, branch in query.queries.items():
            if engine_name in flat_queries and (
                flat_queries[engine_name] is not branch
            ):
                raise ValueError(
                    f"Engine query name {engine_name!r} maps to two "
                    "different SemCQs."
                )
            flat_queries[engine_name] = branch
        ucq_map[name] = list(query.queries)
    return flat_queries, ucq_map


def union_frames(frames: list[pd.DataFrame], selected: list[str]) -> pd.DataFrame:
    """Union of branch tables, deduplicated on ``selected``.

    An all-empty union keeps the first branch's columns (the empty branch
    tables carry the universe schema), so downstream assemblers can still
    reference non-selected columns on empty results.
    """
    non_empty = [f for f in frames if not f.empty]
    if not non_empty:
        return frames[0].head(0) if frames else pd.DataFrame(columns=selected)
    union = pd.concat(non_empty, ignore_index=True)
    return (
        union.drop_duplicates(subset=selected, keep="first")
        .sort_values(selected, kind="stable")
        .reset_index(drop=True)
    )


def intersect_frames(frames: list[pd.DataFrame], selected: list[str]) -> pd.DataFrame:
    """Intersection of branch tables on ``selected``, deduplicated."""
    if not frames:
        return pd.DataFrame(columns=selected)
    out = frames[0][selected].drop_duplicates()
    for frame in frames[1:]:
        out = out.merge(
            frame[selected].drop_duplicates(), on=selected, how="inner"
        )
    return (
        out.sort_values(selected, kind="stable").reset_index(drop=True)
    )


def single(branch: str):
    """Assembler returning one branch's table unchanged."""
    return lambda tables: tables[branch]
