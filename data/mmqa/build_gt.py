#!/usr/bin/env python3
"""Per-tuple ground-truth verifier for the mmqa dataset (2026-11-20).

The official answer key per query is the ``ground_truth`` array in
files/mmqa/query/natural_language/q*.json (the NL spec files; the
per-engine AI.IF semantics live in files/mmqa/query/bigquery/*.sql).
The rowset CSVs under movie/airport ground_truth/ are its
materialization; this script verifies that invariant and regenerates a
WIRED query's CSV only on divergence (official wins):

    movie : Q3a comedies(13) Q3c romances(4) Q3f romantic-comedies(3)
            Q3g biographical-comedies(1)          [wired]
            Q3b sci-fi Q3d horror Q3e heist        [unwired, verify-only]
    airport: Q6a Frankfurt(1) Q6b Germany(1) Q6c Europe(5)  [wired]

Also verifies the _h/_p variant files (legacy engine-variant aliases
from the holistic/per-branch classification era) are byte-identical to
the bare-name files -- they are aliases, not independent GTs, and the
current wiring reads bare names only.

Known disclosed divergence (verify-only, file untouched -- unwired):
    Q3b.csv carries one extra row 'The Disaster Artist' vs the official
    json key (that title belongs to Q3a's comedy GT).

Self-checks (exit 1 on mismatch of any WIRED query or alias):
    csv set == official json ground_truth set, per query
    Q3b divergence disclosed, not fatal
    _h/_p byte-identical to bare, per query having variants

Usage: python3 data/mmqa/build_gt.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
NL_DIR = HERE.parents[1] / "files/mmqa/query/natural_language"

SUBS = {
    "movie": [f"Q3{c}" for c in "abcdefg"],
    "airport": ["Q6a", "Q6b", "Q6c"],
}
WIRED = {"Q3a", "Q3c", "Q3f", "Q3g", "Q6a", "Q6b", "Q6c"}
COL = {"movie": "title", "airport": "Airlines"}

fails = 0
for sub, queries in SUBS.items():
    gt_dir = HERE / sub / "ground_truth"
    for q in queries:
        official = set(json.loads(
            (NL_DIR / f"{q.lower()}.json").read_text())["ground_truth"])
        path = gt_dir / f"{q}.csv"
        csv = pd.read_csv(path, dtype=str)
        mine = set(csv[COL[sub]].dropna())
        ok = mine == official
        wired = q in WIRED
        if not ok and wired:  # regenerate from the official key
            pd.DataFrame({COL[sub]: sorted(official)}).to_csv(
                path, index=False)
            print(f"REGENERATED {sub}/{q}.csv from official json key")
            ok = set(pd.read_csv(path, dtype=str)[COL[sub]]) == official
        if not ok and not wired:
            print(f"DIFF  {sub}/{q}: csv {len(mine)} vs official "
                  f"{len(official)} -- DISCLOSED (unwired, untouched): "
                  f"csv-only {sorted(mine - official)} "
                  f"json-only {sorted(official - mine)}")
            continue
        fails += not ok
        print(f"{'EQUAL' if ok else 'DIFF '} {sub}/{q}: "
              f"json {len(official)} | csv {len(mine)}"
              f"{'' if wired else ' (unwired)'}")
        # _h/_p aliases must stay byte-identical to the bare file
        for suffix in ("_h", "_p"):
            alias = gt_dir / f"{q}{suffix}.csv"
            if alias.exists():
                same = alias.read_bytes() == path.read_bytes()
                fails += not same
                if not same:
                    print(f"DIFF  {sub}/{q}{suffix}.csv != {q}.csv "
                          f"(alias must be byte-identical)")

print(f"RESULT: {'ALL SELF-CHECKS EQUAL' if fails == 0 else f'{fails} FAILURES'}")
sys.exit(1 if fails else 0)
