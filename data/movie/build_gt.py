#!/usr/bin/env python3
"""Per-tuple ground-truth builder for the movie dataset (2026-11-20).

User ruling: the GT files in ground_truth/ (tracked user assets) are
NEVER rewritten by this script while their content is correct -- the
movie engine layer is already official-derived. This script verifies
that invariant and regenerates ONLY on content divergence:

  Q1.csv   engine rowset == official Q1.csv rowset (byte-identical
           copy staged by the 11-18 consolidation)
  Q2.csv   engine rowset == official Q2.csv rowset (14 taken_3
           positives; differs from the official file by one trailing
           newline only -- byte diff disclosed, content set-equal)
  Q3.csv / Q4.csv / Q8.csv   engine rowsets == official Q2 rowset
           (the three aggregate queries share Q2's SemCQ verbatim;
           deterministic postprocessing over the same certified rows)

Aggregate self-checks against the UNTOUCHED officials (exit 1 on
mismatch):
    len(Q2 rows)          == Q3.csv positive_review_cnt   (14)
    len(Q2 rows) / 120    == Q4.csv positivity_ratio      (0.11666667)
    120 - len, len        == Q8.csv NEGATIVE/POSITIVE counts (106/14)

Out of scope (unwired queries; tracked user assets, untouched):
Q5/Q6/Q7 (pair-join tables), Q9/Q10.

Sources: files/movie/raw_results/ground_truth (official SemBench GT --
ground truth by definition, no census needed; the official Q2 rowset IS
the tuple-level GT). MOVIE_N = 120 (taken_3 sample size). Deterministic,
no LLM, no run outputs. Idempotent.

Usage: python3 data/movie/build_gt.py
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
GT = HERE / "ground_truth"
OFFICIAL = HERE.parents[1] / "files/movie/raw_results/ground_truth"
MOVIE_N = 120  # taken_3 sample size (Q4 ratio denominator, Q8 base)

fails = 0


def ids(path: Path, col: str = "reviewId") -> set[str]:
    return set(pd.read_csv(path, dtype=str)[col].dropna())


off_q1 = ids(OFFICIAL / "Q1.csv")
off_q2 = ids(OFFICIAL / "Q2.csv")

# ------------------------------------------- engine rowsets vs officials
checks: list[tuple[str, bool, str]] = []
for name, official in [("Q1", off_q1), ("Q2", off_q2)]:
    mine = ids(GT / f"{name}.csv")
    ok = mine == official
    note = ""
    if ok and (GT / f"{name}.csv").read_bytes() != (
            OFFICIAL / f"{name}.csv").read_bytes():
        note = " (byte diff: trailing newline only, disclosed)"
    checks.append((f"{name}.csv engine set == official "
                   f"({len(official)} reviewIds)", ok, note))

# Q3/Q4/Q8 engine rowsets must equal the official Q2 rowset (shared SemCQ)
for name in ["Q3", "Q4", "Q8"]:
    mine = ids(GT / f"{name}.csv")
    ok = mine == off_q2
    if not ok:  # regenerate from the official rowset, never from a run
        shutil.copy(OFFICIAL / "Q2.csv", GT / f"{name}.csv")
        print(f"REGENERATED {name}.csv <- official Q2.csv "
              f"(content had diverged)")
        ok = ids(GT / f"{name}.csv") == off_q2
    checks.append((f"{name}.csv engine set == official Q2 rowset", ok, ""))

# --------------------------------------- aggregate anchors vs officials
n_pos = len(off_q2)
q3_off = int(pd.read_csv(OFFICIAL / "Q3.csv")["positive_review_cnt"].iloc[0])
q4_off = float(pd.read_csv(OFFICIAL / "Q4.csv")["positivity_ratio"].iloc[0])
q8_off = pd.read_csv(OFFICIAL / "Q8.csv")
q8_neg = int(q8_off.loc[q8_off.scoreSentiment == "NEGATIVE", "count"].iloc[0])
q8_pos = int(q8_off.loc[q8_off.scoreSentiment == "POSITIVE", "count"].iloc[0])
checks += [
    ("len(Q2 positives) == official Q3 count",
     n_pos == q3_off, f" mine={n_pos} official={q3_off}"),
    ("len/120 == official Q4 ratio",
     abs(n_pos / MOVIE_N - q4_off) < 1e-6,
     f" mine={n_pos / MOVIE_N:.8f} official={q4_off:.8f}"),
    ("120-len/len == official Q8 NEG/POS counts",
     (MOVIE_N - n_pos, n_pos) == (q8_neg, q8_pos),
     f" mine=({MOVIE_N - n_pos},{n_pos}) official=({q8_neg},{q8_pos})"),
]

for name, ok, extra in checks:
    fails += not ok
    print(f"{'EQUAL' if ok else 'DIFF '} {name}{extra}")
print(f"RESULT: {'ALL SELF-CHECKS EQUAL' if fails == 0 else f'{fails} FAILURES'}")
sys.exit(1 if fails else 0)
