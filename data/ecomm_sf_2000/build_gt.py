#!/usr/bin/env python3
"""Per-tuple ground-truth builder for the ecomm dataset (2026-11-20).

Medical-pattern port of the census in ground_truth/build_ground_truth.py
(2026-11-18 vintage, left in place untouched) to the standard location.
The official GT is the ``ground_truth`` SQL in files/ecomm/queries/
q{1,2,13}.toml -- there is no official rowset file, so the census IS the
definition; data/ecomm_sf_2000/ground_truth/Q*.csv are its deterministic
materialization intersected with the sf_2000 universe.

The three wired queries are single-branch filters that consume the SAME
rowset at both engine and answer layer (no aggregates, no multi-branch,
no bare-name collision), so this script regenerates the three files
in place, content-identical when nothing drifted:

    Q1.csv   articleType BackPacks... typeName='Backpacks' & Reebok  -> 7
    Q2.csv   Sports Shoes & colours {Yellow,Silver} all present      -> 5
    Q13.csv  Men/Sports Tshirts, Blue|Black, Short Sleeves + Round
             Neck + Striped attributes, season <> Winter             -> 12

Self-checks (exit 1 on mismatch):
    census-in-universe == current GT file (set-equal, per query)
    disclosed: full-census size vs in-universe size (positive-retention
    property of the sf_2000 sample; currently 7/7, 5/5, 12/12)

Sources: files/ecomm/data/fashion_product_images/styles_details.parquet
+ data/ecomm_sf_2000/data_full.csv (universe ids). Deterministic census
(duckdb), no LLM, no run outputs. Idempotent.

Usage: python3 data/ecomm_sf_2000/build_gt.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import duckdb
import pandas as pd

HERE = Path(__file__).resolve().parent
GT = HERE / "ground_truth"
PARQUET = (HERE.parents[1] / "files/ecomm/data/fashion_product_images"
           / "styles_details.parquet")

# official ground_truth SQL, verbatim from files/ecomm/queries/q*.toml
FORMULAS = {
    "Q1": """
        select id from read_parquet('{p}')
        where true
          and articleType.typeName = 'Backpacks'
          and brandName = 'Reebok'
    """,
    "Q2": """
        select id from read_parquet('{p}')
        where true
          and articleType.typeName = 'Sports Shoes'
          and list_has_all([baseColour, colour1, colour2],
                           ['Yellow', 'Silver'])
    """,
    "Q13": """
        SELECT id FROM read_parquet('{p}')
        WHERE true
          AND gender = 'Men'
          AND usage = 'Sports'
          AND articleType.typeName = 'Tshirts'
          AND (baseColour = 'Blue' OR baseColour = 'Black')
          AND 'Short Sleeves' IN articleAttributes['Sleeve Length']
          AND 'Round Neck' IN articleAttributes['Neck']
          AND 'Striped' IN articleAttributes['Pattern']
          AND season <> 'Winter'
    """,
}

universe_ids = set(
    pd.read_csv(HERE / "data_full.csv", dtype=str,
                keep_default_na=False).id)

fails = 0
print(f"universe rows: {len(universe_ids)}")
for q, sql in FORMULAS.items():
    full = duckdb.sql(sql.format(p=PARQUET)).df()
    full_ids = set(full.id.astype(str))
    mine = full_ids & universe_ids
    cur = set(pd.read_csv(GT / f"{q}.csv", dtype=str).id)
    ok = mine == cur
    fails += not ok
    print(f"{q:3}: full-census {len(full_ids)} | in-universe {len(mine)} | "
          f"GT file {len(cur)} | "
          f"{'EQUAL' if ok else 'DIFF'}"
          f"{' (retained all positives)' if mine == full_ids else ''}")
    if not ok:  # regenerate from the census, never from a run
        pd.DataFrame({"id": sorted(mine)}).to_csv(
            GT / f"{q}.csv", index=False)
        print(f"   REGENERATED {q}.csv from census")
print(f"RESULT: {'ALL SELF-CHECKS EQUAL' if fails == 0 else f'{fails} FAILURES'}")
sys.exit(1 if fails else 0)
