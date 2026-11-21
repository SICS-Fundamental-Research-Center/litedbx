#!/usr/bin/env python3
"""Build the OFFICIAL-scale movie universe (2026-11-20, R1.D5.1 fleet fix).

files/movie/data/sf_2000/{Movies,Reviews}.csv IS the official SemBench
scale-2000 universe (paper Table 3: Movie Base Scale = 2,000; verified
11-20: official positives 1487 == data/movie/ground_truth/Q1.csv set,
taken_3 positives 14 == Q2.csv). Prior mounts were NOT this universe:
data/movie = 1.54M full sync (13 h sigma scan, wrong scale);
archived/data/movie_sf_2000 = retired custom 2,262-row sample
  (1,735-row overlap; now under archived/).

Transform mirrors archived/data/movie_sf_2000/build_dataset.py's merge
step with NO resampling: Movies --left-join--> Reviews on id, suffix
("", "_review"), reviewText newline-cleaned. De-sentiment policy (10-26
ablation, R3.D13): the GT-defining leak column scoreSentiment is DROPPED
(paper baselines never see it either -- gemini reads reviewText only).

Naming (2026-11-21, user order): the retired custom 2,262-row sample
formerly at data/movie_sf_2000 moved to archived/data/movie_sf_2000
(its own pre-ablation data_full.csv.bak already lived there), and this
official dir took the freed name data/movie_sf_2000 for consistency
with ecomm_sf_2000 / movie_sf_16000 (was: movie_sf2000_official).

Engine/answer GT stays at data/movie/ground_truth (absolute ENGINE_GT in
workloads/scenarios/movie/_common.py, untouched); this dir carries only
copies for the driver's parity check. Idempotent; asserts purity.

Usage: python3 data/movie_sf_2000/build_data.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
SRC = HERE.parents[1] / "files/movie/data/sf_2000"


def main() -> int:
    movies = pd.read_csv(SRC / "Movies.csv")
    reviews = pd.read_csv(SRC / "Reviews.csv")
    assert len(reviews) == 2000, f"official universe must be 2000 rows, got {len(reviews)}"
    assert "scoreSentiment" in reviews.columns
    # official Movies.csv carries 11 duplicate ids (22 rows, remakes/
    # re-releases) which would fan the join out past 2000; the review is
    # the primary entity, so keep-first dedupe preserves the universe
    movies = movies.drop_duplicates(subset="id", keep="first")

    reviews["reviewText"] = (
        reviews["reviewText"]
        .str.replace("\n", " ", regex=False)
        .str.replace("\r", " ", regex=False)
    )
    merged = movies.merge(reviews, on="id", how="left", suffixes=("", "_review"))
    # de-sentiment: leak column removed at build time (no post-hoc edits)
    merged = merged.drop(columns=["scoreSentiment"])
    assert len(merged) == 2000

    out = HERE / "data_full.csv"
    merged.to_csv(out, index=False)

    # purity + GT self-checks
    back = pd.read_csv(out)
    assert "scoreSentiment" not in back.columns, "leak column must be absent"
    assert len(back) == 2000
    gt1 = set(pd.read_csv(HERE / "ground_truth/Q1.csv", dtype=str)["reviewId"])
    q2 = pd.read_csv(SRC / "Reviews.csv", dtype=str)
    positives = set(q2.loc[q2["scoreSentiment"] == "POSITIVE", "reviewId"])
    print(f"rows={len(back)} cols={len(back.columns)} "
          f"leak-col=absent Q1-GT==official-positives: {gt1 == positives}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
