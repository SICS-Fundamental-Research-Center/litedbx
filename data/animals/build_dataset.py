#!/usr/bin/env python3
"""Universe builder for the animals dataset (2026-11-21).

Replaces the upstream build_dataset.py (deleted from the working tree,
still in git HEAD) which produced the ORIGINAL data_full.csv: a 5-row
city-grouped table (City + comma-joined ImagePath/Species/StationID)
from files/animals/source_data/sf_200/image_data.csv. That format kept
the Species labels inside the table and was retired on 11-16 when the
R1.D3.1 audio campaign rebuilt the universe one row per media item.

What this script (re)builds, byte-reproducibly from the sf_200 raw
tables (the tables the engine never sees):

    data_full.csv  266 rows = 200 image + 66 audio, block order
                   [image, audio]; columns media_id (img_0..199 /
                   aud_0..65), media_type, City, StationID, image_path,
                   audio_path. The opposite modality's path column is
                   empty so per-modality predicates mask naturally on
                   rows of the other modality. Media paths are remapped
                   /home/jiale/SemBench/ -> repo-relative files/... .

Species (image_data.csv) / Animal (audio_data.csv) label columns are
DELIBERATELY excluded from the base table: labels stay GT-side and are
order-joined back by build_gt.py for the census (same family as the
"derived features never in the base table" rule, 2026-11-19).
config.yaml is hand-maintained documentation, not written here.

Self-checks (exit 1 on mismatch): row counts 200/66/266; media_id
uniqueness + img_/aud_ prefixes; order-join City/StationID/paths equal
the sf_200 source cells; cross-modality masking; every non-empty path
exists on disk; no label columns; rebuilt table == current file
byte-wise (the file is rewritten only on drift, never from a run).

Usage: python3 data/animals/build_dataset.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SF200 = REPO / "files/animals/source_data/sf_200"

fails = 0

img = pd.read_csv(SF200 / "image_data.csv")
aud = pd.read_csv(SF200 / "audio_data.csv")


def remap(p: str) -> str:
    return str(p).replace("/home/jiale/SemBench/", "")


rows = []
for _, r in img.iterrows():
    rows.append({"media_id": f"img_{len(rows)}", "media_type": "image",
                 "City": r.City, "StationID": r.StationID,
                 "image_path": remap(r.ImagePath), "audio_path": ""})
for _, r in aud.iterrows():
    rows.append({"media_id": f"aud_{len(rows)}", "media_type": "audio",
                 "City": r.City, "StationID": r.StationID,
                 "image_path": "", "audio_path": remap(r.AudioPath)})
ani = pd.DataFrame(rows)

# ---------------------------------------------------------- self-checks
is_img, is_aud = ani.media_type == "image", ani.media_type == "audio"
CHECKS = [
    ("image rows", int(is_img.sum()), 200),
    ("audio rows", int(is_aud.sum()), 66),
    ("total rows", len(ani), 266),
    ("unique media_id", ani.media_id.nunique(), len(ani)),
    ("img id prefixes img_", bool(ani.loc[is_img, "media_id"]
                                  .str.startswith("img_").all()), True),
    ("aud id prefixes aud_", bool(ani.loc[is_aud, "media_id"]
                                  .str.startswith("aud_").all()), True),
    ("img rows carry no audio_path", set(ani.loc[is_img, "audio_path"]) == {""}, True),
    ("aud rows carry no image_path", set(ani.loc[is_aud, "image_path"]) == {""}, True),
    ("img City order-join", ani.loc[is_img, "City"].tolist()
     == img.City.tolist(), True),
    ("img StationID order-join", ani.loc[is_img, "StationID"].tolist()
     == img.StationID.tolist(), True),
    ("aud City order-join", ani.loc[is_aud, "City"].tolist()
     == aud.City.tolist(), True),
    ("aud StationID order-join", ani.loc[is_aud, "StationID"].tolist()
     == aud.StationID.tolist(), True),
    ("img path order-join", ani.loc[is_img, "image_path"].tolist()
     == [remap(p) for p in img.ImagePath], True),
    ("aud path order-join", ani.loc[is_aud, "audio_path"].tolist()
     == [remap(p) for p in aud.AudioPath], True),
    ("no label columns", not (set(ani.columns)
                              & {"Species", "Animal", "label_species"}), True),
]
for name, mine, want in CHECKS:
    ok = mine == want
    fails += not ok
    print(f"{'EQUAL' if ok else 'DIFF '} {name}: {mine}")

missing = [p for p in [*ani.image_path, *ani.audio_path]
           if p and not (REPO / p).exists()]
fails += bool(missing)
print(f"{'EQUAL' if not missing else 'DIFF '} media files exist on disk"
      + ("" if not missing else f": {missing[:3]} ({len(missing)} missing)")
      + f" ({len([p for p in [*ani.image_path, *ani.audio_path] if p])} paths)")

# --------------------------------------------- regenerate on drift only
cur_path = HERE / "data_full.csv"
new_csv = ani.to_csv(index=False)
cur = cur_path.read_text() if cur_path.exists() else None
if cur != new_csv:
    cur_path.write_text(new_csv)
    print(f"REGENERATED data_full.csv ({len(ani)} rows)")
    fails += 1
else:
    print(f"EQUAL data_full.csv bytes ({len(new_csv)} chars, no write)")

print(f"RESULT: {'ALL SELF-CHECKS EQUAL' if fails == 0 else f'{fails} FAILURES'}")
sys.exit(1 if fails else 0)
