#!/usr/bin/env python3
"""Per-tuple ground-truth builder for the medical dataset (2026-11-20).

User ruling: the official per-query GT files Q1-Q10.csv in ground_truth/
are user assets and are NEVER written by this script. What it
(re)generates is only the tuple-level rowsets the engine layer needs on
top of them:

  aggregate queries -- official GT is scalar/group, no rowset exists:
    Q4__tuple.csv  text_sick & text_diagnosis == "acne"        -> 50
    Q5__tuple.csv  smoking Current & audio-sick & xray-sick    -> 1
  UCQ branches -- each branch runs as a standalone engine query
  (engine-name keyed, same names as workloads/scenarios/medical/q6.py
  / q7.py branch dicts):
    Q7__txt / Q7__xray / Q7__aud / Q7__skin
    Q6__xray / Q6__aud  (x-ray/audio-normal AND >=2 modalities
    available within the official Q6 scope {symptoms, x-ray, audio}
    -- mirrors the DNF Sigma in q6.py; skin never counts, and the
    dropped single-modality patients can never satisfy Q6)

Sources: data_full.csv (patient_id, age, smoking_history) left-joined
1:1 with the answer-key label table
files/medical/data/patient_data_with_labels.csv -- the table the engine
never sees; GT derives from it or the answers would leak. Deterministic
census, no LLM, no run outputs. Idempotent.

Self-checks against the UNTOUCHED official files (exit 1 on mismatch):
    mean(age of Q4__tuple)        == Q4.csv scalar   (64.06)
    len(Q5__tuple)                == Q5.csv count    (1)
    union(Q7 branches)            == Q7.csv patient set (7594)
    text_sick & union(Q6 normal)        == Q6.csv patient set (286)

Usage: python3 data/medical/build_gt.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
GT = HERE / "ground_truth"
LABELS = HERE.parents[1] / "files/medical/data/patient_data_with_labels.csv"

# ---------------------------------------------------------------- sources
base = pd.read_csv(HERE / "data_full.csv", dtype={"patient_id": str},
                    keep_default_na=False)  # FK empties stay "" not NaN
lab = pd.read_csv(LABELS, dtype={"patient_id": str})
m = base[["patient_id", "age", "smoking_history",
            "symptoms", "image_path_xray", "audio_path", "image_path"]].merge(
    lab[["patient_id", "text_diagnosis", "audio_diagnosis",
         "x_ray_diagnosis", "skin_cancer_diagnosis", "is_sick"]],
    on="patient_id", how="left", validate="one_to_one",
)

# ----------------------------------------------------------------- masks
text_sick = m.text_diagnosis.notna() & (m.text_diagnosis != "none")
aud_sick = (m.audio_diagnosis.notna()
            & ~m.audio_diagnosis.isin(["none", "normal"]))
xray_sick = (m.x_ray_diagnosis.notna()
             & ~m.x_ray_diagnosis.isin(["none", "00_normal"]))

# engine-branch rowsets (UCQ per-branch-per-tuple) + agg per-tuple sets
SETS = {
    "Q7__txt": text_sick,
    "Q7__xray": xray_sick,
    "Q7__aud": aud_sick,
    "Q7__skin": m.skin_cancer_diagnosis == "malignant",
    "Q6__xray": (m.x_ray_diagnosis == "00_normal")
    & ((m.audio_path != "") | (m.symptoms != "")),
    "Q6__aud": (m.audio_diagnosis == "normal")
    & ((m.image_path_xray != "") | (m.symptoms != "")),
    "Q4__tuple": text_sick & (m.text_diagnosis == "acne"),
    "Q5__tuple": (m.smoking_history == "Current") & aud_sick & xray_sick,
}
OFFICIAL = {f"Q{i}.csv" for i in range(1, 11)}
assert not OFFICIAL & {f"{n}.csv" for n in SETS}, "official-file collision"


def ids(mask: pd.Series) -> list[str]:
    return sorted(m.loc[mask.fillna(False), "patient_id"])


for name, mask in SETS.items():
    pd.DataFrame({"patient_id": ids(mask)}).to_csv(
        GT / f"{name}.csv", index=False)

# ---------------------------------------------------------- self-checks
def gt_ids(name: str) -> set[str]:
    return set(pd.read_csv(GT / f"{name}.csv", dtype=str).patient_id)


q4_scalar = float(pd.read_csv(GT / "Q4.csv")["average_acne_age"].iloc[0])
q4_mean = (m.set_index("patient_id")
           .loc[ids(SETS["Q4__tuple"]), "age"].astype(float).mean())
q5_count = int(pd.read_csv(GT / "Q5.csv")["count"].iloc[0])
q7_union = (gt_ids("Q7__txt") | gt_ids("Q7__xray")
            | gt_ids("Q7__aud") | gt_ids("Q7__skin"))
q6n_union = gt_ids("Q6__xray") | gt_ids("Q6__aud")
q6_mine = set(m.loc[text_sick, "patient_id"]) & q6n_union

CHECKS = [
    ("Q4__tuple mean age == Q4.csv scalar", round(float(q4_mean), 2),
     q4_scalar),
    ("Q5__tuple count == Q5.csv count", len(gt_ids("Q5__tuple")), q5_count),
    ("union(Q7 branches) == Q7.csv patient set",
     q7_union == set(pd.read_csv(GT / "Q7.csv", dtype=str).patient_id), True),
    ("text_sick & union(Q6 normal) == Q6.csv patient set",
     q6_mine == set(pd.read_csv(GT / "Q6.csv", dtype=str).patient_id), True),
]

fails = 0
print(f"universe rows: {len(m)}")
for name, mask in SETS.items():
    print(f"  {name:12s} {mask.fillna(False).sum():6d}")
for name, mine, official in CHECKS:
    ok = mine == official
    fails += not ok
    print(f"{'EQUAL' if ok else 'DIFF '} {name}: mine={mine} "
          f"official={official}")
print(f"RESULT: {'ALL SELF-CHECKS EQUAL' if fails == 0 else f'{fails} FAILURES'}")
sys.exit(1 if fails else 0)
