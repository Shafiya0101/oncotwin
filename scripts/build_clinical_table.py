#!/usr/bin/env python
"""Turn the TCIA NSCLC-Radiomics clinical CSV (Lung1) into data/real/clinical.csv.

Usage:
  python scripts/build_clinical_table.py path/to/NSCLC-Radiomics-Lung1.clinical.csv

Output columns: PatientID, time_days, event, age, sex, t_stage, n_stage,
m_stage, overall_stage, histology.  (Survival time on TCIA is in days from the
start of treatment; event 1 = dead.)  Column names are matched loosely, and the
script prints what it found so you can check it against your file.
"""
import argparse
import re
import sys
from pathlib import Path

import pandas as pd

ALIASES = {
    "PatientID": ["patientid", "patient", "id", "subjectid"],
    "time_days": ["survivaltime", "survivaltimedays", "time"],
    "event": ["deadstatusevent", "deadstatus", "event", "dead", "status"],
    "age": ["age"],
    "sex": ["gender", "sex"],
    "t_stage": ["clinicaltstage", "tstage"],
    "n_stage": ["clinicalnstage", "nstage"],
    "m_stage": ["clinicalmstage", "mstage"],
    "overall_stage": ["overallstage", "stage"],
    "histology": ["histology"],
}


def norm(s):
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--out", default="data/real/clinical.csv")
    a = ap.parse_args()

    df = pd.read_csv(a.csv)
    print("Columns in your file:", list(df.columns))
    lookup = {norm(c): c for c in df.columns}
    out, missing = pd.DataFrame(), []
    for std, options in ALIASES.items():
        src = next((lookup[o] for o in options if o in lookup), None)
        if src is None:
            missing.append(std)
            continue
        out[std] = df[src]
        print(f"  {std:14s} <- {src}")
    must = {"PatientID", "time_days", "event"}
    if must & set(missing):
        sys.exit(f"Cannot continue, missing required columns: {sorted(must & set(missing))}")
    if missing:
        print("WARNING, not found (will be left out):", missing)

    out["time_days"] = pd.to_numeric(out["time_days"], errors="coerce")
    out["event"] = pd.to_numeric(out["event"], errors="coerce")
    if "age" in out:
        out["age"] = pd.to_numeric(out["age"], errors="coerce")
    for c in ["sex", "t_stage", "n_stage", "m_stage", "overall_stage", "histology"]:
        if c in out:
            out[c] = out[c].astype("string").str.strip()

    before = len(out)
    out = out.drop_duplicates("PatientID")
    bad = out["time_days"].isna() | out["event"].isna() | (out["time_days"] <= 0)
    print(f"\nPatients: {before} rows, {before - len(out)} duplicates dropped, "
          f"{int(bad.sum())} with missing/invalid survival (kept in file, dropped by the analysis)")
    ok = out[~bad]
    print(f"Events: {int(ok['event'].sum())} of {len(ok)} ({ok['event'].mean():.0%}); "
          f"median observed time {ok['time_days'].median():.0f} days")
    print("Missing values per column:\n", out.isna().sum().to_string())
    print("\nCheck: event should be 0/1 with 1 = dead:", sorted(out["event"].dropna().unique())[:5])

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(a.out, index=False)
    print(f"\nWrote {a.out}")


if __name__ == "__main__":
    main()
