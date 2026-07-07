"""Load the real NSCLC-Radiomics (Lung1) clinical data.

Reads TCIA's Lung1.clinical.csv into the twin's own types. Age, stage, and
histology are real; baseline tumor volume, Ki-67 and imaging heterogeneity are
NOT in the clinical file (they come from the scans), so they are filled with
cohort defaults here and overwritten once real imaging is ingested. Survival
time and the death/censoring flag are the real outcomes used for validation.

Columns expected (from TCIA):
  PatientID, age, ..., Overall.Stage, Histology, gender, Survival.time, deadstatus.event
"""
from __future__ import annotations

import csv
from dataclasses import dataclass

from ..domain import PatientFeatures

_STAGE = {"i": 1, "ii": 2, "iii": 3, "iiia": 3, "iiib": 3, "iv": 4}


@dataclass
class ClinicalRecord:
    patient_id: str
    features: PatientFeatures
    survival_days: float
    event: int                 # 1 = death observed, 0 = censored (still alive)


def _parse_stage(s: str | None):
    return _STAGE.get((s or "").strip().lower())


def load_lung1_clinical(path: str, default_volume=40.0, default_ki67=0.30,
                        default_het=0.50) -> list[ClinicalRecord]:
    records: list[ClinicalRecord] = []
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            try:
                age = int(float(row["age"]))
                survival = float(row["Survival.time"])
                event = int(float(row["deadstatus.event"]))
            except (ValueError, TypeError, KeyError):
                continue                                   # skip rows with missing data
            stage = _parse_stage(row.get("Overall.Stage"))
            if stage is None:
                continue
            features = PatientFeatures(
                age=age, stage=stage,
                histology=(row.get("Histology") or "nsclc").strip() or "nsclc",
                baseline_volume_cm3=default_volume,        # from imaging later
                ki67=default_ki67, radiomic_heterogeneity=default_het,
            )
            records.append(ClinicalRecord(row["PatientID"], features, survival, event))
    return records
