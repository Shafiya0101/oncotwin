"""Data adapters: how patient records enter the system.

`SyntheticCohort` generates fully-labelled virtual patients (features + hidden
true parameters + noisy longitudinal scans). It exists so the training and
validation pipelines have something to run on today, and so tests are
deterministic.

`CohortAdapter` is the interface a production loader implements. A real
TCIA + TCGA/cBioPortal adapter would: pull segmented tumor volumes and radiomic
features from imaging, join molecular markers and clinical staging, and yield
`PatientRecord`s with the same shape. Nothing downstream changes.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Protocol
import numpy as np

from ..domain import PatientFeatures, TumorMeasurement
from ..growth import simulate
from ..domain import TreatmentPlan


@dataclass
class PatientRecord:
    patient_id: str
    features: PatientFeatures
    measurements: list[TumorMeasurement]
    true_params: np.ndarray | None = None   # known only for synthetic patients


class CohortAdapter(Protocol):
    def records(self) -> Iterator[PatientRecord]: ...


class SyntheticCohort:
    """Generate virtual patients with known ground truth for training/validation."""

    def __init__(self, n: int = 50, plan: TreatmentPlan | None = None,
                 scan_days=(60, 120, 180, 240), seed: int = 0):
        self.n = n
        self.plan = plan
        self.scan_days = scan_days
        self.rng = np.random.default_rng(seed)

    def _draw_features(self) -> PatientFeatures:
        r = self.rng
        return PatientFeatures(
            age=int(r.integers(45, 82)),
            stage=int(r.integers(1, 5)),
            histology="adenocarcinoma",
            baseline_volume_cm3=float(r.uniform(15, 80)),
            ki67=float(r.uniform(0.1, 0.6)),
            egfr_mutation=bool(r.random() < 0.3),
            radiomic_heterogeneity=float(r.uniform(0.2, 0.9)),
        )

    def _draw_true_params(self, f: PatientFeatures) -> np.ndarray:
        r = self.rng
        growth = 0.010 + 0.006 * f.ki67 + 0.003 * (f.stage - 1) + r.normal(0, 0.003)
        K = max(f.baseline_volume_cm3 * r.uniform(8, 20), 150)
        chemo = max(0.02 + 0.03 * f.egfr_mutation + r.normal(0, 0.02), 1e-3)
        radio = max(0.05 - 0.02 * f.radiomic_heterogeneity + r.normal(0, 0.01), 1e-3)
        return np.array([max(growth, 1e-3), K, chemo, radio])

    def records(self) -> Iterator[PatientRecord]:
        for i in range(self.n):
            f = self._draw_features()
            p = self._draw_true_params(f)
            days = np.array(self.scan_days, dtype=float)
            truth = simulate(f.baseline_volume_cm3, p, self.plan, days)
            meas = [
                TumorMeasurement(float(d), float(max(1.0, v + self.rng.normal(0, 2.0))),
                                 "imaging", 2.0)
                for d, v in zip(days, truth)
            ]
            yield PatientRecord(f"SYN-{i:04d}", f, meas, p)
