"""Domain model for the OncoTwin cancer patient digital twin.

These types define the *shape* of a patient twin independently of how the data
is sourced. Today the values come from a synthetic adapter; in production the
same objects are populated from TCIA imaging features, TCGA/cBioPortal
molecular data, and an EHR clinical feed. Nothing downstream needs to change
when the data source is swapped.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import numpy as np


class TreatmentKind(str, Enum):
    NONE = "none"
    CHEMO = "chemo"
    RADIO = "radiotherapy"


@dataclass(frozen=True)
class TreatmentCourse:
    """A single therapy window applied to the tumor.

    intensity is a unitless relative dose (1.0 = standard of care). It scales
    the therapy-induced kill term in the growth model.
    """
    kind: TreatmentKind
    start_day: float
    end_day: float
    intensity: float = 1.0

    def is_active(self, t: float) -> bool:
        return self.start_day <= t < self.end_day


@dataclass
class TreatmentPlan:
    """An ordered set of therapy courses. Counterfactuals are just different plans."""
    name: str
    courses: list[TreatmentCourse] = field(default_factory=list)

    def active_courses(self, t: float) -> list[TreatmentCourse]:
        return [c for c in self.courses if c.is_active(t)]


@dataclass(frozen=True)
class TumorMeasurement:
    """A real-world observation of tumor burden used to recalibrate the twin."""
    time_days: float
    volume_cm3: float
    source: str = "imaging"          # imaging | pathology | biomarker
    noise_sd: float = 2.0            # measurement uncertainty (cm^3)


@dataclass
class PatientFeatures:
    """Multimodal inputs that drive the patient-specific parameter estimate.

    In production these are the fused outputs of the imaging + clinical + omics
    pipeline. Here they are the interface the ML parameter estimator consumes.
    """
    age: int
    stage: int                       # 1..4
    histology: str                   # e.g. "adenocarcinoma"
    baseline_volume_cm3: float       # from segmentation of the baseline scan
    ki67: float = 0.2                # proliferation index (0..1), imaging/path proxy
    egfr_mutation: bool = False      # example molecular driver
    radiomic_heterogeneity: float = 0.5  # 0..1 texture-based aggressiveness proxy


@dataclass
class ParameterEnsemble:
    """A weighted particle cloud over the mechanistic model parameters.

    Rows are particles; columns are (r, K, chemo_sensitivity, radio_sensitivity).
    Carrying the full ensemble (not just a mean) is what makes forecasts
    probabilistic and makes Bayesian recalibration possible.
    """
    particles: np.ndarray            # shape (N, 4)
    weights: np.ndarray              # shape (N,), sums to 1

    COLS = ("r", "K", "chemo_sensitivity", "radio_sensitivity")

    def mean(self) -> dict[str, float]:
        m = np.average(self.particles, axis=0, weights=self.weights)
        return dict(zip(self.COLS, m))

    def resampled(self) -> "ParameterEnsemble":
        """Systematic resampling -> equal-weight particles (post-update)."""
        n = len(self.weights)
        positions = (np.arange(n) + np.random.random()) / n
        idx = np.searchsorted(np.cumsum(self.weights), positions)
        idx = np.clip(idx, 0, n - 1)
        return ParameterEnsemble(self.particles[idx].copy(), np.full(n, 1.0 / n))


@dataclass
class PatientTwin:
    """The living twin: identity, inputs, current belief, and observed history."""
    patient_id: str
    features: PatientFeatures
    parameters: ParameterEnsemble
    measurements: list[TumorMeasurement] = field(default_factory=list)
    plan: Optional[TreatmentPlan] = None
    version: int = 0                 # increments on every recalibration

    def add_measurement(self, m: TumorMeasurement) -> None:
        self.measurements.append(m)
