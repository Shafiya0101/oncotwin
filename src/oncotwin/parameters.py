"""The ML layer: patient features -> prior belief over mechanistic parameters.

THIS IS THE MAIN MODEL SWAP POINT. Today `HeuristicEstimator` maps fused
multimodal features to parameter distributions using transparent clinical
priors. In production you replace it with a model trained on real cohorts
(TCIA imaging + TCGA molecular + outcomes) that regresses features -> parameter
posteriors. Everything downstream (forecasting, recalibration, API, UI) is
agnostic to which estimator produced the ensemble, because both return a
`ParameterEnsemble`.
"""
from __future__ import annotations

from typing import Protocol
import numpy as np

from .domain import PatientFeatures, ParameterEnsemble


class ParameterEstimator(Protocol):
    def estimate(self, features: PatientFeatures, n_particles: int = 400) -> ParameterEnsemble:
        ...


class HeuristicEstimator:
    """Interpretable priors so the whole system runs end-to-end without a
    trained model. Each rule encodes a widely accepted direction of effect;
    the learned model will refine the magnitudes and correlations."""

    def __init__(self, seed: int | None = None):
        self._rng = np.random.default_rng(seed)

    def estimate(self, f: PatientFeatures, n_particles: int = 400) -> ParameterEnsemble:
        rng = self._rng

        # Growth rate r: higher stage, higher Ki-67, more radiomic heterogeneity
        # -> faster growth. Base ~0.012/day (volume doubling ~ months).
        r_mean = 0.010 + 0.004 * f.ki67 + 0.003 * (f.stage - 1) + 0.004 * f.radiomic_heterogeneity
        r = rng.normal(r_mean, 0.18 * r_mean, n_particles).clip(1e-4, None)

        # Carrying capacity K: scales with baseline burden; wide uncertainty.
        k_mean = max(f.baseline_volume_cm3 * 14.0, 200.0)
        K = rng.lognormal(np.log(k_mean), 0.45, n_particles)

        # Chemo sensitivity: EGFR-mutant adenocarcinoma tends to respond better
        # to targeted/standard therapy in this toy prior; higher Ki-67 (more
        # proliferation) is also more chemo-vulnerable. Tuned so standard dose
        # can overcome early Gompertz growth (net tumor shrinkage on-therapy).
        chemo_mean = 0.055 + 0.020 * f.egfr_mutation + 0.030 * f.ki67
        chemo = rng.normal(chemo_mean, 0.35 * chemo_mean, n_particles).clip(1e-4, None)

        # Radiotherapy sensitivity: modestly higher for less heterogeneous tumors.
        radio_mean = 0.060 - 0.020 * f.radiomic_heterogeneity
        radio = rng.normal(radio_mean, 0.35 * abs(radio_mean), n_particles).clip(1e-4, None)

        particles = np.column_stack([r, K, chemo, radio])
        weights = np.full(n_particles, 1.0 / n_particles)
        return ParameterEnsemble(particles, weights)
