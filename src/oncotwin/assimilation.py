"""Data assimilation: the recalibration loop that makes this a twin, not a
one-shot predictor.

When new tumor measurements arrive, we update our belief over the patient's
parameters by weighting each particle by how well it reproduces the observed
history (a particle filter / Bayesian update). The forecast then shifts and
tightens around the patient's actual behaviour. This is the same principle
weather models use to stay locked onto reality as new observations stream in.
"""
from __future__ import annotations

import numpy as np

from .domain import ParameterEnsemble, TreatmentPlan, TumorMeasurement
from .growth import simulate


def assimilate(
    v0: float,
    ensemble: ParameterEnsemble,
    plan: TreatmentPlan | None,
    measurements: list[TumorMeasurement],
) -> ParameterEnsemble:
    """Return a posterior ensemble conditioned on the observed measurements."""
    if not measurements:
        return ensemble

    times = np.array([m.time_days for m in measurements])
    obs = np.array([m.volume_cm3 for m in measurements])
    sd = np.array([m.noise_sd for m in measurements])

    log_w = np.log(ensemble.weights + 1e-300)
    for i, p in enumerate(ensemble.particles):
        pred = simulate(v0, p, plan, times)
        # Gaussian observation likelihood in log space.
        log_w[i] += -0.5 * np.sum(((pred - obs) / sd) ** 2)

    log_w -= log_w.max()
    w = np.exp(log_w)
    w /= w.sum()

    posterior = ParameterEnsemble(ensemble.particles.copy(), w)
    # Resample so degenerate weights don't collapse future updates, and add a
    # small jitter (regularized particle filter) to keep diversity.
    posterior = posterior.resampled()
    jitter = np.random.default_rng().normal(
        0.0, 0.02 * posterior.particles.std(axis=0), posterior.particles.shape
    )
    posterior.particles = (posterior.particles + jitter).clip(1e-5, None)
    return posterior
