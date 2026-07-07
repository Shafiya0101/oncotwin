"""Probabilistic forecasting: turn the parameter ensemble into a trajectory
distribution with honest uncertainty bands (the 'weather forecast' view)."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .domain import ParameterEnsemble, TreatmentPlan
from .growth import simulate_ensemble


@dataclass
class Forecast:
    plan_name: str
    t: np.ndarray                    # days
    median: np.ndarray
    lower: np.ndarray                # 5th percentile
    upper: np.ndarray                # 95th percentile
    trajectories: np.ndarray         # full ensemble (N, T), kept for downstream use

    def summary(self, horizon_day: float | None = None) -> dict:
        i = -1 if horizon_day is None else int(np.argmin(np.abs(self.t - horizon_day)))
        return {
            "plan": self.plan_name,
            "day": float(self.t[i]),
            "volume_median": round(float(self.median[i]), 1),
            "volume_ci90": (round(float(self.lower[i]), 1), round(float(self.upper[i]), 1)),
        }


def forecast(
    v0: float,
    ensemble: ParameterEnsemble,
    plan: TreatmentPlan | None,
    t_eval: np.ndarray,
    process_noise: float = 0.0,
) -> Forecast:
    """Probabilistic forecast.

    `process_noise` widens the predictive interval with the forecast horizon
    (uncertainty grows as sqrt(time), like any real forecast), representing model
    error and biological drift the parameter ensemble alone doesn't capture.
    Without it the bands reflect only parameter spread and are overconfident —
    see analysis/run_analysis.py, where turning this on moves 90% coverage from
    ~35% toward nominal.
    """
    t_eval = np.asarray(t_eval, dtype=float)
    traj = simulate_ensemble(v0, ensemble.particles, plan, t_eval)

    if process_noise > 0.0:
        rng = np.random.default_rng()
        sigma_t = process_noise * np.sqrt(np.maximum(t_eval, 0.0))     # grows with horizon
        noise = rng.normal(0.0, 1.0, traj.shape) * sigma_t[None, :]
        traj = traj * np.exp(noise - 0.5 * sigma_t[None, :] ** 2)      # median-preserving widening

    w = ensemble.weights
    med = _wquantile(traj, w, 0.50)
    lo = _wquantile(traj, w, 0.05)
    hi = _wquantile(traj, w, 0.95)
    return Forecast(plan.name if plan else "no treatment", t_eval, med, lo, hi, traj)


def _wquantile(traj: np.ndarray, w: np.ndarray, q: float) -> np.ndarray:
    order = np.argsort(traj, axis=0)
    out = np.empty(traj.shape[1])
    for j in range(traj.shape[1]):
        idx = order[:, j]
        cdf = np.cumsum(w[idx])
        k = np.searchsorted(cdf, q)
        out[j] = traj[idx[min(k, len(idx) - 1)], j]
    return out
