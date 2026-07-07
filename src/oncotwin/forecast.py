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
) -> Forecast:
    traj = simulate_ensemble(v0, ensemble.particles, plan, t_eval)
    w = ensemble.weights
    # Weighted quantiles across particles at each time point.
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
