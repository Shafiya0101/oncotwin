"""Mechanistic layer: tumor volume dynamics under treatment.

We use Gompertzian growth, the classic and repeatedly validated model for solid
tumor kinetics, combined with a log-kill therapy term (Skipper/Norton-Simon
tradition). This is deliberately interpretable: every parameter has a biological
meaning, so a clinician can reason about it and the ML layer only has to
estimate a handful of numbers per patient rather than a black-box trajectory.

    dV/dt = r * V * ln(K / V)  -  kill(t) * V

where kill(t) sums the active therapy courses at time t.
"""
from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp

from .domain import TreatmentPlan, TreatmentKind

_V_FLOOR = 1e-3       # cm^3, prevents ln(K/V) blow-up as tumor shrinks
_AB_RATIO = 10.0      # Gy, tumor alpha/beta ratio for the linear-quadratic model
_RESISTANCE_RATE = 0.010  # per day; chemo effect decays as resistance emerges


def _kill_rate(t: float, params: np.ndarray, plan: TreatmentPlan | None) -> float:
    """Instantaneous therapy-induced death rate at time t (per day).

    Two pieces of real oncology beyond a flat log-kill:
      - Acquired chemo resistance: sensitivity decays with time on a course, so
        a tumor that responds early can escape later (a common clinical reality).
      - Linear-quadratic radiotherapy: cell kill scales as d + d^2/(alpha/beta),
        the standard radiobiology model, instead of a flat per-dose term.
    """
    if plan is None:
        return 0.0
    chemo_sens, radio_sens = params[2], params[3]
    rate = 0.0
    for course in plan.active_courses(t):
        if course.kind is TreatmentKind.CHEMO:
            resistance = np.exp(-_RESISTANCE_RATE * max(0.0, t - course.start_day))
            rate += chemo_sens * course.intensity * resistance
        elif course.kind is TreatmentKind.RADIO:
            d = course.intensity
            rate += radio_sens * (d + d * d / _AB_RATIO)
    return rate


def simulate(
    v0: float,
    params: np.ndarray,
    plan: TreatmentPlan | None,
    t_eval: np.ndarray,
) -> np.ndarray:
    """Integrate the ODE for one parameter set. Returns V at each t_eval point.

    Integration always starts from V=v0 at t=0, regardless of the requested
    sample times (which may be a single point, may omit 0, and need not be
    contiguous) — so this works for both forecasting and assimilation.
    """
    r, K = params[0], params[1]
    t_eval = np.asarray(t_eval, dtype=float)
    tf = float(t_eval[-1])
    if tf <= 0.0:                                   # all requested times at/ before baseline
        return np.full(len(t_eval), v0, dtype=float)

    def rhs(t, y):
        v = max(y[0], _V_FLOOR)
        growth = r * v * np.log(K / v)
        death = _kill_rate(t, params, plan) * v
        return [growth - death]

    sol = solve_ivp(
        rhs, (0.0, tf), [v0], t_eval=t_eval, method="LSODA", rtol=1e-6, atol=1e-8,
    )
    y = sol.y[0] if sol.y.size else np.array([])
    if len(y) != len(t_eval):                       # solver returned a different grid
        dense = solve_ivp(rhs, (0.0, tf), [v0], method="LSODA", rtol=1e-6, atol=1e-8)
        y = np.interp(t_eval, dense.t, dense.y[0])
    return np.clip(y, _V_FLOOR, None)


def simulate_ensemble(
    v0: float,
    particles: np.ndarray,
    plan: TreatmentPlan | None,
    t_eval: np.ndarray,
) -> np.ndarray:
    """Simulate every particle. Returns matrix of shape (n_particles, n_times)."""
    out = np.empty((len(particles), len(t_eval)))
    for i, p in enumerate(particles):
        out[i] = simulate(v0, p, plan, t_eval)
    return out
