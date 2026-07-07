"""Validation harness.

A twin is only trustworthy if its forecasts are accurate *and* its uncertainty
is honest. This module backtests both on cohorts with known trajectories:

  - assimilate the first `warmup` scans,
  - forecast the held-out later scans,
  - measure absolute error (accuracy) and 90%-interval coverage (calibration).

Well-calibrated 90% bands should contain the truth ~90% of the time. Run this
on a `SyntheticCohort` today; point it at real longitudinal data later.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .engine import OncoTwinEngine
from .forecast import forecast as _forecast
from .data.adapters import PatientRecord
from .domain import TreatmentPlan


@dataclass
class BacktestResult:
    n_patients: int
    n_forecasts: int
    mae_cm3: float
    coverage_90: float          # fraction of held-out points inside the 90% band

    def __str__(self) -> str:
        return (f"patients={self.n_patients} forecasts={self.n_forecasts} "
                f"MAE={self.mae_cm3:.1f}cm3 coverage90={self.coverage_90:.0%}")


def backtest(records: list[PatientRecord], engine: OncoTwinEngine | None = None,
             plan: TreatmentPlan | None = None, warmup: int = 2,
             process_noise: float = 0.0) -> BacktestResult:
    engine = engine or OncoTwinEngine()
    errors, hits, total = [], 0, 0

    for rec in records:
        if len(rec.measurements) <= warmup:
            continue
        twin = engine.create_twin(rec.patient_id, rec.features)
        engine.assimilate(twin, rec.measurements[:warmup], plan)

        held = rec.measurements[warmup:]
        t_eval = np.array([m.time_days for m in held])
        f = _forecast(rec.features.baseline_volume_cm3, twin.parameters, plan, t_eval,
                      process_noise=process_noise)

        for i, m in enumerate(held):
            errors.append(abs(f.median[i] - m.volume_cm3))
            if f.lower[i] <= m.volume_cm3 <= f.upper[i]:
                hits += 1
            total += 1

    return BacktestResult(
        n_patients=len([r for r in records if len(r.measurements) > warmup]),
        n_forecasts=total,
        mae_cm3=float(np.mean(errors)) if errors else float("nan"),
        coverage_90=hits / total if total else float("nan"),
    )
