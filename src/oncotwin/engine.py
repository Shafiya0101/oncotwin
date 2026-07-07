"""OncoTwinEngine: the orchestration layer the API and UI call into.

One clean surface over the whole system:
    create_twin -> forecast / counterfactuals -> assimilate (recalibrate) -> repeat
"""
from __future__ import annotations

import numpy as np

from .domain import (
    PatientFeatures, PatientTwin, TreatmentPlan, TumorMeasurement,
)
from .parameters import ParameterEstimator, HeuristicEstimator
from .forecast import Forecast, forecast


class OncoTwinEngine:
    def __init__(self, estimator: ParameterEstimator | None = None):
        # Swap HeuristicEstimator for a trained model in production.
        self.estimator = estimator or HeuristicEstimator(seed=7)

    def create_twin(
        self, patient_id: str, features: PatientFeatures,
        plan: TreatmentPlan | None = None, n_particles: int = 400,
    ) -> PatientTwin:
        ensemble = self.estimator.estimate(features, n_particles)
        return PatientTwin(patient_id, features, ensemble, plan=plan)

    def forecast(
        self, twin: PatientTwin, plan: TreatmentPlan | None,
        horizon_days: float = 365, step: float = 5.0,
    ) -> Forecast:
        t = np.arange(0.0, horizon_days + step, step)
        return forecast(twin.features.baseline_volume_cm3, twin.parameters, plan, t)

    def simulate_counterfactuals(
        self, twin: PatientTwin, plans: list[TreatmentPlan],
        horizon_days: float = 365,
    ) -> dict[str, Forecast]:
        return {p.name: self.forecast(twin, p, horizon_days) for p in plans}

    def assimilate(
        self, twin: PatientTwin, new_measurements: list[TumorMeasurement],
        plan: TreatmentPlan | None = None,
    ) -> PatientTwin:
        """Recalibrate the twin against reality. Bumps the version stamp."""
        from .assimilation import assimilate as _assimilate
        for m in new_measurements:
            twin.add_measurement(m)
        active_plan = plan or twin.plan
        twin.parameters = _assimilate(
            twin.features.baseline_volume_cm3, twin.parameters,
            active_plan, twin.measurements,
        )
        twin.version += 1
        return twin

    @staticmethod
    def explain(twin: PatientTwin) -> dict:
        """Report the twin's current belief in human-readable terms (the seed of
        the explainability layer a clinician would see)."""
        p = twin.parameters.mean()
        doubling = np.log(2) / p["r"] if p["r"] > 0 else float("inf")
        return {
            "patient_id": twin.patient_id,
            "version": twin.version,
            "n_observations": len(twin.measurements),
            "growth_rate_per_day": round(float(p["r"]), 4),
            "implied_doubling_time_days": round(float(doubling), 0),
            "chemo_sensitivity": round(float(p["chemo_sensitivity"]), 4),
            "radio_sensitivity": round(float(p["radio_sensitivity"]), 4),
        }
