"""Explainability — which inputs drive a twin's forecast.

A local sensitivity analysis: perturb each patient feature by a standardized
amount, re-estimate parameters, re-forecast, and measure how much the one-year
tumor burden moves. Model-agnostic, so it keeps working when the estimator is
swapped for a trained model. This is the seed of the clinician-facing "why"
panel — a forecast you can interrogate rather than just read.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import numpy as np

from .domain import PatientFeatures, TreatmentPlan


# Standardized perturbations, roughly one clinically meaningful step per feature.
_PERTURB = {
    "ki67": 0.15,
    "stage": 1.0,
    "radiomic_heterogeneity": 0.2,
    "baseline_volume_cm3": 15.0,
    "age": 10.0,
}
_LABEL = {
    "ki67": "Ki-67 proliferation",
    "stage": "tumor stage",
    "radiomic_heterogeneity": "imaging heterogeneity",
    "baseline_volume_cm3": "baseline volume",
    "age": "age",
}


@dataclass
class Driver:
    feature: str
    label: str
    effect_cm3: float          # signed change in 1-year median volume per +1 step
    direction: str             # "increases" | "decreases" | "negligible"


def explain_drivers(engine, features: PatientFeatures,
                    plan: TreatmentPlan | None = None, horizon_days: float = 365) -> list[Driver]:
    """Rank features by their effect on the forecast one-year tumor burden."""
    def one_year(f: PatientFeatures) -> float:
        twin = engine.create_twin("_explain", f)
        return engine.forecast(twin, plan, horizon_days).summary(horizon_days)["volume_median"]

    base = one_year(features)
    drivers: list[Driver] = []
    for field, step in _PERTURB.items():
        bumped = replace(features, **{field: getattr(features, field) + step})
        effect = one_year(bumped) - base
        direction = ("increases" if effect > 1 else "decreases" if effect < -1 else "negligible")
        drivers.append(Driver(field, _LABEL[field], round(float(effect), 1), direction))

    drivers.sort(key=lambda d: abs(d.effect_cm3), reverse=True)
    return drivers
