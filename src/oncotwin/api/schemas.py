"""Pydantic schemas — the API's public contract, decoupled from the domain
dataclasses so the wire format can evolve independently."""
from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field

from ..domain import (
    PatientFeatures, TreatmentPlan, TreatmentCourse, TreatmentKind, TumorMeasurement,
)


class FeaturesIn(BaseModel):
    age: int = 63
    stage: int = Field(3, ge=1, le=4)
    histology: str = "adenocarcinoma"
    baseline_volume_cm3: float = 42.0
    ki67: float = Field(0.35, ge=0, le=1)
    egfr_mutation: bool = False
    radiomic_heterogeneity: float = Field(0.6, ge=0, le=1)

    def to_domain(self) -> PatientFeatures:
        return PatientFeatures(**self.model_dump())


class CourseIn(BaseModel):
    kind: TreatmentKind
    start_day: float
    end_day: float
    intensity: float = 1.0


class PlanIn(BaseModel):
    name: str
    courses: list[CourseIn] = []

    def to_domain(self) -> TreatmentPlan:
        return TreatmentPlan(self.name, [
            TreatmentCourse(c.kind, c.start_day, c.end_day, c.intensity) for c in self.courses])


class MeasurementIn(BaseModel):
    time_days: float
    volume_cm3: float
    source: str = "imaging"
    noise_sd: float = 2.0

    def to_domain(self) -> TumorMeasurement:
        return TumorMeasurement(self.time_days, self.volume_cm3, self.source, self.noise_sd)


class CreateTwinIn(BaseModel):
    patient_id: Optional[str] = None
    features: FeaturesIn = FeaturesIn()
    n_particles: int = 400


class ForecastIn(BaseModel):
    plan: PlanIn
    horizon_days: float = 365
    step: float = 5.0


class CounterfactualsIn(BaseModel):
    plans: list[PlanIn]
    horizon_days: float = 365


class BeliefOut(BaseModel):
    patient_id: str
    version: int
    n_observations: int
    growth_rate_per_day: float
    implied_doubling_time_days: float
    chemo_sensitivity: float
    radio_sensitivity: float


class ForecastOut(BaseModel):
    plan_name: str
    t: list[float]
    median: list[float]
    lower: list[float]
    upper: list[float]
