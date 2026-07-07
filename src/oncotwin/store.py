"""Twin state store: persist and version patient twins.

A minimal, dependency-free store so the API is stateful without pulling in a
database. Twins are held in memory and snapshotted to JSON on disk, so state
survives restarts. Swap `JsonTwinStore` for a real database in production; the
`TwinStore` protocol is all the API depends on.
"""
from __future__ import annotations

import json
import os
from typing import Protocol
import numpy as np

from .domain import (
    PatientTwin, PatientFeatures, ParameterEnsemble,
    TreatmentPlan, TreatmentCourse, TreatmentKind, TumorMeasurement,
)


# ----------------------------- (de)serialization ----------------------------- #

def twin_to_dict(t: PatientTwin) -> dict:
    return {
        "patient_id": t.patient_id,
        "version": t.version,
        "features": vars(t.features),
        "parameters": {
            "particles": t.parameters.particles.tolist(),
            "weights": t.parameters.weights.tolist(),
        },
        "measurements": [vars(m) for m in t.measurements],
        "plan": _plan_to_dict(t.plan) if t.plan else None,
    }


def twin_from_dict(d: dict) -> PatientTwin:
    features = PatientFeatures(**d["features"])
    ensemble = ParameterEnsemble(
        np.array(d["parameters"]["particles"], dtype=float),
        np.array(d["parameters"]["weights"], dtype=float),
    )
    measurements = [TumorMeasurement(**m) for m in d.get("measurements", [])]
    plan = _plan_from_dict(d["plan"]) if d.get("plan") else None
    return PatientTwin(d["patient_id"], features, ensemble, measurements, plan, d["version"])


def _plan_to_dict(p: TreatmentPlan) -> dict:
    return {"name": p.name, "courses": [
        {"kind": c.kind.value, "start_day": c.start_day,
         "end_day": c.end_day, "intensity": c.intensity} for c in p.courses]}


def _plan_from_dict(d: dict) -> TreatmentPlan:
    return TreatmentPlan(d["name"], [
        TreatmentCourse(TreatmentKind(c["kind"]), c["start_day"], c["end_day"], c["intensity"])
        for c in d["courses"]])


# --------------------------------- stores ------------------------------------ #

class TwinStore(Protocol):
    def save(self, twin: PatientTwin) -> None: ...
    def get(self, patient_id: str) -> PatientTwin | None: ...
    def list_ids(self) -> list[str]: ...


class JsonTwinStore:
    def __init__(self, root: str = "data/twins"):
        self.root = root
        os.makedirs(root, exist_ok=True)
        self._cache: dict[str, PatientTwin] = {}

    def _path(self, pid: str) -> str:
        return os.path.join(self.root, f"{pid}.json")

    def save(self, twin: PatientTwin) -> None:
        self._cache[twin.patient_id] = twin
        with open(self._path(twin.patient_id), "w") as f:
            json.dump(twin_to_dict(twin), f, indent=2)

    def get(self, patient_id: str) -> PatientTwin | None:
        if patient_id in self._cache:
            return self._cache[patient_id]
        path = self._path(patient_id)
        if not os.path.exists(path):
            return None
        with open(path) as f:
            twin = twin_from_dict(json.load(f))
        self._cache[patient_id] = twin
        return twin

    def list_ids(self) -> list[str]:
        files = [f[:-5] for f in os.listdir(self.root) if f.endswith(".json")]
        return sorted(files)
