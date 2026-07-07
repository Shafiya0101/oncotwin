"""OncoTwin API — a stateful service around the twin engine.

Endpoints:
    GET  /health
    POST /twins                        create a twin
    GET  /twins                        list twin ids
    GET  /twins/{id}                   belief for a twin
    POST /twins/{id}/forecast          forecast one plan
    POST /twins/{id}/counterfactuals   forecast several plans
    POST /twins/{id}/measurements      assimilate scans, recalibrate

Run:  uvicorn oncotwin.api.main:app --reload
"""
from __future__ import annotations

import uuid
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from ..engine import OncoTwinEngine
from ..store import JsonTwinStore
from ..forecast import Forecast
from .schemas import (
    CreateTwinIn, ForecastIn, CounterfactualsIn, MeasurementIn,
    BeliefOut, ForecastOut,
)

app = FastAPI(title="OncoTwin API", version="0.1.0",
              description="A living, probabilistic cancer patient digital twin.")

# Dev CORS: lets the static web UI (file:// or another port) call the API.
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

engine = OncoTwinEngine()
store = JsonTwinStore()


def _forecast_out(f: Forecast) -> ForecastOut:
    return ForecastOut(
        plan_name=f.plan_name, t=f.t.tolist(),
        median=f.median.tolist(), lower=f.lower.tolist(), upper=f.upper.tolist())


def _require(pid: str):
    twin = store.get(pid)
    if twin is None:
        raise HTTPException(404, f"twin {pid} not found")
    return twin


@app.get("/health")
def health():
    return {"status": "ok", "twins": len(store.list_ids())}


@app.post("/twins", response_model=BeliefOut)
def create_twin(body: CreateTwinIn):
    pid = body.patient_id or f"PT-{uuid.uuid4().hex[:8]}"
    twin = engine.create_twin(pid, body.features.to_domain(), n_particles=body.n_particles)
    store.save(twin)
    return BeliefOut(**engine.explain(twin))


@app.get("/twins")
def list_twins():
    return {"twins": store.list_ids()}


@app.get("/twins/{pid}", response_model=BeliefOut)
def get_twin(pid: str):
    return BeliefOut(**engine.explain(_require(pid)))


@app.post("/twins/{pid}/forecast", response_model=ForecastOut)
def forecast_twin(pid: str, body: ForecastIn):
    twin = _require(pid)
    f = engine.forecast(twin, body.plan.to_domain(), body.horizon_days, body.step)
    return _forecast_out(f)


@app.post("/twins/{pid}/counterfactuals")
def counterfactuals(pid: str, body: CounterfactualsIn):
    twin = _require(pid)
    plans = [p.to_domain() for p in body.plans]
    results = engine.simulate_counterfactuals(twin, plans, body.horizon_days)
    return {name: _forecast_out(f).model_dump() for name, f in results.items()}


@app.post("/twins/{pid}/measurements", response_model=BeliefOut)
def add_measurements(pid: str, body: list[MeasurementIn]):
    twin = _require(pid)
    engine.assimilate(twin, [m.to_domain() for m in body])
    store.save(twin)
    return BeliefOut(**engine.explain(twin))
