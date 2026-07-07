from fastapi.testclient import TestClient
from oncotwin.api.main import app

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_create_forecast_and_assimilate_flow():
    # create
    r = client.post("/twins", json={"patient_id": "PT-API", "features": {}})
    assert r.status_code == 200
    belief0 = r.json()
    assert belief0["version"] == 0

    # forecast
    plan = {"name": "chemo", "courses": [
        {"kind": "chemo", "start_day": 30, "end_day": 200, "intensity": 1.0}]}
    r = client.post("/twins/PT-API/forecast", json={"plan": plan, "horizon_days": 200})
    assert r.status_code == 200
    fc = r.json()
    assert len(fc["t"]) == len(fc["median"]) > 0

    # assimilate
    r = client.post("/twins/PT-API/measurements",
                    json=[{"time_days": 60, "volume_cm3": 120.0}])
    assert r.status_code == 200
    belief1 = r.json()
    assert belief1["version"] == 1
    assert belief1["n_observations"] == 1


def test_missing_twin_404():
    r = client.get("/twins/does-not-exist")
    assert r.status_code == 404
