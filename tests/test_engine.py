import numpy as np
from oncotwin import (
    OncoTwinEngine, PatientFeatures, TreatmentPlan, TreatmentCourse,
    TreatmentKind, TumorMeasurement, SyntheticCohort, backtest,
)


def make_features():
    return PatientFeatures(age=63, stage=3, histology="adenocarcinoma",
                           baseline_volume_cm3=42.0, ki67=0.35)


def test_forecast_bands_are_ordered():
    eng = OncoTwinEngine()
    twin = eng.create_twin("PT-T", make_features())
    f = eng.forecast(twin, None, horizon_days=200)
    assert np.all(f.lower <= f.median + 1e-6)
    assert np.all(f.median <= f.upper + 1e-6)
    assert len(f.t) == len(f.median)


def test_counterfactuals_diverge():
    eng = OncoTwinEngine()
    twin = eng.create_twin("PT-T", make_features())
    none = TreatmentPlan("none", [])
    chemo = TreatmentPlan("chemo", [TreatmentCourse(TreatmentKind.CHEMO, 30, 200, 1.4)])
    res = eng.simulate_counterfactuals(twin, [none, chemo], horizon_days=200)
    # On-treatment, chemo should suppress burden relative to no treatment.
    i150 = int(np.argmin(np.abs(res["none"].t - 150)))
    assert res["chemo"].median[i150] < res["none"].median[i150]


def test_assimilation_shifts_belief_and_bumps_version():
    eng = OncoTwinEngine()
    twin = eng.create_twin("PT-T", make_features())
    prior_fc = eng.forecast(twin, None, horizon_days=150)
    i120 = int(np.argmin(np.abs(prior_fc.t - 120)))
    before = eng.explain(twin)

    # Observations well above the prior's expected trajectory.
    obs = [TumorMeasurement(30, 150.0), TumorMeasurement(60, 380.0)]
    eng.assimilate(twin, obs, None)
    after = eng.explain(twin)
    post_fc = eng.forecast(twin, None, horizon_days=150)

    assert after["version"] == before["version"] + 1
    assert after["n_observations"] == 2
    # Recalibrating on above-trend burden must raise the twin's forecast.
    assert post_fc.median[i120] > prior_fc.median[i120]


def test_backtest_runs_and_is_reasonable():
    cohort = list(SyntheticCohort(n=15, seed=1).records())
    result = backtest(cohort, warmup=2)
    assert result.n_forecasts > 0
    assert result.mae_cm3 >= 0
    assert 0.0 <= result.coverage_90 <= 1.0
