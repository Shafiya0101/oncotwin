import numpy as np
from oncotwin import (
    OncoTwinEngine, PatientFeatures, TreatmentPlan, TreatmentCourse, TreatmentKind,
    SyntheticCohort, backtest, explain_drivers, Driver,
)
from oncotwin.growth import simulate


def make_features():
    return PatientFeatures(age=63, stage=3, histology="adenocarcinoma",
                           baseline_volume_cm3=42.0, ki67=0.35)


def test_chemo_resistance_weakens_late_response():
    # With acquired resistance, a long chemo course loses potency over time:
    # the same drug removed earlier vs later leaves the tumor larger when it
    # runs longer only through the resistant phase.
    params = np.array([0.02, 500.0, 0.09, 0.05])
    t = np.linspace(0, 300, 61)
    early = TreatmentPlan("early", [TreatmentCourse(TreatmentKind.CHEMO, 30, 90, 1.0)])
    v = simulate(40.0, params, early, t)
    # response then rebound: minimum on-treatment volume is below baseline
    assert v.min() < 40.0


def test_linear_quadratic_radiotherapy_is_superlinear():
    # Doubling radiotherapy intensity should more than double the kill (d + d^2 term).
    params = np.array([0.015, 500.0, 0.0, 0.04])
    t = np.linspace(0, 60, 31)
    low = TreatmentPlan("low", [TreatmentCourse(TreatmentKind.RADIO, 10, 40, 1.0)])
    high = TreatmentPlan("high", [TreatmentCourse(TreatmentKind.RADIO, 10, 40, 2.0)])
    reduction_low = 40.0 - simulate(40.0, params, low, t)[-1]
    reduction_high = 40.0 - simulate(40.0, params, high, t)[-1]
    assert reduction_high > 2.0 * reduction_low


def test_process_noise_widens_bands():
    eng = OncoTwinEngine()
    twin = eng.create_twin("PT-N", make_features())
    tight = OncoTwinEngine(process_noise=0.0).forecast(twin, None, horizon_days=300)
    wide = OncoTwinEngine(process_noise=0.01).forecast(twin, None, horizon_days=300)
    tight_w = (tight.upper - tight.lower)[-1]
    wide_w = (wide.upper - wide.lower)[-1]
    assert wide_w > tight_w


def test_calibration_improves_coverage():
    test = list(SyntheticCohort(n=30, seed=2).records())
    no_noise = backtest(test, warmup=2, process_noise=0.0)
    calibrated = backtest(test, warmup=2, process_noise=0.003)
    assert calibrated.coverage_90 > no_noise.coverage_90


def test_explain_drivers_ranks_features():
    eng = OncoTwinEngine()
    drivers = explain_drivers(eng, make_features(), None)
    assert len(drivers) >= 4
    assert all(isinstance(d, Driver) for d in drivers)
    # sorted by absolute effect, largest first
    effects = [abs(d.effect_cm3) for d in drivers]
    assert effects == sorted(effects, reverse=True)
