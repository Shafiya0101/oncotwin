import numpy as np
from oncotwin.domain import TreatmentPlan, TreatmentCourse, TreatmentKind
from oncotwin.growth import simulate


def test_untreated_tumor_grows_toward_carrying_capacity():
    params = np.array([0.02, 500.0, 0.03, 0.03])
    t = np.linspace(0, 365, 74)
    v = simulate(40.0, params, None, t)
    assert v[-1] > v[0]
    assert v[-1] <= 500.0 * 1.01           # never exceeds K


def test_treatment_reduces_burden_versus_no_treatment():
    params = np.array([0.02, 500.0, 0.08, 0.05])
    t = np.linspace(0, 200, 41)
    plan = TreatmentPlan("chemo", [TreatmentCourse(TreatmentKind.CHEMO, 30, 200, 1.0)])
    treated = simulate(40.0, params, plan, t)
    untreated = simulate(40.0, params, None, t)
    assert treated[-1] < untreated[-1]


def test_volume_stays_positive():
    params = np.array([0.02, 500.0, 0.5, 0.5])   # very strong therapy
    t = np.linspace(0, 200, 41)
    plan = TreatmentPlan("chemo", [TreatmentCourse(TreatmentKind.CHEMO, 0, 200, 3.0)])
    v = simulate(40.0, params, plan, t)
    assert np.all(v > 0)
