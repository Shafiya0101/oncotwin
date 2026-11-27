import numpy as np

from oncotwin.survival_eval import (
    CoxRidge, bootstrap_metrics, concordance_index, cross_validate,
    kaplan_meier, permutation_importance_cv, survival_at, time_dependent_auc,
)


def _simulate(n=300, p=5, seed=0, beta=(1.0, -0.8), censor=0.25):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, p))
    lp = x[:, : len(beta)] @ np.array(beta)
    t = rng.exponential(scale=np.exp(-lp))
    c = rng.exponential(scale=np.quantile(t, 1 - censor) * 2, size=n)
    return x, np.minimum(t, c), (t <= c)


def test_concordance_matches_bruteforce():
    rng = np.random.default_rng(1)
    t = rng.integers(1, 30, 60).astype(float)      # includes tied times
    e = rng.random(60) < 0.7
    r = rng.integers(0, 10, 60).astype(float)      # includes tied risks
    conc = comp = 0.0
    for i in range(60):
        for j in range(60):
            if e[i] and t[i] < t[j]:
                comp += 1
                conc += 1.0 if r[i] > r[j] else 0.5 if r[i] == r[j] else 0.0
    assert abs(concordance_index(t, e, r) - conc / comp) < 1e-12


def test_kaplan_meier_hand_example():
    # times 1,2,3,4 ; events 1,0,1,1 -> S(1)=3/4, S(3)=3/4*1/2, S(4)=0
    t = np.array([1, 2, 3, 4.0])
    e = np.array([1, 0, 1, 1], bool)
    assert abs(survival_at(t, e, 1.5) - 0.75) < 1e-12
    assert abs(survival_at(t, e, 3.5) - 0.375) < 1e-12
    assert survival_at(t, e, 10) == 0.0


def test_time_dependent_auc_no_censoring_bruteforce():
    rng = np.random.default_rng(2)
    t = rng.exponential(size=80)
    e = np.ones(80, bool)
    r = -t + rng.normal(scale=0.5, size=80)
    h = np.median(t)
    cases, controls = np.where(t <= h)[0], np.where(t > h)[0]
    s = sum((r[i] > r[j]) + 0.5 * (r[i] == r[j]) for i in cases for j in controls)
    assert abs(time_dependent_auc(t, e, r, h) - s / (len(cases) * len(controls))) < 1e-12


def test_cox_recovers_effect_direction_and_size():
    x, t, e = _simulate(n=1500, p=3, beta=(1.0, -0.8), censor=0.2)
    m = CoxRidge(alpha=1e-6).fit(x, t, e)
    assert abs(m.coef_[0] - 1.0) < 0.15
    assert abs(m.coef_[1] + 0.8) < 0.15
    assert abs(m.coef_[2]) < 0.15


def test_cv_pipeline_beats_chance_on_signal():
    x, t, e = _simulate(n=250, p=8, seed=3)
    cv = cross_validate(x, t, e, n_repeats=2, seed=0, horizons=(np.median(t),))
    risk = cv["oof_lp"].mean(axis=0)
    assert concordance_index(t, e, risk) > 0.65


def test_cv_pipeline_on_pure_noise_is_near_chance():
    # many noise features, few patients: any leakage would push C well above 0.5
    rng = np.random.default_rng(4)
    n = 150
    x = rng.normal(size=(n, 300))
    t = rng.exponential(size=n)
    e = rng.random(n) < 0.7
    cv = cross_validate(x, t, e, n_repeats=2, seed=0)
    c = concordance_index(t, e, cv["oof_lp"].mean(axis=0))
    assert 0.40 < c < 0.60


def test_bootstrap_ci_contains_point_and_pairs_work():
    x, t, e = _simulate(n=200, p=4, seed=5)
    good = x[:, 0] - 0.8 * x[:, 1]
    noise = np.random.default_rng(6).normal(size=200)
    out = bootstrap_metrics(t, e, {"good": good, "noise": noise},
                            horizons=[np.median(t)], n_boot=200, seed=0,
                            pairs=[("good", "noise")])
    lo, hi = out["models"]["good"]["c_ci"]
    assert lo <= out["models"]["good"]["c"] <= hi
    assert out["pairs"]["good - noise"]["delta"] > 0


def test_matches_scikit_survival_if_installed():
    try:
        from sksurv.metrics import concordance_index_censored
    except ImportError:
        return
    x, t, e = _simulate(n=200, p=4, seed=7)
    r = x[:, 0]
    ref = concordance_index_censored(e, t, r)[0]
    assert abs(concordance_index(t, e, r) - ref) < 1e-9


def test_permutation_importance_ranks_signal_above_noise():
    x, t, e = _simulate(n=300, p=4, seed=8, beta=(1.2, 0.0))
    groups = {"signal": [0], "noise_a": [2], "noise_b": [3]}
    imp = permutation_importance_cv(x, t, e, groups, n_repeats=1, n_perm=10, seed=0)
    assert imp["signal"][0] > 0.05
    assert imp["signal"][0] > max(imp["noise_a"][0], imp["noise_b"][0]) + 0.03
