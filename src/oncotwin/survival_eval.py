"""Survival evaluation utilities for the OncoTwin real-cohort experiments.

Dependency-light on purpose (numpy, scipy, scikit-learn only), so the numbers
in the paper can be reproduced without extra survival libraries.

Contents
  - Kaplan-Meier, Harrell's C-index, IPCW time-dependent AUC
  - Ridge-penalised Cox model (Breslow ties) + Breslow baseline hazard
  - In-fold preprocessing (impute, drop constants, correlation filter, scale)
  - Repeated stratified cross-validation with out-of-fold predictions
  - Patient-level bootstrap CIs, including paired differences between models
  - Calibration table (predicted vs Kaplan-Meier observed survival)

Leakage rule: every data-dependent step (imputation, filtering, scaling,
penalty choice) is fitted on training data only, inside each fold.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from sklearn.model_selection import StratifiedKFold

# --------------------------------------------------------------------------
# Kaplan-Meier and metrics
# --------------------------------------------------------------------------

def kaplan_meier(time, event):
    """Kaplan-Meier curve evaluated at every unique observed time."""
    time = np.asarray(time, float)
    event = np.asarray(event, bool)
    ut, inv = np.unique(time, return_inverse=True)
    d = np.bincount(inv, weights=event.astype(float), minlength=len(ut))
    cnt = np.bincount(inv, minlength=len(ut))
    at_risk = len(time) - np.concatenate(([0], np.cumsum(cnt)[:-1]))
    return ut, np.cumprod(1.0 - d / at_risk)


def survival_at(time, event, t0):
    """KM survival at t0. NaN if t0 is beyond follow-up and not yet at zero."""
    ut, s = kaplan_meier(time, event)
    if t0 > ut[-1]:
        return float(s[-1]) if s[-1] == 0 else float("nan")
    i = np.searchsorted(ut, t0, side="right") - 1
    return 1.0 if i < 0 else float(s[i])


def concordance_index(time, event, risk):
    """Harrell's C. Higher risk should mean shorter survival."""
    time = np.asarray(time, float)
    event = np.asarray(event, bool)
    risk = np.asarray(risk, float)
    comp = (time[:, None] < time[None, :]) & event[:, None]
    n_comp = comp.sum()
    if n_comp == 0:
        return float("nan")
    ri, rj = risk[:, None], risk[None, :]
    conc = ((ri > rj) & comp).sum() + 0.5 * ((ri == rj) & comp).sum()
    return float(conc / n_comp)


def censoring_survival_left(time, event, query):
    """G(t-) : Kaplan-Meier of the censoring distribution, left limit."""
    ut, s = kaplan_meier(time, ~np.asarray(event, bool))
    i = np.searchsorted(ut, query, side="left") - 1
    return np.where(i < 0, 1.0, s[np.clip(i, 0, None)])


def time_dependent_auc(time, event, risk, horizon, g=None):
    """IPCW cumulative/dynamic AUC at `horizon` (same units as time).

    Cases: event at or before horizon. Controls: still at risk after horizon.
    Cases are weighted by 1/G(T-) to correct for censoring.
    """
    time = np.asarray(time, float)
    event = np.asarray(event, bool)
    risk = np.asarray(risk, float)
    cases = (time <= horizon) & event
    controls = time > horizon
    if cases.sum() == 0 or controls.sum() == 0:
        return float("nan")
    if g is None:
        g = censoring_survival_left(time, event, time)
    w = 1.0 / np.clip(g[cases], 1e-6, None)
    rc = risk[cases][:, None]
    rn = risk[controls][None, :]
    conc = (rc > rn).astype(float) + 0.5 * (rc == rn)
    return float((w * conc.sum(axis=1)).sum() / (w.sum() * controls.sum()))

# --------------------------------------------------------------------------
# Ridge Cox model
# --------------------------------------------------------------------------

class CoxRidge:
    """Cox proportional hazards with an L2 penalty (Breslow tie handling).

    Objective: -loglik / n_events + 0.5 * alpha * ||beta||^2
    (features are expected to be standardised).
    """

    def __init__(self, alpha: float = 1.0, max_iter: int = 300):
        self.alpha = alpha
        self.max_iter = max_iter

    def fit(self, X, time, event):
        X = np.asarray(X, float)
        time = np.asarray(time, float)
        event = np.asarray(event, bool)
        order = np.argsort(time, kind="stable")
        Xs, ts, es = X[order], time[order], event[order]
        first = np.searchsorted(ts, ts, side="left")
        ev_idx = np.where(es)[0]
        d = len(ev_idx)
        if d == 0:
            raise ValueError("No events in training data.")
        fi = first[ev_idx]
        alpha = self.alpha

        def fun(beta):
            eta = Xs @ beta
            eta = eta - eta.max()
            e = np.exp(eta)
            s0 = np.cumsum(e[::-1])[::-1]
            s1 = np.cumsum((e[:, None] * Xs)[::-1], axis=0)[::-1]
            ll = (eta[ev_idx] - np.log(s0[fi])).sum()
            g = (Xs[ev_idx] - s1[fi] / s0[fi][:, None]).sum(axis=0)
            return (-ll / d + 0.5 * alpha * beta @ beta, -g / d + alpha * beta)

        res = minimize(fun, np.zeros(X.shape[1]), jac=True, method="L-BFGS-B",
                       options={"maxiter": self.max_iter})
        self.coef_ = res.x

        # Breslow baseline cumulative hazard
        lp = np.clip(Xs @ self.coef_, -50, 50)
        s0 = np.cumsum(np.exp(lp)[::-1])[::-1]
        uniq, inv = np.unique(ts[es], return_inverse=True)
        d_k = np.bincount(inv).astype(float)
        self.base_times_ = uniq
        self.base_cumhaz_ = np.cumsum(d_k / s0[np.searchsorted(ts, uniq, side="left")])
        return self

    def predict(self, X):
        return np.asarray(X, float) @ self.coef_

    def predict_survival(self, X, t):
        i = np.searchsorted(self.base_times_, t, side="right") - 1
        h0 = 0.0 if i < 0 else self.base_cumhaz_[i]
        return np.exp(-h0 * np.exp(np.clip(self.predict(X), -50, 50)))

# --------------------------------------------------------------------------
# In-fold preprocessing
# --------------------------------------------------------------------------

class Preprocessor:
    """Median-impute, drop constant columns, drop highly correlated columns
    (greedy, earlier columns win), standardise, clip extreme z-scores."""

    def __init__(self, corr_threshold: float = 0.9, clip: float = 5.0):
        self.corr_threshold = corr_threshold
        self.clip = clip

    def _impute(self, X):
        X = np.array(X, float, copy=True)
        bad = np.isnan(X)
        if bad.any():
            X[bad] = np.take(self.median_, np.where(bad)[1])
        return X

    def fit(self, X):
        X = np.asarray(X, float)
        med = np.nanmedian(X, axis=0) if X.size else np.zeros(X.shape[1])
        self.median_ = np.where(np.isnan(med), 0.0, med)
        xi = self._impute(X)
        sd = xi.std(axis=0)
        keep = np.where(sd > 1e-12)[0]
        z = (xi[:, keep] - xi[:, keep].mean(axis=0)) / sd[keep]
        corr = np.abs(z.T @ z) / len(z)
        selected: list[int] = []
        for j in range(z.shape[1]):
            if not selected or corr[j, selected].max() < self.corr_threshold:
                selected.append(j)
        self.keep_ = keep[selected]
        self.mean_ = xi[:, self.keep_].mean(axis=0)
        self.std_ = xi[:, self.keep_].std(axis=0)
        return self

    def transform(self, X):
        xi = self._impute(np.asarray(X, float))[:, self.keep_]
        return np.clip((xi - self.mean_) / self.std_, -self.clip, self.clip)

# --------------------------------------------------------------------------
# Model fitting and cross-validation
# --------------------------------------------------------------------------

DEFAULT_ALPHAS = (0.003, 0.03, 0.3, 3.0, 30.0)


@dataclass
class FittedModel:
    prep: Preprocessor
    cox: CoxRidge
    alpha: float

    def predict(self, X):
        return self.cox.predict(self.prep.transform(X))

    def predict_survival(self, X, t):
        return self.cox.predict_survival(self.prep.transform(X), t)


def fit_model(X, time, event, alphas=DEFAULT_ALPHAS, seed=0, inner_folds=3):
    """Fit preprocessing + ridge Cox; choose alpha by inner-CV C-index."""
    X = np.asarray(X, float)
    time = np.asarray(time, float)
    event = np.asarray(event, bool)
    alphas = list(alphas)
    alpha = alphas[0]
    if len(alphas) > 1:
        scores = np.zeros(len(alphas))
        skf = StratifiedKFold(inner_folds, shuffle=True, random_state=seed)
        for tr, va in skf.split(X, event):
            prep = Preprocessor().fit(X[tr])
            xtr, xva = prep.transform(X[tr]), prep.transform(X[va])
            for k, a in enumerate(alphas):
                m = CoxRidge(a).fit(xtr, time[tr], event[tr])
                c = concordance_index(time[va], event[va], m.predict(xva))
                scores[k] += 0.5 if np.isnan(c) else c
        alpha = alphas[int(np.argmax(scores))]
    prep = Preprocessor().fit(X)
    cox = CoxRidge(alpha).fit(prep.transform(X), time, event)
    return FittedModel(prep, cox, alpha)


def cross_validate(X, time, event, alphas=DEFAULT_ALPHAS, n_splits=5,
                   n_repeats=10, seed=0, horizons=()):
    """Repeated stratified K-fold with out-of-fold (OOF) predictions.

    Returns a dict:
      oof_lp    (n_repeats, n)  linear predictors, put on a common scale per
                                fold (standardised by the training-set mean/sd)
      oof_surv  {horizon: (n_repeats, n)} predicted survival probabilities
      fold_c    per-fold C-index values
      alphas    chosen penalty per fold
    """
    X = np.asarray(X, float)
    time = np.asarray(time, float)
    event = np.asarray(event, bool)
    n = len(time)
    oof_lp = np.zeros((n_repeats, n))
    oof_surv = {h: np.zeros((n_repeats, n)) for h in horizons}
    fold_c, chosen = [], []
    for r in range(n_repeats):
        skf = StratifiedKFold(n_splits, shuffle=True, random_state=seed + r)
        for tr, te in skf.split(X, event):
            m = fit_model(X[tr], time[tr], event[tr], alphas, seed=seed + r)
            lp_tr, lp_te = m.predict(X[tr]), m.predict(X[te])
            sd = lp_tr.std() if lp_tr.std() > 1e-12 else 1.0
            oof_lp[r, te] = (lp_te - lp_tr.mean()) / sd
            for h in horizons:
                oof_surv[h][r, te] = m.predict_survival(X[te], h)
            fold_c.append(concordance_index(time[te], event[te], lp_te))
            chosen.append(m.alpha)
    return {"oof_lp": oof_lp, "oof_surv": oof_surv,
            "fold_c": np.array(fold_c), "alphas": np.array(chosen)}

# --------------------------------------------------------------------------
# Bootstrap
# --------------------------------------------------------------------------

def bootstrap_metrics(time, event, risk_by_model, horizons, n_boot=1000,
                      seed=0, pairs=()):
    """Patient-level bootstrap for C-index, time-dependent AUC and paired
    C-index differences. The SAME resamples are used for every model, so the
    differences are paired (overlapping CIs do not imply no difference)."""
    time = np.asarray(time, float)
    event = np.asarray(event, bool)
    names = list(risk_by_model)
    risks = {m: np.asarray(risk_by_model[m], float) for m in names}
    n = len(time)
    rng = np.random.default_rng(seed)
    horizons = list(horizons)

    g_full = censoring_survival_left(time, event, time)
    point = {m: {"c": concordance_index(time, event, risks[m]),
                 "auc": [time_dependent_auc(time, event, risks[m], h, g_full)
                         for h in horizons]} for m in names}

    cb = np.zeros((n_boot, len(names)))
    ab = np.full((n_boot, len(names), len(horizons)), np.nan)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        t, e = time[idx], event[idx]
        g = censoring_survival_left(t, e, t)
        for k, m in enumerate(names):
            r = risks[m][idx]
            cb[b, k] = concordance_index(t, e, r)
            for j, h in enumerate(horizons):
                ab[b, k, j] = time_dependent_auc(t, e, r, h, g)

    def ci(a):
        return [float(np.nanpercentile(a, 2.5)), float(np.nanpercentile(a, 97.5))]

    out = {"models": {}, "pairs": {}, "n_boot": n_boot}
    for k, m in enumerate(names):
        out["models"][m] = {
            "c": point[m]["c"], "c_ci": ci(cb[:, k]),
            "auc": point[m]["auc"],
            "auc_ci": [ci(ab[:, k, j]) for j in range(len(horizons))]}
    for a, b_ in pairs:
        ia, ib = names.index(a), names.index(b_)
        delta = cb[:, ia] - cb[:, ib]
        lo, hi = ci(delta)
        out["pairs"][f"{a} - {b_}"] = {
            "delta": point[a]["c"] - point[b_]["c"], "ci": [lo, hi],
            "excludes_zero": bool(lo > 0 or hi < 0)}
    return out

# --------------------------------------------------------------------------
# Calibration
# --------------------------------------------------------------------------

def calibration_table(time, event, pred_surv, horizon, n_bins=5,
                      n_boot=500, seed=0):
    """Group patients by predicted survival at `horizon`; compare with the
    Kaplan-Meier observed survival in each group (bootstrap 95% CI)."""
    time = np.asarray(time, float)
    event = np.asarray(event, bool)
    pred = np.asarray(pred_surv, float)
    rng = np.random.default_rng(seed)
    rows = []
    for grp in np.array_split(np.argsort(pred), n_bins):
        obs = survival_at(time[grp], event[grp], horizon)
        boots = []
        for _ in range(n_boot):
            bi = rng.choice(grp, size=len(grp), replace=True)
            v = survival_at(time[bi], event[bi], horizon)
            if not np.isnan(v):
                boots.append(v)
        rows.append({
            "n": int(len(grp)), "events": int(event[grp].sum()),
            "mean_predicted": float(pred[grp].mean()), "observed_km": obs,
            "obs_ci": ([float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]
                       if len(boots) > 20 else [float("nan")] * 2)})
    valid = [r for r in rows if not np.isnan(r["observed_km"])]
    total = sum(r["n"] for r in valid) or 1
    mace = sum(r["n"] * abs(r["mean_predicted"] - r["observed_km"]) for r in valid) / total
    return {"bins": rows, "mean_abs_calibration_error": float(mace)}


# --------------------------------------------------------------------------
# Permutation importance (held-out folds)
# --------------------------------------------------------------------------

def permutation_importance_cv(X, time, event, groups, alphas=DEFAULT_ALPHAS,
                              n_splits=5, n_repeats=3, n_perm=20, seed=0):
    """Drop in held-out C-index when a variable is shuffled.

    `groups` maps a variable name to the list of its column indices; all columns
    of a group (e.g. the dummy columns of one categorical variable) are shuffled
    together with the same row permutation. Computed on test folds only, with a
    model fitted on the training folds, so it measures what generalises.

    Returns {name: (mean drop, sd of the per-fold drops)}. Bigger = more
    important; about 0 = the model does not use it.
    """
    X = np.asarray(X, float)
    time = np.asarray(time, float)
    event = np.asarray(event, bool)
    rng = np.random.default_rng(seed)
    drops = {g: [] for g in groups}
    for r in range(n_repeats):
        skf = StratifiedKFold(n_splits, shuffle=True, random_state=seed + r)
        for tr, te in skf.split(X, event):
            m = fit_model(X[tr], time[tr], event[tr], alphas, seed=seed + r)
            xte, tte, ete = X[te], time[te], event[te]
            base = concordance_index(tte, ete, m.predict(xte))
            for g, cols in groups.items():
                ds = []
                for _ in range(n_perm):
                    xp = xte.copy()
                    xp[:, cols] = xte[rng.permutation(len(te))][:, cols]
                    ds.append(base - concordance_index(tte, ete, m.predict(xp)))
                drops[g].append(float(np.mean(ds)))
    return {g: (float(np.mean(v)), float(np.std(v))) for g, v in drops.items()}
