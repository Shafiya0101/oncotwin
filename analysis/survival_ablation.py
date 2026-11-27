#!/usr/bin/env python
"""Real-cohort ablation: clinical-only vs imaging-only vs imaging+clinical.

Inputs  (from the two earlier scripts)
  data/real/radiomics.csv   PatientID + radiomic feature columns
  data/real/clinical.csv    PatientID, time_days, event, age, sex, t_stage,
                            n_stage, m_stage, overall_stage, histology

Usage
  python analysis/survival_ablation.py                 # real cohort
  python analysis/survival_ablation.py --demo          # synthetic dry run
  python analysis/survival_ablation.py --baseline-csv twin_risk.csv

Protocol (docs/EVAL_PROTOCOL.md): patient-level repeated stratified 5-fold CV,
all preprocessing inside folds, 1000-resample patient bootstrap, paired
differences between models, time-dependent AUC and calibration.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from oncotwin.survival_eval import (  # noqa: E402
    DEFAULT_ALPHAS, bootstrap_metrics, calibration_table, concordance_index,
    cross_validate, fit_model, permutation_importance_cv,
)

CAT_COLS = ["sex", "t_stage", "n_stage", "m_stage", "overall_stage", "histology"]
DAYS_PER_MONTH = 365.25 / 12


def feature_group(name):
    """Variable a design-matrix column belongs to (dummies of one variable share a group)."""
    for c in CAT_COLS:
        if name.startswith(c + "_"):
            return c
    return name


# ----------------------------------------------------------------- data ----
def make_demo_data(n=399, seed=0):
    """Synthetic stand-in so the pipeline can be tested before real data."""
    rng = np.random.default_rng(seed)
    base = rng.normal(size=(n, 12))
    rad = np.hstack([base + 0.3 * rng.normal(size=(n, 12)) for _ in range(5)])  # correlated blocks
    names = [f"original_feat{i:03d}" for i in range(rad.shape[1])]
    age = rng.normal(68, 10, n).round()
    stage = rng.choice(["I", "II", "IIIa", "IIIb"], n, p=[.25, .15, .3, .3])
    sev = pd.Series(stage).map({"I": 0, "II": .3, "IIIa": .6, "IIIb": .9}).to_numpy()
    lp = 0.9 * sev + 0.02 * (age - 68) + 0.35 * base[:, 0] - 0.25 * base[:, 1]
    t = rng.weibull(1.2, n) * 700 * np.exp(-lp)
    c = rng.exponential(1800, n)
    ids = [f"DEMO-{i:03d}" for i in range(n)]
    clin = pd.DataFrame({
        "PatientID": ids, "time_days": np.minimum(t, c).round(1), "event": (t <= c).astype(int),
        "age": age, "sex": rng.choice(["male", "female"], n),
        "t_stage": rng.choice(["T1", "T2", "T3", "T4"], n),
        "n_stage": rng.choice(["N0", "N1", "N2", "N3"], n),
        "m_stage": "M0", "overall_stage": stage,
        "histology": rng.choice(["adeno", "squamous", "large cell"], n)})
    return pd.DataFrame(rad, columns=names).assign(PatientID=ids), clin


def load_data(args):
    if args.demo:
        return make_demo_data(seed=args.seed)
    return (None if args.clinical_only else pd.read_csv(args.radiomics)), pd.read_csv(args.clinical)


def clinical_matrix(df):
    parts = [pd.to_numeric(df["age"], errors="coerce").rename("age")]
    for c in CAT_COLS:
        if c in df:
            col = df[c].astype(str).str.strip().str.lower().replace({"nan": "unknown", "": "unknown"})
            parts.append(pd.get_dummies(col, prefix=c, dtype=float))
    out = pd.concat(parts, axis=1)
    return out.to_numpy(float), list(out.columns)


def radiomics_matrix(df, id_col):
    drop = [c for c in df.columns if c == id_col or c.startswith(("diagnostics_", "qa_"))]
    num = df.drop(columns=drop).apply(pd.to_numeric, errors="coerce")
    num = num.dropna(axis=1, how="all")
    return num.to_numpy(float), list(num.columns)


# ----------------------------------------------------------------- run -----
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--radiomics", default="data/real/radiomics.csv")
    ap.add_argument("--clinical", default="data/real/clinical.csv")
    ap.add_argument("--baseline-csv", help="CSV with PatientID,risk (e.g. the twin's biology-based risk); evaluated with no training")
    ap.add_argument("--out", default=None)
    ap.add_argument("--id-col", default="PatientID")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--horizons-months", type=float, nargs="+", default=[12, 24, 36])
    ap.add_argument("--calib-month", type=float, default=24)
    ap.add_argument("--demo", action="store_true", help="synthetic data; writes to a separate DEMO folder")
    ap.add_argument("--no-importance", action="store_true", help="skip permutation importance")
    ap.add_argument("--importance-repeats", type=int, default=3)
    ap.add_argument("--clinical-only", action="store_true",
                    help="run only the clinical model (no radiomics file needed); writes to a separate folder")
    args = ap.parse_args()

    default_out = ("analysis/outputs/real_cohort_DEMO" if args.demo else
                   "analysis/outputs/real_cohort_clinical_only" if args.clinical_only else
                   "analysis/outputs/real_cohort")
    out = Path(args.out or default_out)
    out.mkdir(parents=True, exist_ok=True)

    rad, clin = load_data(args)
    if args.clinical_only:
        rad = None
    idc = args.id_col
    n0 = len(clin)
    clin = clin.dropna(subset=["time_days", "event"])
    clin = clin[pd.to_numeric(clin["time_days"], errors="coerce") > 0]
    merged = clin.copy() if rad is None else clin.merge(rad, on=idc, how="inner", suffixes=("", "_rad"))
    print(f"clinical rows: {n0} | usable survival: {len(clin)} | analysed: {len(merged)}"
          + ("" if rad is None else " (those with radiomics)"))
    if len(merged) < 50:
        sys.exit("Too few patients after merging: check that PatientID matches in both files.")

    time = merged["time_days"].to_numpy(float)
    event = merged["event"].to_numpy(int).astype(bool)
    Xc, c_names = clinical_matrix(merged)
    designs = {"clinical": (Xc, c_names)}
    r_names = []
    if rad is not None:
        Xr, r_names = radiomics_matrix(rad.set_index(idc).loc[merged[idc]].reset_index(), idc)
        designs["imaging"] = (Xr, r_names)
        designs["imaging+clinical"] = (np.hstack([Xc, Xr]), c_names + r_names)  # clinical first: wins correlation ties
    best = list(designs)[-1]
    print(f"n={len(merged)}, events={event.sum()} ({event.mean():.0%}), "
          f"clinical features={len(c_names)}, radiomic features={len(r_names)}")

    horizons = [m * DAYS_PER_MONTH for m in args.horizons_months]
    calib_h = args.calib_month * DAYS_PER_MONTH

    # ---- cross-validation ----
    risk, fold_c, surv_pred, alphas_used = {}, {}, {}, {}
    for name, (X, _) in designs.items():
        print(f"cross-validating {name} ...", flush=True)
        cv = cross_validate(X, time, event, DEFAULT_ALPHAS, args.folds, args.repeats,
                            seed=args.seed, horizons=[calib_h])
        risk[name] = cv["oof_lp"].mean(axis=0)               # repeat-averaged OOF risk
        surv_pred[name] = cv["oof_surv"][calib_h].mean(axis=0)
        fold_c[name] = cv["fold_c"]
        alphas_used[name] = float(np.median(cv["alphas"]))

    names = list(designs)
    baseline_name = None
    if args.baseline_csv:
        b = pd.read_csv(args.baseline_csv).set_index(idc)["risk"]
        risk["twin baseline"] = b.reindex(merged[idc]).to_numpy(float)
        if np.isnan(risk["twin baseline"]).any():
            sys.exit("baseline CSV is missing some PatientIDs.")
        baseline_name = "twin baseline"

    # ---- bootstrap ----
    all_names = list(risk)
    pairs = [("imaging", "clinical"), ("imaging+clinical", "clinical"), ("imaging+clinical", "imaging")]
    pairs = [p for p in pairs if p[0] in risk and p[1] in risk]
    if baseline_name:
        pairs.append((best, baseline_name))
    boot = bootstrap_metrics(time, event, risk, horizons, args.bootstrap, args.seed, pairs)

    # ---- calibration ----
    calib = {m: calibration_table(time, event, surv_pred[m], calib_h, seed=args.seed) for m in names}

    # ---- final fit on all data: coefficients of the combined model ----
    X, feat = designs[best]
    fm = fit_model(X, time, event, DEFAULT_ALPHAS, seed=args.seed)
    coefs = pd.DataFrame({"feature": np.array(feat)[fm.prep.keep_], "coef_standardised": fm.cox.coef_})
    coefs["abs"] = coefs["coef_standardised"].abs()
    coefs = coefs.sort_values("abs", ascending=False).drop(columns="abs")
    coefs.to_csv(out / "final_model_coefficients.csv", index=False)

    # ---- permutation importance on held-out folds (best model) ----
    imp = None
    if not args.no_importance:
        print(f"permutation importance ({best}) ...", flush=True)
        groups = {}
        for i, nm in enumerate(feat):
            groups.setdefault(feature_group(nm), []).append(i)
        raw = permutation_importance_cv(X, time, event, groups, DEFAULT_ALPHAS, args.folds,
                                        args.importance_repeats, seed=args.seed)
        imp = sorted(({"variable": g, "mean_c_drop": v[0], "sd_over_folds": v[1]} for g, v in raw.items()),
                     key=lambda d: -d["mean_c_drop"])
        pd.DataFrame(imp).to_csv(out / "feature_importance.csv", index=False)

    results = {
        "config": {**{k: v for k, v in vars(args).items()}, "n_patients": len(merged),
                   "n_events": int(event.sum()), "alphas_grid": list(DEFAULT_ALPHAS),
                   "median_chosen_alpha": alphas_used, "python": platform.python_version(),
                   "numpy": np.__version__},
        "cv_fold_c_mean": {m: float(np.nanmean(v)) for m, v in fold_c.items()},
        "cv_fold_c_sd": {m: float(np.nanstd(v)) for m, v in fold_c.items()},
        "bootstrap": boot, "calibration": calib, "importance": imp, "importance_model": best,
        "horizons_months": args.horizons_months,
    }
    (out / "results.json").write_text(json.dumps(results, indent=2))

    write_markdown(out / "RESULTS_real_cohort.md", results, all_names, names, args.demo)
    make_figures(out, results, all_names, names, args.calib_month)
    print(f"\nWrote results to {out}/")
    print((out / "RESULTS_real_cohort.md").read_text())


# ------------------------------------------------------------ reporting ----
def fmt(v, ci):
    return f"{v:.3f} ({ci[0]:.3f} to {ci[1]:.3f})"


def write_markdown(path, res, all_names, names, demo):
    cfg, bt = res["config"], res["bootstrap"]
    L = []
    if demo:
        L += ["> **SYNTHETIC DEMO DATA: these numbers are not real results.**", ""]
    L += ["# Real-cohort ablation", "",
          f"Patients: {cfg['n_patients']}, events: {cfg['n_events']}. "
          f"Repeated {cfg['folds']}-fold CV x {cfg['repeats']} repeats, seed {cfg['seed']}; "
          f"{bt['n_boot']} bootstrap resamples; 95% percentile intervals.", "",
          "## Discrimination (out-of-fold)", "",
          "| Model | C-index (95% CI) | Mean per-fold C (sd) | " +
          " | ".join(f"AUC {int(h)} mo (95% CI)" for h in res["horizons_months"]) + " |",
          "|---|---|---|" + "---|" * len(res["horizons_months"])]
    for m in all_names:
        r = bt["models"][m]
        fc = (f"{res['cv_fold_c_mean'][m]:.3f} ({res['cv_fold_c_sd'][m]:.3f})" if m in res["cv_fold_c_mean"] else "n/a (no training)")
        L.append(f"| {m} | {fmt(r['c'], r['c_ci'])} | {fc} | " +
                 " | ".join(fmt(a, c) for a, c in zip(r["auc"], r["auc_ci"])) + " |")
    L += ["", "## Paired C-index differences", "", "| Comparison | Delta C (95% CI) | CI excludes 0 |", "|---|---|---|"]
    for k, v in bt["pairs"].items():
        L.append(f"| {k} | {fmt(v['delta'], v['ci'])} | {'yes' if v['excludes_zero'] else 'no'} |")
    L += ["", f"## Calibration at {int(res['config']['calib_month'])} months", "",
          "Predicted survival vs Kaplan-Meier observed survival, 5 risk groups.", ""]
    for m in names:
        c = res["calibration"][m]
        L += [f"**{m}** (mean absolute calibration error {c['mean_abs_calibration_error']:.3f})", "",
              "| Group | n | events | Predicted | Observed KM (95% CI) |", "|---|---|---|---|---|"]
        for i, b in enumerate(c["bins"], 1):
            L.append(f"| {i} | {b['n']} | {b['events']} | {b['mean_predicted']:.3f} | "
                     f"{b['observed_km']:.3f} ({b['obs_ci'][0]:.3f} to {b['obs_ci'][1]:.3f}) |")
        L.append("")
    if res.get("importance"):
        L += ["## Variable importance (permutation, held-out folds)", "",
              f"Model: {res['importance_model']}. Drop in C-index when the variable is shuffled; about 0 means the model does not use it.", "",
              "| Variable | Mean C-index drop | SD over folds |", "|---|---|---|"]
        for d in res["importance"][:12]:
            L.append(f"| {d['variable']} | {d['mean_c_drop']:.4f} | {d['sd_over_folds']:.4f} |")
        L.append("")
    L += ["## Notes", "",
          "- C-index and AUC use repeat-averaged out-of-fold risk scores; each fold's scores are standardised by the training-set mean and SD so folds are comparable.",
          "- Time-dependent AUC is the IPCW cumulative/dynamic estimator.",
          "- All imputation, filtering, scaling and penalty selection happen inside training folds only.",
          "- Top coefficients of the final model: final_model_coefficients.csv (standardised units; not an importance test)."]
    path.write_text("\n".join(L))


def make_figures(out, res, all_names, names, calib_month):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bt = res["bootstrap"]["models"]
    fig, ax = plt.subplots(figsize=(6, 3 + 0.3 * len(all_names)))
    for i, m in enumerate(all_names):
        c, (lo, hi) = bt[m]["c"], bt[m]["c_ci"]
        ax.errorbar(c, i, xerr=[[c - lo], [hi - c]], fmt="o", capsize=4)
    ax.axvline(0.5, ls="--", c="grey")
    ax.set_yticks(range(len(all_names)), all_names)
    ax.set_xlabel("C-index (95% bootstrap CI); dashed = chance")
    ax.invert_yaxis(); fig.tight_layout(); fig.savefig(out / "cindex_ci.png", dpi=200); plt.close(fig)

    if res.get("importance"):
        top = res["importance"][:12][::-1]
        fig, ax = plt.subplots(figsize=(6, 0.4 * len(top) + 1.5))
        ax.barh([d["variable"] for d in top], [d["mean_c_drop"] for d in top],
                xerr=[d["sd_over_folds"] for d in top], capsize=3)
        ax.axvline(0, c="grey", lw=1)
        ax.set_xlabel("Drop in held-out C-index when shuffled")
        fig.tight_layout(); fig.savefig(out / "feature_importance.png", dpi=200); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 5))
    for m in names:
        bins = [b for b in res["calibration"][m]["bins"] if not np.isnan(b["observed_km"])]
        ax.plot([b["mean_predicted"] for b in bins], [b["observed_km"] for b in bins], "o-", label=m)
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel(f"Predicted {int(calib_month)}-month survival"); ax.set_ylabel("Observed (Kaplan-Meier)")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.legend(); fig.tight_layout()
    fig.savefig(out / f"calibration_{int(calib_month)}mo.png", dpi=200); plt.close(fig)


if __name__ == "__main__":
    main()
