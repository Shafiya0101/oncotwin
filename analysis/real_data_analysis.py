"""Validate the twin against REAL patient survival (NSCLC-Radiomics / Lung1).

For each real patient we build a twin from their real age, stage, and histology,
read off the twin's biology-based risk (its implied tumor growth rate), and then
ask a simple, honest question: do patients the twin judges higher-risk actually
die sooner?

We measure this two ways:
  - Concordance index (C-index): fraction of comparable patient pairs the risk
    score orders correctly. 0.5 = random, 1.0 = perfect.
  - Kaplan-Meier survival curves for twin-predicted high- vs low-risk groups.

Run:  python analysis/real_data_analysis.py
Needs: data/real/Lung1.clinical.csv  (the 23 KB file from TCIA)

Honest caveat: without imaging, the twin's risk here is driven mainly by stage
and age (tumor volume / heterogeneity use defaults). Adding real scans is the
next step and should sharpen it.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from oncotwin import OncoTwinEngine
from oncotwin.data.clinical import load_lung1_clinical

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, "..", "data", "real", "Lung1.clinical.csv")
OUT = os.path.join(HERE, "outputs")
os.makedirs(OUT, exist_ok=True)
CORAL, CYAN = "#D85A30", "#1D9E75"


def c_index(risk, t, e):
    risk, t, e = np.asarray(risk), np.asarray(t), np.asarray(e)
    conc = comp = 0.0
    for i in range(len(t)):
        if e[i] != 1:
            continue
        later = t > t[i]                       # i died before these are known to fail
        comp += later.sum()
        conc += (risk[i] > risk[later]).sum() + 0.5 * (risk[i] == risk[later]).sum()
    return conc / comp if comp else float("nan")


def kaplan_meier(t, e):
    t, e = np.asarray(t, float), np.asarray(e, int)
    times = np.unique(t)
    s, curve_t, curve_s = 1.0, [0.0], [1.0]
    for ut in times:
        n_risk = (t >= ut).sum()
        d = ((t == ut) & (e == 1)).sum()
        if n_risk > 0 and d > 0:
            s *= 1 - d / n_risk
        curve_t.append(ut); curve_s.append(s)
    return np.array(curve_t), np.array(curve_s)


def median_survival(t, e):
    ct, cs = kaplan_meier(t, e)
    below = np.where(cs <= 0.5)[0]
    return float(ct[below[0]]) if len(below) else float("nan")


def main():
    if not os.path.exists(CSV):
        print("Could not find", CSV)
        print("Download Lung1.clinical.csv from TCIA and put it in data/real/.")
        return

    records = load_lung1_clinical(CSV)
    print(f"Loaded {len(records)} real patients with complete stage/survival.")

    engine = OncoTwinEngine()
    risk, surv, event, stage = [], [], [], []
    for r in records:
        twin = engine.create_twin(r.patient_id, r.features)
        risk.append(engine.explain(twin)["growth_rate_per_day"])
        surv.append(r.survival_days); event.append(r.event); stage.append(r.features.stage)

    risk, surv, event = np.array(risk), np.array(surv), np.array(event)
    c = c_index(risk, surv, event)
    print(f"Concordance index (twin risk vs real survival): {c:.3f}")

    hi = risk >= np.median(risk)
    hi_t, hi_e = surv[hi], event[hi]
    lo_t, lo_e = surv[~hi], event[~hi]
    m_hi, m_lo = median_survival(hi_t, hi_e), median_survival(lo_t, lo_e)
    print(f"Median survival — twin high-risk: {m_hi:.0f} d   low-risk: {m_lo:.0f} d")

    fig, ax = plt.subplots(figsize=(8, 5))
    for (t, e, label, color) in [(hi_t, hi_e, "twin: higher risk", CORAL),
                                 (lo_t, lo_e, "twin: lower risk", CYAN)]:
        ct, cs = kaplan_meier(t, e)
        ax.step(ct, cs, where="post", color=color, lw=2.2, label=f"{label} (n={len(t)})")
    ax.set_title(f"Real NSCLC survival by twin-predicted risk\nC-index = {c:.3f} "
                 f"(0.5 = chance)", fontsize=12)
    ax.set_xlabel("Days from start of treatment"); ax.set_ylabel("Survival probability")
    ax.set_ylim(0, 1.0); ax.legend(frameon=False); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(f"{OUT}/real_survival_validation.png", dpi=140)
    print(f"Saved {OUT}/real_survival_validation.png")

    verdict = ("does **not** predict real survival better than chance" if c < 0.55
               else "shows **modest** signal for real survival" if c < 0.65
               else "orders real patients' survival **better than chance**")
    with open(os.path.join(HERE, "REAL_RESULTS.md"), "w", encoding="utf-8") as fh:
        fh.write(f"""# Real-data validation (NSCLC-Radiomics / Lung1)

Source: TCIA NSCLC-Radiomics clinical data — {len(records)} real NSCLC patients.
For each patient the twin is built from real age, stage, and histology; its
implied tumor growth rate is used as a risk score and compared to real survival.

- **Concordance index: {c:.3f}** (0.5 = chance). At this value, the twin's
  biology-based risk {verdict}.
- **Median survival** — twin high-risk group: {m_hi:.0f} days;
  low-risk group: {m_lo:.0f} days.

![survival by twin risk](outputs/real_survival_validation.png)

## Why this result — and what would move it

The growth model was never trained on survival (it uses hand-set biological
priors), and the imaging features that carry the prognostic signal — tumor
volume, heterogeneity — are not yet loaded per patient (they use defaults), so
the twin's risk here is driven mainly by stage and age. The landmark study on
this dataset showed radiomic features from the scans do predict survival.
Moving this number means ingesting the real scans with expert masks across the
cohort and training the estimator against survival directly.

## Honest limitations

Tumor volume, Ki-67, and imaging heterogeneity use cohort defaults (no
per-patient imaging yet). This is a simple validation, not a trained survival
model — and the result is reported exactly as it came out.
""")
    print("Wrote analysis/REAL_RESULTS.md")


if __name__ == "__main__":
    main()
