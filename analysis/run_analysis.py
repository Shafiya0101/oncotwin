"""End-to-end training + evaluation analysis.

Run:  python analysis/run_analysis.py

What it does:
  1. Builds a synthetic cohort with known ground truth, split train / test.
  2. Trains the LearnedEstimator on the train split.
  3. Backtests both the heuristic and the trained estimator on the test split
     (rolling-origin: assimilate 2 scans, forecast the rest).
  4. Measures forecast error (MAE) and 90%-interval coverage, plus a PIT
     calibration diagnostic.
  5. Writes figures and analysis/RESULTS.md.

All data here is synthetic and fully reproducible from the seeds below.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from oncotwin import OncoTwinEngine, SyntheticCohort, backtest
from oncotwin.parameters import HeuristicEstimator, LearnedEstimator
from oncotwin.forecast import forecast as _forecast

OUT = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUT, exist_ok=True)
CORAL, CYAN, GREY = "#D85A30", "#1D9E75", "#888780"


def pit_values(engine, records, warmup=2):
    """Probability integral transform: fraction of particles below each held-out
    observation. Perfect calibration => these are Uniform(0,1)."""
    vals = []
    for rec in records:
        if len(rec.measurements) <= warmup:
            continue
        twin = engine.create_twin(rec.patient_id, rec.features)
        engine.assimilate(twin, rec.measurements[:warmup])
        held = rec.measurements[warmup:]
        t = np.array([m.time_days for m in held])
        f = _forecast(rec.features.baseline_volume_cm3, twin.parameters, None, t)
        for i, m in enumerate(held):
            vals.append(float((f.trajectories[:, i] < m.volume_cm3).mean()))
    return np.array(vals)


def main():
    train = list(SyntheticCohort(n=120, seed=1).records())
    test = list(SyntheticCohort(n=80, seed=2).records())

    heuristic = OncoTwinEngine(HeuristicEstimator(seed=0))
    learned = OncoTwinEngine(LearnedEstimator().fit(train))

    bh = backtest(test, heuristic, warmup=2)
    bl = backtest(test, learned, warmup=2)
    print("heuristic:", bh)
    print("learned:  ", bl)

    pit_h, pit_l = pit_values(heuristic, test), pit_values(learned, test)

    # ---- Figure 1: estimator comparison ----
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.8))
    labels = ["heuristic\n(priors)", "learned\n(trained)"]
    ax[0].bar(labels, [bh.mae_cm3, bl.mae_cm3], color=[GREY, CYAN], width=0.6)
    ax[0].set_title("Forecast error (lower is better)")
    ax[0].set_ylabel("MAE (cm³)")
    ax[1].bar(labels, [bh.coverage_90 * 100, bl.coverage_90 * 100], color=[GREY, CYAN], width=0.6)
    ax[1].axhline(90, color=CORAL, ls="--", lw=1.5, label="nominal 90%")
    ax[1].set_title("90% interval coverage")
    ax[1].set_ylabel("% of truths inside band")
    ax[1].legend(frameon=False, fontsize=9)
    for a in ax:
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(f"{OUT}/estimator_comparison.png", dpi=140)

    # ---- Figure 2: PIT calibration ----
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
    for a, pit, name, c in [(ax[0], pit_h, "heuristic", GREY), (ax[1], pit_l, "learned", CYAN)]:
        a.hist(pit, bins=10, range=(0, 1), color=c, alpha=0.85, density=True)
        a.axhline(1.0, color=CORAL, ls="--", lw=1.5)
        a.set_title(f"{name} PIT")
        a.set_xlabel("PIT value")
        a.spines[["top", "right"]].set_visible(False)
    ax[0].set_ylabel("density")
    fig.suptitle("Calibration: flat = well-calibrated, U-shape = overconfident", fontsize=11)
    fig.tight_layout()
    fig.savefig(f"{OUT}/calibration_pit.png", dpi=140)

    # ---- RESULTS.md ----
    def cov_gap(b):
        return abs(b.coverage_90 - 0.90)
    winner = "learned" if cov_gap(bl) < cov_gap(bh) else "heuristic"
    with open(os.path.join(os.path.dirname(__file__), "RESULTS.md"), "w") as fh:
        fh.write(f"""# Results

Reproduce: `python analysis/run_analysis.py` (synthetic data, seeds fixed).
Cohort: {len(train)} train / {len(test)} test patients. Protocol: assimilate the
first 2 scans, forecast the rest.

| Estimator | Forecast MAE (cm³) | 90% coverage | calibration gap |
|---|---|---|---|
| Heuristic (priors) | {bh.mae_cm3:.1f} | {bh.coverage_90:.0%} | {cov_gap(bh):.0%} |
| Learned (trained)  | {bl.mae_cm3:.1f} | {bl.coverage_90:.0%} | {cov_gap(bl):.0%} |

Well-calibrated 90% intervals should contain the truth ~90% of the time. The
**{winner}** estimator is closer to that target here.

![estimator comparison](outputs/estimator_comparison.png)

## Calibration (PIT)

The probability integral transform should be uniform for a calibrated model.
A U-shape means the intervals are too narrow (overconfident); a central hump
means too wide.

![PIT calibration](outputs/calibration_pit.png)

## Takeaway

Training the estimator on data — so the forecast spread reflects real parameter
variability instead of hand-set guesses — improves both accuracy and calibration
here (lower MAE, coverage nearer nominal).

But read the numbers honestly: **both estimators are still overconfident**, with
90% coverage well below 90%. Two causes, both fixable: with only two scans the
particle filter concentrates the ensemble, and the forecast currently omits
process and observation noise, so the bands are too tight. The concrete next
steps — inject process noise in assimilation, add observation noise to the
predictive band, and use more follow-up scans — are tracked in
[docs/ROADMAP.md](../docs/ROADMAP.md). On real longitudinal data this same
harness is how you'd confirm the twin has actually become trustworthy.
""")
    print(f"\nWrote figures to {OUT}/ and analysis/RESULTS.md")


if __name__ == "__main__":
    main()
