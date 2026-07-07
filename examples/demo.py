"""End-to-end demonstration of the OncoTwin engine.

Run:  python demo.py
Produces two figures in outputs/ and prints the twin's evolving belief.

Panel 1 - counterfactuals: one twin, four treatment strategies, each a
    probabilistic trajectory with a 90% band. This is the 'personalized
    strategy' view a tumor board would explore.

Panel 2 - the living twin: we hide a 'true' patient and let reality unfold.
    The twin starts with a wide prior forecast, then recalibrates as two
    follow-up scans arrive, locking onto the patient's actual trajectory.
    This is the difference between a twin and a static predictor.
"""
from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from oncotwin import (
    OncoTwinEngine, PatientFeatures, TreatmentPlan, TreatmentCourse,
    TreatmentKind, TumorMeasurement,
)
from oncotwin.growth import simulate

import os
OUT = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUT, exist_ok=True)
BLUE, CORAL, TEAL, PURPLE, GREY = "#378ADD", "#D85A30", "#1D9E75", "#7F77DD", "#888780"


def make_plans() -> list[TreatmentPlan]:
    return [
        TreatmentPlan("No treatment", []),
        TreatmentPlan("Chemo (standard)", [
            TreatmentCourse(TreatmentKind.CHEMO, 30, 200, intensity=1.0)]),
        TreatmentPlan("Chemo (intensive)", [
            TreatmentCourse(TreatmentKind.CHEMO, 30, 200, intensity=1.6)]),
        TreatmentPlan("Chemo + radiotherapy", [
            TreatmentCourse(TreatmentKind.CHEMO, 30, 200, intensity=1.0),
            TreatmentCourse(TreatmentKind.RADIO, 30, 75, intensity=1.0)]),
    ]


def panel_counterfactuals(engine, twin):
    plans = make_plans()
    forecasts = engine.simulate_counterfactuals(twin, plans, horizon_days=365)

    fig, ax = plt.subplots(figsize=(9, 5.2))
    colors = [GREY, BLUE, PURPLE, TEAL]
    for (name, fc), c in zip(forecasts.items(), colors):
        ax.fill_between(fc.t, fc.lower, fc.upper, color=c, alpha=0.15)
        ax.plot(fc.t, fc.median, color=c, lw=2.2, label=name)
        print("  ", fc.summary(horizon_day=365))

    ax.set_title("Counterfactual treatment strategies for one patient twin\n"
                 "(median trajectory + 90% uncertainty band)", fontsize=12)
    ax.set_xlabel("Days from baseline")
    ax.set_ylabel("Tumor volume (cm$^3$)")
    ax.legend(frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(f"{OUT}/forecast_counterfactuals.png", dpi=140)
    print(f"  saved {OUT}/forecast_counterfactuals.png")


def panel_living_twin(engine, twin):
    """Hidden ground-truth patient; twin recalibrates as scans arrive."""
    plan = TreatmentPlan("Chemo (standard)", [
        TreatmentCourse(TreatmentKind.CHEMO, 30, 200, intensity=1.0)])
    twin.plan = plan

    # A 'true' patient whose parameters differ from the prior mean: faster
    # growth and a weaker-than-expected chemo response (a partial responder).
    # Reality, unknown to the twin until scans arrive.
    true_params = np.array([0.024, 800.0, 0.028, 0.030])
    v0 = twin.features.baseline_volume_cm3
    t = np.arange(0.0, 365 + 5, 5.0)
    truth = simulate(v0, true_params, plan, t)
    rng = np.random.default_rng(1)

    fc_prior = engine.forecast(twin, plan, horizon_days=365)

    # Two follow-up scans at days 60 and 120 (with imaging noise).
    obs_days = [60.0, 120.0]
    measurements = []
    for d in obs_days:
        v = float(np.interp(d, t, truth)) + rng.normal(0, 2.0)
        measurements.append(TumorMeasurement(d, v, "imaging", noise_sd=2.0))

    engine.assimilate(twin, measurements, plan)
    fc_post = engine.forecast(twin, plan, horizon_days=365)

    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.fill_between(fc_prior.t, fc_prior.lower, fc_prior.upper, color=GREY,
                    alpha=0.18, label="Prior forecast (before data)")
    ax.plot(fc_prior.t, fc_prior.median, color=GREY, lw=1.8, ls="--")

    ax.fill_between(fc_post.t, fc_post.lower, fc_post.upper, color=CORAL,
                    alpha=0.22, label="Recalibrated forecast")
    ax.plot(fc_post.t, fc_post.median, color=CORAL, lw=2.4)

    ax.plot(t, truth, color="black", lw=1.6, label="Patient's true trajectory")
    ax.scatter([m.time_days for m in measurements],
               [m.volume_cm3 for m in measurements],
               color="black", zorder=5, s=45, label="Follow-up scans")

    ax.set_title("The living twin: forecast recalibrates as real scans arrive",
                 fontsize=12)
    ax.set_xlabel("Days from baseline")
    ax.set_ylabel("Tumor volume (cm$^3$)")
    ax.legend(frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(f"{OUT}/living_twin_recalibration.png", dpi=140)
    print(f"  saved {OUT}/living_twin_recalibration.png")


def main():
    engine = OncoTwinEngine()
    features = PatientFeatures(
        age=63, stage=3, histology="adenocarcinoma",
        baseline_volume_cm3=42.0, ki67=0.35,
        egfr_mutation=False, radiomic_heterogeneity=0.6,
    )
    twin = engine.create_twin("PT-0001", features)

    print("Twin created. Belief before any follow-up:")
    print("  ", engine.explain(twin))

    print("\nPanel 1 - counterfactual strategies:")
    panel_counterfactuals(engine, twin)

    # Fresh twin for the living-twin panel (same patient, clean prior).
    twin2 = engine.create_twin("PT-0001", features)
    print("\nPanel 2 - living twin recalibration:")
    panel_living_twin(engine, twin2)
    print("\nBelief AFTER assimilating 2 scans:")
    print("  ", engine.explain(twin2))


if __name__ == "__main__":
    main()
