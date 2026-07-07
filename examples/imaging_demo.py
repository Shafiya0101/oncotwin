"""Multimodal demo: a scan becomes a twin.

Run:  python examples/imaging_demo.py

Generates a phantom scan, segments the tumor, extracts radiomic features, fuses
them with clinical data, builds a twin, and forecasts — then saves a figure
showing the whole path.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from oncotwin.imaging import synthetic_scan, segment_tumor, extract_radiomics, fuse
from oncotwin import OncoTwinEngine, TreatmentPlan, TreatmentCourse, TreatmentKind

OUT = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUT, exist_ok=True)
CORAL, CYAN = "#D85A30", "#1D9E75"


def main():
    # 1. imaging
    vol, spacing, true_v = synthetic_scan(seed=2)
    mask = segment_tumor(vol)
    feats = extract_radiomics(vol, mask, spacing)

    # 2. fuse with clinical / molecular data
    clinical = {"age": 63, "stage": 3, "ki67": 0.35, "egfr_mutation": False}
    features = fuse(feats, clinical)

    # 3. twin + forecast
    engine = OncoTwinEngine()
    twin = engine.create_twin("PT-IMG", features)
    plan = TreatmentPlan("Chemo (standard)",
                         [TreatmentCourse(TreatmentKind.CHEMO, 30, 200, 1.0)])
    fc = engine.forecast(twin, plan, horizon_days=365)

    z = int(np.median(np.argwhere(mask)[:, 0]))

    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    ax[0].imshow(vol[z], cmap="gray")
    ax[0].set_title("Scan (axial slice)")
    ax[1].imshow(vol[z], cmap="gray")
    ax[1].contour(mask[z], colors=[CYAN], linewidths=1.5)
    ax[1].set_title("Segmented tumor")
    for a in ax[:2]:
        a.set_xticks([]); a.set_yticks([])

    ax[2].fill_between(fc.t, fc.lower, fc.upper, color=CORAL, alpha=0.18)
    ax[2].plot(fc.t, fc.median, color=CORAL, lw=2.2)
    ax[2].set_title("Twin forecast from fused features")
    ax[2].set_xlabel("Days"); ax[2].set_ylabel("Tumor volume (cm³)")
    ax[2].spines[["top", "right"]].set_visible(False)

    cap = (f"imaging → volume {feats.volume_cm3:.1f} cm³, sphericity {feats.sphericity:.2f}, "
           f"heterogeneity {feats.heterogeneity:.2f}   +   clinical → stage {clinical['stage']}, "
           f"Ki-67 {clinical['ki67']}")
    fig.suptitle("Multimodal pipeline: medical imaging + clinical data → patient twin", fontsize=12)
    fig.text(0.5, 0.005, cap, ha="center", fontsize=9, color="#555")
    fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    fig.savefig(f"{OUT}/multimodal_pipeline.png", dpi=140)

    print(f"true tumor volume:      {true_v:.1f} cm³")
    print(f"segmented volume:       {feats.volume_cm3:.1f} cm³")
    print(f"radiomic features:      {feats.as_dict()}")
    print(f"forecast @ day 365:     {fc.summary(365)}")
    print(f"saved {OUT}/multimodal_pipeline.png")


if __name__ == "__main__":
    main()
