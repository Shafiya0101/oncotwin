"""OncoTwin — Streamlit app.

A hosted interface over the real engine: patient setup, probabilistic forecasting,
treatment comparison, live recalibration, the imaging pipeline, and the 3D twin.

Run locally:   streamlit run streamlit_app.py
Deploy:        push to GitHub, then share.streamlit.io -> point at this file.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st

from oncotwin import (
    OncoTwinEngine, PatientFeatures, TreatmentPlan, TreatmentCourse,
    TreatmentKind, TumorMeasurement,
)
from oncotwin.growth import simulate

CORAL, CYAN, GREY = "#FF7A59", "#35D0BA", "#7C8B90"
TRUE = np.array([0.024, 800.0, 0.028, 0.030])          # hidden "reality" for the scan demo
SCAN_DAYS = [60, 120, 180]

st.set_page_config(page_title="OncoTwin", page_icon="🧬", layout="wide")


def plans():
    return {
        "No treatment": TreatmentPlan("No treatment", []),
        "Chemo (standard)": TreatmentPlan("Chemo (standard)", [
            TreatmentCourse(TreatmentKind.CHEMO, 30, 200, 1.0)]),
        "Chemo (intensive)": TreatmentPlan("Chemo (intensive)", [
            TreatmentCourse(TreatmentKind.CHEMO, 30, 200, 1.6)]),
        "Chemo + radiotherapy": TreatmentPlan("Chemo + radiotherapy", [
            TreatmentCourse(TreatmentKind.CHEMO, 30, 200, 1.0),
            TreatmentCourse(TreatmentKind.RADIO, 30, 75, 1.0)]),
    }


def style(ax):
    ax.set_facecolor("none")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#4A5B61")
    ax.tick_params(colors="#7C8B90")
    ax.xaxis.label.set_color("#7C8B90")
    ax.yaxis.label.set_color("#7C8B90")


engine = OncoTwinEngine()

# ---------------- sidebar: patient setup ---------------- #
st.sidebar.header("Patient")
age = st.sidebar.slider("Age", 40, 85, 63)
stage = st.sidebar.select_slider("Stage", options=[1, 2, 3, 4], value=3)
baseline = st.sidebar.slider("Baseline tumor volume (cm³)", 5.0, 120.0, 42.0)
ki67 = st.sidebar.slider("Ki-67 proliferation index", 0.05, 0.8, 0.35)
het = st.sidebar.slider("Radiomic heterogeneity", 0.0, 1.0, 0.6)
egfr = st.sidebar.checkbox("EGFR mutation", value=False)

features = PatientFeatures(age=age, stage=stage, histology="adenocarcinoma",
                           baseline_volume_cm3=baseline, ki67=ki67,
                           egfr_mutation=egfr, radiomic_heterogeneity=het)

# rebuild the twin when patient inputs change; otherwise keep accumulated scans
sig = (age, stage, baseline, ki67, het, egfr)
if st.session_state.get("sig") != sig:
    st.session_state.sig = sig
    st.session_state.twin = engine.create_twin("PT-APP", features)
    st.session_state.obs = []

twin = st.session_state.twin

st.title("🧬 OncoTwin")
st.caption("A living, probabilistic cancer patient digital twin. "
           "Illustrative model — not a medical device.")

tab_forecast, tab_imaging, tab_3d = st.tabs(
    ["Forecast & treatment", "Imaging → twin", "3D twin"])

# ---------------- tab 1: forecast ---------------- #
with tab_forecast:
    plan_name = st.radio("Treatment strategy", list(plans().keys()),
                         index=1, horizontal=True)
    plan = plans()[plan_name]

    belief = engine.explain(twin)
    resp = ("responder" if belief["chemo_sensitivity"] > 0.05
            else "partial" if belief["chemo_sensitivity"] > 0.03 else "resistant")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Doubling time", f"{belief['implied_doubling_time_days']:.0f} d")
    c2.metric("Chemo response", resp)
    c3.metric("Scans assimilated", belief["n_observations"])
    c4.metric("Twin version", f"v{belief['version']}")

    fc = engine.forecast(twin, plan, horizon_days=365)
    prior = st.session_state.get("prior")

    fig, ax = plt.subplots(figsize=(9, 4))
    fig.patch.set_alpha(0)
    if prior is not None:
        ax.fill_between(prior["t"], prior["lo"], prior["hi"], color=GREY, alpha=0.15)
        ax.plot(prior["t"], prior["med"], color=GREY, ls="--", lw=1.4,
                label="prior (before scans)")
    ax.fill_between(fc.t, fc.lower, fc.upper, color=CORAL, alpha=0.20, label="90% interval")
    ax.plot(fc.t, fc.median, color=CORAL, lw=2.4, label="median forecast")
    for o in st.session_state.obs:
        ax.scatter(o.time_days, o.volume_cm3, color=CYAN, zorder=5, s=45)
    ax.set_xlabel("Days from baseline"); ax.set_ylabel("Tumor volume (cm³)")
    ax.legend(frameon=False, labelcolor="#7C8B90")
    style(ax)
    st.pyplot(fig)

    col_a, col_b = st.columns([1, 1])
    with col_a:
        if st.button("🔬 Simulate follow-up scan", type="primary",
                     disabled=len(st.session_state.obs) >= 3):
            if not st.session_state.obs:
                st.session_state.prior = {"t": fc.t, "med": fc.median,
                                          "lo": fc.lower, "hi": fc.upper}
            day = SCAN_DAYS[len(st.session_state.obs)]
            vol = float(max(1.0, simulate(baseline, TRUE, plan, np.array([day]))[0]
                            + np.random.normal(0, 2)))
            m = TumorMeasurement(day, vol)
            st.session_state.obs.append(m)
            engine.assimilate(twin, [m], plan)
            st.rerun()
    with col_b:
        if st.button("Reset scans"):
            st.session_state.twin = engine.create_twin("PT-APP", features)
            st.session_state.obs = []
            st.session_state.prior = None
            st.rerun()

    st.subheader("Compare all strategies")
    fig2, ax2 = plt.subplots(figsize=(9, 4))
    fig2.patch.set_alpha(0)
    colors = {"No treatment": GREY, "Chemo (standard)": "#378ADD",
              "Chemo (intensive)": "#7F77DD", "Chemo + radiotherapy": CYAN}
    rows = []
    for name, p in plans().items():
        f = engine.forecast(twin, p, horizon_days=365)
        ax2.plot(f.t, f.median, color=colors[name], lw=2, label=name)
        rows.append((name, f.median[-1]))
    ax2.set_xlabel("Days from baseline"); ax2.set_ylabel("Tumor volume (cm³)")
    ax2.legend(frameon=False, labelcolor="#7C8B90", fontsize=8)
    style(ax2)
    st.pyplot(fig2)
    best = min(rows, key=lambda r: r[1])
    st.caption(f"Lowest predicted burden at 1 year: **{best[0]}** "
               f"({best[1]:.0f} cm³ median).")

# ---------------- tab 2: imaging ---------------- #
with tab_imaging:
    st.write("The multimodal pipeline: a scan is segmented, radiomic features are "
             "extracted, and fused with the clinical data in the sidebar to drive the twin.")
    if st.button("Run imaging pipeline on a phantom scan"):
        from oncotwin.imaging import synthetic_scan, segment_tumor, extract_radiomics, fuse
        vol, spacing, true_v = synthetic_scan(seed=2)
        mask = segment_tumor(vol)
        feats = extract_radiomics(vol, mask, spacing)
        z = int(np.median(np.argwhere(mask)[:, 0]))
        cimg, cfeat = st.columns([2, 1])
        with cimg:
            fig3, ax3 = plt.subplots(1, 2, figsize=(7, 3.5))
            fig3.patch.set_alpha(0)
            ax3[0].imshow(vol[z], cmap="gray"); ax3[0].set_title("Scan", color="#E4EAEC")
            ax3[1].imshow(vol[z], cmap="gray")
            ax3[1].contour(mask[z], colors=[CYAN], linewidths=1.5)
            ax3[1].set_title("Segmented tumor", color="#E4EAEC")
            for a in ax3:
                a.set_xticks([]); a.set_yticks([])
            st.pyplot(fig3)
        with cfeat:
            st.metric("Segmented volume", f"{feats.volume_cm3:.1f} cm³")
            st.metric("Sphericity", f"{feats.sphericity:.2f}")
            st.metric("Heterogeneity", f"{feats.heterogeneity:.2f}")
        st.success("These imaging features are fused with the sidebar's clinical "
                   "data to build the twin.")

# ---------------- tab 3: embedded 3D ---------------- #
with tab_3d:
    st.write("The metaverse-inspired 3D view (runs in your browser).")
    html_path = os.path.join(os.path.dirname(__file__), "web", "index.html")
    try:
        with open(html_path, encoding="utf-8") as fh:
            st.components.v1.html(fh.read(), height=900, scrolling=True)
    except FileNotFoundError:
        st.info("3D view file not found (web/index.html).")
