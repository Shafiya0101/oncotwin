import os, sys, pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D
from PIL import Image
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.environ["REPO"], "src")); sys.path.insert(0, HERE)
from exp_calib import make_cohort, SCAN_DAYS
from oncotwin.parameters import LearnedEstimator
from oncotwin.growth import simulate

plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
                     "font.size": 7.5, "axes.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
                     "mathtext.fontset": "stix"})
BLUE, ORANGE, GREEN, GREY, RED, GOLD = "#0072B2", "#D55E00", "#009E73", "#6b6b6b", "#CC3311", "#E69F00"
OUT = os.path.join(HERE, "figs"); os.makedirs(OUT, exist_ok=True)
IM = os.path.join(HERE, "..", "imgs"); SH = os.path.join(HERE, "..", "shot")
A = pickle.load(open(os.path.join(HERE, "out/analysis.pkl"), "rb")); res, sweep, grid = A["res"], A["sweep"], A["grid"]

# ---------- example patient (same selection logic as before) ----------
train = make_cohort("well", 120, 1000); est = LearnedEstimator(seed=0).fit(train); test = make_cohort("well", 100, 3000)
rng = np.random.default_rng(5); PN = 0.0025; tg = np.arange(0, 361, 6.0)
def pipeline(rec):
    ens = est.estimate(rec.features, 400); v0 = rec.features.baseline_volume_cm3
    traj = lambda P: np.array([simulate(v0, p, None, tg) for p in P])
    prior = traj(ens.particles)
    tw = np.array([60., 120.]); yw = np.array([m.volume_cm3 for m in rec.measurements[:2]])
    ll = np.array([-0.5 * np.sum(((simulate(v0, p, None, tw) - yw) / 2.0) ** 2) for p in ens.particles])
    w = np.exp(ll - ll.max()); w /= w.sum()
    pos = (np.arange(400) + rng.random()) / 400
    P = ens.particles[np.clip(np.searchsorted(np.cumsum(w), pos), 0, 399)]
    P = (P + rng.normal(0, 0.02 * P.std(axis=0), P.shape)).clip(1e-5, None)
    post = traj(P); sig = PN * np.sqrt(tg)[None, :]
    return v0, prior, post, post * np.exp(rng.normal(0, 1, post.shape) * sig - 0.5 * sig ** 2)
hi_idx = [np.argmin(abs(tg - d)) for d in (180, 240, 300, 360)]
for k, rec in enumerate(test[:40]):
    v0, prior, post, post_pn = pipeline(rec)
    obs = np.array([m.volume_cm3 for m in rec.measurements[2:]])
    lo, hi = np.quantile(post, [.05, .95], 0); lo2, hi2 = np.quantile(post_pn, [.05, .95], 0)
    miss = sum(not (lo[i] <= o <= hi[i]) for i, o in zip(hi_idx, obs)); keep = sum(lo2[i] <= o <= hi2[i] for i, o in zip(hi_idx, obs))
    if miss >= 2 and keep == 4: break
truth = simulate(v0, rec.true_params, None, tg); obs_t = np.array(SCAN_DAYS, float); obs_y = np.array([m.volume_cm3 for m in rec.measurements])

# ---------- Fig 1: imaging-centred overview ----------
fig = plt.figure(figsize=(7.1, 2.3))
gs = fig.add_gridspec(1, 4, width_ratios=[0.9, 0.9, 2.2, 1.45], wspace=0.3, left=0.005, right=0.995, top=0.9, bottom=0.17)
ct = np.array(Image.open(f"{IM}/p9-001.png")); gtv = np.array(Image.open(f"{IM}/p10-000.png"))
for i, (img, t) in enumerate([(ct, "(a) Real CT, axial"), (gtv, "(b) Expert GTV-1 contour")]):
    ax = fig.add_subplot(gs[i]); ax.imshow(img[110:640, 90:610]); ax.axis("off"); ax.set_title(t, fontsize=7, pad=2)
ax = fig.add_subplot(gs[2]); ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 6.2); ax.set_title("(c) Imaging-to-forecast pipeline", fontsize=7, pad=2)
def box(x, y, w_, h, txt, fc="#eef3f8", ec=BLUE):
    ax.add_patch(FancyBboxPatch((x, y), w_, h, boxstyle="round,pad=0.02,rounding_size=0.12", fc=fc, ec=ec, lw=0.7))
    ax.text(x + w_ / 2, y + h / 2, txt, ha="center", va="center", fontsize=6.3, linespacing=1.15)
def arr(x1, y1, x2, y2, c="black", ls="-"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=6, lw=0.7, color=c, linestyle=ls))
box(0.1, 4.3, 2.9, 1.6, "CT volume (HU)\n+ GTV mask\n(DICOM/NIfTI)")
box(3.5, 4.3, 2.9, 1.6, "Radiomics: volume,\nsurface, sphericity,\ndiameter, GLCM texture")
box(6.9, 4.3, 2.9, 1.6, "Fuse with clinical\nage, stage, Ki-67\n$\\rightarrow$ features $\\mathbf{x}$")
box(6.9, 1.7, 2.9, 1.6, "Estimator\n$p(\\theta|\\mathbf{x})$ +\nGompertz ODE")
box(3.5, 1.7, 2.9, 1.6, "Predictive\ndistribution $V(t)$\n+ process noise", fc="#fbeee6", ec=ORANGE)
box(0.1, 1.7, 2.9, 1.6, "Follow-up CT:\nre-segment,\nre-weight $\\theta$", fc="#e8f5ef", ec=GREEN)
arr(3.0, 5.1, 3.5, 5.1); arr(6.4, 5.1, 6.9, 5.1); arr(8.35, 4.3, 8.35, 3.3); arr(6.9, 2.5, 6.4, 2.5); arr(3.5, 2.5, 3.0, 2.5, c=GREEN); arr(1.55, 3.3, 1.55, 4.3, c=GREEN, ls="--")
ax.text(5.0, 0.6, "segmentation: expert mask (real CT) or Otsu + largest component (phantoms)", fontsize=6.2, ha="center", style="italic", color=GREY)
ax = fig.add_subplot(gs[3])
ax.fill_between(tg, *np.quantile(prior, [.05, .95], 0), color=GREY, alpha=.25, lw=0, label="prior 90%")
ax.fill_between(tg, *np.quantile(post, [.05, .95], 0), color=RED, alpha=.55, lw=0, label="naive posterior")
ax.fill_between(tg, *np.quantile(post_pn, [.05, .95], 0), color=BLUE, alpha=.25, lw=0, label="recalibrated")
ax.plot(tg, np.median(post_pn, 0), color=BLUE, lw=1.0); ax.plot(tg, truth, color="k", lw=0.9, label="true volume")
ax.scatter(obs_t[:2], obs_y[:2], s=12, color="k", zorder=5, label="assimilated")
ax.scatter(obs_t[2:], obs_y[2:], s=12, facecolor="white", edgecolor="k", zorder=5, lw=0.8, label="held out")
ax.axvline(120, color=GREY, lw=0.5, ls=":"); ax.set_xlabel("days from baseline", labelpad=1); ax.set_ylabel("volume (cm$^3$)", labelpad=1)
ax.set_title("(d) Forecast (synthetic patient)", fontsize=7, pad=2); ax.legend(fontsize=5.6, frameon=False, loc="upper left", handlelength=1.2, labelspacing=0.2); ax.set_ylim(0, None)
fig.savefig(f"{OUT}/fig1.pdf"); fig.savefig(f"{OUT}/fig1.png", dpi=200); plt.close(fig)

# ---------- Fig 2: dashboards (screenshots of the app) ----------
tw = Image.open(f"{SH}/full_twin.png").crop((160, 800, 2435, 1885))
im_ = Image.open(f"{SH}/full_imaging.png").crop((160, 235, 2437, 1280))
cl = Image.open(f"{IM}/p11-000.png")
fig = plt.figure(figsize=(7.1, 3.15))
gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 0.95], height_ratios=[tw.size[1] / tw.size[0], im_.size[1] / im_.size[0]], wspace=0.03, hspace=0.12, left=0.005, right=0.995, top=0.94, bottom=0.005)
for (r, c, img, t) in [(0, 0, tw, "(a) Twin tab: 3D segmented tumour, treatment scenarios, feature drivers"),
                       (1, 0, im_, "(b) Imaging tab: axial slice, segmentation, radiomic features"),
                       ]:
    ax = fig.add_subplot(gs[r, c]); ax.imshow(img); ax.axis("off"); ax.set_title(t, fontsize=6.6, pad=1.5, loc="left")
ax = fig.add_subplot(gs[:, 1]); ax.imshow(cl); ax.axis("off"); ax.set_title("(c) Clinician upload app: scans over time, recalibrated forecast", fontsize=6.6, pad=1.5, loc="left")
fig.savefig(f"{OUT}/fig_dash.pdf"); fig.savefig(f"{OUT}/fig_dash.png", dpi=220); plt.close(fig)

# ---------- Fig 3: imaging validation + calibration ----------
rows = pickle.load(open(os.path.join(HERE, "out/imaging.pkl"), "rb"))
fig, axs = plt.subplots(1, 4, figsize=(7.1, 1.85), gridspec_kw=dict(wspace=0.55, left=0.055, right=0.995, top=0.86, bottom=0.2))
cols = {6: "#9ecae1", 9: "#6baed6", 14: "#3182bd", 20: "#08519c", 26: "#08306b"}
ax = axs[0]
for rad, col in cols.items():
    d = [np.mean([x["dice"] for x in rows if x["radius"] == rad and x["ns"] == n]) for n in (1, 2, 3, 4, 5)]
    ax.plot([1, 2, 3, 4, 5], d, color=col, marker="o", ms=2.5, lw=0.9, label=f"{np.mean([x['true_vol'] for x in rows if x['radius']==rad]):.1f} cm$^3$")
ax.set_xlabel("noise level ($\\times$ baseline)", labelpad=1); ax.set_ylabel("Dice", labelpad=1); ax.set_ylim(0.9, 1.0)
ax.set_title("(a) Segmentation", fontsize=7, pad=2); ax.legend(fontsize=5.3, frameon=False, loc="lower left", handlelength=1.2, labelspacing=0.15, title="tumour vol.", title_fontsize=5.3)
ax = axs[1]
for rad, col in cols.items():
    d = [np.mean([abs(x["vol_err"]) for x in rows if x["radius"] == rad and x["ns"] == n]) for n in (1, 2, 3, 4, 5)]
    ax.plot([1, 2, 3, 4, 5], d, color=col, marker="o", ms=2.5, lw=0.9)
ax.set_xlabel("noise level ($\\times$ baseline)", labelpad=1); ax.set_ylabel("|volume error| (%)", labelpad=1); ax.set_title("(b) Volume radiomics", fontsize=7, pad=2)
ax = axs[2]; lv = [0.5, 0.8, 0.9]
mean_cov = lambda w, k: [np.mean([r[c] for r in res[w]["rows"][k]]) for c in ("c50", "c80", "c90")]
ax.plot([0.4, 1], [0.4, 1], color=GREY, lw=0.6, ls="--")
for w, col, mk in (("well", BLUE, "o"), ("mis", ORANGE, "s")):
    ax.plot(lv, mean_cov(w, "A_naive"), ls=":", color=col, marker=mk, ms=3, lw=0.9, mfc="white")
    ax.plot(lv, mean_cov(w, "D_pn_tuned_obs"), color=col, marker=mk, ms=3, lw=0.9)
ax.set_xlabel("nominal level", labelpad=1); ax.set_ylabel("empirical coverage", labelpad=1); ax.set_xlim(0.45, 0.95); ax.set_ylim(0, 1)
ax.set_title("(c) Forecast reliability", fontsize=7, pad=2)
ax.legend(handles=[Line2D([], [], color=BLUE, marker="o", ms=3, lw=0.9, label="well-spec."), Line2D([], [], color=ORANGE, marker="s", ms=3, lw=0.9, label="misspec."),
                   Line2D([], [], color="k", ls=":", lw=0.9, label="naive"), Line2D([], [], color="k", lw=0.9, label="recalibr.")], fontsize=5.3, frameon=False, loc="upper left", labelspacing=0.15, handlelength=1.5)
ax = axs[3]
for w, col in (("well", BLUE), ("mis", ORANGE)):
    ax.plot(grid * 1000, sweep[(w, False)], color=col, lw=1.0); ax.axvline(np.mean(res[w]["tuned"]["C"]) * 1000, color=col, lw=0.5, ls=":")
ax.axhline(0.9, color="k", lw=0.5, ls="--"); ax.axvline(3.0, color=GREY, lw=0.8, ls="-.")
ax.text(3.3, 0.06, "default", fontsize=5.3, color=GREY)
ax.set_xlabel("process noise $\\sigma$ ($\\times10^{-3}$)", labelpad=1); ax.set_ylabel("90% coverage", labelpad=1); ax.set_ylim(0, 1.02); ax.set_xlim(0, 15)
ax.set_title("(d) Coverage vs. $\\sigma$", fontsize=7, pad=2)
fig.savefig(f"{OUT}/fig3.pdf"); fig.savefig(f"{OUT}/fig3.png", dpi=200); plt.close(fig)
print("done")
