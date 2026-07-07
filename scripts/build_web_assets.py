"""Export real pipeline outputs for the web UI as a single assets.js file.

Generates an irregular (multi-lobulated) tumor phantom — closer to real tumor
morphology than a sphere — segments it, and exports:
  - the actual marching-cubes tumor mesh (for the 3D view)
  - base64 scan slice + segmentation overlay (for the Imaging tab)
  - radiomic features and forecast feature-drivers

Loaded via <script src="assets.js"> so it works both from file:// and on a host.
Run:  python scripts/build_web_assets.py
"""
import os, io, json, base64
import numpy as np
from scipy.ndimage import gaussian_filter
from skimage import measure
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from oncotwin.imaging import segment_tumor, extract_radiomics
from oncotwin import (
    OncoTwinEngine, PatientFeatures, explain_drivers,
    TreatmentPlan, TreatmentCourse, TreatmentKind,
)

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def irregular_phantom(shape=(64, 64, 52), spacing=(1.0, 1.0, 1.5), seed=3):
    rng = np.random.default_rng(seed)
    vol = 30.0 + rng.normal(0, 6, shape)
    zz, yy, xx = np.indices(shape)
    c = np.array(shape) / 2.0
    lobes = [((0, 0, 0), 11), ((5, 5, -3), 7), ((-5, 4, 3), 6), ((3, -6, 4), 5)]
    tumor = np.zeros(shape, bool)
    for (dz, dy, dx), r in lobes:
        d = np.sqrt(((zz - c[0] - dz) * spacing[2]) ** 2 +
                    ((yy - c[1] - dy) * spacing[1]) ** 2 +
                    ((xx - c[2] - dx) * spacing[0]) ** 2)
        tumor |= d <= r
    vol[tumor] = 200.0 + rng.normal(0, 35, tumor.sum())
    return gaussian_filter(vol, 0.8), spacing


def to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=90, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def main():
    vol, spacing = irregular_phantom()
    mask = segment_tumor(vol)
    feats = extract_radiomics(vol, mask, spacing)

    verts, faces, _, _ = measure.marching_cubes(mask.astype(float), level=0.5,
                                                spacing=tuple(np.array(spacing)[::-1]))
    verts = verts - verts.mean(0)
    verts = verts / np.abs(verts).max()
    V = [[round(float(x), 3) for x in v] for v in verts]
    F = [[int(i) for i in f] for f in faces]

    z = int(np.median(np.argwhere(mask)[:, 0]))
    fig, ax = plt.subplots(figsize=(3, 3)); ax.imshow(vol[z], cmap="gray"); ax.axis("off")
    scan = to_b64(fig)
    fig, ax = plt.subplots(figsize=(3, 3)); ax.imshow(vol[z], cmap="gray")
    ax.contour(mask[z], colors=["#35D0BA"], linewidths=1.3); ax.axis("off")
    overlay = to_b64(fig)

    eng = OncoTwinEngine()
    features = PatientFeatures(age=63, stage=3, histology="adenocarcinoma",
                               baseline_volume_cm3=feats.volume_cm3, ki67=0.35,
                               radiomic_heterogeneity=feats.heterogeneity)
    plan = TreatmentPlan("Chemo (standard)",
                         [TreatmentCourse(TreatmentKind.CHEMO, 30, 200, 1.0)])
    drivers = [{"label": d.label, "effect": d.effect_cm3, "direction": d.direction}
               for d in explain_drivers(eng, features, plan)]

    out = os.path.join(HERE, "assets.js")
    with open(out, "w") as f:
        f.write("window.TUMOR_MESH=" + json.dumps({"vertices": V, "faces": F}) + ";\n")
        f.write('window.SCAN_SLICE="data:image/png;base64,' + scan + '";\n')
        f.write('window.SCAN_OVERLAY="data:image/png;base64,' + overlay + '";\n')
        f.write("window.IMAGING_FEATURES=" + json.dumps(feats.as_dict()) + ";\n")
        f.write("window.DRIVERS=" + json.dumps(drivers) + ";\n")
    print(f"mesh: {len(V)} verts / {len(F)} faces")
    print(f"radiomics: {feats.as_dict()}")
    print(f"wrote {out} ({os.path.getsize(out)//1024} KB)")


if __name__ == "__main__":
    main()
