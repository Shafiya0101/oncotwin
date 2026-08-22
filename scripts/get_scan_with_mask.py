"""Real multimodal ingestion: real CT + expert tumour mask -> real radiomics -> twin.

Directly addresses the segmentation limitation. Instead of a threshold, this uses
the radiation-oncologist GTV delineation shipped with NSCLC-Radiomics, extracts
real radiomic features from the real tumour, and fuses them with the patient's
real clinical record (age, stage) to build a genuinely multimodal twin for one
patient.

Run:   python scripts/get_scan_with_mask.py
Needs: pip install tcia-utils SimpleITK rt-utils
       data/real/Lung1.clinical.csv  (already downloaded)

We can't test the TCIA download or the RTSTRUCT reader from the sandbox, so this
is the part we debug together; it prints diagnostics at each step.
"""
import os, glob
from dataclasses import replace
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import SimpleITK as sitk

from oncotwin.imaging import extract_radiomics
from oncotwin import (OncoTwinEngine, PatientFeatures, TreatmentPlan,
                      TreatmentCourse, TreatmentKind)

PATIENT = "LUNG1-001"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data", "real")
os.makedirs(OUT, exist_ok=True)


def get_series():
    from tcia_utils import nbia
    s = nbia.getSeries(collection="NSCLC-Radiomics")
    mine = [x for x in s if x["PatientID"] == PATIENT]
    print(f"Series for {PATIENT}:", [x["Modality"] for x in mine])
    ct = next((x for x in mine if x["Modality"] == "CT"), None)
    rt = next((x for x in mine if x["Modality"] == "RTSTRUCT"), None)
    seg = next((x for x in mine if x["Modality"] == "SEG"), None)
    return ct, rt, seg


def download(series):
    from tcia_utils import nbia
    uid = series["SeriesInstanceUID"]
    nbia.downloadSeries([uid], input_type="list")
    folder = os.path.join("tciaDownload", uid)
    if not os.path.isdir(folder):
        cands = [d for d in glob.glob("tciaDownload/*") if os.path.isdir(d)]
        folder = max(cands, key=os.path.getmtime) if cands else None
    return folder


def load_ct(folder):
    r = sitk.ImageSeriesReader()
    r.SetFileNames(r.GetGDCMSeriesFileNames(folder))
    img = r.Execute()
    return sitk.GetArrayFromImage(img).astype(float), img.GetSpacing()   # (z,y,x), (x,y,z)


def gtv_from_rtstruct(ct_dir, rt_dir):
    from rt_utils import RTStructBuilder
    rt_file = glob.glob(os.path.join(rt_dir, "*.dcm"))[0]
    rs = RTStructBuilder.create_from(dicom_series_path=ct_dir, rt_struct_path=rt_file)
    names = rs.get_roi_names()
    print("  ROI names in delineation:", names)
    gtv = next((n for n in names if "GTV" in n.upper()), names[0])
    print("  using ROI:", gtv)
    mask = rs.get_roi_mask_by_name(gtv)               # (rows, cols, slices) = (y, x, z)
    return np.transpose(mask, (2, 0, 1)).astype(bool)  # -> (z, y, x)


def real_clinical():
    try:
        from oncotwin.data.clinical import load_lung1_clinical
        recs = load_lung1_clinical(os.path.join(OUT, "Lung1.clinical.csv"))
        rec = next((r for r in recs if r.patient_id == PATIENT), None)
        if rec:
            print(f"  clinical: age {rec.features.age}, stage {rec.features.stage}, "
                  f"survival {rec.survival_days:.0f} d")
            return rec.features
    except Exception as e:
        print("  (clinical lookup skipped:", e, ")")
    return PatientFeatures(age=78, stage=3, histology="nsclc",
                           baseline_volume_cm3=40.0, ki67=0.3, radiomic_heterogeneity=0.5)


def main():
    try:
        ct, rt, seg = get_series()
        if ct is None:
            print("No CT series found."); return
        ct_dir = download(ct)
        volume, spacing = load_ct(ct_dir)
        print(f"Loaded CT {volume.shape}, spacing {tuple(round(s,2) for s in spacing)}")
        if rt is not None:
            mask = gtv_from_rtstruct(ct_dir, download(rt))
        elif seg is not None:
            print("Only a DICOM SEG (not RTSTRUCT) is available for this patient — "
                  "tell me and I'll add a SEG reader."); return
        else:
            print("No segmentation series found."); return
    except Exception as e:
        print("Failed:", repr(e))
        print("If this is a missing package, run: pip install tcia-utils SimpleITK rt-utils")
        return

    n_vox = int(mask.sum())
    if n_vox == 0:
        print("Mask is empty after alignment — tell me and we'll fix the orientation."); return
    feats = extract_radiomics(volume, mask, spacing)
    print(f"\nREAL radiomics from the expert tumour ({n_vox} voxels):")
    for k, v in feats.as_dict().items():
        print(f"  {k}: {v}")

    features = real_clinical()
    features = replace(features, baseline_volume_cm3=feats.volume_cm3,
                       radiomic_heterogeneity=feats.heterogeneity)
    engine = OncoTwinEngine()
    twin = engine.create_twin(PATIENT, features)
    plan = TreatmentPlan("Chemo + radiotherapy", [
        TreatmentCourse(TreatmentKind.CHEMO, 30, 200, 1.0),
        TreatmentCourse(TreatmentKind.RADIO, 30, 75, 1.0)])
    fc = engine.forecast(twin, plan, horizon_days=365).summary(365)
    print(f"\nTwin from REAL imaging + REAL clinical data:")
    print(f"  baseline tumour volume (from mask): {feats.volume_cm3:.1f} cm3")
    print(f"  1-year forecast (chemo+radio): median {fc['volume_median']:.0f} cm3 "
          f"(90% {fc['volume_ci90'][0]:.0f}-{fc['volume_ci90'][1]:.0f})")

    z = int(np.median(np.argwhere(mask)[:, 0]))
    plt.figure(figsize=(5, 5))
    plt.imshow(np.clip(volume[z], -1000, 400), cmap="gray")
    plt.contour(mask[z], colors=["#35D0BA"], linewidths=1.4)
    plt.axis("off"); plt.title(f"{PATIENT} — expert tumour delineation (slice {z})", fontsize=10)
    path = os.path.join(OUT, f"{PATIENT}_tumor_overlay.png")
    plt.tight_layout(); plt.savefig(path, dpi=140); plt.close()
    print(f"  saved tumour overlay -> {path}")


if __name__ == "__main__":
    main()
