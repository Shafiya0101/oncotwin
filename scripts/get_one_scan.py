"""Ingest ONE real CT scan from TCIA (NSCLC-Radiomics) into the pipeline.

Proves the twin can consume real patient imaging end to end: it downloads a
single patient's CT (~70 MB, public, no login), loads it, saves a real axial
slice you can look at, and reports scan geometry and intensity statistics.

Run:  python scripts/get_one_scan.py
Needs: pip install tcia-utils SimpleITK

Honest scope: this loads the real scan and proves ingestion. Extracting
tumor-specific radiomics needs the expert GTV outline (RTSTRUCT/SEG) that ships
with this dataset — a simple threshold can't find a tumor in a full CT (bone is
brighter than tumor). That masked-radiomics step, run across the cohort, is what
would move the survival C-index, and is the documented next step.
"""
import os
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import SimpleITK as sitk

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "real")
os.makedirs(OUT, exist_ok=True)


def find_ct_series():
    from tcia_utils import nbia
    series = nbia.getSeries(collection="NSCLC-Radiomics")
    for s in series:
        if s.get("Modality") == "CT":
            return s
    raise RuntimeError("No CT series found in NSCLC-Radiomics")


def download(series):
    from tcia_utils import nbia
    uid = series["SeriesInstanceUID"]
    print(f"Downloading CT for patient {series['PatientID']} "
          f"({series.get('ImageCount','?')} images, "
          f"{series.get('FileSize',0)/1e6:.0f} MB)...")
    nbia.downloadSeries([uid], input_type="list")     # -> ./tciaDownload/<uid>/
    folder = os.path.join("tciaDownload", uid)
    if not os.path.isdir(folder):                     # fallback: locate the newest series folder
        cands = [d for d in glob.glob("tciaDownload/*") if os.path.isdir(d)]
        folder = max(cands, key=os.path.getmtime) if cands else None
    if not folder or not os.path.isdir(folder):
        raise RuntimeError("Download folder not found; check the tciaDownload directory.")
    return folder


def load_ct(folder):
    reader = sitk.ImageSeriesReader()
    names = reader.GetGDCMSeriesFileNames(folder)
    if not names:
        raise RuntimeError(f"No DICOM files in {folder}")
    reader.SetFileNames(names)
    image = reader.Execute()
    volume = sitk.GetArrayFromImage(image).astype(float)   # (z, y, x), Hounsfield units
    spacing = image.GetSpacing()                           # (x, y, z) mm
    return volume, spacing


def main():
    try:
        series = find_ct_series()
        folder = download(series)
        volume, spacing = load_ct(folder)
    except Exception as e:
        print("Could not fetch/read the scan:", e)
        print("Make sure you ran: pip install tcia-utils SimpleITK  (and are online)")
        return

    pid = series["PatientID"]
    print(f"\nLoaded real CT for {pid}")
    print(f"  volume shape (z,y,x): {volume.shape}")
    print(f"  voxel spacing (x,y,z) mm: {tuple(round(s,2) for s in spacing)}")
    print(f"  intensity range (HU): {volume.min():.0f} to {volume.max():.0f}")

    # Save a real axial slice with a lung window so you can see the actual scan.
    z = volume.shape[0] // 2
    plt.figure(figsize=(5, 5))
    plt.imshow(np.clip(volume[z], -1000, 400), cmap="gray")
    plt.axis("off"); plt.title(f"{pid} — real CT (axial slice {z})", fontsize=10)
    slice_path = os.path.join(OUT, f"{pid}_ct_slice.png")
    plt.tight_layout(); plt.savefig(slice_path, dpi=140); plt.close()
    print(f"  saved real scan slice -> {slice_path}")

    print("\nReal-scan ingestion works. Next step for real radiomics: load the "
          "expert GTV mask (RTSTRUCT/SEG) shipped with this dataset.")


if __name__ == "__main__":
    main()
