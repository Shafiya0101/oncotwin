#!/usr/bin/env python
"""Extract radiomics for the whole NSCLC-Radiomics cohort -> data/real/radiomics.csv

One patient at a time: download CT + expert RTSTRUCT from TCIA, load the CT,
build the GTV mask, extract features, then DELETE the downloaded scans. So only
about 100 MB is on disk at any moment (no need for the 33 GB full download).

Resumable: re-run the same command and it skips patients already in the CSV.
Failures are logged to data/real/radiomics_failures.csv and the run continues.

Usage (from the repo root, with the .venv active):
  python scripts/extract_cohort_radiomics.py --limit 3        # quick test
  python scripts/extract_cohort_radiomics.py                  # everything
  python scripts/extract_cohort_radiomics.py --only LUNG1-010 LUNG1-011
  python scripts/extract_cohort_radiomics.py --pyradiomics    # also PyRadiomics features (optional)

Needs (same as scripts/get_scan_with_mask.py): tcia-utils SimpleITK rt-utils
Reuses your own oncotwin.imaging.extract_radiomics for the features.
"""
import argparse
import csv
import glob
import os
import shutil
import sys
import time
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

DOWNLOAD_DIR = "tciaDownload"  # tcia_utils default, relative to the current folder


# --------------------------------------------------------------- TCIA ------
def list_series(collection):
    from tcia_utils import nbia
    by_patient = {}
    for s in nbia.getSeries(collection=collection):
        by_patient.setdefault(s["PatientID"], []).append(s)
    return by_patient


def download(uid, retries=3):
    from tcia_utils import nbia
    last = None
    for k in range(retries):
        try:
            nbia.downloadSeries([uid], input_type="list")
            folder = os.path.join(DOWNLOAD_DIR, uid)
            if os.path.isdir(folder):
                return folder
            cands = [d for d in glob.glob(os.path.join(DOWNLOAD_DIR, "*")) if os.path.isdir(d)]
            if cands:  # same fallback as get_scan_with_mask.py
                return max(cands, key=os.path.getmtime)
        except Exception as ex:  # network hiccup: wait and retry
            last = ex
        time.sleep(5 * (k + 1))
    raise RuntimeError(f"download failed for series {uid}: {last}")


# ----------------------------------------------- loading (from your script) -
def load_ct(folder):
    import SimpleITK as sitk
    r = sitk.ImageSeriesReader()
    r.SetFileNames(r.GetGDCMSeriesFileNames(folder))
    img = r.Execute()
    return sitk.GetArrayFromImage(img).astype(float), img.GetSpacing(), img  # (z,y,x), (x,y,z)


def gtv_mask(ct_dir, rt_dir):
    from rt_utils import RTStructBuilder
    rt_file = glob.glob(os.path.join(rt_dir, "*.dcm"))[0]
    rs = RTStructBuilder.create_from(dicom_series_path=ct_dir, rt_struct_path=rt_file)
    names = rs.get_roi_names()
    # prefer the primary tumour contour "GTV-1"; otherwise the first name containing GTV
    exact = [n for n in names if n.strip().upper() == "GTV-1"]
    gtv_like = [n for n in names if "GTV" in n.upper()]
    roi = (exact or gtv_like or names)[0]
    mask = rs.get_roi_mask_by_name(roi)                  # (y, x, z)
    return np.transpose(mask, (2, 0, 1)).astype(bool), roi, names  # -> (z, y, x)


# ----------------------------------------------------------- features ------
def own_features(volume, mask, spacing):
    from oncotwin.imaging import extract_radiomics
    out = {}
    for k, v in extract_radiomics(volume, mask, spacing).as_dict().items():
        try:
            out[f"oncotwin_{k}"] = float(v)
        except (TypeError, ValueError):
            pass  # skip non-numeric entries
    return out


def pyradiomics_features(ct_img, mask):
    import SimpleITK as sitk
    from radiomics import featureextractor
    m = sitk.GetImageFromArray(mask.astype(np.uint8))
    m.CopyInformation(ct_img)
    res = featureextractor.RadiomicsFeatureExtractor().execute(ct_img, m, label=1)
    return {f"pyrad_{k}": float(v) for k, v in res.items() if not k.startswith("diagnostics_")}


# ------------------------------------------------------------- per patient -
def process_patient(pid, series, use_pyradiomics=False):
    ct = next((s for s in series if s["Modality"] == "CT"), None)
    rt = next((s for s in series if s["Modality"] == "RTSTRUCT"), None)
    if ct is None:
        raise ValueError("no CT series for this patient")
    if rt is None:
        raise ValueError("no RTSTRUCT (expert contour) for this patient")
    folders = []
    try:
        ct_dir = download(ct["SeriesInstanceUID"]); folders.append(ct_dir)
        rt_dir = download(rt["SeriesInstanceUID"]); folders.append(rt_dir)
        volume, spacing, ct_img = load_ct(ct_dir)
        mask, roi, roi_names = gtv_mask(ct_dir, rt_dir)
        if mask.shape != volume.shape:
            raise ValueError(f"mask shape {mask.shape} != CT shape {volume.shape}")
        n_vox = int(mask.sum())
        if n_vox == 0:
            raise ValueError("mask is empty after alignment")
        row = {"PatientID": pid}
        row.update(own_features(volume, mask, spacing))
        if use_pyradiomics:
            try:
                row.update(pyradiomics_features(ct_img, mask))
            except Exception as ex:  # keep our own features even if this fails
                row["qa_pyradiomics_error"] = f"{type(ex).__name__}: {ex}"
        row["qa_roi_name"] = roi
        row["qa_all_rois"] = ";".join(roi_names)
        row["qa_mask_voxels"] = n_vox
        row["qa_ct_slices"] = volume.shape[0]
        return row
    finally:
        for f in folders:
            shutil.rmtree(f, ignore_errors=True)


# ------------------------------------------------------------------ main ---
def read_done(path):
    if not path.exists():
        return set(), None
    with path.open(newline="") as f:
        rd = csv.DictReader(f)
        return {r["PatientID"] for r in rd}, rd.fieldnames


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--collection", default="NSCLC-Radiomics")
    ap.add_argument("--out", default="data/real/radiomics.csv")
    ap.add_argument("--failures", default="data/real/radiomics_failures.csv")
    ap.add_argument("--limit", type=int, default=None, help="only the first N patients still to do")
    ap.add_argument("--only", nargs="+", help="only these PatientIDs")
    ap.add_argument("--pyradiomics", action="store_true", help="also compute PyRadiomics features (needs pyradiomics)")
    a = ap.parse_args()

    out, fail = Path(a.out), Path(a.failures)
    out.parent.mkdir(parents=True, exist_ok=True)
    done, fieldnames = read_done(out)

    print("Fetching the patient list from TCIA ...", flush=True)
    by_patient = list_series(a.collection)
    ids = sorted(by_patient)
    if a.only:
        ids = [i for i in ids if i in set(a.only)]
    todo = [i for i in ids if i not in done]
    if a.limit:
        todo = todo[: a.limit]
    print(f"{len(ids)} patients in collection | {len(done)} already done | {len(todo)} to do", flush=True)

    t_start, n_ok, n_fail = time.time(), 0, 0
    for k, pid in enumerate(todo, 1):
        t0 = time.time()
        try:
            row = process_patient(pid, by_patient[pid], a.pyradiomics)
            if fieldnames is None:
                fieldnames = list(row)
            with out.open("a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore", restval="")
                if f.tell() == 0:
                    w.writeheader()
                w.writerow(row)
            n_ok += 1
            status = "ok"
        except KeyboardInterrupt:
            print("\nStopped by you. Re-run the same command to continue.")
            break
        except Exception as ex:
            n_fail += 1
            status = f"FAILED: {type(ex).__name__}: {ex}"
            with fail.open("a", newline="") as f:
                w = csv.writer(f)
                if f.tell() == 0:
                    w.writerow(["PatientID", "error"])
                w.writerow([pid, f"{type(ex).__name__}: {ex}"])
            traceback.print_exc(limit=1)
        per = (time.time() - t_start) / k
        eta_min = per * (len(todo) - k) / 60
        print(f"[{k}/{len(todo)}] {pid} {status} | {time.time() - t0:.0f}s | ~{eta_min:.0f} min left", flush=True)

    print(f"\nFinished: {n_ok} ok, {n_fail} failed this run. Output: {out}"
          + (f" | failures: {fail}" if n_fail else ""))


if __name__ == "__main__":
    main()
