"""Write two demo scan files you can upload into scan_app.py to see it work.

Creates a small 'January' scan (smaller tumour) and a 'March' scan (larger
tumour) as NIfTI files. Upload both to the same patient in the app and watch the
twin recalibrate across the two timepoints.

Run:  python scripts/make_demo_scans.py
"""
import os
import numpy as np
import SimpleITK as sitk

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "demo_scans")
os.makedirs(OUT, exist_ok=True)


def phantom(radius, seed):
    rng = np.random.default_rng(seed)
    shape = (48, 96, 96)
    vol = (rng.random(shape) * 40).astype(np.float32)
    zz, yy, xx = np.indices(shape)
    c = np.array(shape) / 2
    blob = ((zz - c[0]) ** 2 + (yy - c[1]) ** 2 + (xx - c[2]) ** 2) < radius ** 2
    vol[blob] += 300
    img = sitk.GetImageFromArray(vol)
    img.SetSpacing((1.0, 1.0, 1.5))
    return img


for name, r, seed in [("scan_january.nii.gz", 9, 1), ("scan_march.nii.gz", 13, 2)]:
    p = os.path.join(OUT, name)
    sitk.WriteImage(phantom(r, seed), p)
    print("wrote", p)

print("\nNow: run  python scan_app.py  , open http://localhost:8000 , add a patient,")
print("and upload scan_january.nii.gz (date 2026-01-15) then scan_march.nii.gz (2026-03-15).")
