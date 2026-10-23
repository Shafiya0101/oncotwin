"""Imaging-pipeline validation on phantoms with known ground truth (uses the repo's own segment_tumor / extract_radiomics).
Sweeps tumour radius and noise level; reports Dice, volume error, sphericity, max-diameter error."""
import os, sys, pickle
import numpy as np
from scipy import ndimage
sys.path.insert(0, os.path.join(os.environ["REPO"], "src"))
from oncotwin.imaging import segment_tumor, extract_radiomics

SP = (1.0, 1.0, 1.5)
SHAPE = (64, 64, 48)

def phantom(radius, noise_scale, seed):
    rng = np.random.default_rng(seed)
    vol = 30.0 + rng.normal(0, 6 * noise_scale, SHAPE)
    zz, yy, xx = np.indices(SHAPE); c = np.array(SHAPE) / 2.0
    d = np.sqrt(((zz - c[0]) * SP[2]) ** 2 + ((yy - c[1]) * SP[1]) ** 2 + ((xx - c[2]) * SP[0]) ** 2)
    tumor = d <= radius
    vol[tumor] = 200.0 + rng.normal(0, 35 * noise_scale, tumor.sum())
    return ndimage.gaussian_filter(vol, 0.8), tumor

rows = []
for radius in (6, 9, 14, 20, 26):
    for ns in (1, 2, 3, 4, 5):
        for seed in range(10):
            v, gt = phantom(radius, ns, seed)
            m = segment_tumor(v)
            dice = 2 * (m & gt).sum() / (m.sum() + gt.sum() + 1e-9)
            f = extract_radiomics(v, m, SP); g = extract_radiomics(v, gt, SP)
            rows.append(dict(radius=radius, ns=ns, seed=seed, dice=dice,
                             vol_err=100 * (f.volume_cm3 - g.volume_cm3) / g.volume_cm3,
                             sph=f.sphericity, sph_gt=g.sphericity,
                             dia_err=f.max_diameter_mm - g.max_diameter_mm, true_vol=g.volume_cm3, het=f.heterogeneity))
pickle.dump(rows, open("out/imaging.pkl", "wb"))
import collections
for radius in (6, 9, 14, 20, 26):
    for ns in (1, 3, 5):
        r = [x for x in rows if x["radius"] == radius and x["ns"] == ns]
        print(radius, ns, "vol %.1f cm3" % np.mean([x["true_vol"] for x in r]), "dice %.3f" % np.mean([x["dice"] for x in r]),
              "volerr %.1f%% (|.|%.1f)" % (np.mean([x["vol_err"] for x in r]), np.mean([abs(x["vol_err"]) for x in r])),
              "sph %.2f/%.2f" % (np.mean([x["sph"] for x in r]), np.mean([x["sph_gt"] for x in r])), "dia_err %.1f" % np.mean([x["dia_err"] for x in r]))
