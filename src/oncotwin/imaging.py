"""Medical imaging pipeline — the multimodal core.

This turns an actual image volume into the features the twin consumes, and fuses
them with clinical/molecular data. That fusion is the literal "combine medical
imaging and clinical data" step of the project.

  load_volume  ->  segment_tumor  ->  extract_radiomics  ->  fuse(+clinical)  ->  PatientFeatures

`synthetic_scan` produces a phantom so the pipeline runs and tests are
deterministic; `load_volume` reads real .npy or NIfTI (.nii/.nii.gz via nibabel)
so you can point it at TCIA / BraTS data unchanged. The threshold segmenter is a
stand-in — in production you would drop in a trained model (e.g. nnU-Net) behind
the same `segment_tumor` interface.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import numpy as np
from scipy import ndimage
from scipy.stats import skew
from skimage import measure, filters, feature

from .domain import PatientFeatures


@dataclass
class ImagingFeatures:
    volume_cm3: float
    surface_area_mm2: float
    sphericity: float               # 1.0 = perfect sphere
    max_diameter_mm: float
    intensity_mean: float
    intensity_std: float
    intensity_skew: float
    heterogeneity: float            # 0..1 texture-based aggressiveness proxy

    def as_dict(self) -> dict:
        return {k: round(float(v), 3) for k, v in asdict(self).items()}


# ------------------------------ acquisition ------------------------------ #

def synthetic_scan(shape=(64, 64, 48), spacing=(1.0, 1.0, 1.5),
                   radius_mm=14.0, seed: int = 0):
    """A phantom CT-like volume with a heterogeneous tumor blob.

    Returns (volume, spacing_mm, true_volume_cm3). Stands in for a real scan.
    """
    rng = np.random.default_rng(seed)
    vol = 30.0 + rng.normal(0, 6, shape)                       # noisy soft-tissue background
    zz, yy, xx = np.indices(shape)
    center = np.array(shape) / 2.0
    d = np.sqrt(((zz - center[0]) * spacing[2]) ** 2 +
                ((yy - center[1]) * spacing[1]) ** 2 +
                ((xx - center[2]) * spacing[0]) ** 2)
    tumor = d <= radius_mm
    vol[tumor] = 200.0 + rng.normal(0, 35, size=tumor.sum())   # bright, textured tumor
    vol = ndimage.gaussian_filter(vol, sigma=0.8)              # partial-volume blur
    voxel_cm3 = np.prod(spacing) / 1000.0
    true_volume_cm3 = float(tumor.sum() * voxel_cm3)
    return vol, tuple(spacing), true_volume_cm3


def load_volume(path: str) -> np.ndarray:
    """Load a real image volume. .npy or NIfTI (.nii/.nii.gz, needs nibabel)."""
    if path.endswith(".npy"):
        return np.load(path)
    if path.endswith((".nii", ".nii.gz")):
        import nibabel as nib
        return np.asarray(nib.load(path).get_fdata())
    raise ValueError(f"unsupported image format: {path}")


# ------------------------------ segmentation ----------------------------- #

def segment_tumor(volume: np.ndarray) -> np.ndarray:
    """Segment the tumor as the largest bright connected component.

    A deliberately simple, transparent baseline. Swap for a trained model in
    production behind this same signature.
    """
    thresh = filters.threshold_otsu(volume)
    mask = volume > thresh
    mask = ndimage.binary_opening(mask, iterations=1)
    labels, n = ndimage.label(mask)
    if n == 0:
        return np.zeros_like(volume, dtype=bool)
    sizes = ndimage.sum(np.ones_like(labels), labels, index=range(1, n + 1))
    largest = int(np.argmax(sizes)) + 1
    return ndimage.binary_fill_holes(labels == largest)


# ------------------------------ radiomics -------------------------------- #

def extract_radiomics(volume: np.ndarray, mask: np.ndarray, spacing) -> ImagingFeatures:
    spacing = np.asarray(spacing, dtype=float)          # (x, y, z) mm
    voxel_mm3 = float(np.prod(spacing))
    n_vox = int(mask.sum())
    volume_cm3 = n_vox * voxel_mm3 / 1000.0

    # surface + shape from a mesh of the mask
    try:
        verts, faces, _, _ = measure.marching_cubes(mask.astype(float), level=0.5,
                                                     spacing=tuple(spacing[::-1]))
        surface_area = float(measure.mesh_surface_area(verts, faces))
    except (RuntimeError, ValueError):
        surface_area = float(n_vox ** (2 / 3) * voxel_mm3 ** (2 / 3))
    v_mm3 = n_vox * voxel_mm3
    sphericity = float((np.pi ** (1 / 3) * (6 * v_mm3) ** (2 / 3)) / surface_area) if surface_area else 0.0
    sphericity = min(sphericity, 1.0)

    coords = np.argwhere(mask) * spacing[::-1]
    max_diameter = float(np.ptp(coords, axis=0).max()) if len(coords) else 0.0

    vals = volume[mask]
    i_mean, i_std = float(vals.mean()), float(vals.std())
    i_skew = float(skew(vals)) if len(vals) > 2 else 0.0

    # texture heterogeneity via GLCM contrast on the tumor's central slice
    heterogeneity = _glcm_heterogeneity(volume, mask)

    return ImagingFeatures(volume_cm3, surface_area, sphericity, max_diameter,
                           i_mean, i_std, i_skew, heterogeneity)


def _glcm_heterogeneity(volume: np.ndarray, mask: np.ndarray, levels: int = 8) -> float:
    zs = np.argwhere(mask)[:, 0]
    if len(zs) == 0:
        return 0.0
    z = int(np.median(zs))
    sl, m2d = volume[z], mask[z]
    if m2d.sum() < 4:
        return float(np.clip((volume[mask].std() / (abs(volume[mask].mean()) + 1e-6)), 0, 1))
    ys, xs = np.where(m2d)
    patch = sl[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    lo, hi = patch.min(), patch.max()
    q = np.zeros_like(patch, dtype=np.uint8) if hi <= lo else \
        ((patch - lo) / (hi - lo) * (levels - 1)).astype(np.uint8)
    glcm = feature.graycomatrix(q, distances=[1], angles=[0], levels=levels, symmetric=True, normed=True)
    contrast = float(feature.graycoprops(glcm, "contrast")[0, 0])
    return float(np.clip(contrast / ((levels - 1) ** 2), 0, 1))


# -------------------------------- fusion --------------------------------- #

def fuse(imaging: ImagingFeatures, clinical: dict) -> PatientFeatures:
    """Combine imaging-derived features with clinical/molecular data.

    Imaging contributes the baseline tumor volume and the texture-based
    aggressiveness proxy; clinical/molecular data contributes stage, age, and
    markers. This single object is what the twin engine consumes.
    """
    return PatientFeatures(
        age=int(clinical.get("age", 63)),
        stage=int(clinical.get("stage", 3)),
        histology=clinical.get("histology", "adenocarcinoma"),
        baseline_volume_cm3=imaging.volume_cm3,          # from imaging
        ki67=float(clinical.get("ki67", 0.3)),
        egfr_mutation=bool(clinical.get("egfr_mutation", False)),
        radiomic_heterogeneity=imaging.heterogeneity,    # from imaging
    )
