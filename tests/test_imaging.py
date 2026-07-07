import numpy as np
from oncotwin.imaging import (
    synthetic_scan, segment_tumor, extract_radiomics, fuse, ImagingFeatures,
)
from oncotwin.domain import PatientFeatures


def test_segmentation_recovers_tumor_volume():
    vol, spacing, true_v = synthetic_scan(seed=1)
    mask = segment_tumor(vol)
    assert mask.sum() > 0
    seg_v = mask.sum() * np.prod(spacing) / 1000.0
    # segmented volume within 25% of the geometric truth
    assert abs(seg_v - true_v) / true_v < 0.25


def test_radiomics_features_are_sane():
    vol, spacing, _ = synthetic_scan(seed=3)
    feats = extract_radiomics(vol, segment_tumor(vol), spacing)
    assert isinstance(feats, ImagingFeatures)
    assert feats.volume_cm3 > 0
    assert 0.0 < feats.sphericity <= 1.0          # a blob is roughly spherical
    assert 0.0 <= feats.heterogeneity <= 1.0
    assert feats.max_diameter_mm > 0


def test_fusion_produces_patient_features():
    vol, spacing, _ = synthetic_scan(seed=4)
    feats = extract_radiomics(vol, segment_tumor(vol), spacing)
    pf = fuse(feats, {"age": 70, "stage": 4, "ki67": 0.5, "egfr_mutation": True})
    assert isinstance(pf, PatientFeatures)
    assert pf.stage == 4 and pf.egfr_mutation is True
    assert abs(pf.baseline_volume_cm3 - feats.volume_cm3) < 1e-6   # volume comes from imaging
    assert pf.radiomic_heterogeneity == feats.heterogeneity
