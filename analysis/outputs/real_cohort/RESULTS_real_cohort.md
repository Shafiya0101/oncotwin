# Real-cohort ablation

Patients: 421, events: 372. Repeated 5-fold CV x 10 repeats, seed 42; 1000 bootstrap resamples; 95% percentile intervals.

## Discrimination (out-of-fold)

| Model | C-index (95% CI) | Mean per-fold C (sd) | AUC 12 mo (95% CI) | AUC 24 mo (95% CI) | AUC 36 mo (95% CI) |
|---|---|---|---|---|---|
| clinical | 0.540 (0.507 to 0.572) | 0.538 (0.030) | 0.539 (0.480 to 0.601) | 0.560 (0.504 to 0.615) | 0.566 (0.512 to 0.623) |
| imaging | 0.582 (0.549 to 0.613) | 0.581 (0.032) | 0.639 (0.581 to 0.691) | 0.632 (0.575 to 0.690) | 0.643 (0.581 to 0.703) |
| imaging+clinical | 0.581 (0.549 to 0.612) | 0.578 (0.031) | 0.629 (0.567 to 0.680) | 0.624 (0.568 to 0.679) | 0.644 (0.583 to 0.702) |

## Paired C-index differences

| Comparison | Delta C (95% CI) | CI excludes 0 |
|---|---|---|
| imaging - clinical | 0.041 (-0.005 to 0.084) | no |
| imaging+clinical - clinical | 0.041 (0.005 to 0.074) | yes |
| imaging+clinical - imaging | -0.001 (-0.017 to 0.016) | no |

## Calibration at 24 months

Predicted survival vs Kaplan-Meier observed survival, 5 risk groups.

**clinical** (mean absolute calibration error 0.062)

| Group | n | events | Predicted | Observed KM (95% CI) |
|---|---|---|---|---|
| 1 | 85 | 81 | 0.258 | 0.318 (0.224 to 0.424) |
| 2 | 84 | 74 | 0.341 | 0.327 (0.221 to 0.440) |
| 3 | 84 | 73 | 0.405 | 0.500 (0.393 to 0.607) |
| 4 | 84 | 71 | 0.462 | 0.440 (0.345 to 0.536) |
| 5 | 84 | 73 | 0.557 | 0.436 (0.338 to 0.555) |

**imaging** (mean absolute calibration error 0.054)

| Group | n | events | Predicted | Observed KM (95% CI) |
|---|---|---|---|---|
| 1 | 85 | 78 | 0.360 | 0.282 (0.188 to 0.376) |
| 2 | 84 | 73 | 0.391 | 0.376 (0.278 to 0.486) |
| 3 | 84 | 75 | 0.405 | 0.369 (0.262 to 0.476) |
| 4 | 84 | 70 | 0.416 | 0.496 (0.376 to 0.596) |
| 5 | 84 | 76 | 0.437 | 0.500 (0.405 to 0.619) |

**imaging+clinical** (mean absolute calibration error 0.057)

| Group | n | events | Predicted | Observed KM (95% CI) |
|---|---|---|---|---|
| 1 | 85 | 79 | 0.370 | 0.259 (0.170 to 0.341) |
| 2 | 84 | 78 | 0.394 | 0.369 (0.274 to 0.464) |
| 3 | 84 | 71 | 0.404 | 0.444 (0.342 to 0.562) |
| 4 | 84 | 71 | 0.413 | 0.512 (0.405 to 0.619) |
| 5 | 84 | 73 | 0.433 | 0.440 (0.333 to 0.542) |

## Variable importance (permutation, held-out folds)

Model: imaging+clinical. Drop in C-index when the variable is shuffled; about 0 means the model does not use it.

| Variable | Mean C-index drop | SD over folds |
|---|---|---|
| oncotwin_volume_cm3 | 0.0133 | 0.0105 |
| oncotwin_max_diameter_mm | 0.0110 | 0.0073 |
| oncotwin_intensity_mean | 0.0075 | 0.0158 |
| age | 0.0044 | 0.0041 |
| n_stage | 0.0020 | 0.0129 |
| oncotwin_heterogeneity | 0.0017 | 0.0040 |
| sex | 0.0011 | 0.0058 |
| oncotwin_sphericity | 0.0007 | 0.0068 |
| oncotwin_intensity_std | 0.0004 | 0.0040 |
| oncotwin_surface_area_mm2 | 0.0000 | 0.0000 |
| m_stage | -0.0003 | 0.0010 |
| histology | -0.0006 | 0.0065 |

## Notes

- C-index and AUC use repeat-averaged out-of-fold risk scores; each fold's scores are standardised by the training-set mean and SD so folds are comparable.
- Time-dependent AUC is the IPCW cumulative/dynamic estimator.
- All imputation, filtering, scaling and penalty selection happen inside training folds only.
- Top coefficients of the final model: final_model_coefficients.csv (standardised units; not an importance test).