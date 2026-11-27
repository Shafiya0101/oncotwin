# Real-cohort results: TCIA NSCLC-Radiomics (Lung1)

421 patients, 372 deaths. Outcome: overall survival. Repeated 5-fold cross-validation x 10 repeats; every data-dependent step is fitted inside training folds; 1000 patient-level bootstrap resamples (95% percentile intervals). Full protocol: docs/EVAL_PROTOCOL.md.

## Discrimination

- **clinical**: C-index 0.540 (95% CI 0.507 to 0.572), above chance.
- **imaging**: C-index 0.582 (95% CI 0.549 to 0.613), above chance.
- **imaging+clinical**: C-index 0.581 (95% CI 0.549 to 0.612), above chance.

## Does adding information help?

Paired bootstrap differences in C-index (same resamples for both models):

- imaging - clinical: +0.041 (95% CI -0.005 to 0.084); the interval includes zero, so this difference is not established.
- imaging+clinical - clinical: +0.041 (95% CI 0.005 to 0.074); the interval excludes zero.
- imaging+clinical - imaging: -0.001 (95% CI -0.017 to 0.016); the interval includes zero, so this difference is not established.

## Calibration (24 months)

- **clinical**: mean absolute calibration error 0.062 (predicted vs Kaplan-Meier observed survival across 5 risk groups). See the table in analysis/outputs/real_cohort/RESULTS_real_cohort.md.
- **imaging**: mean absolute calibration error 0.054 (predicted vs Kaplan-Meier observed survival across 5 risk groups). See the table in analysis/outputs/real_cohort/RESULTS_real_cohort.md.
- **imaging+clinical**: mean absolute calibration error 0.057 (predicted vs Kaplan-Meier observed survival across 5 risk groups). See the table in analysis/outputs/real_cohort/RESULTS_real_cohort.md.

## Most important variables

Permutation importance on held-out folds (imaging+clinical model), top five with the mean drop in C-index: oncotwin_volume_cm3 (0.013), oncotwin_max_diameter_mm (0.011), oncotwin_intensity_mean (0.007), age (0.004), n_stage (0.002).

## Limitations

- Single retrospective cohort; no external validation.
- Imaging features come from the project's own radiomics code (few features, not the standard IBSI set).
- Ridge-penalised Cox model; no non-linear models were tried.
- Calibration is shown at one horizon (24 months) with wide intervals in small risk groups.
- Molecular (omics) data are not included in this analysis.

## Reproduce

```
python scripts/build_clinical_table.py data/real/Lung1.clinical.csv
python scripts/extract_cohort_radiomics.py
python analysis/survival_ablation.py
python analysis/write_real_results.py
```