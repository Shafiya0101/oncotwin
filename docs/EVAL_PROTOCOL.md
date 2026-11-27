# Evaluation protocol: real-cohort survival experiments

Fixed before the final results were produced. Any change after seeing results must be listed under "Deviations" at the bottom.

## Data
- **Cohort:** TCIA NSCLC-Radiomics (Lung1). Licence CC BY-NC; data are not redistributed (`data/real/` is git-ignored).
- **Outcome:** overall survival. Time in days from the start of treatment (as given by TCIA); event = death.
- **Clinical variables:** age, sex, clinical T, N and M stage, overall stage, histology. Missing categorical values become an "unknown" category; missing age is imputed inside training folds.
- **Imaging features:** computed from the expert GTV contour by `oncotwin.imaging.extract_radiomics` (`oncotwin_*` columns). The contour used is `GTV-1` when present, otherwise the first contour whose name contains "GTV". Every contour name is recorded in `qa_all_rois` so the choice can be audited.
- **Patients:** all patients with usable survival data. Imaging models use the patients whose radiomics were extracted successfully; patients that failed are listed in `radiomics_failures.csv` with the reason. For model comparisons, all models are evaluated on the same patients.

## Models
1. clinical only
2. imaging only
3. imaging + clinical

All use a ridge-penalised Cox model (Breslow ties). The penalty is chosen from {0.003, 0.03, 0.3, 3, 30} by inner 3-fold cross-validation on the training data (C-index).

## Splits and leakage control
- Patient-level, stratified on the event indicator, 5 folds, repeated 10 times, seeds 42 to 51 (base seed 42).
- Everything that learns from data is fitted on the training fold only: median imputation, removal of constant columns, removal of highly correlated columns (|r| > 0.9, greedy), standardisation (z-scores clipped at +/-5), and the penalty.
- A pure-noise test in the test suite checks that the pipeline stays near chance when there is no signal.

## Metrics
- **C-index** (Harrell), from out-of-fold risk scores. Each fold's scores are standardised with the training-set mean and SD, then averaged over the 10 repeats per patient. The mean per-fold C-index is reported as a secondary number.
- **Time-dependent AUC** (IPCW cumulative/dynamic) at 12, 24 and 36 months (1 month = 365.25/12 days).
- **Calibration at 24 months:** predicted survival from the Breslow baseline hazard of each training fold, averaged over repeats; patients grouped into 5 quantile groups; observed survival by Kaplan-Meier with bootstrap intervals.
- **Variable importance:** permutation importance on held-out folds (drop in C-index), variables shuffled as groups (all dummy columns of a variable together); 3 repeats x 5 folds x 20 permutations.

## Uncertainty
- 1,000 patient-level bootstrap resamples; 95% percentile intervals for C-index and AUC.
- Differences between models use the same resamples (paired), because overlapping intervals do not show that two models are equal.
- Baseline: the original twin risk is evaluated with no training on the same patients (`--baseline-csv`).

## Not covered
No external validation; no molecular (omics) features; no non-linear models.

## Reproduce
```
python scripts/build_clinical_table.py data/real/Lung1.clinical.csv
python scripts/extract_cohort_radiomics.py
python analysis/survival_ablation.py
python analysis/write_real_results.py
```
(Windows: `run_real_cohort.bat`. With make: `make real-cohort`.)

## Deviations
None so far.
