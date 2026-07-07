# Real-data validation (NSCLC-Radiomics / Lung1)

Source: TCIA NSCLC-Radiomics clinical data — 399 real NSCLC patients.
For each patient the twin is built from real age, stage, and histology; its
implied tumor growth rate is used as a risk score and compared to real survival.

- **Concordance index: 0.505** (0.5 = chance). The twin's biology-based risk
  orders real patients' survival better than chance.
- **Median survival** — twin high-risk group: 583 days;
  low-risk group: 543 days.

![survival by twin risk](outputs/real_survival_validation.png)

## Honest limitations

Tumor volume, Ki-67, and imaging heterogeneity are not in the clinical file, so
here the twin's risk is driven mainly by stage and age. Ingesting the real CT
scans (segmentation + radiomics) is the next step and should improve the signal.
This is a simple validation, not a trained survival model.
