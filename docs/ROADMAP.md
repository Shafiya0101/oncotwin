# Roadmap

## Done

- Hybrid twin engine: mechanistic Gompertz + treatment, ML-estimated parameters
- Probabilistic forecasting with 90% uncertainty bands
- Bayesian recalibration (particle filter) — the living-twin loop
- Counterfactual treatment simulation
- Versioned twin state store (JSON) with a swappable interface
- FastAPI service + Pydantic contract
- 3D web UI (Three.js) wired to the API with an offline fallback
- Synthetic cohort generator + rolling-origin validation harness
- Test suite + CI

## Known limitation to fix first: calibration

The validation harness already reveals the honest weak spot. On a synthetic
cohort with `warmup=2`, the 90% intervals cover the truth far less than 90% of
the time — the current priors are **overconfident** because the cohort's true
parameters are wider than the `HeuristicEstimator` assumes. This is expected for
a prior-only model and is the first thing real data fixes.

Next steps, roughly in order:

1. **Real-data adapter** — implement a `CohortAdapter` over TCIA imaging
   (segmented volumes + radiomics) joined to TCGA/cBioPortal molecular and
   clinical data. Start with one disease (NSCLC or glioblastoma).
2. **Trained parameter estimator** — replace `HeuristicEstimator` with a model
   that regresses features → parameter posteriors, fit and evaluated with the
   existing backtest. Target: 90% coverage near nominal.
3. **Imaging pipeline** — tumor segmentation → radiomic features feeding the
   estimator (currently assumed as inputs).
4. **Better assimilation** — process noise / drift so the twin tracks changes in
   tumor behaviour over time, not just static parameters.
5. **Explainability layer** — attribute each forecast to the features and
   parameters driving it (the belief readout is the seed of this).
6. **Auth, multi-patient, database store** for a real deployment.

## Out of scope (deliberately)

Prospective clinical validation and regulatory clearance. This is research and
educational scaffolding, not a medical device. The architecture is built to head
toward clinical rigor — interpretable parameters, a validation harness, clean
data boundaries — but those steps require clinical partners and are not part of
this project.
