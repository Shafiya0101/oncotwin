# Roadmap

## Done

- Multimodal imaging pipeline: segment tumor → radiomics → fuse with clinical
  data → twin (`imaging.py`, `make imaging-demo`)
- Hybrid twin engine: mechanistic Gompertz + treatment, ML-estimated parameters
- Probabilistic forecasting with 90% uncertainty bands
- Bayesian recalibration (particle filter) — the living-twin loop
- Counterfactual treatment simulation
- Versioned twin state store (JSON) with a swappable interface
- FastAPI service + Pydantic contract
- 3D web UI (Three.js) wired to the API with an offline fallback
- Synthetic cohort generator + rolling-origin validation harness
- Test suite + CI
- Trained parameter estimator (`LearnedEstimator`) + analysis pipeline
  (`make analysis`), fit and evaluated on synthetic data

## Known limitation to fix first: calibration

The validation harness already reveals the honest weak spot. Backtesting on a
synthetic cohort (`make analysis`), the 90% intervals cover the truth far less
than 90% of the time for both estimators — the forecasts are **overconfident**.
Training the estimator helps (accuracy and coverage both improve) but does not
close the gap on its own.

Concrete fixes, in order:

1. **Process + observation noise** — inject parameter drift in assimilation and
   add measurement noise to the predictive band so intervals aren't artificially
   tight. This is the single biggest calibration lever.
2. **Real-data adapter** — implement a `CohortAdapter` over TCIA imaging
   (segmented volumes + radiomics) joined to TCGA/cBioPortal molecular and
   clinical data, and refit `LearnedEstimator` on it. Start with one disease
   (NSCLC or glioblastoma).
3. **Production imaging** — replace the threshold segmenter with a trained model
   (e.g. nnU-Net) and a full radiomics library (e.g. PyRadiomics), behind the
   existing `segment_tumor` / `extract_radiomics` interfaces.
4. **Explainability layer** — attribute each forecast to the features and
   parameters driving it (the belief readout is the seed of this).
5. **Auth, multi-patient, database store** for a real deployment.

## Out of scope (deliberately)

Prospective clinical validation and regulatory clearance. This is research and
educational scaffolding, not a medical device. The architecture is built to head
toward clinical rigor — interpretable parameters, a validation harness, clean
data boundaries — but those steps require clinical partners and are not part of
this project.
