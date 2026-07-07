# Roadmap

## Done

- Multimodal imaging pipeline: segment tumor → radiomics → fuse with clinical
  data → twin (`imaging.py`, `make imaging-demo`)
- Hybrid twin engine: mechanistic Gompertz + treatment, ML-estimated parameters
- Richer biology: acquired chemo resistance over time + linear-quadratic
  radiotherapy (`growth.py`)
- Probabilistic forecasting with 90% uncertainty bands
- **Calibration fix**: horizon-growing process noise brings 90% coverage from
  ~35% to ~90% without hurting accuracy (`make analysis`)
- Explainability: ranked feature drivers per forecast (`explain.py`)
- Bayesian recalibration (particle filter) — the living-twin loop
- Counterfactual treatment simulation
- Versioned twin state store (JSON) with a swappable interface
- FastAPI service + Pydantic contract
- 3-tab web UI (Twin / Imaging / Compare) with the real segmented tumor mesh in 3D
- Trained parameter estimator + synthetic cohort + validation harness
- Test suite (20 tests) + CI

## Next

The honest remaining boundary is real data and clinical validation:

1. **Real-data adapter** — a `CohortAdapter` over TCIA imaging (segmented volumes
   + radiomics) joined to TCGA/cBioPortal molecular and clinical data, and refit
   `LearnedEstimator` on it. Start with one disease (NSCLC or glioblastoma).
2. **Production imaging** — replace the threshold segmenter with a trained model
   (e.g. nnU-Net) and a full radiomics library (e.g. PyRadiomics), behind the
   existing `segment_tumor` / `extract_radiomics` interfaces.
3. **Auth, multi-patient, database store** for a real deployment (FHIR to pull
   real EHR data).

## Out of scope (deliberately)

Prospective clinical validation and regulatory clearance. This is research and
educational scaffolding, not a medical device. The architecture is built to head
toward clinical rigor — interpretable parameters, a validation harness, clean
data boundaries — but those steps require clinical partners and are not part of
this project.
