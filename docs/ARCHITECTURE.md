# Architecture

OncoTwin is a layered system. Each layer depends only on the interfaces below
it, so any layer can be swapped without rewriting the others.

```
 multimodal features ─▶ parameter estimator (ML)     ← swap point for a trained model
                          │  ParameterEnsemble (weighted particles)
                          ▼
                     forecast engine ───▶ probabilistic trajectories + 90% bands
                          │                         │
                          ▼                         ▼
                  assimilation (recalibrate) ◀── new measurements  ← the twin loop
                          │
                          ▼
                     OncoTwinEngine ───▶ REST API ───▶ web UI (3D twin + charts)
```

## Layers

Imaging (`imaging.py`) — the multimodal front door. Loads a volume (phantom,
`.npy`, or NIfTI), segments the tumor, extracts radiomic features (shape,
intensity, texture), and `fuse()`s them with clinical/molecular data into a
`PatientFeatures`. The segmenter and feature set are transparent baselines with
production swap points (trained segmentation model; a full radiomics library).

Domain (`domain.py`) — the source-agnostic data model: `PatientFeatures`,
`ParameterEnsemble` (particles + weights), `PatientTwin`, treatment plans, and
tumor measurements. Everything else speaks in these types.

Mechanistic model (`growth.py`) — Gompertzian tumor growth with a log-kill
therapy term, integrated with `scipy`. Interpretable: `r` (growth rate), `K`
(carrying capacity), and per-therapy sensitivities. Integration always starts
from baseline at day 0, so it serves both forecasting and assimilation.

Parameter estimator (`parameters.py`) — maps fused multimodal features to a
prior distribution over the mechanistic parameters. `HeuristicEstimator` uses
transparent clinical priors today; it implements the `ParameterEstimator`
protocol, so a model trained on real cohorts drops in unchanged.

Forecasting (`forecast.py`) — simulates every particle and reduces to weighted
median and 5/95 quantile bands. Uncertainty is first-class, not an afterthought.

Assimilation (`assimilation.py`) — a particle filter. New measurements reweight
the particles by observation likelihood; the ensemble is resampled and jittered.
This is the recalibration loop that makes the system a twin.

Engine (`engine.py`) — the orchestration surface: `create_twin`, `forecast`,
`simulate_counterfactuals`, `assimilate`, `explain`.

Store (`store.py`) — versioned twin persistence. `JsonTwinStore` snapshots to
disk; swap for a database via the `TwinStore` protocol.

API (`api/`) — a FastAPI service exposing the engine over REST, with Pydantic
schemas as the wire contract.

Data (`data/`) — cohort adapters. `SyntheticCohort` generates labelled virtual
patients for training and validation; `CohortAdapter` is the interface a real
TCIA + TCGA loader implements.

Validation (`validation.py`) — rolling-origin backtesting of forecast error and
interval calibration.

Web (`web/index.html`) — the interface. Calls the API when it is reachable and
falls back to an in-browser copy of the engine otherwise, so the file always
runs standalone.

## Why hybrid (mechanistic + data-driven)

A pure black-box model needs large longitudinal cohorts and cannot extrapolate
treatment scenarios it never saw. A pure mechanistic model cannot personalize.
The hybrid keeps a small, interpretable mechanistic core and lets the data layer
estimate and continuously correct its few parameters — the approach the
predictive-oncology literature converges on, and the reason counterfactual
treatment simulation is even possible here.
