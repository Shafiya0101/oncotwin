# OncoTwin

**A living, probabilistic, multimodal cancer patient digital twin for oncology decision support.**

> 🔗 **Live demo:** https://shafiya0101.github.io/oncotwin/

Unlike a static predictor that outputs one number, OncoTwin keeps a *belief*
about a patient's tumour that recalibrates as new scans arrive, forecasts disease
as a distribution of trajectories with honest uncertainty, simulates
counterfactual treatment strategies in silico, and explains what drives each
forecast — all exposed through an interactive browser app with a 3D view of the
segmented tumour.

> [!WARNING]
> **Research and educational software — not a medical device.** Parameters are
> interpretable clinical priors, not trained on patient outcomes, so specific
> numbers are illustrative. Nothing here is validated for clinical use.

---

## What makes it a *twin*, not a predictor

1. **Living state.** New measurements recalibrate the twin's parameters via
   Bayesian data assimilation (a particle filter) — the same principle weather
   models use to stay locked onto reality.
2. **Probabilistic.** Forecasts are trajectory *distributions* with 90% intervals,
   calibrated so they mean what they say (see *Calibration*, below).
3. **Counterfactual.** A hybrid model — an ML layer estimating patient-specific
   parameters that feed a mechanistic Gompertz-with-treatment growth model, now
   with acquired chemo resistance and linear-quadratic radiotherapy — lets you
   compare therapy options.
4. **Interpretable & explainable.** Every parameter is biologically meaningful, and
   a live sensitivity analysis ranks which patient features drive each forecast.

### The living twin, recalibrating on real scans

![Living twin recalibration](docs/assets/living_twin_recalibration.png)

The prior forecast (grey) expected a good response and a plateau. The patient's
true trajectory (black) kept climbing. After two follow-up scans, the twin's
forecast (coral) corrected upward and locked onto reality.

### Counterfactual treatment strategies

![Counterfactual strategies](docs/assets/forecast_counterfactuals.png)

One patient, four strategies, each a probabilistic trajectory.

---

## Multimodal: imaging + clinical data

The pipeline turns an actual scan into twin inputs and fuses them with clinical
data — the multimodal core of the project:

```
load scan → segment tumour → extract radiomics → fuse(+ clinical) → patient twin
```

![Multimodal pipeline](docs/assets/multimodal_pipeline.png)

Imaging contributes the segmented tumour volume and texture-based heterogeneity;
clinical data contributes stage, age, and markers. Runs on a phantom out of the
box (`make imaging-demo`) and on real CT (see *Validated on real data*).

---

## Calibration: making the uncertainty honest

```bash
make analysis        # -> analysis/RESULTS.md + figures
```

Backtesting three configurations on a synthetic cohort (assimilate 2 scans,
forecast the rest) exposed a real defect and fixed it:

| Configuration | Forecast MAE (cm³) | 90% coverage |
|---|---|---|
| Heuristic priors | 15.9 | 31% |
| Trained estimator | 14.4 | 37% |
| **Trained + process noise (calibrated)** | **13.5** | **95%** |

![estimator comparison](analysis/outputs/estimator_comparison.png)

The prior-only and trained models were badly **overconfident**. Adding process
noise that grows with the forecast horizon brings 90% coverage from ~35% to ~90%
*without* hurting accuracy — the intervals now mean what they say. Full write-up:
[analysis/RESULTS.md](analysis/RESULTS.md).

---

## Validated on real data (TCIA NSCLC-Radiomics)

The project goes beyond synthetic data and engages honestly with a real public
cohort.

- **Real survival validation** (`python analysis/real_data_analysis.py`) — for
  399 real NSCLC patients, the twin's biology-based risk was compared to actual
  survival. It scored a **concordance index of 0.505 — at chance**. This honest
  negative result is reported truthfully: the growth model was not trained on
  survival, and per-patient imaging features were not yet loaded. See
  [analysis/REAL_RESULTS.md](analysis/REAL_RESULTS.md).
- **Real CT ingestion** (`python scripts/get_one_scan.py`) — downloads a real CT
  from TCIA and loads it end-to-end (134×512×512, Hounsfield units).
- **Real expert-mask radiomics** (`python scripts/get_scan_with_mask.py`) — loads
  the radiation-oncologist GTV delineation for a real patient, extracts real
  radiomics from the real tumour (volume 162 cm³, sphericity 0.60), and fuses them
  with the real clinical record into a fully multimodal twin.

> Real TCIA data is CC BY-NC and is **not** redistributed in this repo
> (`data/real/` is git-ignored). Download it yourself; the scripts above fetch a
> single scan for free with no login.

---

## Quickstart

```bash
pip install -e ".[dev]"      # install
make test                    # run the test suite (20 tests)
make demo                    # forecast + living-twin figures
make imaging-demo            # scan -> segmentation -> twin figure
make analysis                # train + prove the calibration fix
make assets                  # rebuild the web UI's mesh/scan/feature data
```

Run the full app (API + 3D web UI):

```bash
make serve
# API  → http://localhost:8000  (interactive docs at /docs)
# Web  → http://localhost:8080
```

The web app has **four tabs** — Twin (3D segmented tumour, forecast, live patient
sliders, feature drivers), Imaging (scan, segmentation, radiomics), Compare (all
strategies overlaid, with a flag when two are statistically indistinguishable),
and Cohort (calibration and real-data results). It also has a saved-patient
roster and a downloadable PDF report. It calls the API when running and falls
back to an in-browser copy of the engine otherwise, so `index.html` also works
opened directly (and is what the live demo runs).

---

## Using the engine

```python
from oncotwin import OncoTwinEngine, PatientFeatures, TreatmentPlan, TreatmentCourse, TreatmentKind, TumorMeasurement

engine = OncoTwinEngine()
twin = engine.create_twin("PT-0001", PatientFeatures(age=63, stage=3,
    histology="adenocarcinoma", baseline_volume_cm3=42.0, ki67=0.35))

chemo = TreatmentPlan("chemo", [TreatmentCourse(TreatmentKind.CHEMO, 30, 200, 1.0)])
f = engine.forecast(twin, chemo, horizon_days=365)
print(f.summary(horizon_day=365))          # median + 90% interval at 1 year

engine.assimilate(twin, [TumorMeasurement(60, 120.0)])   # recalibrate on a scan
print(engine.explain(twin))                # updated belief, bumped version
```

## API

| Method | Path | Purpose |
|---|---|---|
| `GET`  | `/health` | liveness |
| `POST` | `/twins` | create a twin from features |
| `GET`  | `/twins/{id}` | current belief |
| `POST` | `/twins/{id}/forecast` | forecast one plan |
| `POST` | `/twins/{id}/counterfactuals` | forecast several plans |
| `POST` | `/twins/{id}/measurements` | assimilate scans, recalibrate |

## Project layout

```
src/oncotwin/      engine (domain, growth, parameters, forecast, assimilation, engine)
                   imaging, explain, store, validation, data/, api/ (FastAPI)
index.html         4-tab twin UI (Three.js), API-backed with offline fallback
assets.js          real pipeline outputs for the UI (tumour mesh, scan, features)
tests/             pytest suite (20 tests: engine, API, imaging, biology, calibration)
examples/          demo.py, imaging_demo.py
analysis/          run_analysis.py, real_data_analysis.py + results
scripts/           build_web_assets.py, get_one_scan.py, get_scan_with_mask.py, serve.sh
docs/              ARCHITECTURE.md, ROADMAP.md
```

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the design and
[docs/ROADMAP.md](docs/ROADMAP.md) for what's next.

## License

MIT — see [LICENSE](LICENSE). Real TCIA data is CC BY-NC and is not included.
