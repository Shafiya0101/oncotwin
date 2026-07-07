# Results

Reproduce: `python analysis/run_analysis.py` (synthetic data, seeds fixed).
Cohort: 120 train / 80 test patients. Protocol: assimilate the
first 2 scans, forecast the rest.

| Estimator | Forecast MAE (cm³) | 90% coverage | calibration gap |
|---|---|---|---|
| Heuristic (priors) | 15.9 | 31% | 59% |
| Learned (trained)  | 14.4 | 36% | 54% |

Well-calibrated 90% intervals should contain the truth ~90% of the time. The
**learned** estimator is closer to that target here.

![estimator comparison](outputs/estimator_comparison.png)

## Calibration (PIT)

The probability integral transform should be uniform for a calibrated model.
A U-shape means the intervals are too narrow (overconfident); a central hump
means too wide.

![PIT calibration](outputs/calibration_pit.png)

## Takeaway

Training the estimator on data — so the forecast spread reflects real parameter
variability instead of hand-set guesses — improves both accuracy and calibration
here (lower MAE, coverage nearer nominal).

But read the numbers honestly: **both estimators are still overconfident**, with
90% coverage well below 90%. Two causes, both fixable: with only two scans the
particle filter concentrates the ensemble, and the forecast currently omits
process and observation noise, so the bands are too tight. The concrete next
steps — inject process noise in assimilation, add observation noise to the
predictive band, and use more follow-up scans — are tracked in
[docs/ROADMAP.md](../docs/ROADMAP.md). On real longitudinal data this same
harness is how you'd confirm the twin has actually become trustworthy.
