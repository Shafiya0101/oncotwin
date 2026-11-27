#!/usr/bin/env python
"""Write analysis/REAL_RESULTS.md (and a README snippet) from results.json.

The text is generated from the numbers, so the document can never disagree
with the script output. Usage:
  python analysis/write_real_results.py
  python analysis/write_real_results.py --results analysis/outputs/real_cohort_clinical_only/results.json
"""
import argparse
import json
from pathlib import Path


def fmt(v, ci):
    return f"{v:.3f} (95% CI {ci[0]:.3f} to {ci[1]:.3f})"


def verdict(c, ci):
    if ci[0] > 0.5:
        return "above chance"
    if ci[1] < 0.5:
        return "below chance"
    return "not distinguishable from chance"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="analysis/outputs/real_cohort/results.json")
    ap.add_argument("--out", default="analysis/REAL_RESULTS.md")
    ap.add_argument("--snippet", default=None)
    a = ap.parse_args()

    res = json.loads(Path(a.results).read_text())
    cfg, bt = res["config"], res["bootstrap"]
    models, pairs = bt["models"], bt["pairs"]
    folder = Path(a.results).parent
    snippet_path = Path(a.snippet) if a.snippet else folder / "README_snippet.md"
    demo = cfg.get("demo")

    L = []
    if demo:
        L += ["> **SYNTHETIC DEMO DATA: not real results.**", ""]
    L += ["# Real-cohort results: TCIA NSCLC-Radiomics (Lung1)", "",
          f"{cfg['n_patients']} patients, {cfg['n_events']} deaths. Outcome: overall survival. "
          f"Repeated {cfg['folds']}-fold cross-validation x {cfg['repeats']} repeats; every data-dependent step is "
          f"fitted inside training folds; {bt['n_boot']} patient-level bootstrap resamples (95% percentile intervals). "
          "Full protocol: docs/EVAL_PROTOCOL.md.", "", "## Discrimination", ""]
    for m, r in models.items():
        L.append(f"- **{m}**: C-index {fmt(r['c'], r['c_ci'])}, {verdict(r['c'], r['c_ci'])}.")
    L += [""]
    if pairs:
        L += ["## Does adding information help?", "",
              "Paired bootstrap differences in C-index (same resamples for both models):", ""]
        for k, v in pairs.items():
            sign = "+" if v["delta"] >= 0 else ""
            tail = ("the interval excludes zero." if v["excludes_zero"]
                    else "the interval includes zero, so this difference is not established.")
            L.append(f"- {k}: {sign}{fmt(v['delta'], v['ci'])}; {tail}")
        L += [""]
    if "twin baseline" in models:
        t = models["twin baseline"]
        L += ["## Comparison with the original twin risk", "",
              f"The twin's biology-based risk, evaluated on the same patients without any training, scored "
              f"{fmt(t['c'], t['c_ci'])}.", ""]
    L += ["## Calibration (24 months)", ""]
    for m, c in res["calibration"].items():
        L.append(f"- **{m}**: mean absolute calibration error {c['mean_abs_calibration_error']:.3f} "
                 "(predicted vs Kaplan-Meier observed survival across 5 risk groups). See the table in "
                 f"{folder.as_posix()}/RESULTS_real_cohort.md.")
    L += [""]
    if res.get("importance"):
        top = ", ".join(f"{d['variable']} ({d['mean_c_drop']:.3f})" for d in res["importance"][:5])
        L += ["## Most important variables", "",
              f"Permutation importance on held-out folds ({res['importance_model']} model), top five with the mean "
              f"drop in C-index: {top}.", ""]
    L += ["## Limitations", "",
          "- Single retrospective cohort; no external validation.",
          "- Imaging features come from the project's own radiomics code (few features, not the standard IBSI set).",
          "- Ridge-penalised Cox model; no non-linear models were tried.",
          "- Calibration is shown at one horizon (24 months) with wide intervals in small risk groups.",
          "- Molecular (omics) data are not included in this analysis.", "",
          "## Reproduce", "",
          "```", "python scripts/build_clinical_table.py data/real/Lung1.clinical.csv",
          "python scripts/extract_cohort_radiomics.py", "python analysis/survival_ablation.py",
          "python analysis/write_real_results.py", "```"]
    Path(a.out).write_text("\n".join(L))

    S = ["- **Real survival validation** (`python analysis/survival_ablation.py`): "
         f"{cfg['n_patients']} real NSCLC patients, repeated cross-validation, bootstrap CIs. "]
    S[0] += "; ".join(f"{m}: C-index {fmt(r['c'], r['c_ci'])}" for m, r in models.items() if m != "twin baseline") + "."
    if "twin baseline" in models:
        S[0] += f" The original untrained twin risk scored {models['twin baseline']['c']:.3f}."
    S[0] += " See [analysis/REAL_RESULTS.md](analysis/REAL_RESULTS.md)."
    if demo:
        S = ["SYNTHETIC DEMO OUTPUT: do not paste into the README.", ""] + S
    snippet_path.write_text("\n".join(S))
    print(f"Wrote {a.out} and {snippet_path}")


if __name__ == "__main__":
    main()
