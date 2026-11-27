@echo off
rem Reruns the whole real-cohort analysis. Run from the repo root with the .venv active.
python scripts\build_clinical_table.py data\real\Lung1.clinical.csv || exit /b 1
python scripts\extract_cohort_radiomics.py || exit /b 1
python analysis\survival_ablation.py || exit /b 1
python analysis\write_real_results.py || exit /b 1
