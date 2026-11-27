.PHONY: install api web serve demo test backtest clean

install:
	pip install -e ".[dev]"

api:
	uvicorn oncotwin.api.main:app --reload --port 8000

web:
	python -m http.server 8080

serve:
	./scripts/serve.sh

imaging-demo:
	python examples/imaging_demo.py

demo:
	python examples/demo.py

test:
	pytest -q

backtest:
	python -c "from oncotwin import SyntheticCohort, backtest; print(backtest(list(SyntheticCohort(n=60, seed=3).records())))"

analysis:
	python analysis/run_analysis.py

assets:
	python scripts/build_web_assets.py

clean:
	rm -rf build dist *.egg-info src/*.egg-info .pytest_cache data/twins examples/outputs
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
# Append to your Makefile with:  type Makefile.week1 >> Makefile   (run it ONCE)
.PHONY: clinical radiomics ablation results real-cohort

clinical:
	python scripts/build_clinical_table.py data/real/Lung1.clinical.csv

radiomics:
	python scripts/extract_cohort_radiomics.py

ablation:
	python analysis/survival_ablation.py

results:
	python analysis/write_real_results.py

real-cohort: clinical radiomics ablation results
