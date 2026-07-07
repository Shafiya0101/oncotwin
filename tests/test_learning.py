import numpy as np
from oncotwin import SyntheticCohort, OncoTwinEngine
from oncotwin.parameters import LearnedEstimator


def test_learned_estimator_trains_and_estimates():
    train = list(SyntheticCohort(n=40, seed=1).records())
    est = LearnedEstimator().fit(train)
    assert est.beta is not None and est.resid is not None

    twin = OncoTwinEngine(est).create_twin("PT-L", train[0].features)
    ens = twin.parameters
    assert ens.particles.shape[1] == 4
    assert np.all(ens.particles > 0)
    assert abs(ens.weights.sum() - 1.0) < 1e-6


def test_learned_estimator_requires_fit():
    import pytest
    with pytest.raises(RuntimeError):
        LearnedEstimator().estimate(list(SyntheticCohort(n=1, seed=0).records())[0].features)
