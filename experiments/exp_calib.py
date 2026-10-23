"""Cleaner calibration experiment for the ISBI paper.

Differences from analysis/run_analysis.py in the repo:
  * three disjoint cohorts per replicate: train / calibration / test
  * everything seeded (assimilation resampling + jitter)
  * two data-generating worlds: 'well' (truth = Gompertz, as in repo) and
    'mis' (truth = Gompertz with a random-walk drift in the growth rate, i.e. real model error)
  * stores raw posterior-predictive trajectories so noise variants can be evaluated offline
Usage: python exp_calib.py OUTDIR
"""
import sys, os, pickle, time
import numpy as np
from multiprocessing import Pool

sys.path.insert(0, os.path.join(os.environ["REPO"], "src"))
from oncotwin.data.adapters import SyntheticCohort, PatientRecord
from oncotwin.domain import TumorMeasurement, ParameterEnsemble
from oncotwin.parameters import LearnedEstimator
from oncotwin.growth import simulate

SCAN_DAYS = (60, 120, 180, 240, 300, 360)
WARMUP = 2
OBS_SD = 2.0
DRIFT_SD = 0.02          # per sqrt(day) log-drift of growth rate in 'mis' world


class MisCohort(SyntheticCohort):
    """Same features / parameters as SyntheticCohort, but truth has stochastic growth-rate drift."""
    def records(self):
        for i in range(self.n):
            f = self._draw_features()
            p = self._draw_true_params(f)
            r0, K = p[0], p[1]
            s = np.cumsum(self.rng.normal(0, DRIFT_SD, 400))
            v = f.baseline_volume_cm3
            vol = {}
            for day in range(1, 361):
                rt = r0 * np.exp(s[day - 1])
                v = max(v + rt * v * np.log(K / v), 1e-3)
                vol[day] = v
            meas = [TumorMeasurement(float(d), float(max(1.0, vol[d] + self.rng.normal(0, OBS_SD))), "imaging", OBS_SD)
                    for d in SCAN_DAYS]
            yield PatientRecord(f"MIS-{i:04d}", f, meas, p)


def make_cohort(world, n, seed):
    cls = SyntheticCohort if world == "well" else MisCohort
    return list(cls(n=n, scan_days=SCAN_DAYS, seed=seed).records())


def run_patient(rec, est, n_part, rng, assimilate=True):
    ens = est.estimate(rec.features, n_part)
    P, w = ens.particles, ens.weights
    v0 = rec.features.baseline_volume_cm3
    held = rec.measurements[WARMUP:]
    t_held = np.array([m.time_days for m in held])
    ess = float(n_part)
    if assimilate:
        t_w = np.array([m.time_days for m in rec.measurements[:WARMUP]])
        y_w = np.array([m.volume_cm3 for m in rec.measurements[:WARMUP]])
        sd = np.array([m.noise_sd for m in rec.measurements[:WARMUP]])
        ll = np.array([-0.5 * np.sum(((simulate(v0, p, None, t_w) - y_w) / sd) ** 2) for p in P])
        ww = np.exp(ll - ll.max()); ww /= ww.sum()
        ess = float(1.0 / np.sum(ww ** 2))
        pos = (np.arange(n_part) + rng.random()) / n_part
        idx = np.clip(np.searchsorted(np.cumsum(ww), pos), 0, n_part - 1)
        P = P[idx]
        P = (P + rng.normal(0, 0.02 * P.std(axis=0), P.shape)).clip(1e-5, None)
    traj = np.array([simulate(v0, p, None, t_held) for p in P])          # (N, H)
    return dict(traj=traj.astype(np.float32), obs=np.array([m.volume_cm3 for m in held]),
                t=t_held, ess=ess)


def task(args):
    world, seed, split, n, n_part, assim = args
    base = {"train": 1000, "cal": 2000, "test": 3000}[split]
    recs = make_cohort(world, n, base + seed + (0 if world == "well" else 500))
    train = make_cohort(world, 120, 1000 + seed + (0 if world == "well" else 500))
    est = LearnedEstimator(seed=seed).fit(train)
    rng = np.random.default_rng(10_000 + seed)
    out = [run_patient(r, est, n_part, rng, assim) for r in recs]
    return (world, seed, split, n_part, assim), out


if __name__ == "__main__":
    outdir = sys.argv[1]; os.makedirs(outdir, exist_ok=True)
    seeds = range(5)
    tasks = []
    for world in ("well", "mis"):
        for s in seeds:
            tasks.append((world, s, "cal", 60, 400, True))
            tasks.append((world, s, "test", 100, 400, True))
            tasks.append((world, s, "test", 100, 400, False))     # prior only, no assimilation (cheap)
    # particle-count ablation (well-specified, seed 0, 40 test patients)
    tasks.append(("well", 0, "test", 40, 4000, True))
    t0 = time.time()
    res = {}
    with Pool(2) as pool:
        for k, v in pool.imap_unordered(task, tasks):
            res[k] = v
            print(k, f"{time.time()-t0:.0f}s", flush=True)
            pickle.dump(res, open(os.path.join(outdir, "raw.pkl"), "wb"))
    print("done", time.time() - t0)
