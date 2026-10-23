import pickle, json, sys
import numpy as np

raw = pickle.load(open("out/raw.pkl", "rb"))
OBS_SD = 2.0
LEVELS = (0.5, 0.8, 0.9)
PN_GRID = np.round(np.arange(0, 0.0301, 0.0005), 5)


def samples(p, pn, obsnoise, rng):
    S = p["traj"].astype(np.float64)
    if pn > 0:
        sig = pn * np.sqrt(p["t"])[None, :]
        S = S * np.exp(rng.normal(0, 1, S.shape) * sig - 0.5 * sig ** 2)
    if obsnoise:
        S = S + rng.normal(0, OBS_SD, S.shape)
    return S


def crps(S, y):                      # S (N,H), y (H,)
    N = S.shape[0]
    t1 = np.abs(S - y[None, :]).mean(0)
    Ss = np.sort(S, 0)
    w = (2 * np.arange(1, N + 1) - N - 1)[:, None]
    t2 = (w * Ss).sum(0) / (N * N)   # = 0.5*E|X-X'|
    return t1 - t2


def evaluate(patients, pn, obsnoise, seed=0, idx=None):
    rng = np.random.default_rng(seed)
    cov = {l: [] for l in LEVELS}; width = []; cr = []; ae = []; pit = []
    for k, p in enumerate(patients):
        if idx is not None and k >= idx: break
        S = samples(p, pn, obsnoise, rng); y = p["obs"]
        for l in LEVELS:
            lo, hi = np.quantile(S, [(1 - l) / 2, 1 - (1 - l) / 2], axis=0)
            cov[l].extend(((y >= lo) & (y <= hi)).tolist())
            if l == 0.9: width.extend((hi - lo).tolist())
        cr.extend(crps(S, y).tolist())
        ae.extend(np.abs(np.median(S, 0) - y).tolist())
        pit.extend((S < y[None, :]).mean(0).tolist())
    return dict(c50=np.mean(cov[0.5]), c80=np.mean(cov[0.8]), c90=np.mean(cov[0.9]),
                width90=np.mean(width), crps=np.mean(cr), mae=np.mean(ae), pit=np.array(pit))


def tune_pn(cal, obsnoise, target=0.9):
    for pn in PN_GRID:
        if evaluate(cal, pn, obsnoise, seed=1)["c90"] >= target:
            return float(pn)
    return float(PN_GRID[-1])


res = {}
for world in ("well", "mis"):
    rows = {k: [] for k in ["A_naive", "B_obsnoise", "C_pn_tuned", "D_pn_tuned_obs", "E_pn003", "F_prior_only"]}
    tuned = {"C": [], "D": []}
    ess = []
    for s in range(5):
        cal = raw[(world, s, "cal", 400, True)]
        test = raw[(world, s, "test", 400, True)]
        prior = raw[(world, s, "test", 400, False)]
        ess += [p["ess"] / 400 for p in test]
        pc = tune_pn(cal, False); pd_ = tune_pn(cal, True)
        tuned["C"].append(pc); tuned["D"].append(pd_)
        rows["A_naive"].append(evaluate(test, 0, False))
        rows["B_obsnoise"].append(evaluate(test, 0, True))
        rows["C_pn_tuned"].append(evaluate(test, pc, False))
        rows["D_pn_tuned_obs"].append(evaluate(test, pd_, True))
        rows["E_pn003"].append(evaluate(test, 0.003, False))
        rows["F_prior_only"].append(evaluate(prior, 0, False))
    res[world] = dict(rows=rows, tuned=tuned, ess=np.array(ess))

def summ(rs, key):
    v = np.array([r[key] for r in rs]); return v.mean(), v.std(ddof=1)

print("=== results (mean ± sd over 5 replicates; 100 test patients x 4 held-out scans each) ===")
for world in ("well", "mis"):
    print("\nWORLD:", world, " median ESS/N after assimilation: %.3f (IQR %.3f-%.3f)" % (
        np.median(res[world]["ess"]), *np.quantile(res[world]["ess"], [.25, .75])))
    print(" tuned PN (no obs noise):", res[world]["tuned"]["C"], " (with obs noise):", res[world]["tuned"]["D"])
    for k, rs in res[world]["rows"].items():
        print(f" {k:16s} c50={summ(rs,'c50')[0]:.3f} c80={summ(rs,'c80')[0]:.3f} "
              f"c90={summ(rs,'c90')[0]:.3f}±{summ(rs,'c90')[1]:.3f} w90={summ(rs,'width90')[0]:.1f} "
              f"crps={summ(rs,'crps')[0]:.2f} mae={summ(rs,'mae')[0]:.2f}")

# particle-count ablation
print("\n=== particle ablation (well, seed 0, first 40 test patients) ===")
big = raw[("well", 0, "test", 4000, True)]
small = raw[("well", 0, "test", 400, True)][:40]
for name, P in [("N=400", small), ("N=4000", big)]:
    a = evaluate(P, 0, False); b = evaluate(P, 0, True)
    print(f" {name}: naive c90={a['c90']:.3f} w90={a['width90']:.1f} | +obs noise c90={b['c90']:.3f} | median ESS/N="
          f"{np.median([p['ess'] for p in P])/len(P[0]['traj']):.3f}")

# coverage-vs-PN sweep (pooled over replicates, test) for figure
sweep = {}
for world in ("well", "mis"):
    for obsn in (False, True):
        sweep[(world, obsn)] = [np.mean([evaluate(raw[(world, s, 'test', 400, True)], pn, obsn)["c90"] for s in range(5)])
                                for pn in PN_GRID[::2]]
# horizon-wise coverage for naive vs D
pickle.dump(dict(res=res, sweep=sweep, grid=PN_GRID[::2]), open("out/analysis.pkl", "wb"))
