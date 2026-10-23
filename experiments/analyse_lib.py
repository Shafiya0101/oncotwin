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


