"""Turn saved runs into interval-censored observations and summary rows."""
import glob, json
import numpy as np


def intervals(z):
    """Per replica (lo, hi]: exit between two checks; censored replicas have hi=inf at the last check."""
    grid = z['grid']
    te = z['t_exit']
    last = int(z['last_step'])
    ev = np.isfinite(te)
    c = np.where(ev, te, 0).astype(int)
    lo = np.where(ev, np.where(c > 0, grid[np.maximum(c - 1, 0)], 0), last).astype(float)
    hi = np.where(ev, grid[c].astype(float), np.inf)
    return lo, hi


def load_all(pattern='results/escape/*.npz'):
    out = []
    for f in sorted(glob.glob(pattern)):
        if f.endswith('.tmp.npz') or f.endswith('.fail.npz'):
            continue
        z = np.load(f, allow_pickle=False)
        cfg = json.loads(str(z['cfg']))
        out.append((cfg, z))
    return out
