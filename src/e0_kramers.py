"""E0: controlled double well with an exact Kramers rate.

Validates the simulator and estimators, and maps which mechanism produces Weibull k != 1:
  - start-up transient (start at the minimum vs quasi-stationary start),
  - low barrier (Delta/T small),
  - pooled heterogeneity (mixture of well widths),
  - loss-proportional noise (Mori-type).
V(x) = Delta (x_1^2/w^2 - 1)^2 + (c/2) sum_{j>=2} x_j^2.  Start in the right well; escape = x_1 < -w/2.
Gaussian minibatch noise of variance sigma^2/B is the CLT limit of with-replacement sampling.
"""
import json, os, sys, time
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(__file__))
import survival as sv

OUT = 'results/e0'
R = 1000
D = 8
C_ORTH = 4.0
ETA = 0.002


def grad_v(x, delta, w):
    g = C_ORTH * x
    g[:, 0] = 4 * delta / w**2 * x[:, 0] * (x[:, 0]**2 / w**2 - 1)
    return g


def loss_v(x, delta, w):
    return delta * (x[:, 0]**2 / w**2 - 1)**2 + 0.5 * C_ORTH * (x[:, 1:]**2).sum(1)


def simulate(opt, delta, w, T, seed, t_max, start='min', noise='additive', aligned=False, eta_adam=None, r=R):
    """Run replicas to first exit. T = eta sigma^2/2 sets sigma (B=1 w.l.o.g.)."""
    rng = np.random.default_rng(seed)
    sigma = np.full(D, np.sqrt(2 * T / ETA))
    a = 8 * delta / w**2
    if aligned:  # H-aligned noise: variance proportional to the curvature in each direction
        sig2 = np.full(D, C_ORTH)
        sig2[0] = a
        sigma = np.sqrt(sig2 / sig2.mean()) * sigma
    x = np.zeros((r, D))
    x[:, 0] = w
    if start == 'qsd':  # Gaussian approximation of the well's stationary law
        std = np.sqrt(T / np.r_[a, np.full(D - 1, C_ORTH)])
        if aligned:
            std = sigma * np.sqrt(ETA / 2 / np.r_[a, np.full(D - 1, C_ORTH)])
        x += rng.normal(size=x.shape) * std
    t = np.full(r, np.inf)
    alive = np.arange(r)
    if opt == 'adam':
        m, v = np.zeros((r, D)), np.zeros((r, D))
        b1, b2 = 0.9, 0.999
    for step in range(1, t_max + 1):
        g = grad_v(x, delta, w)
        n = rng.normal(size=x.shape) * sigma
        if noise == 'loss':  # noise std scales with sqrt(1 + V/delta)
            n *= np.sqrt(1 + loss_v(x, delta, w) / delta)[:, None]
        g = g + n
        if opt == 'sgd':
            x -= ETA * g
        else:
            m = b1 * m + (1 - b1) * g
            v = b2 * v + (1 - b2) * g * g
            x -= eta_adam * (m / (1 - b1**step)) / (np.sqrt(v / (1 - b2**step)) + 1e-8)
        done = x[:, 0] < -w / 2
        if done.any():
            t[alive[done]] = step
            keep = ~done
            x, alive = x[keep], alive[keep]
            if opt == 'adam':
                m, v = m[keep], v[keep]
            if len(alive) == 0:
                break
    return t, step


def job(cfg):
    t, last = simulate(cfg['opt'], cfg['delta'], cfg['w'], cfg['T'], cfg['seed'], cfg['t_max'],
                       cfg.get('start', 'min'), cfg.get('noise', 'additive'), cfg.get('aligned', False),
                       cfg.get('eta_adam'))
    lo = np.where(np.isfinite(t), t - 1, cfg['t_max'])
    hi = t.copy()
    return cfg, lo, hi


def kramers(delta, w, T):
    a, b = 8 * delta / w**2, 4 * delta / w**2
    return 2 * np.pi / (ETA * np.sqrt(a * b)) * np.exp(delta / T)


def main():
    os.makedirs(OUT, exist_ok=True)
    cfgs = []
    seed = 100
    # Kramers validation & transient map: barrier/T grid, three widths, two starts.
    for w in [0.5, 1.0, 2.0]:
        for bt in [1, 2, 3, 4, 5, 6]:
            for start in ['min', 'qsd']:
                seed += 1
                T = 1.0 / bt
                cfgs.append(dict(exp='kramers', opt='sgd', delta=1.0, w=w, T=T, start=start, seed=seed,
                                 t_max=int(min(80 * kramers(1.0, w, T) + 2000, 1200000))))
    # Noise model: loss-proportional noise, tests how the batch-size law changes.
    for bt in [1, 2, 3, 4, 5, 6]:
        seed += 1
        cfgs.append(dict(exp='lossnoise', opt='sgd', delta=1.0, w=1.0, T=1.0 / bt, noise='loss',
                         start='min', seed=seed, t_max=1200000))
    # Adam at matched temperature: eta_A sigma/2 = T for isotropic noise (see REPORT).
    for w in [0.5, 2.0]:
        for aligned in [False, True]:
            for bt in [3, 4, 5]:
                T = 1.0 / bt
                sig = np.sqrt(2 * T / ETA)
                if aligned:  # sigma in the stiff/escape coordinate differs; match on coordinate 1
                    a = 8 / w**2
                    sig2 = np.full(D, C_ORTH); sig2[0] = a
                    sig = sig * np.sqrt(a / sig2.mean())
                for opt in ['sgd', 'adam']:
                    seed += 1
                    cfgs.append(dict(exp='adam', opt=opt, delta=1.0, w=w, T=T, aligned=aligned, start='min', seed=seed,
                                     eta_adam=ETA * sig, t_max=400000))
    print(len(cfgs), 'configs', flush=True)
    t0 = time.time()
    rows = []
    with Pool(12) as p:
        for cfg, lo, hi in p.imap_unordered(job, cfgs):
            name = f"{cfg['exp']}_{cfg['seed']}"
            np.savez(f'{OUT}/{name}.npz', lo=lo, hi=hi)
            f = sv.fit_all(lo, hi)
            mean_cens = float(np.where(np.isfinite(hi), hi, lo).mean())
            rows.append(dict(cfg, name=name, fit=f, mean_obs=mean_cens,
                             kramers=kramers(cfg['delta'], cfg['w'], cfg['T'])))
            print(f"{name} w={cfg['w']} bt={1/cfg['T']:.0f} {cfg['opt']} {cfg.get('start')} "
                  f"events={f['n_events']} mean={mean_cens:.0f} kr={rows[-1]['kramers']:.0f} "
                  f"k={f['weibull']['k']:.3f} best={f['best_model']} {time.time()-t0:.0f}s", flush=True)
            json.dump(rows, open(f'{OUT}/summary.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
