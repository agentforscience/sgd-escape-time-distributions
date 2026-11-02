"""Interval-censored parametric survival fits for escape-time data.

An observation is (lo, hi]: the event is known to lie between two checks.
Right-censored replicas have hi = inf and lo = the last time checked.
"""
import numpy as np
from scipy import optimize, stats, special


def _log_diff(log_a, log_b):
    """log(exp(log_a) - exp(log_b)) for log_a >= log_b, safely."""
    d = np.minimum(log_b - log_a, -1e-300)
    return log_a + np.log(-np.expm1(d))


def _ll_from_logsf(logsf, lo, hi):
    """Sum of log-likelihood given a log survival function."""
    ev = np.isfinite(hi)
    ll = 0.0
    if ev.any():
        ll += _log_diff(logsf(lo[ev]), logsf(hi[ev])).sum()
    if (~ev).any():
        ll += logsf(lo[~ev]).sum()
    return ll


def _weibull_logsf(lam, k):
    return lambda t: -np.power(np.maximum(t, 0.0) / lam, k)


def _lognorm_logsf(mu, s):
    return lambda t: stats.norm.logsf((np.log(np.maximum(t, 1e-300)) - mu) / s)


def _gamma_logsf(a, scale):
    return lambda t: np.log(np.maximum(special.gammaincc(a, np.maximum(t, 0.0) / scale), 1e-300))


def _shifted_exp_logsf(t0, lam):
    return lambda t: -np.maximum(t - t0, 0.0) / lam


def init_scale(lo, hi):
    """Crude scale: exponential MLE with censored exposure."""
    ev = np.isfinite(hi)
    mid = np.where(ev, 0.5 * (lo + hi), lo)
    return max(mid.sum() / max(ev.sum(), 1), 1e-9)


def fit_all(lo, hi):
    """Fit exponential, Weibull, gamma, lognormal, shifted exponential.

    Returns a dict of per-model params, ll, aic, plus the LRT p-value of k=1.
    """
    lo = np.asarray(lo, float)
    hi = np.asarray(hi, float)
    n = len(lo)
    s0 = init_scale(lo, hi)
    out = {'n': n, 'n_events': int(np.isfinite(hi).sum())}

    def run(nll, x0s):
        best = None
        for x0 in x0s:
            r = optimize.minimize(nll, x0, method='Nelder-Mead',
                                  options=dict(xatol=1e-4, fatol=1e-4, maxiter=2000))
            if best is None or r.fun < best.fun:
                best = r
        return best

    r = optimize.minimize_scalar(
        lambda u: -_ll_from_logsf(_weibull_logsf(np.exp(u), 1.0), lo, hi),
        bounds=(np.log(s0) - 6, np.log(s0) + 6), method='bounded')
    out['exp'] = dict(lam=float(np.exp(r.x)), ll=float(-r.fun), aic=float(2 * r.fun + 2))

    r = run(lambda x: -_ll_from_logsf(_weibull_logsf(np.exp(x[0]), np.exp(x[1])), lo, hi),
            [[np.log(s0), 0.0], [np.log(s0), -0.5], [np.log(s0), 0.7]])
    out['weibull'] = dict(lam=float(np.exp(r.x[0])), k=float(np.exp(r.x[1])),
                          ll=float(-r.fun), aic=float(2 * r.fun + 4))

    r = run(lambda x: -_ll_from_logsf(_gamma_logsf(np.exp(x[0]), np.exp(x[1])), lo, hi),
            [[0.0, np.log(s0)], [1.0, np.log(s0) - 1.0]])
    out['gamma'] = dict(a=float(np.exp(r.x[0])), scale=float(np.exp(r.x[1])),
                        ll=float(-r.fun), aic=float(2 * r.fun + 4))

    r = run(lambda x: -_ll_from_logsf(_lognorm_logsf(x[0], np.exp(x[1])), lo, hi),
            [[np.log(s0), 0.0], [np.log(s0), -1.0]])
    out['lognormal'] = dict(mu=float(r.x[0]), s=float(np.exp(r.x[1])),
                            ll=float(-r.fun), aic=float(2 * r.fun + 4))

    # Shifted exponential: no mass before t0 (a pure dead time), so t0 <= earliest upper bound.
    tmax0 = float(np.min(hi[np.isfinite(hi)])) if np.isfinite(hi).any() else float(lo.min())
    r = run(lambda x: -_ll_from_logsf(
        _shifted_exp_logsf(tmax0 * special.expit(x[0]), np.exp(x[1])), lo, hi),
        [[-1.0, np.log(s0)], [1.0, np.log(s0)]])
    out['shifted_exp'] = dict(t0=float(tmax0 * special.expit(r.x[0])), lam=float(np.exp(r.x[1])),
                              ll=float(-r.fun), aic=float(2 * r.fun + 4))

    lrt = 2 * (out['weibull']['ll'] - out['exp']['ll'])
    out['lrt_k1_stat'] = float(max(lrt, 0.0))
    out['lrt_k1_p'] = float(stats.chi2.sf(max(lrt, 0.0), 1))
    models = ['exp', 'weibull', 'gamma', 'lognormal', 'shifted_exp']
    out['best_model'] = min(models, key=lambda m: out[m]['aic'])
    return out


def fit_weibull(lo, hi):
    """Weibull-only MLE (fast; used inside bootstraps). Returns (lam, k)."""
    s0 = init_scale(lo, hi)
    r = optimize.minimize(
        lambda x: -_ll_from_logsf(_weibull_logsf(np.exp(x[0]), np.exp(x[1])), lo, hi),
        [np.log(s0), 0.0], method='Nelder-Mead', options=dict(xatol=1e-3, fatol=1e-3, maxiter=600))
    return float(np.exp(r.x[0])), float(np.exp(r.x[1]))


def bootstrap_k(lo, hi, n_boot=200, seed=0):
    """Replica-level percentile bootstrap CI for Weibull k and for log-median."""
    rng = np.random.default_rng(seed)
    n = len(lo)
    ks, meds = [], []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        if not np.isfinite(hi[i]).any():
            continue
        lam, k = fit_weibull(lo[i], hi[i])
        ks.append(k)
        meds.append(lam * np.log(2) ** (1 / k))
    if not ks:
        return (np.nan,) * 4
    return (*np.percentile(ks, [2.5, 97.5]), *np.percentile(meds, [2.5, 97.5]))


def km_median(lo, hi):
    """Median by Turnbull-free midpoint Kaplan-Meier; inf if survival never reaches .5."""
    ev = np.isfinite(hi)
    t = np.where(ev, 0.5 * (lo + hi), lo)
    o = np.argsort(t)
    t, ev = t[o], ev[o]
    at_risk = len(t)
    s = 1.0
    for ti, e in zip(t, ev):
        if e:
            s *= 1 - 1 / at_risk
        at_risk -= 1
        if s <= 0.5:
            return float(ti)
    return float('inf')


def weibull_mean(lam, k):
    return float(lam * special.gamma(1 + 1 / k))


if __name__ == '__main__':
    # Recovery test: interval-censored Weibull data with k in {.7,1,1.3}, 25% censoring.
    rng = np.random.default_rng(0)
    for k_true in [0.7, 1.0, 1.3]:
        ks, ps = [], []
        for rep in range(40):
            t = 1000 * rng.weibull(k_true, 300)
            c = np.quantile(t, 0.75)
            dt = 50
            lo = np.floor(np.minimum(t, c) / dt) * dt
            hi = np.where(t < c, lo + dt, np.inf)
            f = fit_all(lo, hi)
            ks.append(f['weibull']['k'])
            ps.append(f['lrt_k1_p'])
        print(f'k_true={k_true}: mean k_hat={np.mean(ks):.3f} sd={np.std(ks):.3f} '
              f'reject_k1={np.mean(np.array(ps) < .05):.2f}')
