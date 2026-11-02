"""All statistics for the report. Reads results/escape, results/e0, results/e4; writes results/analysis/*.json."""
import glob, json, os, sys
import numpy as np, pandas as pd
from scipy import stats
sys.path.insert(0, 'src')
import load_escape as le, survival as sv

os.makedirs('results/analysis', exist_ok=True)
NB = int(os.environ.get("NB", 200))
rng = np.random.default_rng(0)
sel = json.load(open('results/selected_minima.json'))
geo = {m['name']: m for m in sel['A'] + sel['B']}
calib = {r['min']: r for r in json.load(open('results/calibration_eta0.1.json'))}


def bh(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    q = p[o] * len(p) / (np.arange(len(p)) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty_like(q)
    out[o] = q
    return out


def boot_mean(x, n=10000, seed=1):
    """CI for a mean, resampling minima (the unit of generalization)."""
    x = np.asarray(x, float)
    r = np.random.default_rng(seed)
    m = x[r.integers(0, len(x), (n, len(x)))].mean(1)
    return float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def weibull_median(f):
    return f['weibull']['lam'] * np.log(2) ** (1 / f['weibull']['k'])


def residual_k(lo, hi, q=0.25):
    """Left truncation at the q-quantile of exit: Weibull fit of T - t0 among the survivors of t0.

    A k that falls towards 1 means the excess of k above 1 lives in the early part of the run, a delay.
    """
    ev = np.isfinite(hi)
    if ev.sum() < 40:
        return None
    t0 = float(np.quantile(hi[ev], q))
    keep = lo >= t0
    if keep.sum() < 30 or not np.isfinite(hi[keep]).any():
        return None
    return dict(t0=t0, n=int(keep.sum()), k=sv.fit_weibull(lo[keep] - t0, hi[keep] - t0)[1])


# ---------------------------------------------------------------- per-condition table
rows = []
data = {}
for cfg, z in le.load_all():
    lo, hi = le.intervals(z)
    f = sv.fit_all(lo, hi)
    g = geo[cfg['min']]
    ev = np.isfinite(hi)
    kl, kh, ml, mh = sv.bootstrap_k(lo, hi, n_boot=NB, seed=cfg['seed'])
    l = z['l_exit'][ev]
    d = np.round(l, 4)
    uv, uc = np.unique(d, return_counts=True)
    rk, rk50 = residual_k(lo, hi, .25), residual_k(lo, hi, .5)
    r = dict(id=cfg['id'], exp=cfg['exp'], land=cfg['land'], min=cfg['min'], opt=cfg['opt'], lr=cfg['lr'], B=cfg['B'],
             warm=cfg.get('warm', 0), qmult=cfg.get('qmult', 1), R=cfg['R'], lmax=g['lmax'], trace=g['trace'],
             logdet=g['logdet'], L=g['L'], n_events=f['n_events'], frac_events=f['n_events'] / f['n'],
             km_median=sv.km_median(lo, hi), w_median=weibull_median(f), k=f['weibull']['k'], k_lo=kl, k_hi=kh,
             lam=f['weibull']['lam'], med_lo=ml, med_hi=mh, lrt_p=f['lrt_k1_p'], best=f['best_model'],
             daic_w_exp=f['weibull']['aic'] - f['exp']['aic'],
             daic_gamma=f['gamma']['aic'] - f['weibull']['aic'], daic_lognormal=f['lognormal']['aic'] - f['weibull']['aic'],
             daic_shifted=f['shifted_exp']['aic'] - f['weibull']['aic'], t0_shift=f['shifted_exp']['t0'],
             k_res25=rk['k'] if rk else np.nan, k_res50=rk50['k'] if rk50 else np.nan,
             n_dest=len(uv), top_dest_share=float(uc.max() / uc.sum()) if ev.any() else np.nan,
             dest_lower_share=float((l < 0).mean()) if ev.any() else np.nan,
             bar_share=float((z['b_exit'][ev] > 1e-5).mean()) if ev.any() else np.nan,
             dest_dist=float(np.median(z['d_exit'][ev])) if ev.any() else np.nan,
             eta_lmax=cfg['lr'] * g['lmax'], wall=float(z['wall']))
    rows.append(r)
    data[cfg['id']] = (lo, hi)
df = pd.DataFrame(rows)
# The location statistic everywhere is the Kaplan-Meier median.
df['tau_extrap'] = ~np.isfinite(df.km_median)
# Heavily censored runs give meaningless Weibull extrapolations (up to 1e9), so tau is held at the only thing
# observed, the run length: a lower bound, flagged by tau_extrap and left out of slope and factor fits.
df['tau'] = np.where(df.tau_extrap, 150000.0, df.km_median)
df.to_csv('results/analysis/conditions.csv', index=False)
out = {}

# ---------------------------------------------------------------- H1
sgd = df[(df.opt == 'sgd') & (df.exp == 'E1')].copy()
sgd = sgd[sgd.frac_events >= 0.5]
sgd['q'] = bh(sgd.lrt_p)
h1 = {}
for name, s in [('all', sgd)] + [(f'{l}_eta{e}', sgd[(sgd.land == l) & (sgd.lr == e)]) for l in 'AB' for e in (0.1, 0.2)]:
    if len(s) == 0:
        continue
    rec = dict(n=len(s), k_min=s.k.min(), k_med=s.k.median(), k_max=s.k.max(),
               share_in_range=float(s.k.between(.7, 1.3).mean()), share_gt_1p3=float((s.k > 1.3).mean()),
               share_lt_1=float((s.k < 1).mean()), share_lrt_sig=float((s.q < .05).mean()),
               share_daic10=float((s.daic_w_exp < -10).mean()), best_counts=s.best.value_counts().to_dict(),
               k_res25_med=float(s.k_res25.median()), k_res50_med=float(s.k_res50.median()))
    for g_ in ('lmax', 'trace', 'logdet'):
        x = np.log(s[g_]) if g_ != 'logdet' else s[g_]
        rho, p = stats.spearmanr(x, s.k)
        rec[f'spearman_k_{g_}'] = (float(rho), float(p))
    h1[name] = rec
# Heterogeneity: pooling minima. Replicas of different minima stacked into one sample.
pool = {}
for (l, e), s in sgd.groupby(['land', 'lr']):
    lo = np.concatenate([data[i][0] for i in s.id])
    hi = np.concatenate([data[i][1] for i in s.id])
    f = sv.fit_all(lo, hi)
    pool[f'{l}_eta{e}'] = dict(k=f['weibull']['k'], best=f['best_model'], daic=f['weibull']['aic'] - f['exp']['aic'], n=f['n'],
                               mean_within_k=float(s.k.mean()))
h1['pooled_minima'] = pool
s = sgd[sgd.frac_events > .5]
h1['shape_predictors'] = {c: tuple(map(float, stats.spearmanr(s[c], s.k))) for c in ('n_dest', 'top_dest_share', 'dest_dist')}
h1['spearman_tau_lmax_A_eta0.1'] = tuple(map(float, stats.spearmanr(np.log(df[(df.exp == 'E1') & (df.land == 'A') & (df.lr == .1)].lmax),
                                                                    np.log(df[(df.exp == 'E1') & (df.land == 'A') & (df.lr == .1)].tau))))
out['H1'] = h1

# ---------------------------------------------------------------- H2
def h2_table(land, base_lr):
    """SGD/Adam median ratio per minimum, Adam interpolated in log lr to the matched-temperature lr."""
    a = df[(df.exp == 'E2') & (df.land == land)]
    b = df[(df.exp == 'E1') & (df.land == land) & (df.lr == base_lr)]
    rec = []
    for m, s in b.groupby('min'):
        t_sgd = float(s.tau.iloc[0])
        q = a[a['min'] == m].sort_values('lr')
        if len(q) < 2:
            continue
        c = calib[m]
        lrs = [1e-3, 2e-3, 4e-3, 8e-3, 16e-3]
        ax = [c[f'adam_{l:g}'][0] for l in lrs]
        m_lr = float(np.exp(np.interp(np.log(c['sgd_0.1'][0]), np.log(ax), np.log(lrs))))
        t_adam = float(np.exp(np.interp(np.log(m_lr), np.log(q.lr), np.log(q.tau))))
        rec.append(dict(min=m, lmax=float(s.lmax.iloc[0]), trace=float(s.trace.iloc[0]), tau_sgd=t_sgd, tau_adam=t_adam, matched_lr=m_lr,
                        ratio=t_sgd / t_adam, extrap=bool(q.tau_extrap.any() or s.tau_extrap.iloc[0]),
                        sgd_censored=bool(s.tau_extrap.iloc[0]), adam_censored=bool(q.tau_extrap.any()),
                        k_sgd=float(s.k.iloc[0]), k_adam=float(np.interp(np.log(m_lr), np.log(q.lr), q.k))))
    return pd.DataFrame(rec).sort_values('lmax') if rec else pd.DataFrame()


h2 = {}
for land in 'AB':
    t = h2_table(land, 0.1)
    if len(t) < 4:
        continue
    t.to_csv(f'results/analysis/h2_{land}.csv', index=False)
    n3 = max(1, len(t) // 3)
    w, n = np.log(t.ratio.values[:n3]), np.log(t.ratio.values[-n3:])
    h2[land] = dict(n=len(t), n_group=n3, all=[float(np.exp(v)) for v in boot_mean(np.log(t.ratio))],
                    wide=[float(np.exp(v)) for v in boot_mean(w)], narrow=[float(np.exp(v)) for v in boot_mean(n)],
                    spearman_ratio_lmax=tuple(map(float, stats.spearmanr(np.log(t.lmax), np.log(t.ratio)))),
                    spearman_ratio_trace=tuple(map(float, stats.spearmanr(np.log(t.trace), np.log(t.ratio)))),
                    n_extrap=int(t.extrap.sum()))
    obs = t[~t.extrap]
    h2[land]['observed_only'] = dict(n=len(obs), geo_ratio=[float(np.exp(v)) for v in boot_mean(np.log(obs.ratio))] if len(obs) > 1 else None,
                                     spearman_lmax=tuple(map(float, stats.spearmanr(np.log(obs.lmax), np.log(obs.ratio)))) if len(obs) > 2 else None)
    r = np.random.default_rng(3)
    ri = [np.exp(n[r.integers(0, len(n), len(n))].mean() - w[r.integers(0, len(w), len(w))].mean()) for _ in range(10000)]
    h2[land]['ratio_of_ratios'] = [float(np.exp(n.mean() - w.mean())), *map(float, np.percentile(ri, [2.5, 97.5]))]
abl = []
base = df[(df.exp == 'E1') & (df.land == 'A') & (df.lr == .1)].set_index('min')
for (o, lr, w), s in df[df.exp == 'ABL'].groupby(['opt', 'lr', 'warm']):
    lr_ = np.log(s.tau.values / base.loc[s['min']].tau.values)
    abl.append(dict(opt=o, lr=lr, warm=int(w), n=len(s), events=float(s.frac_events.mean()), tau_over_sgd=[float(np.exp(v)) for v in boot_mean(lr_)],
                    k_med=float(s.k.median()), per_min=dict(zip(s['min'].str[-6:], np.round(s.tau.values / base.loc[s['min']].tau.values, 3).tolist()))))
h2['ablations'] = abl
out['H2'] = h2

# ---------------------------------------------------------------- H3
e3 = df[(df.opt == 'sgd') & (df.lr == .1) & (df.exp.isin(['E1', 'E3'])) & (df.B.isin([4, 8, 16, 32, 64]))]
h3 = {}
for land, s in e3.groupby('land'):
    ms = [m for m, q in s.groupby('min') if len(q) >= 4]
    if len(ms) < 2:
        continue
    s = s[s['min'].isin(ms)].sort_values(['min', 'B'])
    usable = s[~s.tau_extrap]
    slopes = [np.polyfit(np.log2(q.B), np.log2(q.tau), 1)[0] for _, q in usable.groupby('min') if len(q) >= 3]
    fac = []
    for m, q in s.groupby('min'):
        q = q.set_index('B')
        for b in (4, 8, 16, 32):
            if b in q.index and 2 * b in q.index:
                # Events-only comparison: an extrapolated tau is flagged rather than used in the factor table.
                fac.append(dict(min=m, B=b, factor=q.tau[2 * b] / q.tau[b], clean=not (q.tau_extrap[2 * b] or q.tau_extrap[b])))
    fac = pd.DataFrame(fac)
    # Power law vs Arrhenius, same parameter count: minimum intercepts + one slope, on uncensored medians.
    Y = np.log(usable.tau.values)
    D = pd.get_dummies(usable['min']).values.astype(float)
    def aic(x):
        Xd = np.c_[D, x]
        b, *_ = np.linalg.lstsq(Xd, Y, rcond=None)
        rss = ((Y - Xd @ b) ** 2).sum()
        return len(Y) * np.log(rss / len(Y)) + 2 * (Xd.shape[1] + 1), float(b[-1])
    ap, bp = aic(np.log(usable.B.values))
    aa, ba = aic(usable.B.values.astype(float))
    h3[land] = dict(n_minima=len(ms), slopes=list(map(float, slopes)), slope=boot_mean(slopes) if len(slopes) > 1 else None,
                    factors_by_B={str(b): [float(np.exp(np.log(v.factor).mean())), len(v)] for b, v in fac[fac.clean].groupby('B')},
                    factor_overall=[float(np.exp(v)) for v in boot_mean(np.log(fac[fac.clean].groupby('min').factor.apply(lambda x: np.exp(np.log(x).mean()))))],
                    aic_power=ap, aic_arrhenius=aa, slope_power=bp, slope_arrhenius_per_B=ba, frac_events_by_B=s.groupby('B').frac_events.mean().to_dict(),
                    tau_by_B=s.groupby('B').tau.apply(lambda x: float(np.exp(np.log(x).mean()))).to_dict(), censored_by_B=s.groupby('B').tau_extrap.sum().to_dict(), n_extrap=int(s.tau_extrap.sum()))
    # Same eta/B: SDE time tau*eta should be constant.
    trip = df[(df.land == land) & (df.opt == 'sgd') & (((df.lr == .05) & (df.B == 8)) | ((df.lr == .1) & (df.B == 16)) | ((df.lr == .2) & (df.B == 32)))]
    tr = trip.assign(sde=trip.tau * trip.lr).pivot_table(index='min', columns='lr', values='sde').dropna()
    if len(tr):
        h3[land]['same_ratio'] = dict(n=len(tr), sde_mean={str(c): float(np.exp(np.log(tr[c]).mean())) for c in tr},
                                      ratio_02_over_01=[float(np.exp(v)) for v in boot_mean(np.log(tr[.2] / tr[.1]))],
                                      ratio_005_over_01=[float(np.exp(v)) for v in boot_mean(np.log(tr[.05] / tr[.1]))],
                                      steps_ratio_02_over_01=[float(np.exp(v)) for v in boot_mean(np.log(tr[.2] / tr[.1] / 2))])
    # lr doubling at fixed B, for an eta exponent.
    hot = df[(df.land == land) & (df.exp == 'E1') & (df.lr == .2)].set_index('min')
    cold = df[(df.land == land) & (df.exp == 'E1') & (df.lr == .1)].set_index('min')
    ix = hot.index.intersection(cold.index)
    if len(ix):
        h3[land]['eta_doubling_factor'] = [float(np.exp(v)) for v in boot_mean(np.log(cold.tau[ix] / hot.tau[ix]))]
out['H3'] = h3

# ---------------------------------------------------------------- robustness
rob = []
for _, r in df[df.exp == 'ROB'].iterrows():
    b = base.loc[r['min']]
    rob.append(dict(min=r['min'][-6:], lmax=r.lmax, tau_q1=float(b.tau), tau_q3=float(r.tau), ratio=float(r.tau / b.tau), k_q1=float(b.k), k_q3=float(r.k)))
out['robust_quench'] = rob

# ---------------------------------------------------------------- H4
recs = [json.load(open(p)) for p in sorted(glob.glob('results/e4/*.json'))]
h4 = {}
rw = []
for rec in recs:
    lam = np.array(rec['lam'])
    for co in rec['conds']:
        for r, v in co['r'].items():
            if v['n'] < 20:
                continue
            c = np.array(v['counts'], float)
            emp = c / c.sum()
            K = len(lam)
            pf, pg, pn = map(np.array, (v['p_full'], v['p_gap'], v['p_noise']))
            rw.append(dict(saddle=rec['saddle'], modal_share=emp.max(), index=K, eta=co['eta'], B=co['B'], r=float(r), n=v['n'], acc1=emp[0], chance=1 / K,
                           gap_abs=lam[0] - lam[1], gap_rel=(lam[0] - lam[1]) / lam[0], lam1=lam[0], lam2=lam[1],
                           sig_ratio=rec['sig'][0] / rec['sig'][1], frac1=v['frac1'], plus=v['plus_share'], med_t=v['med_t'],
                           full_acc=emp[int(pf.argmax())], noise_acc=emp[int(np.array(rec['sig']).argmax())],
                           p_full1=pf[0], p_gap1=pg[0], p_noise1=pn[0],
                           tv_full=.5 * np.abs(emp - pf).sum(), tv_gap=.5 * np.abs(emp - pg).sum(), tv_noise=.5 * np.abs(emp - pn).sum(),
                           modal_is_1=bool(emp.argmax() == 0)))
rw = pd.DataFrame(rw)
if len(rw):
    rw.to_csv('results/analysis/h4.csv', index=False)
    for (eta, B, r), s in rw[rw.r <= 1.0].groupby(['eta', 'B', 'r']):
        g = dict(n=len(s), modal_share=boot_mean(s.modal_share), share_lt_20=float((s.acc1 < .2).mean()), acc=boot_mean(s.acc1), chance=float(s.chance.mean()), full_acc=boot_mean(s.full_acc), noise_acc=boot_mean(s.noise_acc),
                 tv_full=float(s.tv_full.mean()), tv_gap=float(s.tv_gap.mean()), tv_noise=float(s.tv_noise.mean()),
                 r_gap_rel=tuple(map(float, stats.spearmanr(s.gap_rel, s.acc1))) if s.gap_rel.std() > 0 else None,
                 r_gap_abs=tuple(map(float, stats.spearmanr(s.gap_abs, s.acc1))),
                 r_sig=tuple(map(float, stats.spearmanr(np.log(s.sig_ratio), s.acc1))),
                 r_pfull=tuple(map(float, stats.spearmanr(s.p_full1, s.acc1))),
                 share_modal1=float(s.modal_is_1.mean()), plus=float(s.plus.mean()), frac1=float(s.frac1.mean()), med_t=float(s.med_t.median()))
        h4[f'eta{eta}_B{B}_r{r}'] = g
out['H4'] = h4

# ---------------------------------------------------------------- E0
e0 = json.load(open('results/e0/summary.json'))
kr = [x for x in e0 if x['exp'] == 'kramers']
ratio = [x['fit']['exp']['lam'] / x['kramers'] for x in kr if x['T'] <= 1 / 4]
by = {}
for x in kr:
    by.setdefault((round(1 / x['T']), x['start']), []).append(x['fit']['weibull']['k'])
loss = sorted([(1 / x['T'], x['fit']['exp']['lam'], x['fit']['n_events']) for x in e0 if x['exp'] == 'lossnoise'])
ad = {}
for x in [y for y in e0 if y['exp'] == 'adam']:
    ad.setdefault((x['w'], x['aligned'], round(1 / x['T'])), {})[x['opt']] = x['fit']['exp']['lam']
out['E0'] = dict(tau_over_kramers_highbarrier=[float(np.min(ratio)), float(np.median(ratio)), float(np.max(ratio))],
                 k_by_barrier={f'{b}_{s}': float(np.mean(v)) for (b, s), v in by.items()}, lossnoise=loss,
                 adam={f'w{k[0]}_aligned{k[1]}_bt{k[2]}': v['sgd'] / v['adam'] for k, v in ad.items() if len(v) == 2})
out['counts'] = dict(n_conditions=len(df), by_exp=df.exp.value_counts().to_dict(), replicas=int(df.R.sum()), events=int(df.n_events.sum()), n_saddles=len(recs))
json.dump(out, open('results/analysis/summary.json', 'w'), indent=1, default=float)
print(json.dumps({k: out[k] for k in ('counts', 'H1')}, indent=0, default=float)[:2500])
