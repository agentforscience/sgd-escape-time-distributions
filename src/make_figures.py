"""Report figures. One hue per entity, fixed order; direct labels plus legends; one axis per panel."""
import glob, json, os, sys
import numpy as np, pandas as pd
import matplotlib
import matplotlib.ticker
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, 'src')
import load_escape as le, survival as sv

os.makedirs('figures', exist_ok=True)
C = dict(blue='#2a78d6', orange='#eb6834', aqua='#1baf7a', yellow='#eda100', ink='#0b0b0b', mid='#52514e', grid='#e4e3df')
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.edgecolor': C['mid'],
                     'axes.labelcolor': C['ink'], 'xtick.color': C['mid'], 'ytick.color': C['mid'], 'axes.grid': True,
                     'grid.color': C['grid'], 'grid.linewidth': .6, 'axes.axisbelow': True, 'legend.frameon': False,
                     'figure.dpi': 150, 'savefig.bbox': 'tight', 'lines.linewidth': 2, 'lines.markersize': 7, 'figure.facecolor': '#fcfcfb',
                     'axes.facecolor': '#fcfcfb', 'savefig.facecolor': '#fcfcfb'})
df = pd.read_csv('results/analysis/conditions.csv')
S = json.load(open('results/analysis/summary.json'))
data = {c['id']: (*le.intervals(z), z) for c, z in le.load_all()}


def fig_e0():
    e0 = json.load(open('results/e0/summary.json'))
    kr = [x for x in e0 if x['exp'] == 'kramers']
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    for (w, c, mk) in [(.5, C['blue'], 'o'), (1., C['orange'], 's'), (2., C['aqua'], '^')]:
        s = sorted([x for x in kr if x['w'] == w and x['start'] == 'min'], key=lambda x: 1 / x['T'])
        ax[0].plot([1 / x['T'] for x in s], [x['fit']['exp']['lam'] / x['kramers'] for x in s], marker=mk, color=c, label=f'well half-width w={w}')
    ax[0].axhline(1, color=C['mid'], ls='--', lw=1)
    ax[0].set(xlabel='Barrier / temperature (ΔV/T)', ylabel='Measured τ / Kramers τ', ylim=(.7, 1.3), title='Simulator matches the exact Kramers rate')
    ax[0].legend()
    for st, c, mk, lab in [('min', C['blue'], 'o', 'start at the minimum'), ('qsd', C['orange'], 's', 'start from stationary law')]:
        xs = sorted({round(1 / x['T']) for x in kr})
        ys = [np.mean([x['fit']['weibull']['k'] for x in kr if round(1 / x['T']) == b and x['start'] == st]) for b in xs]
        ax[1].plot(xs, ys, marker=mk, color=c, label=lab)
    ax[1].axhline(1, color=C['mid'], ls='--', lw=1)
    ax[1].set(xlabel='Barrier / temperature (ΔV/T)', ylabel='Weibull k (mean over widths)', title='k → 1 as the barrier grows')
    ax[1].legend()
    fig.tight_layout(); fig.savefig('figures/fig1_e0_validation.png'); plt.close(fig)


def fig_survival():
    s = df[(df.exp == 'E1') & (df.land == 'A') & (df.lr == .1)].sort_values('k')
    picks = [s.iloc[0], s.iloc[len(s) // 2], s.iloc[-1]] if len(s) >= 3 else []
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    for r, c, mk in zip(picks, (C['blue'], C['orange'], C['aqua']), 'os^'):
        lo, hi, _ = data[r.id]
        ev = np.isfinite(hi)
        ts = np.sort(np.unique(hi[ev]))
        surv = np.array([(np.where(ev, hi, np.inf) > t).mean() for t in ts])
        ax[0].step(ts, surv, where='post', color=c, label=f'k={r.k:.2f}, λmax={r.lmax:.2f}')
        tt = np.logspace(np.log10(ts.min()), np.log10(ts.max()), 100)
        ax[0].plot(tt, np.exp(-(tt / r.lam) ** r.k), color=c, ls=':', lw=1.2)
    ax[0].set(xscale='log', xlabel='SGD steps', ylabel='Fraction still in the basin', title='Survival in the basin (dotted: Weibull fit)')
    ax[0].legend()
    for land, e, c, mk, lab in [('A', .1, C['blue'], 'o', 'F-MNIST, η=0.1'), ('A', .2, C['orange'], 's', 'F-MNIST, η=0.2'), ('B', .1, C['aqua'], '^', 'digits, η=0.1'), ('B', .2, C['yellow'], 'D', 'digits, η=0.2')]:
        q = df[(df.exp == 'E1') & (df.land == land) & (df.lr == e) & (df.frac_events >= .5)]
        ax[1].errorbar(q.lmax, q.k, yerr=[np.maximum(q.k - q.k_lo, 0), np.maximum(q.k_hi - q.k, 0)], fmt=mk, color=c, label=lab, ms=5, lw=1)
    ax[1].axhspan(.7, 1.3, color=C['grid'], alpha=.6, lw=0)
    ax[1].text(.98, .03, 'hypothesized range 0.7–1.3', transform=ax[1].transAxes, ha='right', color=C['mid'], fontsize=8)
    ax[1].axhline(1, color=C['mid'], ls='--', lw=1)
    ax[1].set(xscale='log', xlabel='λmax at the minimum', ylabel='Weibull k (95% CI)', title='Shape across minima')
    ax[1].xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax[1].xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax[1].set_xticks([.5, 1, 2, 5])
    ax[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig('figures/fig2_distributions.png'); plt.close(fig)


def fig_hazard():
    s = df[(df.exp == 'E1') & (df.land == 'A') & (df.lr == .1) & (df.frac_events > .8)].sort_values('k')
    picks = [s.iloc[0], s.iloc[len(s) // 2], s.iloc[-1]] if len(s) >= 3 else []
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    for r, c, mk in zip(picks, (C['blue'], C['orange'], C['aqua']), 'os^'):
        lo, hi, _ = data[r.id]
        ev = np.isfinite(hi)
        edges = np.quantile(hi[ev], np.linspace(0, 1, 9))
        edges = np.unique(edges)
        mid, hz = [], []
        for a, b in zip(edges[:-1], edges[1:]):
            risk = (np.where(ev, hi, lo) > a).sum()
            n = ((hi > a) & (hi <= b)).sum()
            if risk >= 10 and n:
                mid.append(np.sqrt(max(a, 1) * b)); hz.append(n / risk / (b - a))
        ax.plot(mid, hz, marker=mk, color=c, label=f'k={r.k:.2f}')
    ax.set(xscale='log', yscale='log', xlabel='SGD steps', ylabel='Exit hazard (per step)', title='Exit hazard is not constant: it rises, then falls')
    ax.legend()
    fig.tight_layout(); fig.savefig('figures/fig3_hazard.png'); plt.close(fig)


def fig_h2():
    p = 'results/analysis/h2_A.csv'
    if not os.path.exists(p):
        return
    t = pd.read_csv(p)
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    ax.scatter(t.lmax[~t.extrap], t.ratio[~t.extrap], color=C['blue'], s=36, label='both medians observed', zorder=3)
    ax.scatter(t.lmax[t.extrap], t.ratio[t.extrap], facecolor='none', edgecolor=C['orange'], s=36, label='a median not reached (bound)', zorder=3)
    ax.axhspan(2, 5, color=C['grid'], alpha=.6, lw=0)
    ax.axhline(1, color=C['mid'], ls='--', lw=1)
    ax.text(.98, .95, 'hypothesized 2–5×', transform=ax.transAxes, ha='right', va='top', color=C['mid'], fontsize=8)
    ax.set(xscale='log', yscale='log', xlabel='λmax at the minimum', ylabel='τ(SGD) / τ(Adam)', title='Matched-temperature Adam vs SGD')
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xticks([.5, 1, 2, 5])
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_yticks([.5, 1, 2, 5])
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig('figures/fig4_adam_sgd.png'); plt.close(fig)


def fig_h3():
    e = df[(df.opt == 'sgd') & (df.lr == .1) & df.exp.isin(['E1', 'E3']) & df.B.isin([4, 8, 16, 32, 64])]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.7), sharey=False)
    for a, land, title in ((ax[0], 'A', 'Fashion-MNIST'), (ax[1], 'B', 'digits')):
        s = e[e.land == land]
        ms = [m for m, q in s.groupby('min') if len(q) >= 4]
        for i, m in enumerate(ms):
            q = s[s['min'] == m].sort_values('B')
            a.plot(q.B, q.tau, color=C['mid'], lw=1, alpha=.55, zorder=1)
            a.scatter(q.B[~q.tau_extrap], q.tau[~q.tau_extrap], color=C['blue'], s=18, zorder=2, label='observed median' if i == 0 else None)
            a.scatter(q.B[q.tau_extrap], q.tau[q.tau_extrap], facecolor='none', edgecolor=C['orange'], s=22, zorder=2, label='median not reached (lower bound)' if i == 0 else None)
        g = s[~s.tau_extrap]
        if len(g):
            b0 = 4
            y0 = np.exp(np.log(g[g.B == g.B.min()].tau).mean())
            bb = np.array([4, 64.])
            a.plot(bb, y0 * (bb / g.B.min()) ** .5, color=C['aqua'], ls='--', label='τ ∝ √B (hypothesis)')
            a.plot(bb, y0 * (bb / g.B.min()), color=C['yellow'], ls=':', label='τ ∝ B')
        a.axhline(150000, color=C['mid'], ls=':', lw=1)
        a.set(xscale='log', yscale='log', xlabel='Batch size B', ylabel='Median escape time (steps)', title=title + ' (η=0.1)')
        a.set_xticks([4, 8, 16, 32, 64], ['4', '8', '16', '32', '64'])
        a.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        a.set_ylim(top=2e6)
        a.legend(fontsize=7, loc='upper left', ncol=2)
    fig.tight_layout(); fig.savefig('figures/fig5_batch_scaling.png'); plt.close(fig)


def fig_h4():
    p = 'results/analysis/h4.csv'
    if not os.path.exists(p):
        return
    h = pd.read_csv(p)
    h = h[h.r <= 1.0]
    s = h[(h.eta == .02) & (h.B == 16)]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.7))
    rs = sorted(s.r.unique())
    for (eta, B), c, mk in zip([(.1, 16), (.02, 16), (.1, 64)], (C['blue'], C['orange'], C['aqua']), 'os^'):
        q = h[(h.eta == eta) & (h.B == B)].groupby('r').acc1.agg(['mean', 'sem'])
        ax[0].errorbar(q.index, q['mean'], yerr=1.96 * q['sem'], marker=mk, color=c, label=f'η={eta}, B={B}', lw=1.5)
    ax[0].axhspan(.7, .85, color=C['grid'], alpha=.6, lw=0)
    ax[0].axhline(h.chance.mean(), color=C['mid'], ls='--', lw=1)
    ax[0].text(.1, h.chance.mean() + .01, 'chance', color=C['mid'], fontsize=8)
    ax[0].set(xscale='log', xlabel='Exit radius r', ylabel='Share exiting along e₁', ylim=(0, 1), title='Most-negative direction wins how often?')
    ax[0].legend(fontsize=8)
    s1 = h[(h.r == 1.0) & (h.eta == .1) & (h.B == 16)]
    ax[1].scatter(s1.gap_rel, s1.acc1, color=C['blue'], s=30, label='measured per saddle')
    ax[1].scatter(s1.gap_rel, s1.p_full1, marker='x', color=C['orange'], s=30, label='linear model')
    ax[1].set(xlabel='Relative gap (|λ₁|−|λ₂|)/|λ₁|', ylabel='Share exiting along e₁', ylim=(0, 1.02), title='Gap vs exit direction (r=1, η=0.1, B=16)')
    ax[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig('figures/fig6_saddles.png'); plt.close(fig)


for f in (fig_e0, fig_survival, fig_hazard, fig_h2, fig_h3, fig_h4):
    f()
print('figures done')
