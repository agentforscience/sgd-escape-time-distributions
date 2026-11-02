"""E4: which negative eigendirection does SGD leave a saddle along?

For one saddle and (eta, B): replicas start exactly at the saddle. The displacement is projected on the
negative eigenvectors each step. At the first crossing of radius r in that subspace the winner is the
eigenvector with the largest |projection|. The saved predictions are a linearized Monte-Carlo model:
  c_{t+1} = (I + eta*Lam) c_t + eta*eps,  eps ~ N(0, Sigma/B),  Sigma measured from per-sample gradients,
in three variants: full, 'gap-only' (isotropic noise, true eigenvalues), 'noise-only' (true noise, equal growth).
"""
import fcntl, glob, json, os, sys, time
import numpy as np, torch
sys.path.insert(0, 'src')
import nnsim as ns, run_escape as re_

RADII = [0.1, 0.3, 1.0]
CONDS = [(0.1, 16), (0.02, 16), (0.1, 64)]
R, T_MAX = 128, 100000


def linear_mc(lam, Sig, eta, B, r, n=3000, seed=0, t_max=400000):
    """Winner probabilities of the linear model at radius r."""
    rng = np.random.default_rng(seed)
    K = len(lam)
    Lc = np.linalg.cholesky(Sig / B + 1e-16 * np.eye(K))
    c = np.zeros((n, K))
    g = 1 + eta * lam
    wins = np.zeros(K)
    left = n
    for t in range(t_max):
        c = c * g + eta * rng.normal(size=c.shape) @ Lc.T
        out = (c * c).sum(1) >= r * r
        if out.any():
            wins += np.bincount(np.abs(c[out]).argmax(1), minlength=K)
            c = c[~out]
            if len(c) == 0:
                break
    return wins / max(wins.sum(), 1)


def run_saddle(path, torch_threads=2):
    torch.set_num_threads(torch_threads)
    land = os.path.basename(path).split('_')[0]
    data, h, wd = re_.LANDSCAPES[land]
    X, y = ns.load_data(data)
    net64 = ns.Net(X, y, h, wd, 'cpu', torch.float64)
    z = np.load(path)
    th = torch.as_tensor(z['th'], dtype=torch.float64)
    e, V = torch.linalg.eigh(net64.hessian(th))
    K = int((e < -1e-7).sum())
    Vn = V[:, :K]                         # ordered most-negative first
    lam = (-e[:K]).numpy()
    # Per-sample gradients, projected on the negative subspace.
    idx = torch.arange(net64.N)[:, None]
    G = net64.grad(th[None].expand(net64.N, -1), idx) @ Vn
    Sig = np.cov(G.numpy().T).reshape(K, K)
    rec = dict(saddle=os.path.basename(path)[:-4], land=land, L=float(z['L']), index=K, lam=lam.tolist(),
               sig=np.sqrt(np.diag(Sig)).tolist(), sig_full_norm=float(G.norm()), conds=[])
    net = ns.Net(X, y, h, wd, 'cpu')
    thf, Vf = th.float(), Vn.float()
    for ci, (eta, B) in enumerate(CONDS):
        torch.manual_seed(int(abs(hash_int(path)) % 100000) + ci)
        x = thf.expand(R, -1).clone()
        alive = torch.arange(R)
        winners = {r: -np.ones(R, int) for r in RADII}
        times = {r: np.full(R, np.nan) for r in RADII}
        frac1 = {r: np.full(R, np.nan) for r in RADII}
        sign1 = {r: np.zeros(R) for r in RADII}
        rstate = torch.zeros((R, len(RADII)), dtype=torch.bool)
        step = 0
        while step < T_MAX and len(alive):
            step += 1
            x = x - eta * net.grad(x, torch.randint(0, net.N, (len(alive), B)))
            c = (x - thf) @ Vf
            n2 = (c * c).sum(1)
            for ri, r in enumerate(RADII):
                hit = (n2 >= r * r) & ~rstate[alive, ri]
                if hit.any():
                    ids = alive[hit]
                    rstate[ids, ri] = True
                    ch = c[hit].numpy()
                    winners[r][ids.numpy()] = np.abs(ch).argmax(1)
                    times[r][ids.numpy()] = step
                    frac1[r][ids.numpy()] = ch[:, 0] ** 2 / (ch ** 2).sum(1)
                    sign1[r][ids.numpy()] = np.sign(ch[:, 0])
            if step % 200 == 0:
                keep = ~rstate[alive, -1]
                if not keep.all():
                    x, alive = x[keep], alive[keep]
        co = dict(eta=eta, B=B, last_step=step, r={})
        for r in RADII:
            w = winners[r]
            ok = w >= 0
            pf = linear_mc(lam, Sig, eta, B, r, seed=ci)
            pg = linear_mc(lam, np.eye(K) * np.trace(Sig) / K, eta, B, r, seed=ci + 10)
            pn = linear_mc(np.full(K, lam.mean()), Sig, eta, B, r, seed=ci + 20)
            co['r'][str(r)] = dict(n=int(ok.sum()), counts=np.bincount(w[ok], minlength=K).tolist(),
                                   med_t=float(np.nanmedian(times[r])) if ok.any() else None,
                                   frac1=float(np.nanmean(frac1[r])) if ok.any() else None,
                                   plus_share=float((sign1[r][ok] > 0).mean()) if ok.any() else None,
                                   p_full=pf.tolist(), p_gap=pg.tolist(), p_noise=pn.tolist())
        rec['conds'].append(co)
        json.dump(rec, open(f"results/e4/{rec['saddle']}.json.partial", 'w'))
    os.replace(f"results/e4/{rec['saddle']}.json.partial", f"results/e4/{rec['saddle']}.json")
    return rec


def hash_int(s):
    import hashlib
    return int(hashlib.md5(s.encode()).hexdigest()[:8], 16)


def dedupe(paths):
    """Same loss and spectrum means the same saddle up to a symmetry of the network: keep the first."""
    seen, out = set(), []
    for p in sorted(paths):
        z = np.load(p)
        key = (round(float(z['L']), 7), round(float(z['ev'][0]), 7))
        if key not in seen:
            seen.add(key)
            out.append(p)
    return out


if __name__ == '__main__':
    os.makedirs('results/e4', exist_ok=True)
    max_n = int(sys.argv[1])
    for p in dedupe(glob.glob('results/saddles/S_*.npz'))[:max_n]:
        name = os.path.basename(p)[:-4]
        out = f'results/e4/{name}.json'
        if os.path.exists(out):
            continue
        lk = open(f'{out}.lock', 'w')
        try:
            fcntl.flock(lk, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            continue
        t0 = time.time()
        r = run_saddle(p)
        c = r['conds'][0]['r']['1.0']
        print(name, 'index', r['index'], 'lam', np.round(r['lam'][:3], 5), 'r=1 counts', c['counts'], 'p_full', np.round(c['p_full'], 2),
              f'{time.time()-t0:.0f}s', flush=True)
