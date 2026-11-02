"""Find generic saddles of index >= 2 by index-preserving, trust-region Newton (mode following).

Start: a descent trajectory stopped at a random time, kept only if it has 2-8 negative modes. Then
d = -V s |lambda|^-1 V' g, ascending along the k lowest modes and descending along the rest (adaptive trust
region, accepted only if the gradient norm falls). Converges to a critical point of index k. Symmetric saddles (unit-zero, unit-duplicate) are not used: per-sample gradients
vanish along their escape directions, so SGD noise could never select a direction there.
"""
import json, os, sys, time
import numpy as np, torch
sys.path.insert(0, 'src')
import nnsim as ns, run_escape as re_

land, wid, n_try = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
torch.set_num_threads(2)
os.makedirs('results/saddles', exist_ok=True)
data, h, wd = re_.LANDSCAPES[land]
X, y = ns.load_data(data)
net = ns.Net(X, y, h, wd, 'cpu', torch.float64)
rng = np.random.default_rng(2000 + wid)
for t in range(n_try):
    torch.manual_seed(int(rng.integers(1 << 30)))
    th = torch.randn(net.P, dtype=torch.float64) * rng.choice([0.2, 0.5, 1.0])
    # Descent lingers near low-index saddles; stop at a random time and keep only low-index starts.
    for _ in range(int(rng.choice([300, 1000, 3000, 8000]))):
        th = th - 0.5 * net.grad(th[None])[0]
    e = torch.linalg.eigvalsh(net.hessian(th))
    k0 = int((e < 0).sum())
    if k0 < 2 or k0 > 8:
        print(f'w{wid} t{t} skip k0={k0}', flush=True)
        continue
    t0, ok, k, trust = time.time(), False, k0, 0.2
    g = net.grad(th[None])[0]
    for it in range(250):
        gn = g.norm().item()
        if gn < 1e-9:
            ok = True
            break
        e, V = torch.linalg.eigh(net.hessian(th))
        s = torch.ones_like(e)
        s[:k] = -1.0
        d0 = -(V @ (s * (V.T @ g) / e.abs().clamp(min=1e-7)))
        while trust > 1e-7:   # accept only steps that reduce the gradient norm
            d = d0 * min(1.0, trust / d0.norm().item())
            g2 = net.grad((th + d)[None])[0]
            if g2.norm() < g.norm():
                th, g = th + d, g2
                trust = min(trust * 2, 1.0)
                break
            trust *= 0.4
        if trust <= 1e-7:
            break
    if not ok:
        print(f'w{wid} t{t} no-conv k0={k0} gn={gn:.1e} it={it} {time.time()-t0:.0f}s', flush=True)
        continue
    ev = torch.linalg.eigvalsh(net.hessian(th)).numpy()
    idx = int((ev < -1e-7).sum())
    L = net.flat_loss_single(th).item()
    print(f'w{wid} t{t} L={L:.4f} index={idx} ev={ev[:4].round(4)} gn={gn:.1e} it={it} {time.time()-t0:.0f}s', flush=True)
    if idx >= 2:
        np.savez(f'results/saddles/{land}_w{wid}_t{t}.npz', th=th.numpy(), ev=ev, L=L, index=idx, gn=gn)
