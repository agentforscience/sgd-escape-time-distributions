"""Find and polish strict minima with a wide spread of sharpness.

Sharpness is varied by the search protocol (init scale and optimizer), not by changing the loss:
every minimum of a dataset lives in the same landscape (same h, wd).
Saves th*, the Hessian spectrum and a geometry summary to results/minima/.
"""
import json, os, sys, time
import numpy as np, torch
sys.path.insert(0, 'src')
import nnsim as ns

data, h, wd, n_seeds = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4])
os.makedirs('results/minima', exist_ok=True)
dev = ns.pick_gpu()
X, y = ns.load_data(data)
net = ns.Net(X, y, h, wd, dev)
net64 = ns.Net(X, y, h, wd, dev, torch.float64)
rows = []
# (init scale, optimizer, lr, steps): large inits with plain descent end in sharper minima.
protocols = [(0.1, 'adam', 0.01, 3000), (0.3, 'adam', 0.01, 3000), (0.3, 'sgd', 0.3, 4000), (1.0, 'adam', 0.003, 3000),
             (2.0, 'adam', 0.003, 3000), (2.0, 'sgd', 0.1, 4000), (4.0, 'adam', 0.003, 3000), (4.0, 'sgd', 0.03, 4000)]
for seed in range(n_seeds):
    for pi, (sc, opt, lr, steps) in enumerate(protocols):
        torch.manual_seed(1000 * seed + pi)
        th = torch.randn(1, net.P, device=dev) * sc / np.sqrt(h)
        st = ns.make_opt(opt, th, lr)
        for _ in range(steps):
            th = st(th, net.grad(th))
        ths, H, gn = ns.polish_min(net64, th[0].double(), maxit=100)
        ev = torch.linalg.eigvalsh(H).cpu().numpy()
        L = net64.flat_loss_single(ths).item()
        acc = (net64.logits(ths[None], net64.X).argmax(-1) == net64.y).float().mean().item()
        if gn > 1e-7 or ev[0] <= 0 or not np.isfinite(L):
            print('reject', seed, pi, gn, ev[0], flush=True); continue
        name = f'{data}_wd{wd:g}_s{seed}_p{pi}'
        np.savez(f'results/minima/{name}.npz', th=ths.cpu().numpy(), ev=ev)
        r = dict(name=name, data=data, h=h, wd=wd, seed=seed, proto=pi, init=sc, opt=opt, L=L, acc=acc, gn=gn,
                 lmin=float(ev[0]), lmax=float(ev[-1]), trace=float(ev.sum()), logdet=float(np.log(ev).sum()),
                 norm=float(ths.norm()))
        rows.append(r)
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items() if k in ('name', 'L', 'acc', 'lmax', 'trace', 'norm')}, flush=True)
        json.dump(rows, open(f'results/minima/{data}_wd{wd:g}.json', 'w'), indent=1)
