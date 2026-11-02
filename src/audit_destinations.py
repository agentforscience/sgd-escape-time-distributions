"""Audit of the primary escape call: is each reported destination really a different strict minimum?

Each sampled destination is Newton-polished in double precision from the float32 quench end point, then
checked for zero gradient, a positive-definite Hessian, and a distance from th*. Failure of any of these
would mean an in-basin replica had been counted as an exit.
"""
import json, sys
import numpy as np, torch
sys.path.insert(0, 'src')
import nnsim as ns, run_escape as re_, load_escape as le

torch.set_num_threads(2)
n_per = int(sys.argv[1])
rng = np.random.default_rng(0)
nets, rows = {}, []
for cfg, z in le.load_all('results/escape/E1__*.npz'):
    land = cfg['land']
    if land not in nets:
        d, h, wd = re_.LANDSCAPES[land]
        X, y = ns.load_data(d)
        nets[land] = ns.Net(X, y, h, wd, 'cpu', torch.float64)
    net = nets[land]
    ths = torch.as_tensor(np.load(f"results/minima/{cfg['min']}.npz")['th'], dtype=torch.float64)
    Ls = net.flat_loss_single(ths).item()
    ids = np.flatnonzero(np.isfinite(z['t_exit']))
    for i in rng.choice(ids, min(n_per, len(ids)), replace=False):
        th0 = torch.as_tensor(z['dest'][i], dtype=torch.float64)
        th, H, gn = ns.polish_min(net, th0, maxit=40)
        ev = torch.linalg.eigvalsh(H)
        d = (th - ths).norm().item()
        rows.append(dict(job=cfg['id'], rep=int(i), d_quench=float(z['d_exit'][i]), d_polish=d, moved=(th - th0).norm().item(),
                         gn=gn, lmin=ev[0].item(), dL=net.flat_loss_single(th).item() - Ls))
    json.dump(rows, open('results/destination_audit.json', 'w'), indent=0)
r = rows
print('n', len(r))
print('strict minimum (gn<1e-7, lmin>0):', np.mean([x['gn'] < 1e-7 and x['lmin'] > 0 for x in r]))
print('distinct from th* (d>0.05):', np.mean([x['d_polish'] > 0.05 for x in r]))
print('fell back to th* (d<1e-3):', np.mean([x['d_polish'] < 1e-3 for x in r]))
