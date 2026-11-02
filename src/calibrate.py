"""Find the Adam LR that matches SGD's in-basin thermal level (mean excess full loss).

Measured over steps 500-2000, well before any escape, per minimum, so temperature is matched in the
basin itself rather than assumed from theory. Also records the default Adam lr=1e-3 for comparison.
"""
import json, sys, numpy as np, torch
sys.path.insert(0, 'src')
import nnsim as ns, run_escape as re_
dev = sys.argv[1]
sel = json.load(open('results/selected_minima.json'))
SGD_LR, B, R = 0.1, 16, 64
adam_lrs = [1e-3, 2e-3, 4e-3, 8e-3, 16e-3]
out = []
for land in 'AB':
    net = re_.load_landscape(land, dev)
    for m in sel[land]:
        th = torch.as_tensor(np.load(f"results/minima/{m['name']}.npz")['th'], dtype=torch.float32, device=dev)
        Ls = net.flat_loss_single(th).item()
        rec = dict(min=m['name'], land=land, lmax=m['lmax'])
        for kind, lrs in (('sgd', [SGD_LR]), ('adam', adam_lrs)):
            for lr in lrs:
                torch.manual_seed(7)
                x = th.expand(R, -1).clone()
                op = ns.make_opt(kind, x, lr)
                vals, d = [], []
                for s in range(1, 2001):
                    x = op(x, net.grad(x, torch.randint(0, net.N, (R, B), device=dev)))
                    if s >= 500 and s % 100 == 0:
                        vals.append((net.full_loss(x).mean() - Ls).item())
                        d.append((x - th).norm(dim=1).mean().item())
                rec[f'{kind}_{lr:g}'] = (float(np.mean(vals)), float(np.mean(d)))
        print(rec, flush=True)
        out.append(rec)
        json.dump(out, open('results/calibration_eta0.1.json', 'w'), indent=1)
