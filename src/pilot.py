"""Pilot: are NN minima strict, and does SGD leave them at practical noise levels?"""
import sys, time, torch, numpy as np
sys.path.insert(0, 'src')
import nnsim as ns
torch.manual_seed(0); np.random.seed(0)
dev = ns.pick_gpu(); print('device', dev)
data, h, wd = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])
X, y = ns.load_data(data)
net = ns.Net(X, y, h, wd, dev)
net64 = ns.Net(X, y, h, wd, dev, torch.float64)
print(data, X.shape, 'P', net.P)
for seed in range(3):
    torch.manual_seed(seed)
    th = torch.randn(1, net.P, device=dev) * 0.3
    st = ns.make_opt('adam', th, 0.01)
    t0 = time.time()
    for i in range(4000):
        th = st(th, net.grad(th))
    th_s, H, gn = ns.polish_min(net64, th[0].double())
    ev = torch.linalg.eigvalsh(H)
    acc = (net64.logits(th_s[None], net64.X).argmax(-1) == net64.y).float().mean().item()
    print(f'seed{seed} L={net64.flat_loss_single(th_s):.5f} acc={acc:.3f} gn={gn:.1e} lmin={ev[0]:.2e} lmax={ev[-1]:.3f} tr={ev.sum():.2f} t={time.time()-t0:.1f}s', flush=True)
th_star = th_s.float()
Lstar = net.flat_loss_single(th_star).item(); lmax = ev[-1].item()
R = 128
for lr, B in [(0.05, 16), (0.2, 16), (0.5, 16), (1.0, 16)]:
    if lr * lmax > 1.9: print('skip unstable', lr); continue
    x = th_star.expand(R, -1).clone()
    t0 = time.time()
    for step in range(1, 60001):
        idx = torch.randint(0, net.N, (R, B), device=dev)
        x = x - lr * net.grad(x, idx)
        if step in (100, 1000, 5000, 20000, 60000):
            q = ns.quench(net, x, 1.0 / lmax * 0.5, 200)
            bar = ns.segment_barrier(net, q, th_star)
            print(f'lr={lr} B={B} step={step} dL={(net.full_loss(x).mean()-Lstar):.4f} dist={(x-th_star).norm(dim=1).mean():.3f} '
                  f'qL-L*={(net.full_loss(q)-Lstar).median():.5f} qdist={(q-th_star).norm(dim=1).median():.3f} barrier>1e-3:{(bar>1e-3).float().mean():.2f} t={time.time()-t0:.0f}s', flush=True)
