"""Run escape-time conditions from a job file.

Each job is one optimizer/LR/batch setting at one polished minimum, with R independent replicas.
At geometrically spaced checks every live replica is quenched to see which minimum it falls into.
Basin exit is decided by a converged quench. Loss and distance trajectories are saved, so loss-level and\nradius first-passage definitions can be applied post hoc.
"""
import fcntl, json, os, sys, time
import numpy as np, torch
sys.path.insert(0, 'src')
import nnsim as ns

GATE = 0.3     # distance after the gate quench above which the long quench is run
R_OUT = 0.05   # distance after the long quench above which the replica may have left the basin
BAR_MIN, DL_MIN = 1e-5, 2e-5   # an exit also needs a segment barrier or a change of minimum loss
LANDSCAPES = {'A': ('fmnist', 8, 1e-4), 'B': ('digits', 8, 1e-3), 'S': ('fmnist', 3, 1e-4)}


def check_grid(t_max, ratio=1.1, gap=100):
    g, t = [], 0.0
    while t < t_max:
        t = max(t + gap, t * ratio)
        g.append(int(min(t, t_max)))
    return g


def load_landscape(key, dev, cache={}):
    k = (key, dev)
    if k not in cache:
        data, h, wd = LANDSCAPES[key]
        X, y = ns.load_data(data)
        cache[k] = ns.Net(X, y, h, wd, dev)
    return cache[k]


def run_job(j, dev):
    net = load_landscape(j['land'], dev)
    z = np.load(f"results/minima/{j['min']}.npz")
    th_star = torch.as_tensor(z['th'], dtype=torch.float32, device=dev)
    lmax = float(z['ev'][-1])
    torch.manual_seed(j['seed'])
    R, B = j['R'], j['B']
    x = th_star.expand(R, -1).clone()
    opt = ns.make_opt(j['opt'], x, j['lr'], net=net, warm_batch=B if j.get('warm') else None)
    grid = check_grid(j['t_max'])
    C = len(grid)
    nanarr = lambda: np.full((R, C), np.nan, dtype=np.float32)
    loss, dist, gate = nanarr(), nanarr(), nanarr()
    t_exit = np.full(R, np.nan)          # first check at which the replica was out of the basin
    d_exit, l_exit, b_exit = np.full(R, np.nan), np.full(R, np.nan), np.full(R, np.nan)
    dest = np.full((R, net.P), np.nan, dtype=np.float32)
    alive = torch.arange(R, device=dev)
    Lstar = net.flat_loss_single(th_star).item()
    # Gate from a pilot on two minima: after q_gate quench steps, in-basin replicas sit <0.04 from th* and escaped
    # ones >2.5. Replicas past GATE get the long quench, whose converged end point decides.
    qm = j.get('qmult', 1)
    q_lr, q_gate, q_long = 1.0 / lmax, int(qm * 1500 * max(1, lmax)), int(qm * 5000 * max(1, lmax))
    t0, step = time.time(), 0
    for c, target in enumerate(grid):
        while step < target:
            step += 1
            idx = torch.randint(0, net.N, (len(alive), B), device=dev)
            x = opt(x, net.grad(x, idx))
        if not torch.isfinite(x).all():
            raise FloatingPointError('non-finite parameters')
        ids = alive.cpu().numpy()
        with torch.no_grad():
            loss[ids, c] = (net.full_loss(x) - Lstar).cpu().numpy()
            dist[ids, c] = (x - th_star).norm(dim=1).cpu().numpy()
        Q = ns.quench(net, x, q_lr, q_gate)
        dq = (Q - th_star).norm(dim=1)
        gate[ids, c] = dq.cpu().numpy()
        cand = dq > GATE
        keep = torch.ones(len(alive), dtype=torch.bool, device=dev)
        if cand.any():
            Ql = ns.quench(net, Q[cand], q_lr, q_long - q_gate)
            dl = (Ql - th_star).norm(dim=1)
            # Slow soft-mode convergence can leave an in-basin point past R_OUT. Newton polishing showed such
            # points have no barrier AND no loss change; a real neighbour always shows at least one of them.
            out = dl > R_OUT
            if out.any():
                bq = torch.zeros_like(dl)
                lq = torch.zeros_like(dl)
                bq[out] = ns.segment_barrier(net, Ql[out], th_star)
                lq[out] = net.full_loss(Ql[out]) - Lstar
                out = out & ((bq > BAR_MIN) | (lq.abs() > DL_MIN))
            if out.any():
                cid = ids[cand.cpu().numpy()]
                o = out.cpu().numpy()
                t_exit[cid[o]] = c
                d_exit[cid[o]] = dl[out].cpu().numpy()
                l_exit[cid[o]] = lq[out].cpu().numpy()
                b_exit[cid[o]] = bq[out].cpu().numpy()
                dest[cid[o]] = Ql[out].cpu().numpy()
                keep[torch.nonzero(cand).squeeze(1)[out]] = False
        if not keep.all():
            x, alive = x[keep], alive[keep]
            opt.subset(keep)
        if len(alive) == 0:
            break
    return dict(grid=np.array(grid), loss=loss, dist=dist, gate=gate, t_exit=t_exit, d_exit=d_exit, l_exit=l_exit,
                b_exit=b_exit, dest=dest, Lstar=Lstar, lmax=lmax, wall=time.time() - t0, last_step=step,
                n_alive=len(alive))


def main():
    jobfile, dev = sys.argv[1], sys.argv[2]
    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', 2)))
    os.makedirs('results/escape', exist_ok=True)
    locks = {}
    while True:
        # Re-read after every job so the queue can be extended or reordered while workers run.
        j = None
        for c in json.load(open(jobfile)):
            out = f"results/escape/{c['id']}.npz"
            if os.path.exists(out) or c['id'] in locks:
                continue
            lock = open(f'{out}.lock', 'w')
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                lock.close()
                continue
            if os.path.exists(out):
                lock.close()
                continue
            j, locks[c['id']] = c, lock
            break
        if j is None:
            return
        while True:  # on a shared GPU, wait out OOMs on this job instead of skipping to the next
            try:
                r = run_job(j, dev)
                break
            except RuntimeError as e:
                if 'out of memory' not in str(e).lower():
                    print('FAIL', j['id'], repr(e)[:300], flush=True)
                    r = None
                    break
                print('OOM wait', j['id'], flush=True)
                torch.cuda.empty_cache()
                time.sleep(60)
        if r is None:
            continue
        tmp = f'{out}.tmp.npz'
        np.savez_compressed(tmp, cfg=json.dumps(j), **r)
        os.replace(tmp, out)
        print(f"done {j['id']} alive={r['n_alive']}/{j['R']} wall={r['wall']:.0f}s", flush=True)


if __name__ == '__main__':
    main()
