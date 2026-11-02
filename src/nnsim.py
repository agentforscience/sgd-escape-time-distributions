"""Batched replica simulator for one-hidden-layer tanh MLPs.

R replicas are stored as one tensor theta[R, P]. All replicas share the training set, each draws
its own minibatches, and autograd on the sum of replica losses gives each replica its own gradient.
"""
import gzip, os, struct
import numpy as np
import torch
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA


def load_data(name):
    """Return (X, y) as float64/int64 numpy arrays, standardized."""
    if name == 'digits':
        d = load_digits()
        X, y = d.data.astype(np.float64) / 16.0, d.target.astype(np.int64)
    elif name == 'fmnist':
        root = 'datasets/raw_fmnist'
        with gzip.open(f'{root}/train-images-idx3-ubyte.gz', 'rb') as f:
            _, n, r, c = struct.unpack('>IIII', f.read(16))
            X = np.frombuffer(f.read(), dtype=np.uint8).reshape(n, r * c).astype(np.float64) / 255.0
        with gzip.open(f'{root}/train-labels-idx1-ubyte.gz', 'rb') as f:
            f.read(8)
            y = np.frombuffer(f.read(), dtype=np.uint8).astype(np.int64)
        idx = np.random.default_rng(0).permutation(len(y))[:2000]
        X, y = X[idx], y[idx]
        X = PCA(n_components=24, random_state=0).fit_transform(X)
    else:
        raise ValueError(name)
    X = (X - X.mean(0)) / (X.std(0) + 1e-8)
    return X, y


class Net:
    """Tanh MLP d -> h -> c. Weight decay wd is part of the loss, so minima are strict."""

    def __init__(self, X, y, h, wd, device, dtype=torch.float32):
        self.X = torch.as_tensor(X, dtype=dtype, device=device)
        self.y = torch.as_tensor(y, device=device)
        self.N, self.d = self.X.shape
        self.c = int(self.y.max()) + 1
        self.h, self.wd, self.device, self.dtype = h, wd, device, dtype
        d, c = self.d, self.c
        self.sizes = [d * h, h, h * c, c]
        self.P = sum(self.sizes)

    def _split(self, th):
        s = th.shape[:-1]
        a, b, e, f = torch.split(th, self.sizes, dim=-1)
        return (a.reshape(*s, self.d, self.h), b.unsqueeze(-2), e.reshape(*s, self.h, self.c), f.unsqueeze(-2))

    def logits(self, th, X):
        """th[R,P], X[N,d] (shared) or [R,B,d] (per replica)."""
        W1, b1, W2, b2 = self._split(th)
        return torch.tanh(X @ W1 + b1) @ W2 + b2

    def data_loss(self, th, X, y):
        """Per-replica mean cross entropy."""
        lg = self.logits(th, X)
        if y.dim() == 1:
            y = y.expand(th.shape[0], -1)
        lp = lg - lg.logsumexp(-1, keepdim=True)
        return -lp.gather(-1, y.unsqueeze(-1)).squeeze(-1).mean(-1)

    def loss(self, th, X=None, y=None):
        X = self.X if X is None else X
        y = self.y if y is None else y
        return self.data_loss(th, X, y) + 0.5 * self.wd * (th * th).sum(-1)

    def grad(self, th, idx=None):
        """Per-replica gradient; idx[R,B] selects minibatches, None means full batch.

        Hand-derived backprop (no autograd graph): about 3x faster on CPU. grad_autograd checks it.
        """
        X, y = (self.X, self.y) if idx is None else (self.X[idx], self.y[idx])
        W1, b1, W2, b2 = self._split(th)
        s = torch.tanh(X @ W1 + b1)
        r = torch.softmax(s @ W2 + b2, dim=-1)
        if y.dim() == 1:
            r[:, torch.arange(y.shape[0], device=y.device), y] -= 1.0
        else:
            r.scatter_add_(-1, y.unsqueeze(-1), -torch.ones_like(y, dtype=r.dtype).unsqueeze(-1))
        n = X.shape[-2]
        gW2 = (s.transpose(-1, -2) @ r) / n
        gb2 = r.mean(-2)
        u = (r @ W2.transpose(-1, -2)) * (1 - s * s)
        gW1 = (X.transpose(-1, -2) @ u) / n
        gb1 = u.mean(-2)
        R = th.shape[0]
        return torch.cat([gW1.reshape(R, -1), gb1, gW2.reshape(R, -1), gb2], dim=-1) + self.wd * th

    def grad_autograd(self, th, idx=None):
        """Reference gradient for testing, and for double-precision Hessian code paths."""
        th = th.detach().requires_grad_(True)
        L = self.loss(th) if idx is None else self.loss(th, self.X[idx], self.y[idx])
        g, = torch.autograd.grad(L.sum(), th)
        return g.detach()

    def full_loss(self, th, chunk=2048):
        return torch.cat([self.loss(t) for t in th.split(chunk)])

    def flat_loss_single(self, th):
        return self.loss(th.unsqueeze(0)).squeeze(0)

    def hessian(self, th):
        """Exact Hessian at one point."""
        return torch.func.hessian(self.flat_loss_single)(th)


def to64(net, X, y, device):
    return Net(X, y, net.h, net.wd, device, torch.float64)


def polish_min(net64, th, maxit=60, tol=1e-9):
    """Damped Newton to a strict minimum with eigenvalue-shift regularization. Returns (th, H, gnorm)."""
    th = th.clone().to(net64.device, net64.dtype)
    for it in range(maxit):
        g = net64.grad(th.unsqueeze(0)).squeeze(0)
        gn = g.norm().item()
        H = net64.hessian(th)
        ev = torch.linalg.eigvalsh(H)
        if gn < tol and ev[0] > 0:
            return th, H, gn
        mu = max(0.0, -ev[0].item() * 1.5) + 1e-9
        step = torch.linalg.solve(H + mu * torch.eye(net64.P, device=H.device, dtype=H.dtype), g)
        f0 = net64.flat_loss_single(th).item()
        a = 1.0
        for _ in range(30):
            if net64.flat_loss_single(th - a * step).item() <= f0 + 1e-12:
                break
            a *= 0.5
        th = th - a * step
    return th, net64.hessian(th), gn


def adam_step(th, g, st, lr, b1=0.9, b2=0.999, eps=1e-8):
    st['t'] += 1
    st['m'].mul_(b1).add_(g, alpha=1 - b1)
    st['v'].mul_(b2).addcmul_(g, g, value=1 - b2)
    mh = st['m'] / (1 - b1 ** st['t'])
    vh = st['v'] / (1 - b2 ** st['t'])
    return th - lr * mh / (vh.sqrt() + eps)


def make_opt(kind, th, lr, net=None, warm_batch=None, warm_n=200):
    """Stateful optimizer closure on replicas, supports subsetting after replicas exit."""
    st = dict(t=0, m=torch.zeros_like(th), v=torch.zeros_like(th))
    if kind in ('adam', 'rmsprop') and warm_batch:
        # Stationary start: pre-fill the second moment from gradients evaluated at the start point.
        v = torch.zeros_like(th[:1])
        for _ in range(warm_n):
            idx = torch.randint(0, net.N, (64, warm_batch), device=th.device)
            v = v + (net.grad(th[:1].expand(64, -1), idx) ** 2).mean(0, keepdim=True)
        st['v'][:] = v / warm_n
        st['t'] = 10 ** 6

    def step(x, g):
        if kind == 'sgd':
            return x - lr * g
        if kind == 'sgdm':
            st['m'].mul_(0.9).add_(g)
            return x - lr * st['m']
        if kind == 'adam':
            return adam_step(x, g, st, lr)
        if kind == 'rmsprop':
            st['t'] += 1
            st['v'].mul_(0.999).addcmul_(g, g, value=0.001)
            return x - lr * g / ((st['v'] / (1 - 0.999 ** st['t'])).sqrt() + 1e-8)
        raise ValueError(kind)

    def subset(keep):
        st['m'], st['v'] = st['m'][keep], st['v'][keep]

    step.subset = subset
    return step


def quench(net, th, lr, steps, beta=0.9):
    """Full-batch heavy-ball descent; stiff directions relax so the point falls into its own basin."""
    x = th.clone()
    m = torch.zeros_like(x)
    for _ in range(steps):
        g = net.grad(x)
        m = beta * m + g
        x = x - lr * m
    return x


def hinv_cached(name, net64):
    """H*^-1 at a polished minimum; cached since Hessians are slow on a loaded CPU."""
    path = f'results/minima/{name}.hinv.npy'
    if os.path.exists(path):
        return np.load(path)
    th = torch.as_tensor(np.load(f'results/minima/{name}.npz')['th'], device=net64.device, dtype=net64.dtype)
    Hi = torch.linalg.inv(net64.hessian(th)).cpu().numpy().astype(np.float32)
    tmp = f'{path}.{os.getpid()}.npy'
    np.save(tmp, Hi)
    os.replace(tmp, path)
    return Hi


def classify(net, x, th_star, Hinv, lmax, q0=150, chord=25, q_long=5000, r_in=0.02, r_out=0.05):
    """Converged-quench basin test. Returns (Q, escaped) with Q the quenched points.

    Stage 1 certifies 'still in the basin of th*' cheaply: a short quench, then chord-Newton with the
    fixed H*^-1 converges to th* only from inside its neighbourhood. Stage 2 runs the full long quench
    on the replicas stage 1 could not certify, so it can only turn a false alarm back into 'in basin'.
    """
    Q0 = quench(net, x, 1.0 / lmax, q0)
    Q = Q0.clone()
    for _ in range(chord):
        s = net.grad(Q) @ Hinv
        s = s * (2.0 / s.norm(dim=1, keepdim=True).clamp(min=2.0))
        Q = Q - s
    near = (Q - th_star).norm(dim=1) < r_in
    Qf = Q
    escaped = torch.zeros(len(x), dtype=torch.bool, device=x.device)
    if (~near).any():
        Ql = quench(net, Q0[~near], 1.0 / lmax, int(q_long * max(1.0, lmax)))
        Qf = Q.clone()
        Qf[~near] = Ql
        escaped[~near] = (Ql - th_star).norm(dim=1) > r_out
    return Qf, escaped


def chord_refine(net, Q, Hinv, iters=60, clip=0.5):
    """Refine quench end points with the fixed H*^-1. Every fixed point is a true critical point, so a
    slowly converging in-basin point collapses onto th*, while a genuine neighbouring minimum stays."""
    Q = Q.clone()
    for _ in range(iters):
        s = net.grad(Q) @ Hinv
        s = s * (clip / s.norm(dim=1, keepdim=True).clamp(min=clip))
        Q = Q - s
    return Q


def segment_barrier(net, th_a, th_star, n=9):
    """Max loss on the segment th_a -> th_star, above the higher endpoint. Zero in a star-convex basin."""
    al = torch.linspace(0, 1, n, device=th_a.device, dtype=th_a.dtype).view(n, 1, 1)
    pts = (al * th_a.unsqueeze(0) + (1 - al) * th_star.view(1, 1, -1)).reshape(-1, th_a.shape[1])
    L = net.full_loss(pts).reshape(n, -1)
    return L.max(0).values - L[[0, -1]].max(0).values


def pick_gpu(min_free_mb=1200):
    """Least-loaded GPU with enough free memory, else CPU."""
    if not torch.cuda.is_available():
        return 'cpu'
    import subprocess
    out = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.free,utilization.gpu',
                                   '--format=csv,noheader,nounits']).decode()
    rows = [list(map(int, l.split(','))) for l in out.strip().splitlines()]
    rows = [r for r in rows if r[1] >= min_free_mb]
    if not rows:
        return 'cpu'
    rows.sort(key=lambda r: (r[2], -r[1]))
    return f'cuda:{rows[0][0]}'
