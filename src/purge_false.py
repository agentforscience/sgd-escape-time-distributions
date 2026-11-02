"""Delete finished runs containing exits the corrected rule rejects, so they rerun with the same seed."""
import glob, json, os, sys
import numpy as np
sys.path.insert(0, 'src')
n = t = 0
for f in sorted(glob.glob('results/escape/*.npz')):
    if f.endswith('.tmp.npz'):
        continue
    z = np.load(f)
    ev = np.isfinite(z['t_exit'])
    bad = ev & ~((z['b_exit'] > 1e-5) | (np.abs(z['l_exit']) > 2e-5))
    t += ev.sum()
    n += bad.sum()
    if bad.any():
        print(os.path.basename(f)[:60], int(bad.sum()), '/', int(ev.sum()))
        if len(sys.argv) > 1:
            os.remove(f)
print('false', n, 'of', t)
