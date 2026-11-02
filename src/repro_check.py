"""Re-run one finished condition with the same seed and compare exit checks replica by replica."""
import json, sys, numpy as np
sys.path.insert(0, 'src')
import run_escape as re_
dev, jid = sys.argv[1], sys.argv[2]
j = next(x for x in json.load(open('results/jobs.json')) if x['id'] == jid)
r = re_.run_job(j, dev)
z = np.load(f'results/escape/{jid}.npz')
a, b = z['t_exit'], r['t_exit']
same = np.mean((a == b) | (np.isnan(a) & np.isnan(b)))
out = dict(job=jid, device=dev, share_identical=float(same), median_a=float(np.nanmedian(a)), median_b=float(np.nanmedian(b)))
print(out)
json.dump(out, open('results/repro_check.json', 'w'), indent=1)
