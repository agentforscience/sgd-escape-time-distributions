"""Fix the set of minima BEFORE any escape measurement: evenly spaced in lmax rank, seeded."""
import json, numpy as np
rs = sorted(json.load(open('results/minima/fmnist_wd0.0001.json')), key=lambda r: r['lmax'])
idx = np.unique(np.round(np.linspace(0, len(rs) - 1, 12)).astype(int))
A = [rs[i] for i in idx]
rd = sorted(json.load(open('results/minima/digits_wd0.001.json')), key=lambda r: r['lmax'])
idx = np.unique(np.round(np.linspace(0, len(rd) - 1, 6)).astype(int))
B = [rd[i] for i in idx]
json.dump(dict(A=A, B=B), open('results/selected_minima.json', 'w'), indent=1)
for r in A + B:
    print(r['name'], round(r['lmax'], 2), round(r['trace'], 1), round(r['L'], 4), round(r['acc'], 3), round(r['logdet'], 1))
