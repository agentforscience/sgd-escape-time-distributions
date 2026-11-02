# Escape Time Distributions in SGD

Measures how long SGD and Adam stay in loss basins of small tanh networks, using hundreds of replicas from exact minima and saddles, and tests four numerical
claims about those times. Basin exit is decided by converged quenching, and exits are audited by Newton polishing.

## Key findings
- **Exponential is rejected, but k is not 0.7–1.3:** only 32% of conditions are in range (median 1.47). k does not track sharpness. After removing the start-up delay, median k is 1.03; pooling minima gives 0.52–0.74.
- **Batch size:** τ grows about as B² (×5.1 per doubling on Fashion-MNIST), not √2. τ·η is nearly constant at fixed η/B on Fashion-MNIST, less so on digits.
- **Adam:** about 2× faster at matched in-basin loss, for narrow and wide basins alike. Adaptive scaling is the cause, not momentum. At default lr=0.001 it did not escape at all.
- **Saddles:** exit follows the top negative eigenvector 54% of the time at high noise and 82–86% at low noise (r=1). The eigenvalue gap does not predict this.

Full details: [REPORT.md](REPORT.md).

## Reproduce
```bash
uv venv --python 3.12 && source .venv/bin/activate && uv pip install -r requirements.txt
export NUMPY_MADVISE_HUGEPAGE=0     # needed on hosts where large numpy allocations stall
# Fetch Fashion-MNIST: see datasets/README.md
python src/e0_kramers.py                                   # double well (about 5 min, 12 processes)
python src/find_minima.py fmnist 8 1e-4 7; python src/find_minima.py digits 8 1e-3 4
python src/select_minima.py; python src/calibrate.py cuda:0
python src/make_jobs.py 0.008,0.004,0.016 0.002,0.001
./src/worker_loop.sh cuda:0 a                              # run several; about 30 worker-hours in total
python src/find_saddles.py S 0 2000; python src/e4_saddles.py 60    # run several in parallel
python src/audit_destinations.py 6
python src/analyze.py; python src/make_figures.py
```
Seeds are fixed per job. A same-seed GPU rerun of one condition reproduced every replica (`results/repro_check.json`).

## Layout
```
planning.md, literature_review.md, resources.md, REPORT.md
src/nnsim.py          batched MLP, Newton polish, quench, optimizers
src/run_escape.py     escape runs and worker queue (E1-E3)
src/e0_kramers.py     double-well validation
src/e4_saddles.py     saddle exit direction; src/find_saddles.py finds saddles
src/survival.py       interval-censored fits
src/analyze.py, src/make_figures.py
results/ figures/ papers/ datasets/ logs/
```
