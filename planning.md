# Research Plan: Escape-Time Distributions of SGD and Adam

## Motivation & Novelty Assessment

### Why This Research Matters
How long an optimizer stays in a basin underlies theories of implicit regularization, flat-minima bias and LR/batch-size
schedules. Those theories rest on *mean* escape-time formulae, derived asymptotically. The full distribution is what tells us
whether escape is memoryless (Poisson-like), and hence whether the SDE picture applies at practical noise levels.

### Gap in Existing Work
Prior work (Xie 2021, Mori 2022, Ibayashi 2023) measures mean rates. Distribution *shape* in real networks with censoring and
model comparison appears unmeasured. Adam-vs-SGD escape is theorised with opposite signs (Zhou 2020 vs H2), with no measured
ratio by basin width. Which negative eigendirection SGD follows out of a true saddle is not measured against linear theory.

### Our Novel Contribution
A batched replica simulator with hundreds of independent runs from *the same* exact minimum/saddle; basin membership defined by
quench; interval-censored distribution fits; decomposition of non-exponentiality into transient vs heterogeneity vs memoryless;
a calibration-robust Adam/SGD comparison; direct falsification tests of the specific numerical claims in the hypothesis.

### Experiment Justification
- E0 (controlled potentials with known Kramers rate): validates simulator and estimators against an exact answer, and maps which
  mechanism yields k≠1. Without it, an NN "k=0.9" is uninterpretable.
- E1 (NN minima): measures the shape parameter k per minimum vs geometry. Tests H1.
- E2 (Adam vs SGD at narrow/wide minima, two LR calibrations). Tests H2.
- E3 (batch/LR grid). Tests H3 including the rival Arrhenius form.
- E4 (true saddles, exact Hessian). Tests H4 with chance and theory baselines.

## Research Question
What is the distribution of first-exit times of SGD/Adam from basins in small neural nets; and do its shape, optimizer dependence,
batch-size dependence, and saddle exit direction follow the four numerical claims below?

## Hypothesis Decomposition (preregistered criteria, fixed before any experiment)
- **H1a** Weibull fits better than exponential: LRT k=1 rejected (BH-FDR 5%) in a majority of NN conditions, ΔAIC>10.
- **H1b** Shape range: per-condition k within [0.7, 1.3] in ≥90% of conditions, with both k<1 and k>1 present.
- **H1c** Geometry dependence: |Spearman ρ(k, log sharpness)| significant across minima (p<.05).
- **H2a** Narrow: τ_SGD/τ_Adam 95% CI inside [2,5] at matched-temperature calibration.
- **H2b** Wide: ratio CI inside [0.67, 1.5].
- **H2c** (calibration-free) ratio-of-ratios narrow/wide > 1.
- **H3a** slope of log₂τ on log₂B CI contains 0.5 and excludes 0 and 1, on a constant-slope (power law) model.
- **H3b** Power-law log τ~log B fits better than the Arrhenius log τ~B by AIC.
- **H3c** Same η/B → same SDE time τ·η (within CI), for pairs (η,B),(2η,2B),(4η,4B).
- **H4a** Pooled top-eigenvector exit accuracy CI overlaps [70,85]% and exceeds chance (1/index).
- **H4b** Accuracy increases with eigenvalue gap (logistic slope > 0).

## Proposed Methodology

### Approach
Real optimizers on real data. Tanh MLPs (smooth, exact Hessians) with L2 penalty on Fashion-MNIST/digits: underparameterized so minima
are strict and per-example gradients nonzero. Each condition: R independent replicas start at the *same* polished minimum θ* (or saddle).

**Escape definition** (primary): basin membership by quench: replicas are periodically relaxed with full-batch descent, then tested
for a loss barrier on the segment to θ*. Secondary (robustness): first passage of loss level L*+δ, and of radius ‖θ−θ*‖=R.

### Experimental Steps
1. E0. d-dim double well, finite-sum per-sample linear noise (additive) and loss-proportional noise; compare τ with Kramers.
2. Minima: find ≥ 12 per dataset with varied protocols (init scale etc.); polish; exact Hessian.
3. E1/E2/E3 grids (≥ 256 replicas per condition).
4. E4: find saddles of index ≥ 2 with Newton on ‖∇L‖²; ≥ 100 SGD runs per saddle; project exit displacement on eigenvectors.

### Baselines
Exponential (Kramers); Kramers rate on E0; chance (1/index) and linear-theory direction prediction on E4.

### Evaluation Metrics
Weibull k; AIC among exponential/Weibull/gamma/lognormal; mean/median τ (restricted mean where censored); slope; exit accuracy.

### Statistical Analysis Plan
Interval-censored MLE. Bootstrap CIs resampling *replicas*, and for cross-minimum summaries resampling *minima* (the replicas of one
minimum are not independent draws of the population). BH-FDR for LRTs. α=.05. Censoring-robust time: median, or Weibull-implied mean.
A mix of censored conditions is reported, never dropped.

## Expected Outcomes
Theory-based expectation: k≈1 at high barrier; k>1 from a start-up transient; k<1 from pooled heterogeneity; log τ linear in B. H3 is
expected to fail; H2 direction uncertain.

## Timeline and Milestones
Planning 20m; simulator + pilot 45m; E0–E3 90m; E4 40m; analysis 40m; docs 30m.

## Potential Challenges
- No escapes within budget / immediate escapes (too low or too high T): pilot to choose T window; report the regime.
- Replica non-independence across minima: bootstrap at minimum level.
- Adam LR not commensurable with SGD's: two calibrations and H2c.
- Shared GPUs: replicas fit a 1–3 GB slice; CPU fallback.

## Success Criteria
All four sub-hypotheses receive a clear verdict with CIs, simulator validated against Kramers on E0, reported irrespective of direction.

## Deviations From This Plan (recorded during execution)
1. **Escape definition.** The plan's short quench + segment-barrier test was replaced before production. Tests showed its verdict depended on
   quench length: 400 steps reported 0–34% exited where the converged quench reported up to 97%. Production uses a converged quench.
2. **False-exit correction.** A Newton-polish audit found 495 of the first 6,337 exits (7.8%) were slow-converging in-basin replicas. An extra rule
   (barrier or loss change required) was added and the 10 affected runs were rerun with the same seeds.
3. **Base condition** moved from η=0.2 to η=0.1 (at η=0.2 escape competes with in-well relaxation). η=0.2 kept as the hot condition.
4. **Sharpness range.** Weight decay made digits minima near-uniform, so the main landscape became Fashion-MNIST at wd=1e-4 (λmax 0.62–5.48, 9×).
5. **Saddles** use a smaller network (h=3, P=115). Mode-following Newton did not converge at P=290. Symmetric saddles were rejected: per-sample
   gradients vanish along their escape directions.
6. **Secondary definitions** (loss level, radius first passage) were not analysed. Quench-length and exit-rule robustness were analysed instead.
7. **Scope.** 165 of 166 jobs finished. Missing: default-LR Adam at the sharpest minimum (stopped at the time limit).
8. **H4 radius** was not fixed in advance, so all three radii and all three noise levels are reported.
