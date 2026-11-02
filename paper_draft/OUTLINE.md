# Outline and evidence map

Skeleton: Intro; Related Work (Kramers, SDE, Adam vs SGD, saddles, sharpness); Method (formulation, models, exit rule, E0-E4, baselines, statistics);
Results (validation, hops, H1-H4); Discussion (limitations inside); Conclusion; Appendix (preregistration, deviations).

Evidence: all numbers come from results/analysis/{summary.json,conditions.csv,h2_A.csv,h4.csv}, figures from figures/.

## Where the paper differs from REPORT.md (recomputed from results/)
- AIC gaps: REPORT's 2.1/0.1/4.9 are medians over all 36 E1 conditions. On the 28 H1 conditions, gamma and lognormal beat Weibull by 5.1 and 6.4, and Weibull beats shifted exponential by 6.9. So "differences are small" was dropped.
- H3c Fashion-MNIST: one minimum is capped at all three settings, so its ratios are 2 and 0.5 by construction. Paper leads with the 5 observed minima: 1.03 [0.97, 1.11], 0.89 [0.86, 0.93] (bootstrap over minima, geometric mean). The all-6 values are also given.
- H3c digits: all (0.05,8) medians are capped, and one minimum has both (0.1,16) and (0.2,32) observed (ratio 1.03). Verdict changed from "partly supported" to "inconclusive".
- eta doubling 9.4x includes 2 capped minima; 8.5x without them.
- Adam narrow/wide: added observed-only 2.01 (n=3) and 1.91 (n=3). Ablations: added observed-only 0.50 and 0.41.
- Rounding: 0.9346 is 0.93 (not 0.94); k at Delta/T=1 is 1.18 (not 1.19).
- Saddle |lambda_1| is 2e-4 to 0.03 in h4.csv (REPORT: 1e-4 to 1e-3).
- Citations: venues and years corrected; authors and titles checked against arXiv metadata. Ibayashi & Imaizumi, Jastrzebski et al. cited as arXiv.
