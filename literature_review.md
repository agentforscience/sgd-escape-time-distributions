# Literature Review: Escape Times of Stochastic Optimizers from Loss Basins

*Note: the resource-finding stage produced no output for this workspace. This review was compiled during
the research session. Every paper below had its arXiv metadata checked via the arXiv API on 2026-10-30;
the formulas quoted from Xie 2021, Mori 2022, Xie 2022 and Zhou 2020 were read from the PDFs in `papers/`.
An arXiv search for "escape time" + Weibull + "stochastic gradient" returned nothing, as did
"first passage" + SGD + "loss landscape". That is evidence of a gap but not proof of one (arXiv full-text
keywords only).*

## The theoretical frame: Kramers escape
Kramers (1940) gives the rate for overdamped Langevin dynamics at temperature T over a barrier ΔE:
κ = (√(a|b|)/2π) exp(−ΔE/T), with a, b the curvatures at the well and saddle. At high barrier the escape is a
Poisson process, so **first-passage times are exponential** (Weibull k = 1). Any deviation must come from outside
that limit: a relaxation transient, a low barrier, or a mixture of rates.

## SGD as an SDE
- **Mandt et al. 2017** and **Smith & Le 2018**, **Jastrzębski et al. 2018**: for SGD, the temperature scales
  as T ∝ η/B. This is the origin of "the ratio of batch size to learning rate controls temperature".
- **Zhu et al. 2019 (arXiv:1803.00195)**: SGD noise is anisotropic and aligned with sharp directions, which speeds escape
  from sharp minima relative to isotropic noise.
- **Xie et al. 2021 (arXiv:2002.03495)**: with noise covariance ≈ Hessian, mean escape time is exponential in
  (B/η)·ΔL/H. "The ratio of the batch size and the learning rate exponentially matters."
- **Mori et al. 2022 (arXiv:2105.09557)**: with noise ∝ L(θ), the rate is a power law in the loss ratio,
  κ ~ (√(h*|h_s|)/2π)·(L(θ_s)/L(θ*))^−(B/(ηh*) + 1 − n/2). Note it is a power law in **L(θ_s)/L(θ*)**, whereas
  log τ remains **linear in B/η**.
- **Ibayashi & Imaizumi 2023 (arXiv:2111.04004)**: exponential escape efficiency in a non-stationary regime without a
  stationary-distribution assumption.
- **Şimşekli et al. 2019 (arXiv:1901.06053)**: heavy-tailed gradient noise; Lévy jump models predict polynomial, not
  exponential, dependence of escape time on barrier *height* and scale with basin *width*.

**Implication for this project.** No theory above predicts τ ∝ B^{1/2}. Arrhenius-type theories predict
log τ linear in B/η, i.e. doubling-B factors that *grow with B*. A barrier-free diffusive first passage over a fixed
distance predicts τ ∝ B/η² (factor 2 per doubling). √2 corresponds to no standard regime and is therefore a substantive
and falsifiable claim.

## Adam vs SGD
- **Zhou et al. 2020 (arXiv:2010.05627)**: Lévy-SDE analysis; Adam's escape time is larger than SGD's because Adam's
  basin-measure is larger: "SGD is more locally unstable" at sharp minima. This **contradicts** H2's direction.
- **Xie et al. 2022 (arXiv:2006.15815)**: "Adaptive Inertia": Adam's adaptive learning rate accelerates saddle
  escape but *hurts* flat-minima selection relative to SGD; momentum helps with saddles, not minima selection.
- No direct measurement of an Adam:SGD escape-time ratio split by basin width was found. Any such ratio depends on
  how the two learning rates are calibrated: that confound has to be handled explicitly.

## Saddles
- **Jin et al. 2017 (arXiv:1703.00887)**, **Daneshmand et al. 2018 (arXiv:1803.05999)**: rate-of-escape results via
  perturbation or intrinsic gradient noise; neither claims which negative eigendirection wins.
- Linear theory (our derivation): near a saddle, the component along eigenvector i grows as
  ~ z_i σ_i (1 + η|λ_i|)^t, with z_i Gaussian. So the most-negative direction wins *in probability*, with odds set by both
  the eigenvalue gap (relative to the log-time of escape) and the noise projection σ_i. The gap alone is an incomplete
  predictor, a testable refinement of H4.
- **Fukumizu & Amari 2000**, **Şimşek et al. 2021 (arXiv:2105.12221)**: critical points by unit duplication and
  symmetry give a source of true saddles in small networks.

## Sharpness caveats
- **Keskar et al. 2017** and **Dinh et al. 2017 (arXiv:1703.04933)**: sharpness is not reparametrization-invariant. We
  report λ_max and trace as measured in a fixed parameterization, and use them descriptively only.

## Metric conventions
- Yaida 2019 (arXiv:1810.00004): fluctuation-dissipation relations, a check for stationarity of SGD.
- Survival analysis: Weibull with interval censoring. AIC-based model selection is the standard way of comparing
  Weibull, exponential, gamma and lognormal fits.
