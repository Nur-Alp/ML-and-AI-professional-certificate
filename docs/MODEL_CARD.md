# Model Card

Following the structure of the [Google model cards](https://modelcards.withgoogle.com/model-reports), per capstone component 25.3, and the spirit of [Model Cards for Model Reporting](https://arxiv.org/abs/1810.03993).

## Model Description

**Input:** For each of the eight functions, a real-valued vector in
`[0.000000, 0.999999]^d`, where `d` ranges from 2 (Functions 1, 2) to 8
(Function 8). Submitted to the capstone portal as a hyphen-separated string
with six decimal places, e.g. `0.812353-0.531722` for a 2D function.

**Output:** A single scalar per query, the true hidden function's value at
that input, returned by the portal. All eight functions are framed as
maximisation problems within this project.

**Model architecture:** A Gaussian Process (GP) regression surrogate,
`sklearn.gaussian_process.GaussianProcessRegressor`, with a `Matern(nu=2.5)`
kernel (automatic relevance determination, one length-scale per input
dimension) plus a `WhiteKernel` noise term, and `normalize_y=True`. One
independent GP is fit per function, refit from scratch each round on all
data collected so far (23-53 points per function by the final round). The
next query point is chosen by maximising an Upper Confidence Bound
acquisition function, `mean + kappa * std` with `kappa = 2.0` fixed, over a
pool of 20,000 random candidates plus 2,000 points sampled near the current
best observation, with any candidate within 0.02 of an already-observed
point excluded (added at round 10 to prevent wasted duplicate queries).
Implementation: `notebooks/bbo_capstone.ipynb`, built with scikit-learn.

## Performance

There is no traditional train/test split, performance is measured by the
best real value returned by the portal within the query budget (one query
per function per week, 13 rounds total). Final results:

| Function | Best value | Achieved | Improvement from round 1 |
|---|---|---|---|
| 1 | 0.000000 | round 6 | none (stayed at noise floor) |
| 2 | 0.659839 | round 10 | +294% vs. round 1 (0.167) |
| 3 | -0.029742 | round 11 | improved after 10 stagnant rounds |
| 4 | 0.655593 | round 12 | improved from a negative starting value |
| 5 | 8662.405001 | round 6 | +281% vs. round 1 (2271.5) |
| 6 | -0.176172 | round 9 | improved from -0.887 |
| 7 | 2.731447 | round 13 (final round) | +132% vs. round 1 (1.176) |
| 8 | 9.993834 | round 12 | +1.4% vs. round 1 (9.857, already near-ceiling) |

See `data/convergence_plots.png` for the full running-best-value curve per
function across all 13 rounds. A secondary, interpretability-oriented
metric is each function's fitted per-dimension length-scale (see
[DATASHEET.md](DATASHEET.md)), used to diagnose which dimensions the model
has confidently learned are sensitive versus which remain unresolved.

## Limitations

- **Smoothness assumption:** the Matern kernel assumes a reasonably smooth
  underlying function. Function 1's near-zero-everywhere behaviour is a
  poor match, and its length-scale diagnostics confirm the model never
  developed confident structure for it across all 13 rounds.
- **Small sample sizes relative to dimensionality:** the higher-dimensional
  functions (7 at 6D, 8 at 8D) remain sparse relative to their input space
  even at 43 and 53 points respectively; Function 8 still has one
  dimension with an unresolved (bound-saturated) length-scale at the final
  round.
- **No independent noise characterisation** beyond one directly confirmed
  case: Function 5 returned an identical output for an identical input in
  two separate rounds, confirming it is deterministic (or has negligible
  noise) at that point; this was not tested for the other seven functions.
- **Non-gradient acquisition search:** maximising UCB over random candidates
  rather than a continuous optimiser (e.g. L-BFGS-B) may miss narrow
  optima in the acquisition surface, particularly for Functions 7 and 8.
- **A real failure mode was discovered and fixed mid-project:** before
  round 10, the pipeline had no safeguard against proposing an exact or
  near-duplicate of an already-queried point, a genuine risk for any
  deterministic function, since it would silently waste a query. Fixed by
  excluding near-duplicate candidates before maximising UCB.

## Trade-offs

- **Fixed `kappa` across all eight functions.** Applying the same
  exploration weight to Function 1 (near-zero, essentially unstructured)
  and Function 5 (single dominant direction, strongly correlated with
  output, PCA correlation 0.90) is a real simplification. A
  function-specific or annealed kappa would likely have improved
  efficiency, particularly for functions that revealed clear structure
  early (5, 8) versus those that never did (1).
- **Exploitation versus robustness at the search boundary.** By the final
  rounds, Function 5's proposals repeatedly pinned multiple dimensions at
  the exact domain boundary (0.999999), a confident bet that traded
  robustness (hedging against being wrong about the optimum's location)
  for performance (concentrating the remaining budget on the
  highest-expected-value region).
- **Verification overhead versus speed.** Explicitly checking each proposed
  point's distance to prior observations, and periodically running cluster
  and PCA diagnostics before trusting a raw proposal, added manual
  overhead each round. This trade-off paid off directly: it caught the
  round 10 duplicate-query bug before it wasted a query, a failure that
  would otherwise have been invisible, since the buggy proposal looked
  like a perfectly ordinary, well-formatted output.
