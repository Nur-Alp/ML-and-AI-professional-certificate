# Model Card: GP + UCB Bayesian Optimisation Pipeline

Following the spirit of [Model Cards for Model Reporting](https://arxiv.org/abs/1810.03993), structured per capstone component 21.2.

## Overview

- **Name:** BBO Capstone GP+UCB Pipeline
- **Type:** Gaussian Process regression surrogate model with an Upper
  Confidence Bound (UCB) acquisition function, one independently fit model
  per function, implemented in `notebooks/bbo_capstone.ipynb` with
  scikit-learn.
- **Version:** as of round 10 (10 rounds of real portal feedback collected,
  most recent methodology change: near-duplicate candidate filtering added
  to `propose_next_point`).

## Intended use

This pipeline proposes the next query point for each of the eight functions
in the BBO capstone challenge, one query per function per round, under a
tight, expensive-evaluation budget. It is a coursework exercise, not
intended for use outside this challenge, on real optimisation problems, or
in any production system. It should not be used as a general-purpose
Bayesian optimisation library, its defaults (fixed kernel, fixed `kappa`,
random-candidate acquisition search) were chosen for this specific setting
of very few, very expensive evaluations, not tuned or validated for other
problem scales.

## Details: strategy across the ten rounds

The core approach has stayed constant throughout: fit a Matern(nu=2.5) +
WhiteKernel Gaussian Process per function, maximise `mean + 2.0 * std`
(UCB) over a pool of 20,000 random candidates plus 2,000 points sampled
near the current best observation, and submit the resulting point. What
changed round to round was the data feeding that process, and one real
methodology fix:

- **Rounds 1-4:** established the baseline loop. Round-to-round proposals
  used a different random seed purely for candidate-pool diversity, no
  change to the underlying method.
- **Round 5:** re-fit on the full accumulated dataset after rounds 3-4's
  real results came back; the README's living log was introduced here to
  track round-by-round reasoning.
- **Round 6-8:** continued the same loop as more data arrived; several
  rounds intentionally explored rather than purely exploited, consistent
  with UCB's design.
- **Round 7:** an empirical discovery, Function 5 returned an identical
  output for an identical input to round 6, confirming that function is
  deterministic (or has negligible observation noise) at that point.
- **Round 9-10:** a deep diagnostic pass ahead of round 10 found that the
  raw UCB proposal for Function 5 was an exact duplicate of the
  already-queried, confirmed-deterministic round 6/7 point, which would
  have wasted a query for zero new information. Fixed by adding a
  near-duplicate filter to `propose_next_point` (excluding any candidate
  within 0.02 of an already-observed point), which also corrected a
  near-duplicate proposal for Function 2. This is the one point where the
  method itself, not just the data, evolved.

## Performance

There is no traditional train/test split, this is a maximisation task
scored by the best real value returned by the portal within budget. Best
observed value per function through round 9, and which round achieved it:

| Function | Best observed | Achieved | Notes |
|---|---|---|---|
| 1 | ~0.000000 | all rounds | signal remains indistinguishable from zero throughout |
| 2 | 0.639762 | round 4 | |
| 3 | -0.034835 | initial data | no round 1-9 query has beaten the instructor-provided initial data |
| 4 | 0.619288 | round 6 | |
| 5 | 8662.405001 | rounds 6 and 7 (tied, deterministic) | more than tripled from round 1's 2271.5 |
| 6 | -0.176172 | round 9 | most recent function to reach a new best |
| 7 | 2.500510 | round 9 | steady improvement across the last four rounds |
| 8 | 9.955947 | round 5 | |

A secondary, interpretability-oriented metric is each function's fitted
per-dimension length-scale: a small length-scale indicates the model has
found real sensitivity in that dimension, while a length-scale pinned at
the search bound (10.0) indicates no sensitivity has been confidently
detected yet. As of round 9, Function 8 still has one of its eight
dimensions bound-saturated despite 49 points, the clearest case of a
function needing more data than its budget realistically provides.

## Assumptions and limitations

- **Smoothness assumption:** the Matern kernel assumes a reasonably smooth
  underlying function. Function 1's near-zero-everywhere behaviour is a
  poor match for this assumption, and its length-scale diagnostics confirm
  the model has learned essentially nothing confident about it.
- **Fixed exploration weight:** `kappa=2.0` is applied identically across
  eight functions with very different scales (Function 1 near zero,
  Function 5 past 8,000) and different observed volatility. A
  function-specific or annealed kappa would likely be more efficient, but
  hasn't been implemented.
- **Exploration-biased candidate search:** the local-refinement component of
  candidate search always samples near the current best point, so the
  search is structurally weighted toward confirming known good regions.
  Function 5's proposals repeatedly pinning multiple dimensions at the
  search boundary across several rounds is a direct symptom.
- **Non-gradient acquisition search:** maximising UCB over random candidates
  rather than a continuous optimiser (e.g. L-BFGS-B) may miss narrow optima
  in the acquisition surface, particularly for the higher-dimensional
  Functions 7 and 8.
- **Failure mode discovered in practice:** before the round 10 fix, the
  pipeline had no safeguard against proposing an exact or near-duplicate of
  an already-queried point, a real failure mode for any function that turns
  out to be deterministic, since it would silently waste a round.

## Ethical considerations

The data is entirely synthetic and contains no personal, sensitive or
identifiable information, so there are no fairness or bias concerns in the
traditional sense. The main consideration this project surfaces is about
transparency itself: documenting the method, the data, and the one real bug
found and fixed mid-project (rather than only presenting a clean final
result) is what makes the approach reproducible and lets another researcher
judge where to trust its proposals and where not to. Several of the
challenge's real-world analogies (drug discovery, manufacturing) involve
real costs of being wrong that a purely synthetic score cannot capture, a
reminder that in a genuine deployment, the exploration/exploitation
trade-off, and any fixed hyperparameter like `kappa`, would need to account
for the real-world cost of a bad query, not just its informational value.

Would more detail improve this card? Marginally, a per-function
recommended `kappa` would be genuinely useful, but hasn't been derived, so
adding a placeholder for it would create false precision rather than real
clarity. The current structure, documenting what was actually done,
observed, and fixed, is deliberately kept tied to real evidence from the
project rather than speculative detail that isn't yet backed by results.
