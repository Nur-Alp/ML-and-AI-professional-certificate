# Bayesian Black-Box Optimisation (BBO) Capstone

## Section 1: project overview

This project tackles eight hidden mathematical functions, each standing in for a
real-world optimisation problem, locating a radiation source, tuning a noisy
model, minimising adverse drug reactions, where only a handful of measurements
can be taken before committing to the best input. Rather than guessing
randomly, the project uses Bayesian optimisation: a statistical model learns
from every past measurement to predict where the best result is likely to be,
and each new measurement is chosen to balance testing promising areas against
exploring uncertain ones.

This mirrors a genuinely common industry constraint, expensive-to-evaluate
systems (physical experiments, long-running simulations, A/B tests) where
brute-force grid search isn't an option, which is directly relevant to future
ML engineering work involving hyperparameter tuning or experiment design under
a tight query budget.

## Section 2: inputs and outputs

Each of the eight functions takes an input vector with dimensionality specific
to that function, ranging from 2 dimensions (Functions 1-2) up to 8 dimensions
(Function 8). Every input value must lie in `[0.000000, 0.999999]` and is
submitted through the capstone portal as a hyphen-separated string with six
decimal places, for example `0.812353-0.531722` for a 2D function. The portal
returns a single scalar output value, the (possibly negated, to keep the task
a maximisation problem) result of evaluating the true hidden function at that
point.

## Section 3: challenge objectives

All eight functions are framed as maximisation problems. The core constraint
is query budget, one new measurement per function per week, so the strategy
has to extract maximum information from every single query rather than
exploring broadly and cheaply. There's also response delay to account for,
results can take time to come back from the portal, and the function
structure itself is entirely unknown beyond a rough real-world analogy and
dimensionality, so no assumptions about smoothness or unimodality can be taken
for granted beyond what the data itself reveals.

## Section 4: technical approach

The approach uses a Gaussian Process (GP) surrogate model,
`sklearn.gaussian_process.GaussianProcessRegressor` with a Matern kernel plus
a white-noise term, one independently fit per function. A GP was chosen over
a simpler regression or an SVM because it naturally provides a calibrated
uncertainty estimate at every candidate point, not just a prediction, which is
exactly what's needed to balance exploration and exploitation. That balance is
handled by an Upper Confidence Bound acquisition function
(`mean + kappa * std`, kappa=2.0), which explicitly rewards both high
predicted value and high uncertainty, so a candidate can be proposed either
because the model thinks it's good or because the model doesn't know enough
about that region yet.

This is a living record, updated as the approach evolves and more rounds come
back:

- **Rounds 1-2:** established the baseline GP+UCB loop. Function 5 already
  showed the exploration budget paying off, dipping from 2271.5 to 1261.8
  between rounds before jumping to a new best.
- **Rounds 3-4:** confirmed the pattern. Function 5 jumped again to 4956.2 in
  round 3, the largest single-round gain seen so far, while Function 1
  continued showing almost no usable signal (near-zero output across nearly
  all queries), consistent with its documented "sparse spike" behaviour in
  `docs/DATASHEET.md`.
- **Round 5:** re-fit on the full accumulated dataset (14-44 points per
  function depending on dimensionality). Several functions (2, 3, 4, 6) still
  carry wide predicted uncertainty relative to their predicted mean, so
  current proposals lean exploratory rather than confidently exploitative.
- **Round 5 results:** Function 5 jumped again, from 4039.6 to 8297.1, more
  than doubling and easily the largest single-round gain so far. Functions 7
  and 8 both reached new bests (1.74 and 9.96 respectively). Function 1
  remains essentially flat at zero, consistent with its documented sparse,
  spike-only signal.
- **Round 6 results:** a mixed round. Functions 4 and 5 both reached new
  bests (0.619 and 8662.4 respectively, Function 5's fourth consecutive
  improvement), while Functions 6, 7 and 8 came in below their current best,
  expected behaviour under UCB, which deliberately spends some queries on
  exploration rather than pure exploitation.
- **Round 7 results:** Functions 6 and 7 both reached new bests (-0.397 and
  2.235 respectively). Function 5 returned the exact same value as round 6
  (8662.4), since both rounds proposed the identical point, confirming the
  function is deterministic (no observation noise) at that location.
- **Round 8 results:** Function 7 reached another new best (2.273), while
  Functions 5, 6 and 8 came in below their current bests, another
  exploration-heavy round under UCB rather than a regression in the
  underlying model.
- **Round 9 results:** Functions 6 and 7 both reached new bests (-0.176 and
  2.501 respectively). A deep diagnostic pass ahead of round 10 caught a real
  bug: the raw UCB proposal for Function 5 was an exact duplicate of the
  point already queried in rounds 6 and 7 (distance to nearest observation =
  0.0000), which would have wasted a query on a confirmed-deterministic
  function. Fixed by excluding near-duplicate candidates
  (`propose_next_point` now filters out any candidate within 0.02 of an
  already-observed point) before maximising UCB, this also corrected a
  near-duplicate proposal for Function 2.
- **Round 10 results:** Functions 2 and 8 reached new bests (0.660 and
  9.971 respectively). Function 5 came in second-best (8266.8, below the
  8662.4 tied best from rounds 6-7), confirming the true optimum along that
  dimension sits slightly off the exact search boundary rather than exactly
  on it.

## Repository structure

```
notebooks/bbo_capstone.ipynb   Main notebook: fits a GP surrogate per function,
                                proposes the next query, and plots progress.
data/initial_data/             Per-function input/output arrays, updated as
                                each week's new query result comes back.
docs/DATASHEET.md              Description of the data used, its origin and limitations.
docs/MODEL_CARD.md             Model behaviour, assumptions and limitations.
```

## How to run

```
pip install numpy scikit-learn scipy matplotlib jupyter
jupyter nbconvert --to notebook --execute --inplace notebooks/bbo_capstone.ipynb
```

The notebook reloads whatever data currently exists under `data/initial_data/`,
so re-running it after appending a new week's result automatically produces an
updated proposal and convergence plot, no code changes needed.

## Further documentation

See [docs/DATASHEET.md](docs/DATASHEET.md) for details on the data,
[docs/MODEL_CARD.md](docs/MODEL_CARD.md) for the model's assumptions and
limitations, and [docs/METHODOLOGY.md](docs/METHODOLOGY.md) for the technical
justification behind the approach, relevant prior research, and library
choices.

## Course reference material

`course-reference/` holds supplementary, unrelated course materials (kept
separate from the capstone project itself); see
[course-reference/README.md](course-reference/README.md) for details.
