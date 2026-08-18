# Model Card: GP + UCB Bayesian Optimisation Pipeline

Following the spirit of [Model Cards for Model Reporting](https://arxiv.org/abs/1810.03993).

## Model details

- **Surrogate model:** Gaussian Process Regression (`sklearn.gaussian_process.GaussianProcessRegressor`),
  with a `Matern(nu=2.5)` kernel (automatic relevance determination, one
  length-scale per input dimension) plus a `WhiteKernel` noise term, and
  `normalize_y=True`.
- **Acquisition function:** Upper Confidence Bound, `mean + kappa * std`,
  with `kappa = 2.0` fixed across all functions and rounds.
- **Candidate search:** the acquisition function is maximised over a pool of
  20,000 uniformly random candidates plus 2,000 points sampled near the
  current best observation (for local refinement), rather than by gradient-based
  continuous optimisation.
- **One independent model is fit per function** (eight functions, refit from
  scratch each round using all data collected so far).
- **Implementation:** `notebooks/bbo_capstone.ipynb`, built with `scikit-learn`.

## Intended use

This pipeline proposes the next query point for each of the eight functions
in the BBO capstone challenge, one query per function per round. It is a
coursework exercise and is not intended for use outside this challenge, on
real optimisation problems, or in any production system.

## Factors

Model behaviour varies meaningfully by function, primarily due to:

- **Input dimensionality** (2D to 8D): the GP's reliability degrades as
  dimensionality increases relative to the number of observed points, since
  the input space grows exponentially while data grows only linearly.
- **Signal sparsity:** functions with mostly flat/near-zero observed output
  so far (Function 1) give the GP very little to learn from.
- **True response shape:** functions described as unimodal (Function 5) are
  a good match for a smooth GP surrogate; functions with sharp, localised
  structure are a poor match.

## Metrics

- **Running best value observed** (per function, plotted in the notebook) is
  the primary progress metric, this challenge is a maximisation task, and
  the metric that matters is simply the best value found within the query
  budget, not a held-out prediction error.
- **Fitted per-dimension length-scale** is used as a secondary,
  interpretability-oriented diagnostic: a small length-scale indicates the
  function is sensitive to that input dimension, a length-scale pinned at
  the upper search bound (10.0) indicates the model found no strong evidence
  of sensitivity to that dimension given the current data.

## Training data

See [DATASHEET.md](DATASHEET.md). In short: eight independent synthetic
functions, 10-40 initial points each, growing by one point per function per
round.

## Evaluation

There is no traditional train/test split, the "evaluation" is whether each
round's real, portal-returned output improves on the previous best. By this
measure, two of the eight functions improved substantially after the first
round of real feedback (Function 4 improved from -4.03 to -0.23; Function 5
more than doubled, from 1088.86 to 2271.54), while others did not improve on
that particular round, consistent with UCB deliberately allocating some
queries to exploration rather than pure exploitation.

## Ethical considerations

The data is entirely synthetic and contains no personal, sensitive or
identifiable information. There are no fairness or bias concerns in the
traditional sense. The main "real-world" consideration this project
surfaces is a general one: several of the challenge's real-world analogies
(drug discovery, manufacturing) involve costs of being wrong that a purely
synthetic score cannot capture, a reminder that in a genuine deployment,
the acquisition function's exploration/exploitation trade-off would need to
account for the real-world cost of a bad query, not just its informational
value.

## Caveats and recommendations

- **Fixed `kappa`:** using the same exploration weight (`kappa=2.0`) for
  every function ignores that they have very different scales, noise levels
  and dimensionalities. A function-specific or annealed `kappa` (higher
  early, lower as data accumulates) would likely be more efficient.
- **Smoothness assumption:** the Matern kernel assumes a reasonably smooth
  underlying function. Function 1's near-zero-everywhere-except-spikes
  behaviour is a poor match for this assumption, and proposals for that
  function should be treated as closer to informed exploration than a
  confident prediction.
- **Length-scale bound saturation:** several functions (1, 2, 3, 5, 7, 8)
  have at least one dimension whose fitted length-scale sits at the search
  bound (10.0), meaning the model could not confidently estimate sensitivity
  in that dimension from the available data. This should improve as more
  rounds of data are collected, but current proposals along those dimensions
  are less trustworthy than proposals along dimensions with a clearly
  estimated, smaller length-scale.
- **Non-gradient acquisition search:** maximising the acquisition function
  over random candidates rather than via gradient-based continuous
  optimisation is simple and reliable but may miss narrow optima in the
  acquisition surface, particularly in higher dimensions (Functions 7 and
  8). A larger candidate pool or a proper continuous optimiser
  (e.g. L-BFGS-B with multiple restarts) would likely improve proposal
  quality at higher dimensionality.
