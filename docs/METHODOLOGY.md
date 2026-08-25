# Methodology and technical justification

## Main technical justification

The core approach, Gaussian Process (GP) regression with an Upper Confidence
Bound (UCB) acquisition function, is the standard, well-established method for
sample-efficient black-box optimisation when queries are expensive and the
objective's structure is unknown. GPs are the right surrogate here specifically
because they provide a full posterior (mean and variance) rather than a point
estimate, which UCB needs directly to trade off exploitation against
exploration at every candidate point.

## Relevant research

The most directly relevant paper is Cowen-Rivers et al., ["HEBO: Heteroscedastic
Evolutionary Bayesian Optimisation"](https://arxiv.org/abs/2012.03826) (NeurIPS
2020), which won that year's official NeurIPS Black-Box Optimisation
Challenge, the same competition format this capstone is modelled on. HEBO's
core insight is that real black-box objectives often have heteroscedastic
noise and non-stationary structure, and it addresses this with non-linear
input/output warping plus an ensemble of acquisition functions evaluated via
multi-objective optimisation to select Pareto-optimal candidates.

This project deliberately adopts HEBO's foundational choice, a GP surrogate
driving a principled acquisition function, but not its more elaborate
machinery. The input/output warping and multi-objective acquisition ensemble
are scoped for the much larger query budgets (100+ evaluations) HEBO targets,
whereas this project gets one query per function per week. A fixed Matern
kernel with a single UCB acquisition is a deliberate, budget-appropriate
simplification of the same underlying idea, not an oversight.

As more rounds accumulate, HEBO's non-linear warping becomes increasingly
worth revisiting, particularly for Function 5, which has shown large,
non-stationary jumps in observed value across rounds (2271.5 to 1261.8 to
4956.2 to 8297.1 across rounds 1-5), exactly the kind of behaviour HEBO's
approach was designed to handle better than a plain GP.

## Libraries and alternatives

The core stack is scikit-learn (`GaussianProcessRegressor` with `Matern`,
`WhiteKernel`, `ConstantKernel`), NumPy, SciPy (`cdist` for the uncertainty
diagnostics), and Matplotlib. Deliberately not PyTorch, TensorFlow, or
GPU-oriented GP libraries like GPyTorch or BoTorch: those exist specifically
to scale Gaussian Processes to thousands or millions of points via
approximate inference, but every function here has at most a few dozen
observed points (15 to 45 depending on dimensionality as of round 5), well
within the range where scikit-learn's exact, CPU-based GP implementation is
both sufficient and simpler to reason about. Choosing a deep-learning
framework here would trade away simplicity and determinism for scalability
this project doesn't need.

## Documentation plan

These justifications live here, linked from the README, alongside
[DATASHEET.md](DATASHEET.md) (data provenance and limitations) and
[MODEL_CARD.md](MODEL_CARD.md) (model behaviour and caveats). The README's
own "technical approach" section is kept as a living, round-by-round log, so
a reader gets the narrative there and the deeper justification here.

## Looking ahead

Beyond HEBO's warping techniques, worth consulting as the project continues:
constant-liar and local-penalization strategies for batch/parallel Bayesian
optimisation (not currently relevant at one query per function per week, but
useful if the query cadence ever changes), and a proper continuous
optimiser (e.g. L-BFGS-B with multiple restarts) for maximising the
acquisition function in the higher-dimensional functions (7 and 8), where the
current random-candidate-pool search is more likely to miss narrow optima in
the acquisition surface.
