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

Across the full, completed project, Function 5 showed exactly the kind of
large, non-stationary behaviour HEBO's warping was designed for: its best
value moved 2271.5, 1261.8, 4956.2, 8297.1, 8662.4, 8662.4 (confirmed
deterministic repeat), 6794.9, 6543.9, 8266.8, 8282.8, 8281.2, 7924.2 across
rounds 1-13, more than tripling from its starting point before settling
into a plateau. That non-monotonic, threshold-then-plateau shape is
precisely what a plain GP with a fixed kernel struggles to represent
cleanly compared to HEBO's non-linear warping. In hindsight, HEBO's
approach was never adopted in practice, the fixed Matern kernel plus UCB
was sufficient to capture the project's biggest win (Function 5) and its
final-round win (Function 7, still improving as of round 13), so the
budget-appropriate simplification held up across the whole project, not
just its early rounds.

## Libraries and alternatives

The core stack is scikit-learn (`GaussianProcessRegressor` with `Matern`,
`WhiteKernel`, `ConstantKernel`), NumPy, SciPy (`cdist` for the uncertainty
diagnostics), and Matplotlib. Deliberately not PyTorch, TensorFlow, or
GPU-oriented GP libraries like GPyTorch or BoTorch: those exist specifically
to scale Gaussian Processes to thousands or millions of points via
approximate inference, but every function here finished the project with at
most 53 observed points (23 to 53 depending on dimensionality, after all 13
rounds), well within the range where scikit-learn's exact, CPU-based GP
implementation was both sufficient and simpler to reason about throughout.
Choosing a deep-learning framework here would have traded away simplicity
and determinism for scalability this project never needed, even at its
final data scale.

## Documentation plan

These justifications live here, linked from the README, alongside
[DATASHEET.md](DATASHEET.md) (data provenance and limitations) and
[MODEL_CARD.md](MODEL_CARD.md) (model behaviour and caveats). The README's
own "Technical approach" section is kept as the full, final round-by-round
log (rounds 1-13), so a reader gets the narrative there and the deeper
justification here.

## What would be worth trying with more query budget

The optimisation phase for this capstone is complete (13 of 13 rounds
submitted), so these were not implemented, but would be the natural next
steps for a longer-running version of this project: constant-liar and
local-penalization strategies for batch/parallel Bayesian optimisation (not
relevant at one query per function per week, but useful at a higher query
cadence), a proper continuous optimiser (e.g. L-BFGS-B with multiple
restarts) for maximising the acquisition function in the higher-dimensional
functions (7 and 8), where the random-candidate-pool search used throughout
this project is more likely to miss narrow optima, and a function-specific
or annealed `kappa` in place of the fixed value of 2.0 used across all
eight functions for all 13 rounds, the single most consistently flagged
limitation across this project's own reflections.
