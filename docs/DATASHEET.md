# Datasheet

This datasheet documents the optimisation decisions, learning and reasoning
for every function in the BBO capstone project, following the template
provided for capstone component 25.3. All eight functions were optimised
using the same underlying pipeline; questions that vary by function are
answered per function below, questions that apply identically across the
whole project (data handling, ethics) are answered once.

## Function overview

| Function | Real-world scenario | Dimensionality | Initial points | Output represents |
|---|---|---|---|---|
| 1 | Radiation/contamination source detection | 2D | 10 | Proximity-based detection signal |
| 2 | Noisy log-likelihood optimisation of a mystery model | 2D | 10 | Log-likelihood value |
| 3 | Drug discovery (minimising adverse reactions, negated) | 3D | 15 | Negated adverse-reaction severity |
| 4 | Warehouse product placement (fast surrogate model) | 4D | 30 | Placement efficiency score |
| 5 | Chemical process yield (described as typically unimodal) | 4D | 20 | Process yield |
| 6 | Recipe optimisation (multi-factor score, negated) | 5D | 20 | Negated recipe quality score |
| 7 | ML hyperparameter tuning (accuracy/F1 objective) | 6D | 30 | Model accuracy/F1 score |
| 8 | ML hyperparameter tuning, higher-dimensional | 8D | 40 | Model accuracy/F1 score |

All eight functions were treated as maximisation problems; several
real-world analogies are naturally minimisation problems and their outputs
were pre-negated by the challenge itself so that higher is always better
within this project.

## Nature of the data

**Structure.** Each function's data is a pair of NumPy arrays,
`initial_inputs.npy` (shape `(n, d)`) and `initial_outputs.npy` (shape
`(n,)`), stored under `data/initial_data/function_{1-8}/`. Every input
dimension lies in `[0.000000, 0.999999]`.

**Evolution.** The dataset grew by exactly one new (input, output) pair per
function per week, across 13 rounds of real portal feedback, taking every
function from its initial 10-40 points to a final 23-53 points. Each new
query was chosen by the GP+UCB pipeline (see [MODEL_CARD.md](MODEL_CARD.md)),
so the sampling pattern is not uniform, it concentrates near regions the
model believes are good, with progressively less exploration for functions
that revealed a clear, well-separated optimum early (Function 5 by round 6)
and sustained broader exploration for functions that didn't (Function 1,
throughout; Function 3, until round 11).

**Noise.** One function was directly confirmed deterministic in practice:
Function 5 returned the exact same output (8662.405001248297) for the exact
same input in both round 6 and round 7, and again the pipeline proposed
that identical point once more before a round 10 fix prevented it. No other
function was directly tested for repeat-query determinism. Fitted GP noise
levels (from the final model fit per function) ranged from effectively zero
(Functions 1, 3, 7, 8) to modest but non-trivial (Functions 2 and 6, noise
level ~0.02-0.03), suggesting those two carry genuine observation noise
rather than being purely deterministic.

**Shape assessment**, based on final fitted GP length-scales and observed
behaviour:

| Function | Assessment | Evidence |
|---|---|---|
| 1 | Effectively flat/unresolved | One dimension's length-scale still saturated at the search bound after 23 points; best value stayed at 0.000000 across the entire project |
| 2 | Smooth, one dominant dimension | One length-scale small (0.073) and confidently estimated, the other saturated |
| 3 | Smooth, one dominant dimension | Similar pattern to Function 2; one saturated dimension |
| 4 | Smooth, moderate sensitivity across all dimensions | All four length-scales in a similar, moderate range (~1.7), no saturation |
| 5 | Smooth, moderate sensitivity across all dimensions, matches "unimodal" description | All four length-scales similar (~1.7-1.9), no saturation; PCA on inputs shows one dominant component correlating 0.90 with output |
| 6 | Smooth with one weaker dimension | Four dimensions similarly sensitive, fifth notably less so (length-scale 2.67 vs ~1.2-1.6) |
| 7 | Smooth, all dimensions contribute | All six length-scales well-resolved and no saturation, the best-behaved higher-dimensional function |
| 8 | Sparse relative to dimensionality | One of eight dimensions still saturated even at 53 points; several others (length-scale 6-8.6) only weakly resolved |

## Your optimisation strategy

**Method.** Gaussian Process regression (Matern nu=2.5 kernel plus a
`WhiteKernel` noise term) with Upper Confidence Bound acquisition, one
independently fit model per function, refit from scratch each round on all
data collected so far. See [MODEL_CARD.md](MODEL_CARD.md) for full
implementation details.

**Why this method.** A GP was chosen over simpler regression or an SVM
because it provides a calibrated uncertainty estimate at every candidate
point, not just a prediction, which UCB needs directly to trade off
exploitation against exploration. This mattered differently by function:
for Functions 5 and 8, where a single dominant, well-correlated direction
existed, the model's confidence let proposals concentrate tightly once that
structure emerged; for Functions 1 and 3, persistently wide uncertainty
correctly kept the search broad rather than converging prematurely on noise.

**Exploration/exploitation balance.** Handled via UCB's `mean + kappa * std`
formula with `kappa = 2.0`, fixed across all eight functions and all 13
rounds. This is a known, documented limitation (see
[MODEL_CARD.md](MODEL_CARD.md)), a function-specific or annealed kappa
would likely have been more efficient, particularly given how differently
the eight functions behaved.

**Did the strategy change over the weeks?** Yes, in two concrete ways.
First, at round 10, a diagnostic check (distance from any proposed point to
every existing observation) was added permanently to the pipeline after it
caught the algorithm proposing an exact duplicate of an already-queried,
confirmed-deterministic Function 5 point, which would have wasted that
round entirely. Second, from round 11 onward, cluster analysis (k-means)
and PCA on each function's accumulated inputs were added as a diagnostic
layer, specifically to check whether a function's apparent "best region"
was real, separated structure or statistically indistinguishable noise,
before trusting the raw proposal.

## Data handling and preprocessing

No rescaling or encoding was applied to the inputs; they arrive already
continuous and normalised to `[0, 1)` by the challenge's own design. The
one transformation applied is at the acquisition-search stage, not to the
data itself: `propose_next_point` excludes any candidate within 0.02 of an
already-observed point (added at round 10, see above). The GP surrogate's
own hyperparameters (per-dimension length-scale, noise level) are optimised
automatically via gradient-based restarts (`n_restarts_optimizer=5`) each
time a model is fit, this is the only "training" step involved. No explicit
outlier handling was applied, the `WhiteKernel` noise term absorbs mild
observation noise, and no data point was ever excluded from a function's
dataset.

## Weekly iteration and learning

New data points reshaped understanding most visibly through the length-scale
diagnostics: dimensions that started pinned at the search bound (no
detected sensitivity) resolved into confidently small length-scales as data
accumulated for several functions, directly changing where proposals
concentrated. Local optima were not directly detected via a formal test,
but Function 5's trajectory (peaking at round 6, then plateauing and
slightly declining through rounds 10-13) is consistent with the search
having settled near a real optimum rather than continuing to find better
regions. The most informative queried points were consistently the
boundary-adjacent and high-uncertainty ones, Function 5's repeated
proposals at the domain boundary (0.999999) were only confirmed as correct,
rather than an artefact, once round 10 showed backing off slightly from
that exact corner performed worse. If restarted, the duplicate-check
safeguard and the per-function cluster/PCA diagnostics would be built in
from round 1 rather than added midway.

## Performance and results

| Function | Best value achieved | Achieved | Input vector |
|---|---|---|---|
| 1 | 0.000000 | round 6 | `[0.535814, 0.577742]` |
| 2 | 0.659839 | round 10 | `[0.690468, 0.978181]` |
| 3 | -0.029742 | round 11 | `[0.037808, 0.924717, 0.430483]` |
| 4 | 0.655593 | round 12 | `[0.422645, 0.369708, 0.410643, 0.407544]` |
| 5 | 8662.405001 | round 6 | `[0.999999, 0.999999, 0.999999, 0.999999]` |
| 6 | -0.176172 | round 9 | `[0.470661, 0.374915, 0.675722, 0.729856, 0.0]` |
| 7 | 2.731447 | round 13 (final) | `[0.0, 0.09348, 0.525245, 0.255483, 0.348463, 0.647678]` |
| 8 | 9.993834 | round 12 | `[0.10737, 0.114218, 0.125626, 0.146633, 0.815302, 0.508034, 0.244412, 0.676281]` |

**Confidence in these being near-global-optima varies substantially.**
Highest confidence: Functions 5 and 8, where a single dominant direction was
identified (PCA correlation 0.90 for both) and the search stabilised near
the boundary/best region for multiple consecutive rounds without further
improvement. Moderate confidence: Functions 2, 4, 6, 7, which improved
steadily and, in Function 7's case, was still improving in the final round,
suggesting the true optimum may not yet be fully reached. Low confidence:
Function 3, which only broke through after ten stagnant rounds, and
Function 1, which never developed usable signal above noise across the
entire project, its reported "best" value is not meaningfully distinguishable
from zero and should not be read as a located optimum.

Results generally aligned with the functions' described real-world
analogies, Function 5's smooth, unimodal-consistent improvement matched its
"typically unimodal" chemical-yield description, and Function 1's
persistent near-zero signal matched its framing as a proximity-only
detection problem.

## Ethical, practical and general considerations

This task mirrors real expensive-evaluation optimisation problems (drug
formulation testing, manufacturing parameter tuning, hyperparameter search)
where each evaluation costs real time or money, so the core lesson,
extracting maximum information from a tightly budgeted number of queries,
transfers directly. The main limitation from the synthetic nature of the
data is that ground truth is never observed, all shape and behaviour
assessments above are inferred from sampled points and fitted diagnostics,
not verified against the true underlying functions. The strategy would
scale to more serious or expensive problems with modification, the fixed
`kappa` and random-candidate acquisition search are simplifications
appropriate for a small, cheap-to-compute exercise, and would need
per-problem tuning and a proper continuous optimiser (e.g. L-BFGS-B) at
higher stakes. The main pitfall a future user should be aware of: a
"successful-looking" proposal is not automatically a good one, the round 10
bug showed that a plausible, well-formatted query can still be silently
worthless, and verifying proposals against accumulated data before acting
on them is not optional overhead, it is what caught a real, otherwise
invisible failure.
