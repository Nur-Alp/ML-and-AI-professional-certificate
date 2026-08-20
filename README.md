# Bayesian Black-Box Optimisation (BBO) Capstone

## Summary

This project tackles eight hidden mathematical functions, each standing in for a
real-world optimisation problem (locating a radiation source, tuning a chemical
process, choosing hyperparameters) where only a handful of measurements can be
taken before settling on the best inputs. Rather than guessing randomly, the
project uses Bayesian optimisation: a statistical model learns from every past
measurement to predict where the best result is likely to be, and each new
measurement is chosen to balance testing promising areas against exploring
uncertain ones. Results have already improved substantially for several
functions after just one or two rounds of real feedback, showing the approach
is working as intended.

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

See [docs/DATASHEET.md](docs/DATASHEET.md) for details on the data, and
[docs/MODEL_CARD.md](docs/MODEL_CARD.md) for the model's assumptions and
limitations.

## Course reference material

`course-reference/` holds supplementary, unrelated course materials (kept
separate from the capstone project itself); see
[course-reference/README.md](course-reference/README.md) for details.
