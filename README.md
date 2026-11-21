# Causal_Driver

Code for recovering the ancestral causal order of genes from pooled fitness screens, where each knockout is recorded as one scalar per cell line. The method regresses the per-sample readouts on standardized expression, normalizes the estimated matrix by its diagonal, and orders genes with a feedback-arc-set rule. A certified version of the rule marks the pairwise relations whose order is guaranteed once a data-driven tolerance exceeds the estimation error.

All scripts sit in one directory and import each other by module name, so run them from the repository root. Most scripts use a GPU through PyTorch when one is available and fall back to the CPU otherwise. The `theory_*.py` scripts need only NumPy.

## Installation

```
pip install -r requirements.txt
```

## Core modules

| File | Contents |
| --- | --- |
| `common.py` | Linear algebra helpers, reachability matrix, lasso solver, correlation utilities |
| `synth.py` | Random directed acyclic graphs, linear Gaussian covariates, quadratic outcome |
| `oriented.py` | Ridge estimate of the identified matrix, normalization, and the ordering rules (score rule, peeling, Copeland, feedback arc set) |
| `theory_sim.py` | Self-contained NumPy version of the generator and estimator, the certified rule, and the delta-method tolerance |

## Synthetic experiments

| File | Experiment |
| --- | --- |
| `regimes.py` | Scalar readout against interventional marginals with a reduced profiling budget |
| `variants.py` | Ordering rules and their margins over the sample-permutation control |
| `ordtest.py` | Normalization exponent and the unnormalized pairwise rule |
| `baselines.py` | Variance sorting, variance explained, and column-mean baselines |
| `finite.py` | Fixed finite interventions and knockout readouts |
| `misspec.py` | Off-diagonal curvature, non-Gaussian noise, cubic outcome terms, partial knockouts |
| `frontier2.py`, `pairfrontier.py` | Noise and sample-size grid, and ratio of samples to genes |
| `fig_frontier.py` | Margin and edge-consistency figure from the output of `frontier2.py` |
| `transport2.py`, `perm_control.py` | Mechanism recovery and transport to a rewired target population |
| `theory_bound.py` | Entrywise estimation error against its finite-sample bound, and the score-rule perturbation bound |
| `theory_certified.py` | Certified rule with the data-driven tolerance: coverage, certified fraction, precision |
| `theory_per_replicate.py` | Replicate-level comparison of the certified and uncertified rules |

Every synthetic experiment uses seeded generators, so each output row is reproduced by its sample size, noise level, and replicate index.

## Cancer Dependency Map experiments

1. Download release 24Q4 from the public figshare mirror:
   ```
   python fetch_depmap_figshare.py --data data/depmap24q4
   ```
2. Download the reference networks into `ref/`: the human TRRUST v2 table as `ref/trrust_rawdata.human.tsv` (https://www.grnpedia.org/trrust/) and CollecTRI as `ref/collectri.tsv`, for example exported with the `decoupler` or `OmniPath` Python packages.
3. Build the aligned expression and gene-effect panels, which are written to `cache/`:
   ```
   python prep.py --data data/depmap24q4 --tcga TCGA_EXPRESSION.tsv --out cache
   ```
   Here `TCGA_EXPRESSION.tsv` is a tab-separated TCGA expression table.
4. Run the analyses:

| File | Analysis |
| --- | --- |
| `real_ref.py` | Recovered order against TRRUST with the sample-permutation control |
| `real_ref2.py`, `real_ref3.py` | Reference-free checks (cross-library and split-half agreement) and controls for the expression divisor |
| `real_ref4.py` | Comparison with the column-mean order |
| `real_ref5.py` | Gene labels shuffled within each cell line |
| `realrules.py` | All four ordering rules on one estimated matrix |
| `alphareal.py` | Normalization exponent on the screen |
| `lineage.py` | Lineage residualization and per-lineage fits |
| `noise_rescaled.py`, `noise_rescaled2.py` | Between-library reliability of the raw and rescaled readouts |
| `real3.py`, `real4.py` | Final real-data runs: TRRUST and CollecTRI agreement, regulator-blocked bootstrap, and power study |

The panels and reference edge lists are fixed before any order is fitted, and no quantity derived from a reference network enters the estimation.

## License

MIT, see `LICENSE`.
