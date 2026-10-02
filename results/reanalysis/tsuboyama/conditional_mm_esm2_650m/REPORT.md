# Conditional masked-marginal ESM2-650M across Tsuboyama assays

Analyzed **50 assays** and **51,924 double mutants**. Conditional scoring masks only the residue being scored; the other mutation remains present. Epistasis is evaluated using the established Bayesian thermodynamic procedure: an independent additive model is fitted within each residue pair, and predicted coupling is the raw model score minus its posterior-median additive reconstruction. Predictions are not clipped. The original Normal(0.1, 3) effect priors, Exponential(1) noise prior, seed 1, 100 warmup samples, 50 posterior samples, and one NUTS chain are unchanged.

## Main results

| Task | Subset | Assays | Default mean ρ | Conditional mean ρ | Mean Δρ (95% bootstrap CI) | Improved / worse | Paired Wilcoxon p |
|---|---:|---:|---:|---:|---:|---:|---:|
| fitness | all_doubles | 50 | 0.227 | 0.267 | +0.040 [+0.019, +0.064] | 33 / 17 | 0.000726 |
| epistasis | all_doubles | 50 | -0.002 | 0.153 | +0.155 [+0.117, +0.193] | 42 / 8 | 5.32e-10 |
| fitness | epistatic | 49 | 0.236 | 0.297 | +0.061 [+0.038, +0.085] | 36 / 12 | 5.27e-06 |
| epistasis | epistatic | 49 | -0.028 | 0.215 | +0.242 [+0.159, +0.324] | 41 / 8 | 1.66e-07 |

## Conclusion

Across the predefined epistatic variants, conditional MM **improves epistasis prediction** relative to default MM (mean Δρ +0.242; paired Wilcoxon p=1.66e-07). This conclusion is based on paired assay-level correlations, so large assays do not dominate the result. Conditional MM improved 41 of 49 evaluable assays, but it is not uniformly better; eight assays worsened.

## Diagnostics and interpretation

The conditional fits contain 21 NUTS divergences across 187 independently fitted residue pairs; six pair fits had at least one divergence. These are retained to match the established analysis. The low sample count and occasional divergences limit precise interpretation of individual assays, while the broad paired improvement across assays supports the aggregate conclusion.

The direct `E_cond` column is retained for mechanistic inspection, but it is not substituted for the project’s established fitted-coupling definition in the headline epistasis result.

## Files

- `tables/per_assay_metrics.csv`: every assay/task/subset result.
- `tables/summary.csv`: aggregate paired statistics.
- `figures/paired_assay_spearman.png`: paired assay comparison.
- `figures/delta_spearman_distributions.png`: distribution of assay-level changes.
- `scores/*.csv`: all requested directional intermediate quantities.
- `fits/*/couplings.csv`: conditional Bayesian additive reconstructions and couplings.
