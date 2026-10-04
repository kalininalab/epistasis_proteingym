# ESM2-650M epistasis comparison: masked marginals, conditional MM, and PLL

Compared **50 Tsuboyama assays** and **51,924 double mutants** with identical assay/variant filtering. The predefined epistatic subset is evaluable in 49 assays.

## Main result (mean assay-level Spearman ρ)

| Method | All doubles | Epistatic subset | Δρ vs original on epistatic subset | Improved / worse assays | Paired p |
|---|---:|---:|---:|---:|---:|
| Original ProteinGym | -0.002 | -0.028 | +0.000 | 0 / 0 | — |
| Direct E_cond | 0.036 | 0.067 | +0.095 | 37 / 12 | 0.00025 |
| Fitted conditional MM | 0.152 | 0.215 | +0.243 | 41 / 8 | 9.77e-08 |
| Direct epsilon_PLL | 0.028 | 0.061 | +0.089 | 34 / 15 | 0.0123 |
| Fitted PLL(AB) | 0.007 | -0.009 | +0.019 | 27 / 21 | 0.34 |

## Conclusion

Corrected fitted conditional MM is the strongest of the tested ESM2-650M epistasis scores. On the epistatic subset it exceeds fitted `PLL(AB)` by mean Δρ +0.224 (42/49 assays; paired p=9.05e-08). Direct conditional interaction and direct PLL inclusion–exclusion both carry modest positive signal; direct `E_cond` is slightly better than direct `epsilon_PLL` by mean Δρ +0.006 (34/49 assays; p=0.0256). Fitting an additive model to raw `PLL(AB)` does not improve epistasis prediction over the original ProteinGym-derived baseline. Full-sequence PLL therefore does not justify its much larger inference cost for this task.

## Runtime and diagnostics

PLL inference consumed **21.35 aggregate GPU-hours** (21.80 including model loading) for 3,351,842 masked examples. The 10-GPU DAG completed in about four wall-clock hours. PLL additive fits contained **0 NUTS divergences** across 187 position-pair fits.

## Files

- `tables/per_assay_correlations.csv`: all five methods per assay and subset.
- `tables/aggregate_summary.csv`: aggregate statistics versus original.
- `tables/paired_method_comparisons.csv`: direct paired comparisons among conditional and PLL methods.
- `figures/all_methods_assay_spearman.png`: complete method comparison.
- `figures/original_vs_corrected_conditional_paired.png`: requested paired assay plot.
- `tables/pll_runtime_per_assay.csv`: measured PLL runtime by assay.
