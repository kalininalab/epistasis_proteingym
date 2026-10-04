# ESM2-650M scoring comparison by Tsuboyama assay

The figures compare three scores on every Tsuboyama double-mutant assay:

- **Default MM:** the original ProteinGym masked-marginal score.
- **Conditional MM:** `S_cond_direct`, which conditions each mutation on the other.
- **PLL:** pseudo-log-likelihood of the complete double-mutant sequence, `PLL_AB`.

`raw_fitness_three_scores` reports Spearman correlation between each raw score
and experimental `dG`. `fitted_epistasis_three_scores` reports Spearman
correlation between the score's fitted additive residual and the experimental
thermodynamic coupling. Each figure shows all double mutants and the predefined
epistatic subset in separate panels. All methods use identical variants within
each assay and panel.

The analysis contains 51,924 double mutants across 50 assays. The epistatic
subset contains 7,285 variants and has a defined correlation for 49 assays.

Files:

- `per_assay_correlations.csv`: every plotted value and its sample size.
- `summary.csv`: mean and median assay-level correlations.
- PNG files: slide-ready raster figures.
- PDF files: vector figures.

Reproduce from the repository root:

```bash
/nethome/akolchina/miniconda3/envs/protease-pipeline/bin/python \
  scripts/reanalysis/three_scoring_comparison/plot.py
```
