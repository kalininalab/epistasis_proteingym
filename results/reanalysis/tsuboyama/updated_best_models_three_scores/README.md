# Updated best-model epistasis heatmaps

The figures compare three fitted-epistasis scoring procedures for the 39
best-family ProteinGym representatives across 50 Tsuboyama assays:

1. original ProteinGym-derived scoring;
2. conditional masked-marginal scoring;
3. pseudo-log-likelihood (PLL).

Rows in **bold** have at least one newly recomputed method for all 50 assays.
Gray cells mean that the method is unavailable or its full run is still in
progress. VESPA, VESPAl, and VespaG have complete conditional-MM results but
no PLL results. One assay has no valid predefined epistatic subset, so summary
counts are 50 assays for all doubles and 49 for the epistatic subset.

## Main observations

- Conditional MM is consistently more informative for fitted epistasis than
  PLL across the completed models.
- ESM3 has the strongest mean conditional-MM correlation: 0.217 across all
  doubles and 0.334 on the predefined epistatic subset.
- ESMC-300M ranks second: 0.175 and 0.295, respectively.
- The best mean PLL result is much smaller: ESM3 reaches 0.035 across all
  doubles and 0.040 on the epistatic subset.

## Files

- `three_score_epistasis_heatmap_all_doubles.{png,pdf}`
- `three_score_epistasis_heatmap_epistatic.{png,pdf}`
- `model_method_summary.csv`: assay-aggregated means and medians.
- `recomputed_model_coverage.csv`: method availability per model.
- `recomputed_per_assay_metrics.csv`: source values plotted in the new panels.

Reproduce with:

```bash
/nethome/akolchina/miniconda3/envs/protease-pipeline/bin/python \
  scripts/reanalysis/best_models_epistasis/plot_updated_three_scores.py
```
