# Updated best-model epistasis heatmaps

Six separate heatmaps compare three scoring procedures on all double mutants and on the predefined epistatic subset:

- `original_epistasis_heatmap_*.png`: original ProteinGym scores for every model.
- `conditional_mm_epistasis_heatmap_*.png`: conditional masked-marginal scores for sitewise models.
- `pll_epistasis_heatmap_*.png`: pseudo-log-likelihood scores for sitewise models.

In the conditional-MM and PLL figures, joint models are shown first and retain their original ProteinGym scores. The explicit-pairwise model also retains its original score. Horizontal lines separate joint, sitewise, and explicit-pairwise models. Bold model names mark models for which new scores were recomputed; gray cells indicate unavailable or incomplete results.

Each cell is Spearman correlation between predicted and experimental thermodynamic-coupling residuals for one Tsuboyama assay. PDF versions are provided beside every PNG. Supporting values are in `recomputed_per_assay_metrics.csv`, `model_method_summary.csv`, and `recomputed_model_coverage.csv`.

## Main observations

- Conditional MM is consistently more informative for fitted epistasis than PLL across the completed models.
- ESM3 has the strongest mean conditional-MM correlation: 0.217 across all doubles and 0.334 on the predefined epistatic subset.
- ESMC-300M ranks second: 0.175 and 0.295, respectively.
- ProtSSN conditional MM reaches mean rho 0.121 across all doubles and 0.168 on the epistatic subset; its fitted PLL correlations are approximately zero on average.
- The best mean PLL result is much smaller: ESM3 reaches 0.035 across all doubles and 0.040 on the epistatic subset.

Reproduce with:

```bash
/nethome/akolchina/miniconda3/envs/protease-pipeline/bin/python   scripts/reanalysis/best_models_epistasis/plot_updated_three_scores.py
```
