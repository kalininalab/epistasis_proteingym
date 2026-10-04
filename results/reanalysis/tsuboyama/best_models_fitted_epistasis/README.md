# Best ProteinGym models: fitted epistasis by Tsuboyama assay

This analysis uses the representative models listed in
`tsuboyama_best_models.csv`, across all scoring categories. For each model and
assay, the existing Bayesian additive fit is used to define predicted
epistasis as the raw model score minus its posterior-median additive
reconstruction. Spearman correlation is calculated against experimental
thermodynamic coupling for:

- all valid double mutants;
- the predefined epistatic subset.

The analysis includes **39 models from 36 families across 50 assays**:
24 sitewise models, 14 joint models, and one explicit-pairwise model. The
`Linear_regression` and `MLP` rows are listed in `excluded_best_models.csv`:
they are project-specific user models and have no columns in the existing
ProteinGym Tsuboyama prediction files.

## Figures

- `by_dataset/`: one plot per assay, with models on the x axis.
- `by_model/`: one plot per model, with assays on the x axis.
- `by_dataset_all_models.pdf`: all 50 assay plots in one PDF.
- `by_model_all_datasets.pdf`: all 39 model plots in one PDF.
- `overview_heatmap_all_doubles.*` and `overview_heatmap_epistatic.*`:
  compact overviews of every model-assay correlation.

Blue bars show all double mutants; pink bars show the predefined epistatic
subset. Every bar is an assay-level Spearman correlation computed on fitted
epistasis residuals, not on raw fitness scores. All bar plots use the same
fixed y-axis range of -1 to 1.

## Tables

- `per_assay_model_correlations.csv`: plotted correlations, sample counts,
  fit diagnostics, model family, and scoring category.
- `model_summary.csv`: mean and median assay-level correlation per model.
- `excluded_best_models.csv`: requested best-model rows without existing
  ProteinGym predictions.

## Reproduce

The source fits were already complete, so this analysis required no new model
inference or Condor jobs. The plotting script supports bounded batches:

```bash
/nethome/akolchina/miniconda3/envs/protease-pipeline/bin/python \
  scripts/reanalysis/best_models_epistasis/plot.py --mode dataset --start 0 --end 50

/nethome/akolchina/miniconda3/envs/protease-pipeline/bin/python \
  scripts/reanalysis/best_models_epistasis/plot.py --mode model --start 0 --end 39

/nethome/akolchina/miniconda3/envs/protease-pipeline/bin/python \
  scripts/reanalysis/best_models_epistasis/plot.py --mode finish
```
