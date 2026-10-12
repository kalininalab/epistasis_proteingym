# Mutation association analysis: D7PM05 CLYGR

For a concise, presentation-ready interpretation, see [`BOSS_REPORT.md`](BOSS_REPORT.md). The report includes the sequence-level cgreGFP association map and the main findings in plain language.

Analyzed **24,515 variants** and **1,651 distinct substitutions**. Continuous analyses use log10 fluorescence. Activity was defined by the intersection of a two-component Gaussian mixture fitted to log10 fluorescence: **3.610** (**4,076.2 raw units**), producing 12,999 active and 11,516 inactive variants. The existing ProteinGym label instead changes near 12,500 raw units; agreement with the mixture label is 0.957.

## Predictive evaluation

Performance uses nested 5-fold outer / 3-fold inner cross-validation. Hyperparameters are tuned only inside each training fold.

- Elastic Net logistic ROC-AUC: **0.962**
- Elastic Net linear R²: **0.601**
- Elastic Net linear Spearman ρ: **0.848**

## Outputs

- `mutation_fluorescence_associations.csv`: frequency, odds ratio, Fisher p/q, point-biserial r/p/q, and Spearman rho/p/q for every mutation.
- `mutation_mutation_associations.csv.gz`: pairwise phi, Fisher p/q, frequencies, and co-occurrences for mutations observed at least 10 times.
- `strong_mutation_pairs.csv`: pairs with |phi| ≥ 0.5 and Fisher q < 0.05.
- `elastic_net_mutation_coefficients.csv`: fitted standardized coefficients and univariate statistics.
- `elastic_net_oof_predictions.csv`: out-of-fold predictions used for performance estimates.
- `variant_activity_labels.csv`: reproducible per-variant log10 fluorescence and activity label.
- PNG figures show the threshold, mutation associations, mutation-pair heatmap, and regression predictions/coefficients.

Association does not by itself establish a causal mutation effect: library design, linked mutations, and epistasis can produce correlations. Elastic Net coefficients are conditional associations within this dataset.

## Reproduce

```bash
/nethome/akolchina/miniconda3/envs/protease-pipeline/bin/python \
  scripts/somermeyer/mutation_association.py \
  --input data/somermeyer/D7PM05_CLYGR_Somermeyer_2022.csv \
  --output results/reanalysis/somermeyer/D7PM05_CLYGR_mutation_associations
```

Use `--threshold-log10 VALUE` to override the mixture-derived activity threshold. `--resume` reuses completed association tables when only the multivariate analysis or plots need to be regenerated.

Generate the sequence map, position summary, validated wild-type FASTA, and executive report with:

```bash
/nethome/akolchina/miniconda3/envs/protease-pipeline/bin/python \
  scripts/somermeyer/plot_sequence_mutation_associations.py \
  --data data/somermeyer/D7PM05_CLYGR_Somermeyer_2022.csv \
  --results results/reanalysis/somermeyer/D7PM05_CLYGR_mutation_associations
```
