# Executive report: mutation associations with cgreGFP fluorescence

## Executive summary

We analyzed **24,515 cgreGFP variants** containing **1,651 distinct amino-acid substitutions**. Mutations were tested individually for association with fluorescence, mutation pairs were tested for co-occurrence, and Elastic Net models measured how well the complete mutation pattern predicts fluorescence.

The strongest signals are predominantly **loss-of-fluorescence substitutions**. Mutations around positions **65–69, 97, 107, 168, 186, and 231** repeatedly show negative associations. Positive associations exist but have smaller univariate effect sizes. The models predict held-out variants well, showing that the combined mutation pattern contains substantial fluorescence information.

![Sequence-level mutation association map](cgre_sequence_mutation_heatmap.png)

## What we did

1. Converted each variant into a binary vector indicating which substitutions are present.
2. Log10-transformed fluorescence and fitted a two-component Gaussian mixture. The intersection between the two fitted modes, **log10 3.610** or **4,076 raw units**, defines active versus inactive fluorescence. This agrees with **95.7%** of the existing ProteinGym binary labels.
3. For each mutation, calculated an active/inactive odds ratio and Fisher exact test, plus point-biserial and Spearman correlations with continuous fluorescence. Each test family was FDR-corrected with Benjamini–Hochberg.
4. For mutation pairs observed often enough for stable analysis, calculated phi correlation, Fisher exact p-values, and global FDR q-values.
5. Evaluated Elastic Net logistic and linear models with nested **5-fold outer / 3-fold inner cross-validation**. All reported performance is based on out-of-fold predictions.

## Main findings

### Individual mutations

Among the **1,123 substitutions observed in at least 10 variants**, **577** have a Spearman FDR q-value below 0.05 and **509** have a Fisher FDR q-value below 0.05. Statistical significance is common because the dataset is large; correlation magnitude and recurrence across substitutions at a position are more useful for prioritization.

Strongest negative associations: **R97C** (ρ=-0.121), **K168E** (ρ=-0.116), **Y69C** (ρ=-0.095), **R97S** (ρ=-0.091), **S65P** (ρ=-0.087), **Y69H** (ρ=-0.087).

Strongest positive associations: **L198P** (ρ=0.039), **Y144N** (ρ=0.035), **N117D** (ρ=0.034), **I153V** (ρ=0.033), **A169V** (ρ=0.031), **K163R** (ρ=0.030).

The strongest positive correlation is only **ρ=0.039**, compared with **ρ=-0.121** for the strongest negative association. This library therefore provides clearer evidence for fluorescence-disrupting mutations than for individually beneficial mutations.

### Mutation–mutation associations

We tested **630,003 mutation pairs** among 1,123 substitutions observed at least 10 times. Only **3 pairs** meet the prespecified strong-pair definition, |phi| ≥ 0.5 and Fisher q < 0.05. They connect **T233R, Y234T, and R235A** (phi 0.83–0.96), indicating a tightly linked C-terminal mutation block. This likely reflects library design or jointly introduced variants and should not automatically be interpreted as biochemical interaction.

![Strongest mutation-pair correlations](mutation_mutation_phi_heatmap_strongest_pairs.png)

### Multivariate prediction

- Active/inactive Elastic Net ROC-AUC: **0.962**
- Continuous log10-fluorescence Elastic Net R²: **0.601**
- Continuous prediction Spearman ρ: **0.848**

![Elastic Net out-of-fold predictions and coefficients](elastic_net_regression.png)

These values show strong predictive signal, but they do not prove that every selected coefficient is causal. Linked mutations, unequal library sampling, and epistasis can redistribute coefficients between correlated substitutions.

## Recommended interpretation

- Use the sequence map to prioritize positions with repeated negative signals across several substitutions.
- Treat the positive mutations as candidates for experimental validation rather than confirmed brightness improvements.
- Validate candidate substitutions individually or in controlled backgrounds before making mechanistic claims.
- For publication, describe these results as **associations in a combinatorial library**, not single-mutation causal effects.

## Files

- `cgre_sequence_mutation_heatmap.png`: sequence-level substitution map.
- `position_association_summary.csv`: strongest association and tested substitutions at every sequence position.
- `mutation_fluorescence_associations.csv`: complete univariate mutation results.
- `mutation_mutation_associations.csv.gz`: complete pairwise results.
- `elastic_net_mutation_coefficients.csv` and `elastic_net_oof_predictions.csv`: multivariate results.
- `summary.json`: machine-readable settings and performance.

The reproducible analysis commands are documented in `README.md`.
