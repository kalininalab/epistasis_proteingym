# Direct conditional-MM validation

Validated **50 assays** and **51,924 double mutants**. No previous results were overwritten.

## Per-assay epistasis results

The complete table reports original ProteinGym-derived fitted epistasis, fitted conditional-MM epistasis, direct `E_cond`, and `rho_conditional - rho_original` for both all doubles and the predefined epistatic subset.

| Subset | Original mean ρ | Fitted conditional mean ρ | Direct E_cond mean ρ | Fitted Δρ | Paired p |
|---|---:|---:|---:|---:|---:|
| All doubles (50 assays) | -0.002 | 0.153 | 0.036 | +0.155 | 5.32e-10 |
| Epistatic (49 assays) | -0.028 | 0.215 | 0.067 | +0.242 | 1.66e-07 |

Direct `E_cond` improves over original epistasis in **37/49** evaluable epistatic-subset assays (mean Δρ +0.095; paired p=0.00025). Its mean ρ is lower than fitted conditional MM, showing that the additive fit removes main-effect structure that remains in the direct conditional score.

## Score compatibility

ProteinGym ESM2 masked-marginal scores and the four conditional terms use the same natural-log probability difference, `log p(mutant residue) - log p(WT residue)`. Larger values therefore have the same sign convention. The stored double-mutant score is the sum of the two WT-background masked marginals.

Across all variants, stored versus recomputed default scores have Spearman **0.9999993**, slope **0.999882**, and mean absolute difference **0.0059** log-score units, compared with default-score SD **6.59**. The small discrepancy is consistent with mixed-precision inference and CSV precision.

Algebraically, `S_default + E_cond` equals the mean of the two mutation-order paths:

`0.5 * [(ΔA_WT + ΔB_A) + (ΔB_WT + ΔA_B)]`.

Thus adding `E_cond` is mathematically compatible: it replaces the purely WT-background additive path with the symmetric average conditional path. The analysis anchors this correction to the stored ProteinGym score to avoid treating small numerical recomputation differences as biological interaction.

## Outputs

- `tables/per_assay_epistasis_correlations.csv`
- `tables/aggregate_summary.csv`
- `tables/score_compatibility.csv`
- `figures/original_vs_conditional_paired.png`
