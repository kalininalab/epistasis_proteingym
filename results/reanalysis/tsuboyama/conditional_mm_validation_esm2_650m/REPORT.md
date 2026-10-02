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

The ProteinGym implementation constructs a WT-context masked-marginal table by masking each sequence position independently. Its `label_row` function loops through all substitutions in a variant and sums `log p(mutant residue) - log p(WT residue)`. Thus, for a double mutant, the ProteinGym formula is exactly `S_default_AB = ΔA_WT + ΔB_WT`. Larger values use the same sign convention as our components.

Across all variants, stored ProteinGym versus newly recomputed default scores have Spearman **0.9999993**, slope **0.999882**, and mean absolute difference **0.0059** log-score units, compared with default-score SD **6.59**. The formula is identical; the numerical discrepancy comes from a separate Hugging Face FP16 recomputation rather than the original ProteinGym fair-esm inference.

We define the conditional score directly as the mean of the two mutation-order paths:

`0.5 * [(ΔA_WT + ΔB_A) + (ΔB_WT + ΔA_B)]`.

By algebra, this direct score also equals `(ΔA_WT + ΔB_WT) + E_cond` when every term comes from the same forward-pass implementation. It should not be constructed by mixing the stored ProteinGym score with newly recomputed `E_cond`, because the two implementations have small numerical differences.

## Outputs

- `tables/per_assay_epistasis_correlations.csv`
- `tables/aggregate_summary.csv`
- `tables/score_compatibility.csv`
- `figures/original_vs_conditional_paired.png`
