# Conditional score-definition audit

No ESM inference was run. This audit uses the four saved conditional masked-marginal components for all 51,924 double mutants in 50 assays.

The ProteinGym ESM implementation masks each WT position independently, then `label_row` sums `log p(mut) - log p(WT)` over the substitutions. Its exact double-mutant definition is therefore:

`S_default_AB = ΔA_WT + ΔB_WT`.

The conditional score is now defined directly as:

`S_cond_direct = 0.5 * [(ΔA_WT + ΔB_A) + (ΔB_WT + ΔA_B)]`.

Using components from the same inference, this equals `(ΔA_WT + ΔB_WT) + E_cond` to floating-point precision: maximum absolute identity error **3.55e-15**.

The formerly fitted construction mixed the stored ProteinGym score with newly computed Hugging Face FP16 components: `S_default_stored + E_cond`. Compared with `S_cond_direct`, its maximum absolute difference is **0.048354149** log-score units, mean absolute difference **0.005852384**, and RMSE **0.007883523**.

The formulas are compatible, but the implementations are not bit-identical. Future fits use `S_cond_direct`; existing fitted outputs remain untouched and are explicitly treated as the former construction.
