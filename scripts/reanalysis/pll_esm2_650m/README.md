# ESM2-650M pseudo-log-likelihood scoring

For each sequence `S`, `score.py` masks one residue at a time and sums the log
probability assigned to the observed residue:

`PLL(S) = sum_i log p(s_i | S with i masked)`.

Every unique sequence is scored once per assay and persisted in
`sequence_pll_cache.csv`. The double-mutant output retains `PLL_WT`, `PLL_A`,
`PLL_B`, `PLL_AB`, and `epsilon_PLL = PLL_AB - PLL_A - PLL_B + PLL_WT`.

`fit.py` uses `PLL_AB` as the double-mutant score, fits the same Bayesian
additive model used by the Tsuboyama conditional-MM analysis within every
position-pair matrix, and saves `PLL_AB - fitted_additive_PLL` as predicted
epistasis.

The pilot and its measured runtime are documented under
`results/reanalysis/tsuboyama/pll_esm2_650m_pilot/REPORT.md`. The full assay run
must not be submitted until that pilot runtime has been reviewed.
