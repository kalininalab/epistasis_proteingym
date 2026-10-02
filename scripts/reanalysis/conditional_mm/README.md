# Conditional masked-marginal ESM2-650M analysis

This isolated workflow compares ProteinGym's default ESM2-650M masked-marginal
score with symmetric conditional masked marginals on every Tsuboyama assay used
by the project. It does not overwrite `data/tsuboyama` or the established
`analysis2/raw_unclipped` results.

For a double mutant `A:B`, `score_all.py` masks only A while B is present, then
masks only B while A is present. It saves all four mutation deltas, both
directional interactions, and their symmetric mean `E_cond`. Conditional
fitness is defined directly from the saved components as
`0.5 * [(delta_A_WT + delta_B_A) + (delta_B_WT + delta_A_B)]`. This avoids
mixing the stored fair-esm ProteinGym score with new Hugging Face FP16 forward
passes. Although the formulas are algebraically compatible, the two inference
implementations are not bit-identical.

`fit.py` sends conditional fitness through the same implementation used by
`tsuboyama_analysis2.py`: a Bayesian additive model is fitted separately for
each residue pair and coupling is raw score minus posterior-median additive
reconstruction. Priors, seed, NUTS sampling, and unclipped predictions are
unchanged.

Run from the repository root:

```bash
/nethome/akolchina/miniconda3/envs/ofs/bin/python scripts/reanalysis/conditional_mm/prepare.py
ssh -o BatchMode=yes lsv-submit \
  'cd /data/users/akolchina/epistasis_proteingym && condor_submit_dag results/reanalysis/tsuboyama/conditional_mm_esm2_650m/condor/analysis.dag'
/nethome/akolchina/miniconda3/envs/epi_env/bin/python scripts/reanalysis/conditional_mm/finalize.py
```

The DAG uses 10 GPU scoring shards followed by one CPU fit per assay. Completed
score and fit files are skipped. The final report is
`results/reanalysis/tsuboyama/conditional_mm_esm2_650m/REPORT.md`.
