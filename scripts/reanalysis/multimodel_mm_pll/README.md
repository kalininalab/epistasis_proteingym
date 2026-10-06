# Multimodel conditional-MM and PLL analysis

This workflow extends the completed ESM2-650M experiment to the representative
masked ESM models in `tsuboyama_best_models.csv`:

- ESM1b;
- ESM1v ensemble (five component checkpoints, averaged before fitting);
- ESM2-150M.

ESM2-650M is already complete and is not recomputed. `models.csv` tracks the
implemented models and the next ESM-family adapters. The production DAG has
350 GPU scoring jobs and at most **50 run concurrently**. Each scoring job
computes conditional MM and cached PLL together; dependent CPU jobs aggregate
ESM1v and fit the established additive model.

## Reproducible model weights

Weights are external inputs and are never committed to Git. Condor reads them
from `/data/users/akolchina/.cache/huggingface`; `/nethome` must not be used for
model caches. Every revision is pinned in `models.csv`.

Official sources:

- [ESM1b](https://huggingface.co/facebook/esm1b_t33_650M_UR50S/tree/7b37824baec4d3658e1df7479222a7c79b465b76)
- [ESM1v component 1](https://huggingface.co/facebook/esm1v_t33_650M_UR90S_1/tree/8bfdb1892536cc77bd0760b9c25ddced2cd0b4c8)
- [ESM1v component 2](https://huggingface.co/facebook/esm1v_t33_650M_UR90S_2/tree/3c1e9e64480f069b163e456c361bfd80a8bab04c)
- [ESM1v component 3](https://huggingface.co/facebook/esm1v_t33_650M_UR90S_3/tree/0b00fd112e63f6b5e70a9cd8484d4e660312ce70)
- [ESM1v component 4](https://huggingface.co/facebook/esm1v_t33_650M_UR90S_4/tree/443968f644da132d323bbc6321a6d443149fa57b)
- [ESM1v component 5](https://huggingface.co/facebook/esm1v_t33_650M_UR90S_5/tree/fb2e51cb0f605cbad2c4ca3bb784be7bc8e2f4a8)
- [ESM2-150M](https://huggingface.co/facebook/esm2_t30_150M_UR50D/tree/a695f6045e2e32885fa60af20c13cb35398ce30c)
- [ESM2-650M](https://huggingface.co/facebook/esm2_t33_650M_UR50D/tree/08e4846e537177426273712802403f7ba8261b6c)

Download or restore the pinned weights directly to `/data/users`:

```bash
export HF_HOME=/data/users/akolchina/.cache/huggingface
/nethome/akolchina/miniconda3/envs/ofs/bin/python \
  scripts/reanalysis/multimodel_mm_pll/download_models.py
```

## Run

Prepare the isolated eight-variant pilot and the full DAG:

```bash
/nethome/akolchina/miniconda3/envs/protease-pipeline/bin/python \
  scripts/reanalysis/multimodel_mm_pll/prepare.py --pilot
/nethome/akolchina/miniconda3/envs/protease-pipeline/bin/python \
  scripts/reanalysis/multimodel_mm_pll/prepare.py
```

Submit the pilot first. Submit the full DAG only after every pilot score,
ensemble aggregation, and additive fit succeeds. Pilot outputs are isolated
under `multimodel_mm_pll/pilot/` and cannot be mistaken for full results.

## Saved quantities

Every double mutant retains the four conditional components, `E_cond`, the
complete `S_cond_direct`, `PLL_WT`, `PLL_A`, `PLL_B`, `PLL_AB`, and
`epsilon_PLL`. The fit stage saves conditional-MM and PLL additive
reconstructions, residual epistasis, diagnostics, and correlations for all
doubles and the predefined epistatic subset.

## Final comparisons

`results/reanalysis/tsuboyama/multimodel_mm_pll/summary/` contains the unified
assay-level tables. `three_score_barplots/<model>/` contains matching grouped
barplots for raw double-mutant fitness and fitted epistatic residuals: three
bars per assay for default MM, conditional MM, and PLL.
