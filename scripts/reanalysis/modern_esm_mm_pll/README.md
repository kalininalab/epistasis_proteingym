# ESMC/ESM3 conditional-MM and PLL

Scores all prepared Tsuboyama double-mutant assays with ESMC-300M and ESM3-open-small. Each variant receives directional conditional masked-marginal components and full-sequence PLL values for WT, both singles, and the double; the shared fit stage then computes the thermodynamic/additive residual analysis.

Reproducibility:
- Python package: `esm==3.2.1`
- ESMC checkpoint: `EvolutionaryScale/esmc-300m-2024-12`, revision `7f10b20ae75017b2dbc884070e03434515709a8d`
- ESM3 checkpoint: `EvolutionaryScale/esm3-sm-open-v1`, revision `47f0545b2b6daf26a93439a3cd610f4f7f3d5478`
- Set `HF_HOME=/data/users/akolchina/huggingface`; checkpoints and outputs remain under `/data/users` and are not committed.
- Jobs require `GPUs_Capability >= 8.0` because the official loaders use bfloat16.

Run:
```bash
scripts/reanalysis/modern_esm_mm_pll/prepare.py --pilot  # omit --pilot for all assays
cd results/reanalysis/tsuboyama/modern_esm_mm_pll/condor/pilot
condor_submit_dag -maxjobs 2 analysis.dag
```
