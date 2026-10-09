# Next sitewise conditional-MM and PLL models

This workflow extends the 50-assay Tsuboyama analysis to the remaining
representative sitewise models from `tsuboyama_best_models.csv`.

- MSA Transformer uses `esm_msa1b_t12_100M_UR50S` and ProteinGym's exact
  five MSA subsampling seeds (1--5), 400 sequences per seed. Conditional-MM
  and PLL keep the sampled homologs fixed and change only the query sequence.
- xTrimoPGLM-10B-MLM uses `biomap-research/proteinglm-10b-mlm`, revision
  `dc34acccaa493211e55bd2deeecdfc11b5d8a93f`, in bfloat16. It is restricted
  to GPUs with at least 30 GB memory.
- MULAN-small uses `DFrolova/MULAN-small`, revision
  `a5c39a6df528d292a702540af2c7230b36f5c545`. The experimental sequence is
  changed while the assay structure remains fixed; only the scored residue and
  its structural inputs are masked.

MSAs and weights come from ProteinGym v1.3, Zenodo record `15293562`.
Checkpoints and data live under `/data/users/akolchina`; they are not committed.

```bash
python scripts/reanalysis/next_sitewise_mm_pll/prepare.py --pilot
cd results/reanalysis/tsuboyama/next_sitewise_mm_pll/condor/pilot
condor_submit_dag -maxjobs 2 analysis.dag
```

Pilot jobs use eight double mutants. Full jobs omit `--pilot` and retain a
maximum of ten simultaneous GPU jobs. Every output preserves conditional-MM
components, sequence PLL values, runtime, and the shared fitted epistasis
metrics.
