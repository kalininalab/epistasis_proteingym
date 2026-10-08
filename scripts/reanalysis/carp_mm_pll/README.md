# CARP conditional-MM and PLL

Scores all prepared Tsuboyama double-mutant assays with CARP-76M, then uses the shared thermodynamic/additive fitting workflow.

Reproducibility:
- Code: `https://github.com/microsoft/protein-sequence-models`, commit `af695772c4a1c056d930c95ec7e6428aa042f5cd`
- Checkpoint: `https://zenodo.org/record/6564798/files/carp_76M.pt?download=1`
- Local checkpoint path is configured in `run.sh`; downloaded weights and outputs remain under `/data/users` and are not committed.

Run:
```bash
scripts/reanalysis/carp_mm_pll/prepare.py --pilot  # omit --pilot for all assays
cd results/reanalysis/tsuboyama/carp_mm_pll/condor/pilot
condor_submit_dag -maxjobs 50 analysis.dag
```
