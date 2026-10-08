# ProtT5 and VESPA-family extension

ProtT5 uses the official VESPA reconstruction convention and supports default
MM, conditional MM, and PLL. The checkpoint is
[`Rostlab/prot_t5_xl_uniref50` at revision `973be27c`](https://huggingface.co/Rostlab/prot_t5_xl_uniref50/tree/973be27c52ee6474de9c945952a8008aeb2a1a73).

VESPA, VESPAl, and VespaG are evaluated separately with conditional
variant-score interactions because their outputs are effect-predictor scores,
not log probabilities; PLL is therefore undefined for these predictors.
Official pinned sources are VESPA
[`34fd1ab`](https://github.com/Rostlab/VESPA/tree/34fd1ab1b7fdc2b4e1a241fdae1db5c2f09d8944)
and VespaG [`3d47582`](https://github.com/JSchlensok/VespaG/tree/3d4758252dbd423249e694d6f7d195c707f72a92).

## What is conditional for these predictors?

For mutations A and B, each model scores A in WT and B backgrounds and B in
WT and A backgrounds. VESPA and VESPAl use ProteinGym's additive aggregation;
VespaG uses ProteinGym's product aggregation. The two mutation orders are
averaged. Because current official VESPA inference does not exactly reproduce
the historical stored ProteinGym singles, the conditional change is anchored
to the stored default double score:

`conditional = stored_default + (current_conditional_path - current_default_path)`

Both direct current-model paths and anchored scores are saved, so this choice
can be audited. The fitted epistasis analysis uses the anchored score.

## Checkpoints

- ProtT5: `Rostlab/prot_t5_xl_uniref50`, revision `973be27c52ee6474de9c945952a8008aeb2a1a73`
- VespaG encoder: `facebook/esm2_t36_3B_UR50D`, revision `476b639933c8baad5ad09a60ac1a87f987b656fc`
- Predictor heads are in the pinned VESPA and VespaG repositories above.

Large checkpoints belong in `$HF_HOME` on `/data/users`; they are never
committed. Recreate the environment with `environment.yml` in this directory.

## Run

```bash
conda env create -f scripts/reanalysis/vespa_family/environment.yml
python scripts/reanalysis/vespa_family/prepare.py
python scripts/reanalysis/vespa_family/prepare_vespa.py
python scripts/reanalysis/vespa_family/prepare_vespag.py

cd results/reanalysis/tsuboyama/vespa_family/condor/full && condor_submit_dag analysis.dag
cd ../vespa_full && condor_submit_dag analysis.dag
cd ../vespag_full && condor_submit_dag analysis.dag
```

Each DAG has one GPU scoring node and one dependent CPU thermodynamic-fit node
per assay. Outputs are under `results/reanalysis/tsuboyama/vespa_family/`.
