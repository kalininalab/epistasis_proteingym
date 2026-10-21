# ProtSSN conditional-MM and PLL

Runs the representative ProteinGym model `ProtSSN_k30_h1280` on the 50
Tsuboyama double-mutant assays. It uses the official ProteinGym ProtSSN code,
`protssn_k30_h1280.pt`, ESM2-650M sequence embeddings, ProteinGym AF2
structures, the official k=30 graph normalization, and the shared additive
thermodynamic-fit workflow.

When scoring residue `i`, only `i` is masked in the ESM input. The fixed AF2
structure is passed to the ProtSSN graph network; the other mutation remains
in the sequence for conditional scoring. PLL masks every sequence position in
turn. Checkpoint: https://huggingface.co/tyang816/ProtSSN/blob/main/protssn_k30_h1280.pt

The local scorer contains two compatibility adapters for the current inference
environment: ProteinGym imports `seaborn` only for unused plotting utilities,
and official ProtSSN uses the pre-PyG-2.5 `BaseTransform.__call__` and
`Inspector.distribute` APIs. The adapters preserve the original transform and
message-passing calculations; they only bridge renamed framework interfaces.

```bash
scripts/reanalysis/protssn_mm_pll/prepare.py --pilot
cd results/reanalysis/tsuboyama/protssn_mm_pll/condor/pilot
condor_submit_dag -maxjobs 1 analysis.dag
```

Omit `--pilot` for all assays. Outputs are stored separately under
`results/reanalysis/tsuboyama/protssn_mm_pll/`.
