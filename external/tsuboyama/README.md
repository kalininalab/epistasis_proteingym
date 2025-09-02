# Tsuboyama module (external)

Purpose: reproduce ΔG reconstruction and thresholds used in Tsuboyama et al.

Environment:
conda env create -f protease-pipeline.yml
protease pipeline, jaxlib version and sklearn was changed as it didnt work
conda activate prote FIIIIX

Run:
python scripts/run_tsuboyama.py --input data/... --out results/tables/tsuboyama_...

Attribution:
Code adapted from [<paper/repo>](https://github.com/Rocklin-Lab/cdna-display-proteolysis-pipeline/blob/main/Pipeline_figure_model/Additive_model_Fig4.ipynb). License: <MIT/GPL/...>. See LICENSE or link.
