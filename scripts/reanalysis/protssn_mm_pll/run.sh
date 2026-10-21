#!/usr/bin/env bash
set -euo pipefail
R=/data/users/akolchina/epistasis_proteingym
export HF_HOME=/data/users/akolchina/huggingface TORCH_HOME=/data/users/akolchina/torch TOKENIZERS_PARALLELISM=false
mode=$1; ds=$2; namespace=$3; limit=${4:-none}; out="$R/results/reanalysis/tsuboyama/protssn_mm_pll/$namespace/protssn_k30_h1280/$ds"
if [[ $mode == score ]]; then
  extra=(); [[ $limit != none ]] && extra=(--max-doubles "$limit")
  exec /data/users/akolchina/envs/esm_modern310/bin/python -u "$R/scripts/reanalysis/protssn_mm_pll/score.py" --dataset "$ds" --output "$out" --batch-size 4 "${extra[@]}"
else
  export JAX_PLATFORM_NAME=cpu XLA_FLAGS='--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1'
  exec /nethome/akolchina/miniconda3/envs/protease-pipeline/bin/python -u "$R/scripts/reanalysis/multimodel_mm_pll/fit.py" --scores "$out/scores.csv" --output "$out/fits"
fi
