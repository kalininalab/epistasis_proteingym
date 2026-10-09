#!/usr/bin/env bash
set -euo pipefail
R=/data/users/akolchina/epistasis_proteingym
export HF_HOME=/data/users/akolchina/huggingface TORCH_HOME=/data/users/akolchina/torch TOKENIZERS_PARALLELISM=false
mode=$1; dataset=$2; model=$3; namespace=$4; limit=${5:-none}
slug=$(printf %s "$model" | tr '[:upper:]' '[:lower:]' | tr -c 'a-z0-9' '_')
out="$R/results/reanalysis/tsuboyama/next_sitewise_mm_pll/$namespace/$slug/$dataset"
if [[ $mode == score ]]; then
  extra=(); [[ $limit != none ]] && extra=(--max-doubles "$limit")
  if [[ $model == xTrimoPGLM-10B-MLM ]]; then
    exec /data/users/akolchina/envs/esm_modern310/bin/python -u "$R/scripts/reanalysis/next_sitewise_mm_pll/score_xtrimo.py" --dataset "$dataset" --output "$out" --batch-size 2 "${extra[@]}"
  elif [[ $model == MSA_Transformer_ensemble ]]; then
    exec /nethome/akolchina/miniconda3/envs/ofs/bin/python -u "$R/scripts/reanalysis/next_sitewise_mm_pll/score_msa_transformer.py" --dataset "$dataset" --output "$out" --batch-size 1 "${extra[@]}"
  elif [[ $model == MULAN_small ]]; then
    export PYTHONPATH=/data/users/akolchina/software/ProteinGym/proteingym/baselines/mulan:/data/users/akolchina/software/mulan_deps:${PYTHONPATH:-}
    exec /data/users/akolchina/envs/esm_modern310/bin/python -u "$R/scripts/reanalysis/next_sitewise_mm_pll/score_mulan.py" --dataset "$dataset" --output "$out" --batch-size 16 "${extra[@]}"
  else
    echo "Unknown model: $model" >&2; exit 2
  fi
else
  export JAX_PLATFORM_NAME=cpu XLA_FLAGS='--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1'
  exec /nethome/akolchina/miniconda3/envs/protease-pipeline/bin/python -u "$R/scripts/reanalysis/multimodel_mm_pll/fit.py" --scores "$out/scores.csv" --output "$out/fits"
fi
