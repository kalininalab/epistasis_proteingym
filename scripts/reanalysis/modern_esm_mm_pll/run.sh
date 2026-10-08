#!/usr/bin/env bash
set -euo pipefail
ROOT=/data/users/akolchina/epistasis_proteingym
ENV=/data/users/akolchina/envs/esm_modern310
export HF_HOME=/data/users/akolchina/huggingface TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$ENV/lib/python3.10/site-packages:${PYTHONPATH:-}"
mode=$1 dataset=$2 model=$3 namespace=$4 limit=${5:-none}
slug=$(printf '%s' "$model"|tr '[:upper:]' '[:lower:]'|tr -c 'a-z0-9' '_');out="$ROOT/results/reanalysis/tsuboyama/modern_esm_mm_pll/$namespace/$slug/$dataset"
if [[ $mode == score ]];then extra=();[[ $limit != none ]]&&extra=(--max-doubles "$limit");batch=8;[[ $model == ESM3 ]]&&batch=2;exec "$ENV/bin/python" "$ROOT/scripts/reanalysis/modern_esm_mm_pll/score.py" --dataset "$dataset" --model "$model" --output "$out" --batch-size "$batch" "${extra[@]}"
elif [[ $mode == fit ]];then export JAX_PLATFORM_NAME=cpu XLA_FLAGS='--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1';exec /nethome/akolchina/miniconda3/envs/protease-pipeline/bin/python "$ROOT/scripts/reanalysis/multimodel_mm_pll/fit.py" --scores "$out/scores.csv" --output "$out/fits";else exit 2;fi
