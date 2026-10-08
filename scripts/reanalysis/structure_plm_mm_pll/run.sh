#!/usr/bin/env bash
set -euo pipefail
R=/data/users/akolchina/epistasis_proteingym; E=/data/users/akolchina/envs/esm_modern310
export HF_HOME=/data/users/akolchina/huggingface TORCH_HOME=/data/users/akolchina/torch TOKENIZERS_PARALLELISM=false
mode=$1;ds=$2;model=$3;ns=$4;limit=${5:-none};slug=$(printf %s "$model"|tr '[:upper:]' '[:lower:]'|tr -c 'a-z0-9' '_');out="$R/results/reanalysis/tsuboyama/structure_plm_mm_pll/$ns/$slug/$ds"
if [[ $mode == score ]];then
 extra=();[[ $limit != none ]]&&extra=(--max-doubles "$limit")
 if [[ $model == MIF || $model == MIFST ]];then w=/data/users/akolchina/model_weights/mif/$(printf %s "$model"|tr '[:upper:]' '[:lower:]').pt;exec /nethome/akolchina/miniconda3/envs/ofs/bin/python -u "$R/scripts/reanalysis/structure_plm_mm_pll/score_mif.py" --dataset "$ds" --model "$model" --weights "$w" --output "$out" --batch-size 16 "${extra[@]}"
 else exec "$E/bin/python" -u "$R/scripts/reanalysis/structure_plm_mm_pll/score_hf.py" --dataset "$ds" --model "$model" --output "$out" --batch-size 8 "${extra[@]}";fi
else export JAX_PLATFORM_NAME=cpu XLA_FLAGS='--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1';exec /nethome/akolchina/miniconda3/envs/protease-pipeline/bin/python -u "$R/scripts/reanalysis/multimodel_mm_pll/fit.py" --scores "$out/scores.csv" --output "$out/fits";fi
