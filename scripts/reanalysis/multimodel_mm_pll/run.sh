#!/usr/bin/env bash
set -euo pipefail
root=/data/users/akolchina/epistasis_proteingym
base="$root/results/reanalysis/tsuboyama/multimodel_mm_pll"
export HF_HOME=/data/users/akolchina/.cache/huggingface
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
case "$1" in
  score)
    key="$2"; checkpoint="$3"; revision="$4"; default="$5"; dataset="$6"; max_doubles="${7:-none}"; namespace="${8:-full}"
    [[ "$namespace" == "full" ]] || base="$base/$namespace"
    extra=(); [[ "$max_doubles" == "none" ]] || extra=(--max-doubles "$max_doubles")
    exec /nethome/akolchina/miniconda3/envs/ofs/bin/python -u "$root/scripts/reanalysis/multimodel_mm_pll/score.py" --model-key "$key" --checkpoint "$checkpoint" --revision "$revision" --default-column "$default" --dataset "$dataset" --output "$base/scores/$key/$dataset" --batch-size 16 "${extra[@]}"
    ;;
  aggregate)
    dataset="$2"; namespace="${3:-full}"; [[ "$namespace" == "full" ]] || base="$base/$namespace"; out="$base/scores/ESM1v_ensemble/$dataset/scores.csv"; args=()
    for i in 1 2 3 4 5; do args+=(--input "$base/scores/ESM1v_${i}/$dataset/scores.csv"); done
    exec /nethome/akolchina/miniconda3/envs/protease-pipeline/bin/python -u "$root/scripts/reanalysis/multimodel_mm_pll/aggregate_ensemble.py" "${args[@]}" --output "$out" --model-key ESM1v_ensemble
    ;;
  fit)
    key="$2"; dataset="$3"; namespace="${4:-full}"; [[ "$namespace" == "full" ]] || base="$base/$namespace"; export JAX_PLATFORM_NAME=cpu XLA_FLAGS='--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
    exec /nethome/akolchina/miniconda3/envs/protease-pipeline/bin/python -u "$root/scripts/reanalysis/multimodel_mm_pll/fit.py" --scores "$base/scores/$key/$dataset/scores.csv" --output "$base/fits/$key/$dataset"
    ;;
  *) echo 'expected score, aggregate, or fit' >&2; exit 2;;
esac
