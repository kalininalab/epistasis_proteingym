#!/usr/bin/env bash
set -euo pipefail

# Run from your repo root 
DATA_DIR="data"
mkdir -p "$DATA_DIR"

echo "Downloading ProteinGym zero-shot scores zip..."
curl -L \
  -o "$DATA_DIR/zero_shot_substitutions_scores.zip" \
  "https://marks.hms.harvard.edu/proteingym/ProteinGym_v1.3/zero_shot_substitutions_scores.zip"

echo "Downloading GFP fitness peaks CSV (GitHub raw)..."
curl -L \
  -o "$DATA_DIR/amacGFP_cgreGFP_ppluGFP2__final_nucleotide_genotypes_to_brightness.csv" \
  "https://raw.githubusercontent.com/aequorea238/Orthologous_GFP_Fitness_Peaks/master/data/final_datasets/amacGFP_cgreGFP_ppluGFP2__final_nucleotide_genotypes_to_brightness.csv"

echo "Downloading GFP alignment FASTA (GitHub raw)..."
curl -L \
  -o "$DATA_DIR/avGFP_amacGFP_cgreGFP_ppluGFP2__protein_sequences.fasta" \
  "https://raw.githubusercontent.com/aequorea238/Orthologous_GFP_Fitness_Peaks/master/data/alignments/avGFP_amacGFP_cgreGFP_ppluGFP2__protein_sequences.fasta"

echo "Downloading Zenodo K50 processed datasets zip..."
curl -L \
  -o "$DATA_DIR/Processed_K50_dG_datasets.zip" \
  "https://zenodo.org/records/7992926/files/Processed_K50_dG_datasets.zip?download=1"

echo "Unzipping archives..."
unzip -q -o "$DATA_DIR/zero_shot_substitutions_scores.zip" -d "$DATA_DIR/zero_shot_substitutions_scores"
unzip -q -o "$DATA_DIR/Processed_K50_dG_datasets.zip" -d "$DATA_DIR/Processed_K50_dG_datasets"

echo "Done. Downloaded into: $DATA_DIR"
echo "Contents:"
find "$DATA_DIR" -maxdepth 3 -type f | sed 's|^| - |'