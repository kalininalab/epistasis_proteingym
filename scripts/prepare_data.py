#!/usr/bin/env python3
from __future__ import annotations

import shutil
from pathlib import Path


def delete_all_zips(data_dir: Path) -> int:
    """Delete all .zip files anywhere under data_dir."""
    count = 0
    for z in data_dir.rglob("*.zip"):
        if z.is_file():
            z.unlink()
            count += 1
    return count


def rename_fasta_and_trim_last_2_lines(data_dir: Path) -> None:
    """Rename the fasta and delete the last 2 lines (rows)."""
    src = data_dir / "avGFP_amacGFP_cgreGFP_ppluGFP2__protein_sequences.fasta"
    dst = data_dir / "protein_seqs.fa"

    if not src.exists():
        raise FileNotFoundError(f"FASTA not found: {src}")

    # If destination exists, we avoid silent overwrite.
    if dst.exists():
        raise FileExistsError(f"Destination already exists: {dst}")

    src.rename(dst)

    # Remove last 2 lines (if file has fewer than 2 lines, it becomes empty).
    lines = dst.read_text(encoding="utf-8", errors="replace").splitlines(True)  # keep line endings
    if len(lines) >= 2:
        lines = lines[:-2]
    else:
        lines = []
    dst.write_text("".join(lines), encoding="utf-8")


def move_csv_and_remove_processed_folder(data_dir: Path) -> None:
    """
    Move specific CSV out of nested Processed_K50_dG_datasets folder into data/,
    then delete the processed folder.
    """
    processed_root = data_dir / "Processed_K50_dG_datasets"
    src_csv = processed_root / "Processed_K50_dG_datasets" / "Tsuboyama2023_Dataset2_Dataset3_20230416.csv"
    dst_csv = data_dir / "Tsuboyama2023_Dataset2_Dataset3_20230416.csv"

    if not src_csv.exists():
        raise FileNotFoundError(f"Source CSV not found: {src_csv}")

    # Move CSV (overwrite is blocked to avoid accidents)
    if dst_csv.exists():
        raise FileExistsError(f"Destination CSV already exists: {dst_csv}")

    dst_csv.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src_csv), str(dst_csv))

    # Remove the entire processed folder tree
    if processed_root.exists():
        shutil.rmtree(processed_root)


def main() -> None:
    data_dir = Path("data").resolve()
    if not data_dir.exists() or not data_dir.is_dir():
        raise FileNotFoundError(f"'data' directory not found at: {data_dir}")

    zips_deleted = delete_all_zips(data_dir)
    rename_fasta_and_trim_last_2_lines(data_dir)
    move_csv_and_remove_processed_folder(data_dir)

    print("Done")
    print(f"Deleted .zip files: {zips_deleted}")
    print("Renamed fasta -> data/protein_seqs.fa and removed last 2 lines.")
    print("Moved Tsuboyama CSV into data/ and removed data/Processed_K50_dG_datasets/.")


if __name__ == "__main__":
    main()