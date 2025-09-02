import shutil
import pandas as pd
import tempfile
import numpy as np
from scipy.stats import norm
import os
from pathlib import Path
from sklearn.preprocessing import FunctionTransformer
from Bio import SeqIO
import matplotlib.pyplot as plt
from .data_processing import table_parser, genotype_to_seq
from .plotting import dotplot_triplet, tsuboyama_dotplot_single
from .utils import convert_name_tsuboyama

def epistasis_detection_somermeyer(input_dir, output_dir, N=1):
    file_path = os.path.join(input_dir, "amacGFP_cgreGFP_ppluGFP2__final_nucleotide_genotypes_to_brightness.csv")
    big_df = pd.read_csv(file_path)
    big_df.rename(columns={
        "pseudocell_count": "total_cell_count",
        "aa_genotype_native": "aa_genotype",
        "replicates_mean_brightness": "brightness",
        "replicates_stdev_weighted": "brightness_stdev"
    }, inplace=True)

    proteins = big_df["gene"].unique()
    for protein in proteins:
        df = big_df[big_df["gene"] == protein].copy()
        df = table_parser(df)   # your existing parser

        if len(df) == 0:
            print(f"No data for {protein}, skipping.")
            continue

        singles = df[df['num_mutations'] == 1].copy()
        multi   = df[df['num_mutations'] > 1].copy()

        # ensure columns exist on BOTH so we can concat safely
        singles.loc[:, "sum_brightness"] = pd.NA
        singles.loc[:, "epistatic"]      = pd.NA

        multi.loc[:, "sum_brightness"] = pd.NA
        multi.loc[:, "epistatic"]      = pd.NA

        # --- compute additivity + epistasis for multi ---
        for i, row in multi.iterrows():
            muts = row["aa_genotype"].split(":")
            sum_brightness = 0
            missing = False

            for mut in muts:
                match = singles[singles["aa_genotype"] == mut]
                if match.empty:
                    missing = True
                    break
                sum_brightness += match["brightness"].values[0]

            if missing:
                continue

            # write expected sum and compute total error
            multi.loc[i, "sum_brightness"] = sum_brightness
            sum_errors = sum(
                singles.loc[singles["aa_genotype"] == mut, "brightness_stdev"].values[0]
                for mut in muts
            )
            sum_errors += row["brightness_stdev"]

            # NOTE: this line forces expected sum not below the min observed brightness in multi.
            # Keep if intentional; otherwise consider removing.
            sum_brightness = max(sum_brightness, float(multi["brightness"].min()))

            is_epistatic = abs(row["brightness"] - sum_brightness) > N * sum_errors
            multi.loc[i, "epistatic"] = is_epistatic

        # --- derive sequences for BOTH singles and multi ---
        wt_seqs = {}
        input_fa = os.path.join(input_dir, "protein_seqs.fa")
        fasta_sequences = SeqIO.parse(open(input_fa), 'fasta')
        for fasta in fasta_sequences:
            wt_seqs[fasta.id] = str(fasta.seq)

        # drop nonsense* genotypes everywhere
        singles = singles[~singles["aa_genotype"].str.contains(r"\*")]
        multi   = multi[~multi["aa_genotype"].str.contains(r"\*")]

        get_seq_from_genotype = FunctionTransformer(
            genotype_to_seq, kw_args={'wt_seqs': wt_seqs, 'gene_name': protein}
        )
        singles.loc[:, "sequence"] = get_seq_from_genotype.fit_transform(singles)
        multi.loc[:,   "sequence"] = get_seq_from_genotype.fit_transform(multi)

        # --- merge and save (now includes singles) ---
        result = pd.concat([singles, multi], ignore_index=True)
        out_path = Path(output_dir) / f"{protein}.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(out_path, index=False)

        # quick counts (multi only, as before)
        print(protein)
        print("Number of all (non-epistatic) multi:", (multi["epistatic"] == False).sum())
        print("Number of epistatic multi:", (multi["epistatic"] == True).sum())


def explore_N_values(input_dir, output_dir):
    fig, axes = plt.subplots(3, 3, figsize=(24, 24))
    with tempfile.TemporaryDirectory(dir=input_dir) as tmpdir:
        tmp_path = Path(tmpdir)
        for row, N in enumerate([1, 2, 3]):
            print("N =", N)
            # Produce three *_GFP*.csv files into temp_output_dir for this N
            epistasis_detection_somermeyer(input_dir, tmp_path, N=N)
            # Fill this row with the triplet
            _, _ = dotplot_triplet(tmp_path, axes=axes[row, :], hist=True, title=str(N))
            # clear tmp_path for next iteration
            for item in tmp_path.iterdir():
                if item.is_dir():
                    shutil.rmtree(item)
                else:
                    item.unlink()

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "1_GFP_N_explored.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    
    
def epistasis_detection_tsuboyama(input_dir, output_dir, N=6):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for file in os.listdir(input_dir):
        file_path = os.path.join(input_dir, file)
        df = pd.read_csv(file_path)

        singles = df[df['num_mutations'] == 1].copy()
        doubles = df[df['num_mutations'] == 2].copy()

        # add the new columns so concat works cleanly
        for col in ["epistatic", "symmetric_1", "symmetric_2", "symmetric_double", "dG_std"]:
            singles[col] = None
            doubles[col] = None

        for i, row in doubles.iterrows():
            mut1 = row["mutant"].split(":")[0]
            mut2 = row["mutant"].split(":")[1]

            if len(singles[singles["mutant"] == mut1]) == 0 or \
               len(singles[singles["mutant"] == mut2]) == 0:
                continue

            # ---- single mutant 1 ----
            median = singles.loc[singles["mutant"] == mut1, "dG"].values[0]
            low = singles.loc[singles["mutant"] == mut1, "deltaG_95CI_low"].values[0]
            high = singles.loc[singles["mutant"] == mut1, "deltaG_95CI_high"].values[0]
            left, right = median - low, high - median
            symmetric = min(left, right) / max(left, right) >= 0.9
            doubles.at[i, "symmetric_1"] = symmetric
            error1 = (high - low) / (2 * 1.96) if symmetric else (left + right) / (2 * 1.96)

            # ---- single mutant 2 ----
            median = singles.loc[singles["mutant"] == mut2, "dG"].values[0]
            low = singles.loc[singles["mutant"] == mut2, "deltaG_95CI_low"].values[0]
            high = singles.loc[singles["mutant"] == mut2, "deltaG_95CI_high"].values[0]
            left, right = median - low, high - median
            symmetric = min(left, right) / max(left, right) >= 0.9
            doubles.at[i, "symmetric_2"] = symmetric
            error2 = (high - low) / (2 * 1.96) if symmetric else (left + right) / (2 * 1.96)

            # ---- double mutant ----
            median = row["dG"]
            low = row["deltaG_95CI_low"]
            high = row["deltaG_95CI_high"]
            left, right = median - low, high - median
            symmetric = min(left, right) / max(left, right) >= 0.9
            doubles.at[i, "symmetric_double"] = symmetric
            error_double = (high - low) / (2 * 1.96) if symmetric else (left + right) / (2 * 1.96)
            doubles.at[i, "dG_std"] = error_double

            # ---- epistasis check ----
            is_epi = abs(row["thermodynamic_coupling"]) > N * (error_double + error1 + error2)
            doubles.at[i, "epistatic"] = is_epi

        # merge singles + doubles
        result = pd.concat([singles, doubles], ignore_index=True)
        result.to_csv(output_dir / file, index=False)


def explore_N_values_tsuboyama(input_dir, output_dir, filename=None, pattern="*.csv"):
    """
    Explore N values (1..6) for a single Tsuboyama dataset and plot results in a 2x3 panel.

    Parameters
    ----------
    input_dir : str | Path
        Directory with input dataset(s).
    output_dir : str | Path
        Directory to save the final figure.
    filename : str | None
        Specific CSV file to use. If None, the first file matching `pattern` is used.
    pattern : str
        Glob pattern to find file if filename is None.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # choose dataset
    if filename is not None:
        src_file = input_dir / filename
        if not src_file.exists():
            raise FileNotFoundError(f"File not found: {src_file}")
    else:
        matches = sorted(input_dir.glob(pattern))
        src_file = matches[0]

    fig, axes = plt.subplots(2, 3, figsize=(20, 14))
    axes = axes.ravel()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        for idx, N in enumerate([1, 2, 3, 4, 5, 6]):
            tmp_in = tmpdir / f"in_N{N}"
            tmp_out = tmpdir / f"out_N{N}"
            tmp_in.mkdir(parents=True, exist_ok=True)
            tmp_out.mkdir(parents=True, exist_ok=True)

            # copy one dataset into tmp_in
            shutil.copy2(src_file, tmp_in / src_file.name)

            # run detection
            epistasis_detection_tsuboyama(tmp_in, tmp_out, N=N)

            # get the annotated CSV
            out_csv = tmp_out / src_file.name
            if not out_csv.exists():
                raise RuntimeError(f"Expected output not found: {out_csv}")

            # plot with your existing single-dotplot function
            tsuboyama_dotplot_single(
                ax=axes[idx],
                csv_path=out_csv,
                title=f"{convert_name_tsuboyama(src_file)}, N={N}",
                hist=True
            )

    plt.tight_layout()
    fig.savefig(output_dir / "3_Tsuboyama_N_explored.png", dpi=300, bbox_inches="tight")
    plt.close(fig)