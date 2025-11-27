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
from scipy.stats import spearmanr
from .data_processing import table_parser, genotype_to_seq
from .plotting import dotplot_triplet, tsuboyama_dotplot_single, combined_plot_tsuboyama
from .utils import convert_name_tsuboyama, convert_name_to_gfp, shift_mutation_positions_up
from src.models.linear_regression import linear_regression
from src.models.mlp import mlp
from .constants import SEED, set_seed
from matplotlib.lines import Line2D

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
    
    
def epistasis_detection_tsuboyama(input_dir, output_dir, N=3):
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
    
# for notebook 03    
    
def compute_result_table(model_dir, epi_dir, dataset_name, seed=SEED, dataset_set=None):
    input_dir = model_dir / dataset_name
    # Open 1 dataset just to get model names
    file_path = next(input_dir.glob("*.csv"))
    df = pd.read_csv(file_path)
    df["num_mutations"] = df["mutant"].apply(lambda x: len(x.split(":")))
    singles = df[df['num_mutations'] == 1].copy()
    model_columns = list(singles.columns)
    values_to_remove = ['mutant', 'mutated_sequence', 'DMS_score', 'DMS_score_bin', 'num_mutations']
    model_columns = [x for x in model_columns if x not in values_to_remove]
    model_columns.extend(["Linear_regression", "MLP"])
    cols = []
    datasets = []

    for file_path in input_dir.glob("*.csv"): # iterate over datasets
        df = pd.read_csv(file_path)
        dataset = file_path.stem
        if dataset_set is not None and dataset not in dataset_set:
            continue
        datasets.append(dataset)
        cols.append(dataset + '_all')
        cols.append(dataset + '_epistatic')
    result_all = pd.DataFrame(index=model_columns, columns=cols)
    df_counts = pd.DataFrame(index=["counts"], columns=datasets)

    for file_path in input_dir.glob("*.csv"): # iterate over datasets
        df = pd.read_csv(file_path)
        df["num_mutations"] = df["mutant"].apply(lambda x: len(x.split(":")))
        singles = df[df['num_mutations'] == 1].copy()
        multis = df[df['num_mutations'] > 1].copy()
        dataset = file_path.stem
        
        if dataset_set is not None and dataset not in dataset_set:
            continue
        
        if dataset_name == "somermeyer":
            selected = pd.read_csv(epi_dir / f"{convert_name_to_gfp(dataset)}.csv")
            selected.rename(columns={"aa_genotype": "mutant"}, inplace=True)
            selected['mutant'] = selected['mutant'].apply(shift_mutation_positions_up)
        else:
            selected = pd.read_csv(epi_dir / file_path.name)
        
        selected = selected[['mutant', 'epistatic']]
        multis = multis.merge(selected, on='mutant', how='left')
        epistatic = multis[multis["epistatic"].notna() & multis["epistatic"]]
        df_counts.loc["counts", dataset] = len(epistatic)
        
        if len(epistatic) == 0: # no epistasis detected for the dataset
            result_all.drop(columns=[dataset + '_all', dataset + '_epistatic'], inplace=True)
            df_counts.drop(columns=[dataset], inplace=True)
            continue

        values_to_remove = ["Linear_regression", "MLP"]
        model_columns = [x for x in model_columns if x not in values_to_remove]
        for model in model_columns: # calculate for all model predictions
            spearman_epistatic = spearmanr(epistatic["DMS_score"], epistatic[model])[0]

            result_all.loc[model, dataset + '_epistatic'] = spearman_epistatic
            # sample as the same size as epistatic points
            sample = multis.sample(n=len(epistatic), random_state=seed)
            spearman_all = spearmanr(sample['DMS_score'], sample[model])[0]
            result_all.loc[model, dataset + '_all'] = spearman_all

    if dataset_name == "tsuboyama":
        return result_all, df_counts
    return result_all


def select_best_models(dir, dataset_name):
    # Load CSV with model names as index
    data = pd.read_csv(dir / f"{dataset_name}_all_models.csv", index_col=0)

    # Fill missing values with 0
    data.fillna(0, inplace=True)

    # Pick columns that end with '_all' to compute mean scores
    columns_to_consider = [col for col in data.columns if col.endswith('_all')]

    # Create a new column for category (e.g. "MLP_v2" → "MLP")
    data['category'] = data.index.to_series().str.split(r'[_-]').str[0]

    # Prepare list to collect best models
    best_rows = []

    # For each category, find the best full model (by mean(abs(...)))
    for category, group in data.groupby('category'):
        # Find model name (i.e. index) with highest mean score
        best_model_name = group[columns_to_consider].abs().mean(axis=1).idxmax()
        best_row = data.loc[best_model_name].copy()
        best_row.name = best_model_name  # ensure full model name is preserved
        best_rows.append(best_row)

    # Combine best rows into a new DataFrame
    best_models_df = pd.DataFrame(best_rows)

    # Drop the helper column if you don't want it
    best_models_df.drop(columns=['category'], inplace=True)

    # Save final result
    best_models_df.to_csv(dir / f"{dataset_name}_best_models.csv", index=True)
    

def combined_plot_somermeyer_with_whiskers(
    models_eval_dir: Path,
    output_dir: Path,
    model_dir: Path,
    epi_dir: Path,
    seeds=(0,1,2,3,4),
):
    models_eval_dir = Path(models_eval_dir)
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)

    # Which models to show (your best models file)
    df_best = pd.read_csv(models_eval_dir / "somermeyer_best_models.csv", index_col=0)
    model_list = df_best.index.tolist()

    # Dataset IDs from columns in best-models
    datasets = df_best.columns.str.replace("_all","",regex=False).str.replace("_epistatic","",regex=False).unique()

    # Re-run pipeline for each seed; collect the per-seed tables (floats)
    per_seed = []
    for seed in seeds:
        res = compute_result_table(model_dir, epi_dir, "somermeyer", seed=seed)
        res = linear_regression(res, model_dir, epi_dir, "somermeyer", seeds=1, seed_provided=seed)
        res = mlp(res, model_dir, epi_dir, "somermeyer", seeds=1, seed_provided=seed)
        res = res.apply(pd.to_numeric, errors="coerce")
        per_seed.append(res)

    # Per-dataset figures
    for dataset in datasets:
        col_all = f"{dataset}_all"
        col_epi = f"{dataset}_epistatic"

        # Stack per-seed columns (rows=models, cols=seeds)
        all_mat = []
        epi_mat = []
        for res in per_seed:
            a = res[col_all] if col_all in res.columns else pd.Series(index=res.index, dtype=float)
            e = res[col_epi] if col_epi in res.columns else pd.Series(index=res.index, dtype=float)
            all_mat.append(a)
            epi_mat.append(e)
        all_mat = pd.concat(all_mat, axis=1)
        epi_mat = pd.concat(epi_mat, axis=1)

        # keep only selected models; abs values
        all_mat = np.abs(all_mat.loc[model_list])
        epi_mat = np.abs(epi_mat.loc[model_list])

        # bars = mean across seeds; whiskers (ALL) = std across seeds
        all_mean = all_mat.mean(axis=1, skipna=True)
        epi_mean = epi_mat.mean(axis=1, skipna=True)
        all_std  = all_mat.std(axis=1, ddof=0, skipna=True)

        df_temp = pd.DataFrame({
            "model": all_mean.index,
            "all": all_mean.values,
            "epistatic": epi_mean.values,
            "delta": all_mean.values - epi_mean.values,
            "all_std": all_std.values,
        })

        # sort: baselines first, then by 'all' desc
        df_temp["sort_key"] = df_temp["model"].apply(lambda x: 1 if x in ["Linear_regression", "MLP"] else 0)
        df_temp = df_temp.sort_values(by=["sort_key","all"], ascending=[True, False]).reset_index(drop=True)

        x = np.arange(len(df_temp))
        width = 0.4
        cmap = plt.get_cmap("Paired")
        dataset_name = convert_name_to_gfp(dataset)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12), sharex=True,
                                        gridspec_kw={"height_ratios":[2,1]})

        # ---- bars
        bar_all = ax1.bar(x - width/2, df_temp["all"],       width=width, color=cmap(0), label="All points")
        bar_epi = ax1.bar(x + width/2, df_temp["epistatic"], width=width, color=cmap(1), label="Epistatic points")

        # ---- whiskers (explicit errorbar so they sit above bars)
        centers  = x - width/2
        y_all    = df_temp["all"].to_numpy(dtype=float)
        yerr_all = df_temp["all_std"].to_numpy(dtype=float)
        mask = np.isfinite(y_all) & np.isfinite(yerr_all)
        if mask.any():
            ax1.errorbar(centers[mask], y_all[mask], yerr=yerr_all[mask],
                         fmt="none", ecolor="dimgray", elinewidth=2.0, capsize=5, capthick=2.0, zorder=5)

        # ---- cosmetics
        ax1.set_ylabel("Spearman correlation", fontsize=14)
        ax1.set_title(f"Spearman correlation for {dataset_name}", fontsize=16)
        ax1.grid(True, axis="y"); ax1.tick_params(axis="y", labelsize=12)

        # global medians as ref lines
        med_all = float(np.nanmedian(df_temp["all"]))
        med_epi = float(np.nanmedian(df_temp["epistatic"]))
        ax1.axhline(med_all, color="red",   linestyle="-", linewidth=1.8)
        ax1.axhline(med_epi, color="green", linestyle="-", linewidth=1.8)

        # legend
        from matplotlib.lines import Line2D
        custom_handles = [
            bar_all.patches[0], bar_epi.patches[0],
            Line2D([0],[0], color="red",   linestyle="-", linewidth=1.8),
            Line2D([0],[0], color="green", linestyle="-", linewidth=1.8),
        ]
        ax1.legend(custom_handles, ["All points", "Epistatic points", "Median (all)", "Median (epistatic)"],
                   loc="upper left", prop={"size":12})

        # ---- delta subplot
        ax2.bar(x, df_temp["delta"], color="cornflowerblue")
        ax2.set_ylabel("Δ (All - Epistatic)", fontsize=14)
        ax2.grid(True, axis="y"); ax2.tick_params(axis="y", labelsize=12)
        ax2.set_xticks(x)
        ax2.set_xticklabels([f"$\\bf{{{m}}}$" if m in ["Linear_regression","MLP"] else m
                             for m in df_temp["model"]], fontsize=12, rotation=90)
        ax2.set_xlabel("Model", fontsize=14)

        plt.tight_layout()
        plt.savefig(output_dir / f"2_combined_plot_{dataset_name}_whiskers.png", bbox_inches="tight", dpi=300)
        plt.close()
    

def somermeyer_top_models(input_dir, output_dir):
    # Load your table
    df = pd.read_csv(input_dir / "somermeyer_best_models.csv", index_col=0)

    # collect dataset names (strip the suffixes)
    datasets = df.columns.str.replace("_all", "", regex=False).str.replace("_epistatic", "", regex=False).unique()

    # build result table
    result_rows = []
    for dataset in datasets:
        epi_scores = df[dataset + "_epistatic"].astype(float)  # use epistatic values
        top5 = epi_scores.sort_values(ascending=False).head(5)

        row = {"Dataset": convert_name_to_gfp(dataset)}
        for i, (model, score) in enumerate(top5.items(), start=1):
            row[f"Top-{i}"] = f"{model} ({score:.3f})"
        result_rows.append(row)

    result_table = pd.DataFrame(result_rows)
    result_table = result_table.sort_values(by="Dataset")  # sort by Dataset alphabetically

    result_table.to_csv(output_dir / "1_somermeyer_top5_epistatic.csv", index=False)


def build_tsuboyama_N_series(
    raw_tsuboyama_dir: Path,
    model_root_dir: Path,         # usually data_dir
    work_root_dir: Path,          # usually interm_tables_dir
    figures_dir: Path,            # usually main_figures_dir
):
    """
    For N = 1..6:
      - recompute epistatic labels for ALL Tsuboyama CSVs
      - recompute model tables (compute_result_table -> LR -> MLP -> select_best)
      - render combined plot
    Saves: figures_dir / '3_tsuboyama_combined_plot_N{N}.png'
    """
    raw_tsuboyama_dir = Path(raw_tsuboyama_dir)
    model_root_dir    = Path(model_root_dir)
    work_root_dir     = Path(work_root_dir)
    figures_dir       = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    for N in [1, 2, 3, 4, 5, 6]:
        print(f"\n=== N = {N} ===")

        # Where we put epistatic-annotated CSVs for this N
        epi_dir = work_root_dir / f"tsuboyama_epistatic_N{N}"
        if epi_dir.exists():
            shutil.rmtree(epi_dir)
        epi_dir.mkdir(parents=True, exist_ok=True)

        # 1) Recompute epistasis for ALL datasets at this N
        epistasis_detection_tsuboyama(raw_tsuboyama_dir, epi_dir, N=N)

        # 2) Build results for this N in an isolated run dir layout
        perN_run = work_root_dir / f"eval_tsuboyama_N{N}"
        me_dir   = perN_run / "models_evaluation"
        me_dir.mkdir(parents=True, exist_ok=True)

        # compute base tables
        results, counts = compute_result_table(model_root_dir, epi_dir, "tsuboyama")
        (perN_run / "counts.csv").write_text(counts.to_csv(index=True))

        # baselines
        results = linear_regression(results, model_root_dir, epi_dir, "tsuboyama")
        results = mlp(results,            model_root_dir, epi_dir, "tsuboyama")

        # save all models and pick best
        results.to_csv(me_dir / "tsuboyama_all_models.csv", index=True)
        select_best_models(me_dir, "tsuboyama")

        # 3) Make combined plot for this N (function writes 3_tsuboyama_combined_plot.png)
        combined_plot_tsuboyama(perN_run, figures_dir, N=N)

        # rename to include N (to keep all 6)
        src = figures_dir / "4_tsuboyama_combined_plot.png"
        dst = figures_dir / f"4_tsuboyama_combined_plot_N{N}.png"
        if src.exists():
            src.replace(dst)
        print(f"[OK] Saved {dst}")
        
        
def tsuboyama_top_models(input_dir, output_dir, N=6):
    for i in range(1, N+1):
        file_path = input_dir / f"eval_tsuboyama_N{i}" / "models_evaluation" / "tsuboyama_best_models.csv"
        df_scores = pd.read_csv(file_path, index_col=0)
        
        file_path = input_dir / f"eval_tsuboyama_N{i}" / "counts.csv"
        df_counts = pd.read_csv(file_path, index_col=0)

        # collect dataset names (strip the suffixes)
        datasets = df_scores.columns.str.replace("_all", "", regex=False).str.replace("_epistatic", "", regex=False).unique()
        # sort datasets by counts and select top 10
        dataset_counts = {dataset: df_counts.at["counts", dataset] for dataset in datasets}
        sorted_datasets = sorted(dataset_counts, key=lambda k: dataset_counts[k], reverse=True)[:10]
        datasets = sorted_datasets

        # build result table
        result_rows = []
        for dataset in datasets:
            epi_scores = df_scores[dataset + "_epistatic"].astype(float)  # use epistatic values
            top5 = epi_scores.sort_values(ascending=False).head(5)

            row = {"Dataset": convert_name_tsuboyama(dataset)}
            for j, (model, score) in enumerate(top5.items(), start=1):
                row[f"Top-{j}"] = f"{model} ({score:.3f})"
            result_rows.append(row)

        result_table = pd.DataFrame(result_rows)
        result_table = result_table.sort_values(by="Dataset")  # sort by Dataset alphabetically

        result_table.to_csv(output_dir / f"1_tsuboyama_top5_epistatic_N{i}.csv", index=False)
        

def combined_plot_tsuboyama_with_whiskers(
    input_dir: Path, # intermediate
    output_dir: Path,
    model_dir: Path,
    epi_dir: Path,
    seeds=(0,1,2,3,4),
    n_sequences=100
):
    models_eval_dir = input_dir / "models_evaluation"
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)

    # Which models to show (your best models file)
    df_best = pd.read_csv(models_eval_dir / "tsuboyama_best_models.csv", index_col=0)
    model_list = df_best.index.tolist()
    df_counts = pd.read_csv(input_dir / "counts.csv", index_col=0)

    # Dataset IDs from columns in best-models
    datasets = df_best.columns.str.replace("_all","",regex=False).str.replace("_epistatic","",regex=False).unique()
    dataset_counts = {dataset: df_counts.at["counts", dataset] for dataset in datasets}
    subset = [d for d in datasets if dataset_counts[d] >= n_sequences]

    # Re-run pipeline for each seed; collect the per-seed tables (floats)
    per_seed = []
    for seed in seeds:
        res, counts = compute_result_table(model_dir, epi_dir, "tsuboyama", seed=seed, dataset_set=subset)
        res = linear_regression(res, model_dir, epi_dir, "tsuboyama", seeds=1, seed_provided=seed, dataset_set=subset)
        res = mlp(res, model_dir, epi_dir, "tsuboyama", seeds=1, seed_provided=seed, dataset_set=subset)
        res = res.apply(pd.to_numeric, errors="coerce")
        per_seed.append(res)

    # Per-dataset figures
    for dataset in subset:
        col_all = f"{dataset}_all"
        col_epi = f"{dataset}_epistatic"

        # Stack per-seed columns (rows=models, cols=seeds)
        all_mat = []
        epi_mat = []
        for res in per_seed:
            a = res[col_all] if col_all in res.columns else pd.Series(index=res.index, dtype=float)
            e = res[col_epi] if col_epi in res.columns else pd.Series(index=res.index, dtype=float)
            all_mat.append(a)
            epi_mat.append(e)
        all_mat = pd.concat(all_mat, axis=1)
        epi_mat = pd.concat(epi_mat, axis=1)

        # keep only selected models; abs values
        all_mat = np.abs(all_mat.loc[model_list])
        epi_mat = np.abs(epi_mat.loc[model_list])

        # bars = mean across seeds; whiskers (ALL) = std across seeds
        all_mean = all_mat.mean(axis=1, skipna=True)
        epi_mean = epi_mat.mean(axis=1, skipna=True)
        all_std  = all_mat.std(axis=1, ddof=0, skipna=True)

        df_temp = pd.DataFrame({
            "model": all_mean.index,
            "all": all_mean.values,
            "epistatic": epi_mean.values,
            "delta": all_mean.values - epi_mean.values,
            "all_std": all_std.values,
        })

        # sort: baselines first, then by 'all' desc
        df_temp["sort_key"] = df_temp["model"].apply(lambda x: 1 if x in ["Linear_regression", "MLP"] else 0)
        df_temp = df_temp.sort_values(by=["sort_key","all"], ascending=[True, False]).reset_index(drop=True)

        x = np.arange(len(df_temp))
        width = 0.4
        cmap = plt.get_cmap("Paired")
        dataset_name = convert_name_to_gfp(dataset)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12), sharex=True,
                                        gridspec_kw={"height_ratios":[2,1]})

        # ---- bars
        bar_all = ax1.bar(x - width/2, df_temp["all"],       width=width, color=cmap(0), label="All points")
        bar_epi = ax1.bar(x + width/2, df_temp["epistatic"], width=width, color=cmap(1), label="Epistatic points")

        # ---- whiskers (explicit errorbar so they sit above bars)
        centers  = x - width/2
        y_all    = df_temp["all"].to_numpy(dtype=float)
        yerr_all = df_temp["all_std"].to_numpy(dtype=float)
        mask = np.isfinite(y_all) & np.isfinite(yerr_all)
        if mask.any():
            ax1.errorbar(centers[mask], y_all[mask], yerr=yerr_all[mask],
                         fmt="none", ecolor="dimgray", elinewidth=2.0, capsize=5, capthick=2.0, zorder=5)

        # ---- cosmetics
        ax1.set_ylabel("Spearman correlation", fontsize=14)
        ax1.set_title(f"Spearman correlation for {dataset_name}", fontsize=16)
        ax1.grid(True, axis="y"); ax1.tick_params(axis="y", labelsize=12)

        # global medians as ref lines
        med_all = float(np.nanmedian(df_temp["all"]))
        med_epi = float(np.nanmedian(df_temp["epistatic"]))
        ax1.axhline(med_all, color="red",   linestyle="-", linewidth=1.8)
        ax1.axhline(med_epi, color="green", linestyle="-", linewidth=1.8)

        # legend
        from matplotlib.lines import Line2D
        custom_handles = [
            bar_all.patches[0], bar_epi.patches[0],
            Line2D([0],[0], color="red",   linestyle="-", linewidth=1.8),
            Line2D([0],[0], color="green", linestyle="-", linewidth=1.8),
        ]
        ax1.legend(custom_handles, ["All points", "Epistatic points", "Median (all)", "Median (epistatic)"],
                   loc="upper left", prop={"size":12})

        # ---- delta subplot
        ax2.bar(x, df_temp["delta"], color="cornflowerblue")
        ax2.set_ylabel("Δ (All - Epistatic)", fontsize=14)
        ax2.grid(True, axis="y"); ax2.tick_params(axis="y", labelsize=12)
        ax2.set_xticks(x)
        ax2.set_xticklabels([f"$\\bf{{{m}}}$" if m in ["Linear_regression","MLP"] else m
                             for m in df_temp["model"]], fontsize=12, rotation=90)
        ax2.set_xlabel("Model", fontsize=14)

        plt.tight_layout()
        plt.savefig(output_dir / f"4_combined_plot_{dataset_name}_whiskers.png", bbox_inches="tight", dpi=300)
        plt.close()
        

def tsuboyama_top_models_by_seq(input_dir, output_dir, i=3, n_sequences=400):
    file_path = input_dir / f"eval_tsuboyama_N{i}" / "models_evaluation" / "tsuboyama_best_models.csv"
    df_scores = pd.read_csv(file_path, index_col=0)
    
    file_path = input_dir / f"eval_tsuboyama_N{i}" / "counts.csv"
    df_counts = pd.read_csv(file_path, index_col=0)

    # collect dataset names (strip the suffixes)
    datasets = df_scores.columns.str.replace("_all", "", regex=False).str.replace("_epistatic", "", regex=False).unique()
    # sort datasets by counts and select top 10
    dataset_counts = {dataset: df_counts.at["counts", dataset] for dataset in datasets}
    sorted_datasets = sorted(dataset_counts, key=lambda k: dataset_counts[k], reverse=True)[:6]
    datasets = sorted_datasets

    # build result table
    result_rows = []
    for dataset in datasets:
        epi_scores = df_scores[dataset + "_epistatic"].astype(float)  # use epistatic values
        top10 = epi_scores.sort_values(ascending=False).head(10)

        selected = pd.read_csv(input_dir / f"tsuboyama_epistatic_N{i}" / f"{dataset}.csv")

        row = {"Dataset": convert_name_tsuboyama(dataset),
               "All sequences": len(selected[selected["num_mutations"] > 1]),
               "Epistatic sequences": (selected["epistatic"] == True).sum()}
        for j, (model, score) in enumerate(top10.items(), start=1):
            row[f"Top-{j}"] = f"{model} ({score:.3f})"
        result_rows.append(row)

    result_table = pd.DataFrame(result_rows)
    result_table = result_table.sort_values(by="Dataset")  # sort by Dataset alphabetically

    result_table.to_csv(output_dir / f"2_tsuboyama_top10_{n_sequences}.csv", index=False)
    
    
def combined_plot_tsuboyama_top50(
    model_root_dir: Path,          # usually data_dir (has model score tables under data_dir/tsuboyama)
    epi_dir: Path,                 # your existing epistasis dir (already has thermodynamic_coupling)
    dataset_name: str,             # exact stem or unique substring in the epi_dir CSV name
    models_eval_dir: Path,         # interm_tables_dir / "models_evaluation" (to read tsuboyama_best_models.csv)
    output_dir: Path,              # where to save figure
    top_n: int = 50,
    seeds=(0, 1, 2, 3, 4),
):
    model_root_dir  = Path(model_root_dir)
    epi_dir         = Path(epi_dir)
    models_eval_dir = Path(models_eval_dir)
    output_dir      = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)

    # --- which models to display (filter), same as your combined plots
    df_best = pd.read_csv(models_eval_dir / "tsuboyama_best_models.csv", index_col=0)
    best_models = df_best.index.tolist()

    # --- locate dataset CSV INSIDE epi_dir (not model dir!)
    if (epi_dir / f"{dataset_name}.csv").exists():
        src_csv = epi_dir / f"{dataset_name}.csv"
        ds_stem = dataset_name
    else:
        matches = sorted([p for p in epi_dir.glob("*.csv") if dataset_name in p.stem])
        if not matches:
            raise FileNotFoundError(f"No CSV in {epi_dir} matching '{dataset_name}'")
        src_csv = matches[0]
        ds_stem = src_csv.stem

    # --- build a temporary epi labels dir using top-N |thermodynamic_coupling| from epi_dir CSV
    with tempfile.TemporaryDirectory() as tmp:
        epi_topN_dir = Path(tmp) / "epi_topN"
        epi_topN_dir.mkdir(parents=True, exist_ok=True)

        df = pd.read_csv(src_csv)
        if "num_mutations" not in df.columns:
            df["num_mutations"] = df["mutant"].apply(lambda x: len(str(x).split(":")))
        doubles = df[df["num_mutations"] == 2].copy()
        if "thermodynamic_coupling" not in doubles.columns:
            raise ValueError(
                f"'thermodynamic_coupling' not found in {src_csv}. "
                "Make sure reconstruction already ran."
            )

        doubles["abs_tc"] = np.abs(doubles["thermodynamic_coupling"].astype(float))
        doubles = doubles.sort_values("abs_tc", ascending=False)
        doubles["epistatic"] = False
        doubles.loc[doubles.head(top_n).index, "epistatic"] = True

        # labels file that compute_result_table expects (mutant,epistatic)
        selected = doubles[["mutant", "epistatic"]].copy()
        selected.to_csv(epi_topN_dir / f"{ds_stem}.csv", index=False)

        # --- recompute the result tables across seeds, restricted to THIS dataset only
        dataset_set = [ds_stem]
        per_seed = []
        for seed in seeds:
            res_df, _ = compute_result_table(
                model_root_dir, epi_topN_dir, "tsuboyama",
                seed=seed, dataset_set=dataset_set
            )
            res_df = linear_regression(
                res_df, model_root_dir, epi_topN_dir, "tsuboyama",
                seeds=1, seed_provided=seed, dataset_set=dataset_set
            )
            res_df = mlp(
                res_df, model_root_dir, epi_topN_dir, "tsuboyama",
                seeds=1, seed_provided=seed, dataset_set=dataset_set
            )
            res_df = res_df.apply(pd.to_numeric, errors="coerce")
            per_seed.append(res_df)

    # --- stack across seeds and prepare plotting
    col_all = f"{ds_stem}_all"
    col_epi = f"{ds_stem}_epistatic"

    # union of model indices across seeds
    all_models_idx = per_seed[0].index
    for r in per_seed[1:]:
        all_models_idx = all_models_idx.union(r.index)

    mats_all, mats_epi = [], []
    for r in per_seed:
        a = r[col_all] if col_all in r.columns else pd.Series(index=r.index, dtype=float)
        e = r[col_epi] if col_epi in r.columns else pd.Series(index=r.index, dtype=float)
        mats_all.append(a.reindex(all_models_idx))
        mats_epi.append(e.reindex(all_models_idx))

    mat_all = np.abs(pd.concat(mats_all, axis=1))
    mat_epi = np.abs(pd.concat(mats_epi, axis=1))

    # filter to best models, sort locally like combined plots (baselines first, then mean all desc)
    mat_all = mat_all.reindex(best_models)
    mat_epi = mat_epi.reindex(best_models)

    mean_all = mat_all.mean(axis=1, skipna=True)
    mean_epi = mat_epi.mean(axis=1, skipna=True)
    std_all  = mat_all.std(axis=1, ddof=0, skipna=True)

    df_plot = pd.DataFrame({
        "model": mean_all.index,
        "all": mean_all.values,
        "epistatic": mean_epi.values,
        "delta": mean_all.values - mean_epi.values,
        "all_std": std_all.values,
    })

    df_plot["sort_key"] = df_plot["model"].apply(lambda x: 1 if x in ["Linear_regression", "MLP"] else 0)
    df_plot = df_plot.sort_values(by=["sort_key", "all"], ascending=[True, False]).reset_index(drop=True)

    # --- plot
    x = np.arange(len(df_plot))
    width = 0.4
    cmap = plt.get_cmap("Paired")
    pretty = convert_name_tsuboyama(ds_stem)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12), sharex=True,
                                    gridspec_kw={"height_ratios": [2, 1]})

    bar_all = ax1.bar(x - width/2, df_plot["all"],       width=width, color=cmap(0), label="All points")
    bar_epi = ax1.bar(x + width/2, df_plot["epistatic"], width=width, color=cmap(1), label="Epistatic points")

    # whiskers: std across seeds for the 'All' bars
    centers  = x - width/2
    y_all    = df_plot["all"].to_numpy(dtype=float)
    yerr_all = df_plot["all_std"].to_numpy(dtype=float)
    mask = np.isfinite(y_all) & np.isfinite(yerr_all)
    if mask.any():
        ax1.errorbar(centers[mask], y_all[mask], yerr=yerr_all[mask],
                     fmt="none", ecolor="dimgray", elinewidth=2.0, capsize=5, capthick=2.0, zorder=5)

    ax1.set_ylabel("Spearman correlation", fontsize=14)
    ax1.set_title(f"Spearman for {pretty} (Top {top_n} by $|\Delta \Delta G|$)", fontsize=16)
    ax1.grid(True, axis="y"); ax1.tick_params(axis="y", labelsize=12)

    # global medians
    med_all = float(np.nanmedian(df_plot["all"]))
    med_epi = float(np.nanmedian(df_plot["epistatic"]))
    ax1.axhline(med_all, color="red",   linestyle="-", linewidth=1.8)
    ax1.axhline(med_epi, color="green", linestyle="-", linewidth=1.8)

    from matplotlib.lines import Line2D
    ax1.legend(
        [bar_all.patches[0], bar_epi.patches[0],
         Line2D([0],[0], color="red",   linestyle="-", linewidth=1.8),
         Line2D([0],[0], color="green", linestyle="-", linewidth=1.8)],
        ["All points", "Epistatic points", "Median (all)", "Median (epistatic)"],
        loc="upper left", prop={"size": 12}
    )

    ax2.bar(x, df_plot["delta"], color="cornflowerblue")
    ax2.set_ylabel(r'$\Delta$ (All - Epistatic)', fontsize=14)
    ax2.grid(True, axis="y"); ax2.tick_params(axis="y", labelsize=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels(
        [f"$\\bf{{{m}}}$" if m in ["Linear_regression", "MLP"] else m for m in df_plot["model"]],
        fontsize=12, rotation=90
    )
    ax2.set_xlabel("Model", fontsize=14)

    plt.tight_layout()
    out_path = output_dir / f"6_combined_plot_{pretty}_top{top_n}_whiskers.png"
    plt.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"[OK] saved {out_path}")