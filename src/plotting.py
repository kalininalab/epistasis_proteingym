import os
from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from .utils import convert_name_to_gfp, convert_name_tsuboyama, extract_dataset_name

# for notebook 01

def mut_hist_somermeyer(file, ax=None, max_k=10):
    """
    Plot a histogram of mutation counts for 1..max_k.
    Also shows how many variants have > max_k mutations.
    """
    df = pd.read_csv(file)
    # assume the column with mutations is named "mutant"
    # and mutations are separated by a colon
    df["num_mutations"] = df["mutant"].apply(lambda x: len(str(x).split(":")))

    # count frequencies of all mutation counts
    counts_all = df["num_mutations"].astype(int).value_counts().sort_index()

    # prepare categories 1..max_k (fill with 0 if absent)
    x = list(range(1, max_k + 1))
    y = [int(counts_all.get(k, 0)) for k in x]

    # count how many variants are in the "tail" (> max_k)
    tail_n = int(df.loc[df["num_mutations"] > max_k].shape[0])

    # plot bars
    ax.bar(x, y, width=0.8)

    # labels and style
    ax.set_xticks(x)
    ax.set_xlabel("Number of mutations", fontsize=12, labelpad=6)
    ax.set_ylabel("Frequency", fontsize=12, labelpad=6)
    ax.tick_params(axis="both", labelsize=11)

    name = file.name.split(".")[0]
    name = convert_name_to_gfp(name)
    ax.set_title(f"Distribution of the number of mutations for {name}", fontsize=13)

    # annotate how many variants are beyond max_k
    ax.text(
        0.98, 0.92,
        f"{tail_n} variants with >{max_k} mutations",
        ha="right", va="top", transform=ax.transAxes, fontsize=10
    )
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    return ax


def mut_hist_tsuboyama(tsuboyama_data_dir, save_path=None):
    names_dict = {}

    for file in Path(tsuboyama_data_dir).glob("*.csv"):
        # short dataset name
        name = "_".join(file.stem.split("_")[:2])
        df = pd.read_csv(file)
        df["num_mutations"] = df["mutant"].apply(lambda x: len(x.split(":")))
        counts = df["num_mutations"].value_counts().sort_index()
        # store counts for 1 and 2 mutations (fill missing with 0)
        names_dict[name] = {1: counts.get(1, 0), 2: counts.get(2, 0)}

    # convert to DataFrame
    result_df = pd.DataFrame(names_dict).T
    result_df = result_df[[1, 2]]  # ensure column order

    # plot
    fig, ax = plt.subplots(figsize=(12, 6))
    result_df.plot(kind="bar", ax=ax)
    ax.set_xlabel("Dataset", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.set_title("Distribution of mutations per dataset", fontsize=13)
    ax.legend(title="Mutations")
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        plt.close(fig)
    else:
        plt.show()


def val_hist_somermeyer(somermeyer_data_dir, save_path=None):
    plt.figure(figsize=(6, 6))

    for file in Path(somermeyer_data_dir).glob("*.csv"):
        df = pd.read_csv(file)
        name = file.stem
        name = convert_name_to_gfp(name)
        plt.hist(df["DMS_score"].dropna(),
                bins=50,
                alpha=0.5,
                label=name)

    plt.xlabel("DMS_score")
    plt.ylabel("Frequency")
    plt.legend()
    plt.title("Distribution of DMS_score per dataset")
    plt.grid(linestyle='--', alpha=0.7)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()
        
        
def val_boxplot_tsuboyama(tsuboyama_data_dir, save_path=None):
    dict_names = {}
    for file in Path(tsuboyama_data_dir).glob("*.csv"):
        # Take first two parts of filename as dataset name
        name = "_".join(file.stem.split("_")[:2])
        df = pd.read_csv(file)
        dict_names[name] = df["DMS_score"].dropna()

    # Plot boxplot
    plt.figure(figsize=(12, 6))
    plt.boxplot(dict_names.values(),
                labels=dict_names.keys(),
                patch_artist=True,
                showfliers=False,
                boxprops=dict(facecolor="lightblue", color="black"),   
                capprops=dict(color="black"),
                whiskerprops=dict(color="black"),
                medianprops=dict(color="black"))
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("DMS_score")
    plt.title("Distribution of DMS_score across datasets")
    plt.tight_layout()
    plt.grid(axis="y", linestyle='--', alpha=0.7)

    if save_path:
        plt.savefig(save_path, dpi=300)
    else:
        plt.show()

    plt.close()
    
    
# for notebook 02

def dotplot_single(ax, file_path, title, hist=False):
    df = pd.read_csv(file_path)

    epi_x = df[df["epistatic"] == True]["brightness"].values
    epi_y = df[df["epistatic"] == True]["sum_brightness"].values
    x = df[df["epistatic"] == False]["brightness"].values
    y = df[df["epistatic"] == False]["sum_brightness"].values

    cmap = plt.get_cmap("Paired")
    ax.scatter(epi_x, epi_y, color=cmap(1), alpha=0.6, label="epistasis")
    ax.scatter(x, y, color=cmap(0), alpha=0.6, label="no epistasis")
    
    if hist:
        # Bar plot for counts of x and epi_x
        counts = [len(x), len(epi_x)]
        bar_labels = [len(x), len(epi_x)]
        bar_colors = [cmap(0), cmap(1)]
        # Position the bar plot in the right bottom corner
        inset_ax = ax.inset_axes([0.75, 0.05, 0.2, 0.2])
        inset_ax.bar(range(len(bar_labels)), counts, color=bar_colors, alpha=0.8)
        inset_ax.set_xticks(range(len(bar_labels)))
        inset_ax.set_xticklabels(bar_labels, fontsize=14)
        inset_ax.get_yaxis().set_visible(False)
        inset_ax.set_title('Sequence count', fontsize=12)
        inset_ax.tick_params(axis='both', which='major', labelsize=12)
        ax.legend(loc="lower center", fontsize=14)

    ax.set_title(title, fontsize=18)
    ax.set_xlabel("Brightness value of a multi mutant", fontsize=16)
    ax.set_ylabel("Sum of brightness values of single mutants", fontsize=16)
    if not hist:
        ax.legend(fontsize=16)
    ax.grid(True)
    

def dotplot_triplet(directory, save_path=None, hist=True, pattern="*GFP*.csv", axes=None, title=None):
    directory = Path(directory)
    files = sorted(directory.glob(pattern))[:3]
    
    created_fig = None
    if axes is None:
        created_fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(24, 8))

    for idx, file in enumerate(files):
        if title is not None:
            dotplot_single(axes[idx], file, file.stem + ', N=' + title, hist=hist)
        else:
            dotplot_single(axes[idx], file, file.stem, hist=hist)

    if created_fig is not None:
        plt.tight_layout()
        if save_path:
            created_fig.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close(created_fig)
        else:
            plt.show()

    return created_fig, axes
    
    
def std_distribution_single(ax, file_path, title, bins=30):
    df = pd.read_csv(file_path)
    epi = df[df["epistatic"] == True]["brightness_stdev"].values
    non = df[df["epistatic"] == False]["brightness_stdev"].values

    ax.hist(epi, bins=bins, alpha=0.5, color="blue", label="epistatic mutants", density=True)
    ax.hist(non, bins=bins, alpha=0.5, color="orange", label="non-epistatic mutants", density=True)

    ax.set_title(title, fontsize=16)
    ax.set_xlabel("Standard deviation of brightness", fontsize=14)
    ax.set_ylabel("Density", fontsize=14)
    ax.grid(True)     
    ax.legend(fontsize=14)


def std_distribution_triplet(directory, save_path=None, bins=30):
    directory = Path(directory)
    file_paths = sorted(directory.glob("*GFP*.csv"))[:3]

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(24, 8))

    for idx, file in enumerate(file_paths):
        title = file.stem
        std_distribution_single(axes[idx], file, title, bins=bins)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def tsuboyama_dotplot_single(ax, csv_path, title=None, hist=False):
    """
    One dataset: observed dG (x) vs reconstructed dG (y),
    with optional inset bar of counts (non-epistatic vs epistatic).
    """
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)

    epi_x  = df[df["epistatic"] == True]["dG"].values
    epi_y  = df[df["epistatic"] == True]["recon_dg"].values
    non_x  = df[df["epistatic"] == False]["dG"].values
    non_y  = df[df["epistatic"] == False]["recon_dg"].values

    cmap = plt.get_cmap("Paired")
    # keep color mapping consistent with your earlier plots
    ax.scatter(epi_x, epi_y, color=cmap(1), alpha=0.8, label="epistasis")
    ax.scatter(non_x, non_y, color=cmap(0), alpha=0.8, label="no epistasis")

    if hist:
        # inset count bar (non-epistatic first, then epistatic)
        counts = [len(non_x), len(epi_x)]
        bar_colors = [cmap(0), cmap(1)]
        inset_ax = ax.inset_axes([0.75, 0.05, 0.2, 0.2])
        inset_ax.bar([0, 1], counts, color=bar_colors, alpha=0.8)
        inset_ax.set_xticks([0, 1])
        inset_ax.set_xticklabels([str(c) for c in counts], fontsize=14)
        inset_ax.get_yaxis().set_visible(False)
        inset_ax.set_title("Sequence count", fontsize=12)
        inset_ax.tick_params(axis="both", which="major", labelsize=12)
        # local legend placement same as before when hist=True
        ax.legend(loc="lower center", fontsize=14)

    ax.set_title(title or convert_name_tsuboyama(csv_path), fontsize=18)
    ax.set_xlabel("dG of a double mutant", fontsize=16)
    ax.set_ylabel("Reconstructed dG of a double mutant", fontsize=16)
    if not hist:
        ax.legend(fontsize=16)
    ax.grid(True)
    

def tsuboyama_dotplot_grid(input_dir, out_dir, chunk_size=12, rows=4, cols=3, hist=True):
    """
    Paginate all CSVs in input_dir into pages of rows×cols scatter plots.
    Saves to out_dir/part_{idx}.png
    """
    input_dir = Path(input_dir)
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(input_dir.glob("*.csv"))
    
    pages = [files[i:i+chunk_size] for i in range(0, len(files), chunk_size)]

    for page_idx, page_files in enumerate(pages):
        fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(20, 30))
        axes = axes.reshape(rows, cols)

        first_handles_labels = None

        for k, csv_path in enumerate(page_files):
            r, c = divmod(k, cols)
            tsuboyama_dotplot_single(axes[r, c], csv_path, title=convert_name_tsuboyama(csv_path), hist=hist)
            if first_handles_labels is None:
                first_handles_labels = axes[r, c].get_legend_handles_labels()

        # hide unused axes on the last page
        for k in range(len(page_files), rows * cols):
            r, c = divmod(k, cols)
            axes[r, c].axis("off")

        plt.tight_layout()
        out_path = out_dir / f"part_{page_idx}.png"
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        

# def tsuboyama_std_distribution_single(ax, file_path, title, bins=30, density=True):
#     df = pd.read_csv(file_path)
#     epi = df[df["epistatic"] == True]["dG_std"].values
#     non = df[df["epistatic"] == False]["dG_std"].values

#     ax.hist(epi, bins=bins, alpha=0.5, color="blue",   label="epistatic mutants",     density=density)
#     ax.hist(non, bins=bins, alpha=0.5, color="orange", label="non-epistatic mutants", density=density)

#     ax.set_title(title, fontsize=16)
#     ax.set_xlabel("Standard deviation of dG", fontsize=14)
#     ax.set_ylabel("Density" if density else "Count", fontsize=14)
#     ax.grid(True)
#     ax.legend(fontsize=14)
        

# def tsuboyama_std_distribution_grid(
#     input_dir,
#     out_dir,
#     rows=4,
#     cols=3,
#     bins=30,
#     density=True
# ):
#     """
#     Paginate all CSVs in input_dir into pages of rows×cols histograms
#     (dG_std for epistatic vs non-epistatic). Saves to out_dir/{prefix}_{idx}.png
#     """
#     input_dir = Path(input_dir)
#     out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)

#     files = sorted(input_dir.glob("*.csv"))
#     chunk_size = rows * cols
#     pages = [files[i:i+chunk_size] for i in range(0, len(files), chunk_size)]

#     for page_idx, page_files in enumerate(pages):
#         fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(20, 30))
#         axes = axes.reshape(rows, cols)

#         first_handles_labels = None

#         for k, csv_path in enumerate(page_files):
#             r, c = divmod(k, cols)
#             title = csv_path.stem
#             tsuboyama_std_distribution_single(
#                 ax=axes[r, c],
#                 file_path=csv_path,
#                 title=title,
#                 bins=bins,
#                 density=density,
#             )
#             if first_handles_labels is None:
#                 first_handles_labels = axes[r, c].get_legend_handles_labels()

#         # hide unused axes on the last page
#         for k in range(len(page_files), rows * cols):
#             r, c = divmod(k, cols)
#             axes[r, c].axis("off")

#         plt.tight_layout()
#         out_path = out_dir / f"part_{page_idx}.png"
#         fig.savefig(out_path, dpi=300, bbox_inches="tight")
#         plt.close(fig)


# for notebook 03

# def combined_plot(input_dir, output_dir):

#     df = pd.read_csv(input_dir / 'somermeyer_best_models.csv', index_col=0)
#     datasets = df.columns.str.replace('_all', '').str.replace('_epistatic', '').unique()

#     for dataset in datasets:
#         all_values = np.abs(df[dataset + '_all'])
#         epistatic_values = np.abs(df[dataset + '_epistatic'])
#         deltas = all_values - epistatic_values

#         dataset_name = convert_name_to_gfp(dataset)

#         df_temp = pd.DataFrame({
#             'model': df.index,
#             'all': all_values,
#             'epistatic': epistatic_values,
#             'delta': deltas
#         }).reset_index(drop=True)

#         df_temp['sort_key'] = df_temp['model'].apply(lambda x: 1 if x in ['Linear_regression', 'MLP'] else 0)
#         df_temp = df_temp.sort_values(by=['sort_key', 'all'], ascending=[True, False])

#         x = np.arange(len(df_temp))
#         width = 0.4
#         cmap = plt.get_cmap('Paired')

#         fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12), sharex=True, gridspec_kw={'height_ratios': [2, 1]})

#         # upper plot
#         ax1.bar(x - width/2, df_temp['all'], width=width, color=cmap(0), label='All points')
#         ax1.bar(x + width/2, df_temp['epistatic'], width=width, color=cmap(1), label='Epistatic points')
#         ax1.set_ylabel('Spearman correlation', fontsize=14)
#         ax1.set_title(f'Spearman correlation for {dataset_name}', fontsize=16)
#         ax1.grid(True, axis='y')
#         ax1.legend(loc='upper left', prop={'size': 12})
#         ax1.tick_params(axis='y', labelsize=12)

#         # lower plot: delta
#         ax2.bar(x, df_temp['delta'], color='cornflowerblue')
#         ax2.set_ylabel('Δ (All - Epistatic)', fontsize=14)
#         ax2.grid(True, axis='y')
#         ax2.tick_params(axis='y', labelsize=12)

#         # common ax
#         ax2.set_xticks(x)
#         ax2.set_xticklabels(
#             [f"$\\bf{{{m}}}$" if m in ['Linear_regression', 'MLP'] else m for m in df_temp['model']],
#             fontsize=12,
#             rotation=90
#         )
#         ax2.set_xlabel('Model', fontsize=14)

#         plt.tight_layout()
#         plt.savefig(output_dir / f'2_combined_plot_{dataset_name}.png', bbox_inches='tight')
#         plt.close()

def combined_plot_somermeyer(input_dir, output_dir):
    df = pd.read_csv(input_dir / 'somermeyer_best_models.csv', index_col=0)
    datasets = df.columns.str.replace('_all', '', regex=False).str.replace('_epistatic', '', regex=False).unique()

    for dataset in datasets:
        all_values = np.abs(df[dataset + '_all'].astype(float))
        epistatic_values = np.abs(df[dataset + '_epistatic'].astype(float))
        deltas = all_values - epistatic_values

        dataset_name = convert_name_to_gfp(dataset)

        df_temp = pd.DataFrame({
            'model': df.index,
            'all': all_values,
            'epistatic': epistatic_values,
            'delta': deltas
        }).reset_index(drop=True)

        # sort: baselines first, then by 'all' desc
        df_temp['sort_key'] = df_temp['model'].apply(lambda x: 1 if x in ['Linear_regression', 'MLP'] else 0)
        df_temp = df_temp.sort_values(by=['sort_key', 'all'], ascending=[True, False])

        x = np.arange(len(df_temp))
        width = 0.4
        cmap = plt.get_cmap('Paired')

        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(16, 12), sharex=True,
            gridspec_kw={'height_ratios': [2, 1]}
        )

        # ---- upper plot: bars ----
        bar_all = ax1.bar(x - width/2, df_temp['all'],       width=width, color=cmap(0), label='All points')
        bar_epi = ax1.bar(x + width/2, df_temp['epistatic'], width=width, color=cmap(1), label='Epistatic points')

        ax1.set_ylabel('Spearman correlation', fontsize=14)
        ax1.set_title(f'Spearman correlation for {dataset_name}', fontsize=16)
        ax1.grid(True, axis='y')
        ax1.tick_params(axis='y', labelsize=12)

        # ---- medians (black lines) ----
        med_all = float(np.nanmedian(df_temp['all'].values))
        med_epi = float(np.nanmedian(df_temp['epistatic'].values))

        line_all = ax1.axhline(med_all, color='red', linestyle='-', linewidth=1.8)
        line_epi = ax1.axhline(med_epi, color='green', linestyle='-', linewidth=1.8)

        # Legend: include bars + median lines
        custom_handles = [
            bar_all.patches[0],  # proxy for "All points" bars
            bar_epi.patches[0],  # proxy for "Epistatic points" bars
            Line2D([0], [0], color='red', linestyle='-', linewidth=1.8),  # median all
            Line2D([0], [0], color='green', linestyle='-', linewidth=1.8),  # median epi
        ]
        custom_labels = ['All points', 'Epistatic points', 'Median (all)', 'Median (epistatic)']
        ax1.legend(custom_handles, custom_labels, loc='upper left', prop={'size': 12})

        # ---- lower plot: delta ----
        ax2.bar(x, df_temp['delta'], color='cornflowerblue')
        ax2.set_ylabel('Δ (All - Epistatic)', fontsize=14)
        ax2.grid(True, axis='y')
        ax2.tick_params(axis='y', labelsize=12)

        # ---- common x labels ----
        ax2.set_xticks(x)
        ax2.set_xticklabels(
            [f"$\\bf{{{m}}}$" if m in ['Linear_regression', 'MLP'] else m for m in df_temp['model']],
            fontsize=12,
            rotation=90
        )
        ax2.set_xlabel('Model', fontsize=14)

        plt.tight_layout()
        plt.savefig(output_dir / f'2_combined_plot_{dataset_name}.png', bbox_inches='tight', dpi=300)
        plt.close()
        

def combined_plot_tsuboyama(input_dir, output_dir, N=None):
    df_models = pd.read_csv(input_dir / "models_evaluation" / "tsuboyama_best_models.csv", index_col=0)
    # Drop baselines if you don’t want them in the boxplot
    df_models.drop(labels=["MLP", "Linear_regression"], axis=0, inplace=True) 
    df_counts = pd.read_csv(input_dir / "counts.csv", index_col=0)

    # Split all vs epistatic
    df_all = np.abs(df_models.filter(like="_all")).copy()
    df_epi = np.abs(df_models.filter(like="_epistatic")).copy()

    # Harmonize column names
    df_all.columns = [extract_dataset_name(c) for c in df_all.columns]
    df_epi.columns = [extract_dataset_name(c) for c in df_epi.columns]

    # Keep only datasets present in both
    common_cols = sorted(set(df_all.columns) & set(df_epi.columns))
    df_all = df_all[common_cols]
    df_epi = df_epi[common_cols]

    # Drop empty datasets
    non_empty_cols = [
        col for col in common_cols
        if not (df_all[col].fillna(0).eq(0).all() or df_epi[col].fillna(0).eq(0).all())
    ]
    df_all = df_all[non_empty_cols]
    df_epi = df_epi[non_empty_cols]

    # Order by median of "all"
    # median_order = df_all.median().sort_values(ascending=False).index.tolist()
    # df_all = df_all[median_order]
    # df_epi = df_epi[median_order]

    # Long format for seaborn
    df_all_melt = df_all.melt(var_name="Model", value_name="Spearman")
    df_all_melt["Point type"] = "All points"
    df_epi_melt = df_epi.melt(var_name="Model", value_name="Spearman")
    df_epi_melt["Point type"] = "Epistatic points"
    df_plot = pd.concat([df_all_melt, df_epi_melt], ignore_index=True)
    
    # df_plot["Model"] = pd.Categorical(df_plot["Model"], categories=median_order, ordered=True)

    # Counts bar (align columns)
    df_counts.columns = [extract_dataset_name(c) for c in df_counts.columns]
    df_counts = df_counts[non_empty_cols]

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(21, 8),
        gridspec_kw={"height_ratios": [3, 1]},
        sharex=True
    )

    # ----- Boxplot
    hue_order = ["All points", "Epistatic points"]
    palette = plt.get_cmap("Paired").colors[:2]

    sns.boxplot(
        data=df_plot,
        x="Model",
        y="Spearman",
        hue="Point type",
        hue_order=hue_order,
        palette=palette,
        dodge=True,
        width=0.6,
        linewidth=1,
        ax=ax1
    )

    # ----- Global median lines
    all_median = df_plot[df_plot["Point type"] == "All points"]["Spearman"].median()
    epi_median = df_plot[df_plot["Point type"] == "Epistatic points"]["Spearman"].median()

    for line in ax1.lines:
        if ax1.lines.index(line) % 6 == 4:
            line.set_linewidth(3)
            line.set_color('black')
            
    ax1.axhline(all_median, color="red", linestyle="-", linewidth=2.5, label="Median (all)")
    ax1.axhline(epi_median, color="green", linestyle="-", linewidth=2.5, label="Median (epistatic)")

    # ----- Legend
    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(handles, labels)

    # ----- Counts bar plot
    ax2.grid(axis="y")
    ax2.bar(df_counts.columns, list(df_counts.loc["counts", :]))
    ax2.set_ylabel("Number of points", fontsize=12)
    ax2.set_xlabel("Dataset", fontsize=14)

    # ----- Labels / titles
    ax1.set_ylabel("Spearman correlation", fontsize=14)
    if N is not None:
        ax1.set_title(f"Boxplot of all vs. epistatic points across models (Tsuboyama datasets, N={N})", fontsize=16)
    else:
        ax1.set_title("Boxplot of all vs. epistatic points across models (Tsuboyama datasets)", fontsize=16)
    ax1.grid(axis="y")

    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(output_dir / "3_tsuboyama_combined_plot.png", bbox_inches="tight", dpi=300)
    plt.close()


def individual_combined_plot_tsuboyama(input_dir, output_dir):
    df = pd.read_csv(input_dir / "models_evaluation" / "tsuboyama_best_models.csv", index_col=0)
    datasets = df.columns.str.replace('_all', '', regex=False).str.replace('_epistatic', '', regex=False).unique()
    df_counts = pd.read_csv(input_dir / "counts.csv", index_col=0)

    for dataset in datasets:
        if df_counts.at["counts", dataset] < 400:
            continue
        all_values = np.abs(df[dataset + '_all'].astype(float))
        epistatic_values = np.abs(df[dataset + '_epistatic'].astype(float))
        deltas = all_values - epistatic_values

        dataset_name = convert_name_tsuboyama(dataset)

        df_temp = pd.DataFrame({
            'model': df.index,
            'all': all_values,
            'epistatic': epistatic_values,
            'delta': deltas
        }).reset_index(drop=True)

        # sort: baselines first, then by 'all' desc
        df_temp['sort_key'] = df_temp['model'].apply(lambda x: 1 if x in ['Linear_regression', 'MLP'] else 0)
        df_temp = df_temp.sort_values(by=['sort_key', 'all'], ascending=[True, False])

        x = np.arange(len(df_temp))
        width = 0.4
        cmap = plt.get_cmap('Paired')

        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(16, 12), sharex=True,
            gridspec_kw={'height_ratios': [2, 1]}
        )

        # ---- upper plot: bars ----
        bar_all = ax1.bar(x - width/2, df_temp['all'],       width=width, color=cmap(0), label='All points')
        bar_epi = ax1.bar(x + width/2, df_temp['epistatic'], width=width, color=cmap(1), label='Epistatic points')

        ax1.set_ylabel('Spearman correlation', fontsize=14)
        ax1.set_title(f'Spearman correlation for {dataset_name}', fontsize=16)
        ax1.grid(True, axis='y')
        ax1.tick_params(axis='y', labelsize=12)

        # ---- medians (black lines) ----
        med_all = float(np.nanmedian(df_temp['all'].values))
        med_epi = float(np.nanmedian(df_temp['epistatic'].values))

        line_all = ax1.axhline(med_all, color='red', linestyle='-', linewidth=1.8)
        line_epi = ax1.axhline(med_epi, color='green', linestyle='-', linewidth=1.8)

        # Legend: include bars + median lines
        custom_handles = [
            bar_all.patches[0],  # proxy for "All points" bars
            bar_epi.patches[0],  # proxy for "Epistatic points" bars
            Line2D([0], [0], color='red', linestyle='-', linewidth=1.8),  # median all
            Line2D([0], [0], color='green', linestyle='-', linewidth=1.8),  # median epi
        ]
        custom_labels = ['All points', 'Epistatic points', 'Median (all)', 'Median (epistatic)']
        ax1.legend(custom_handles, custom_labels, loc='upper left', prop={'size': 12})

        # ---- lower plot: delta ----
        ax2.bar(x, df_temp['delta'], color='cornflowerblue')
        ax2.set_ylabel('Δ (All - Epistatic)', fontsize=14)
        ax2.grid(True, axis='y')
        ax2.tick_params(axis='y', labelsize=12)

        # ---- common x labels ----
        ax2.set_xticks(x)
        ax2.set_xticklabels(
            [f"$\\bf{{{m}}}$" if m in ['Linear_regression', 'MLP'] else m for m in df_temp['model']],
            fontsize=12,
            rotation=90
        )
        ax2.set_xlabel('Model', fontsize=14)

        plt.tight_layout()
        plt.savefig(output_dir / f'4_combined_plot_{dataset_name}.png', bbox_inches='tight', dpi=300)
        plt.close()
        
        
def _hist_mode(x, bins=70):
    """Histogram-based mode (bin center with max count)."""
    if len(x) == 0:
        return None
    counts, edges = np.histogram(x, bins=bins)
    i = np.argmax(counts)
    return 0.5 * (edges[i] + edges[i+1])

def _overlay_fit(ax, x, color, method="kde", label=None, bins=70):
    """
    Draw a smooth density curve on top of the histogram.
    method: 'kde' (preferred), 'gaussian', or None
    """
    if len(x) == 0 or method is None:
        return

    x = np.asarray(x)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return

    xx = np.linspace(x.min(), x.max(), 300)

    if method == "kde":
        # try scipy KDE, otherwise fallback to smoothed histogram
        try:
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(x)
            yy = kde(xx)
        except Exception:
            # normalized histogram -> simple interpolation
            h, edges = np.histogram(x, bins=bins, density=True)
            centers = 0.5 * (edges[:-1] + edges[1:])
            yy = np.interp(xx, centers, h)
    elif method == "gaussian":
        # fit a normal N(mu, sigma)
        mu = np.mean(x)
        sigma = np.std(x)
        if sigma <= 0:
            return
        coef = 1.0 / (np.sqrt(2*np.pi) * sigma)
        yy = coef * np.exp(-0.5 * ((xx - mu) / sigma)**2)
    else:
        return

    ax.plot(xx, yy, lw=2, color=color, alpha=0.9, label=label)
    
    
def tsuboyama_epi_single(
    ax,
    csv_path,
    title=None,
    hist=False,
    bins=70,
    show_modes=True,
    fit=None,  # None | "kde" | "gaussian"
    density=True,
    right_limit=5
):
    df = pd.read_csv(csv_path)

    epi_x  = np.abs(df[df["epistatic"] == True]["thermodynamic_coupling"].values)
    non_x  = np.abs(df[df["epistatic"] == False]["thermodynamic_coupling"].values)

    cmap = plt.get_cmap("Paired")

    # Plot histograms as DENSITY so the curves align in scale
    bins = np.linspace(0, max(epi_x.max() if len(epi_x) else 0, non_x.max() if len(non_x) else 0), 70)

    # Combined distribution (epi + non-epi) in gray
    try:
        all_x = np.concatenate([epi_x, non_x])
    except Exception:
        all_x = epi_x if len(epi_x) else non_x
    if len(all_x):
        ax.hist(all_x, bins=bins, density=density, color="black", alpha=0.8, label="all")

    # Class-specific histograms
    if len(epi_x):
        ax.hist(epi_x, bins=bins, density=density, color="sandybrown", alpha=0.5, label="epistasis")
    if len(non_x):
        ax.hist(non_x, bins=bins, density=density, color=cmap(1), alpha=0.5, label="no epistasis")

    # Optional fitted curves
    _overlay_fit(ax, epi_x, color="sandybrown", method=fit, label=("epi fit" if fit else None), bins=bins)
    _overlay_fit(ax, non_x, color=cmap(1), method=fit, label=("non-epi fit" if fit else None), bins=bins)

    # Optional mode markers (separate for epi/non-epi)
    if show_modes:
        m_epi = _hist_mode(epi_x, bins=bins)
        m_non = _hist_mode(non_x, bins=bins)
        if m_epi is not None:
            ax.axvline(m_epi, color="sandybrown", ls="--", lw=2, alpha=0.9)
            ax.text(m_epi, ax.get_ylim()[1]*0.9, "mode (epi)", color="sandybrown", ha="center", va="top", fontsize=10)
        if m_non is not None:
            ax.axvline(m_non, color=cmap(1), ls="--", lw=2, alpha=0.9)
            ax.text(m_non, ax.get_ylim()[1]*0.82, "mode (non-epi)", color=cmap(1), ha="center", va="top", fontsize=10)

    # Inset count bar (still shows absolute counts)
    if hist:
        counts = [len(non_x), len(epi_x)]
        bar_colors = [cmap(1), "sandybrown"]
        inset_ax = ax.inset_axes([0.74, 0.08, 0.22, 0.22])
        inset_ax.bar([0, 1], counts, color=bar_colors, alpha=0.8)
        inset_ax.set_xticks([0, 1])
        inset_ax.set_xticklabels([str(c) for c in counts], fontsize=12)
        inset_ax.get_yaxis().set_visible(False)
        inset_ax.set_title("Sequence count", fontsize=11)
        inset_ax.tick_params(axis="both", which="major", labelsize=11)
        ax.legend(loc="upper right", fontsize=12)
    else:
        ax.legend(fontsize=12)

    ax.set_title(title or convert_name_tsuboyama(csv_path), fontsize=16)
    ax.set_xlabel("|Thermodynamic coupling|", fontsize=14)
    if density:
        ax.set_ylabel("Density", fontsize=14)  
    else:
        ax.set_ylabel("Counts", fontsize=14)
    ax.set_xlim(right=right_limit)
    ax.grid(True)


def tsuboyama_epi_distribution_plots(
    input_dir,
    output_dir,
    chunk_size=12,
    rows=4,
    cols=3,
    hist=True,
    bins=70,
    show_modes=True,
    fit=None  # None | "kde" | "gaussian"
):
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(Path(input_dir).glob("*.csv"))
    pages = [files[i:i+chunk_size] for i in range(0, len(files), chunk_size)]

    for page_idx, page_files in enumerate(pages):
        fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(30, 20))
        axes = np.asarray(axes).reshape(rows, cols)

        for k, csv_path in enumerate(page_files):
            r, c = divmod(k, cols)
            tsuboyama_epi_single(
                axes[r, c],
                csv_path,
                title=convert_name_tsuboyama(csv_path),
                hist=hist,
                bins=bins,
                show_modes=show_modes,
                fit=fit
            )

        # hide unused axes on the last page
        for k in range(len(page_files), rows * cols):
            r, c = divmod(k, cols)
            axes[r, c].axis("off")

        plt.tight_layout()
        out_path = Path(output_dir) / f"part_{page_idx}.png"
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        
# trash

def _auto_grid(n):
    if n == 0:
        return 1, 1
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    return rows, cols

def _collect_files_by_epi(input_dir, min_epi=None, max_epi=None):
    files = sorted(Path(input_dir).glob("*.csv"))
    keep = []
    for f in files:
        df = pd.read_csv(f)
        epi_count = (df["epistatic"] == True).sum()
        if (min_epi is not None and epi_count < min_epi):
            continue
        if (max_epi is not None and epi_count > max_epi):
            continue
        keep.append((f, epi_count))
    return keep  # list of (path, epi_count)

def plot_epi_hist_gt400_onepage(
    input_dir,
    out_png,
    hist=True,
    bins=70,
    show_modes=True,
    fit=None,      # None | "kde" | "gaussian"
    density=True,
    figsize_scale=6
):

    selected = _collect_files_by_epi(input_dir, min_epi=401, max_epi=None)
    n = len(selected)
    if n == 0:
        print("No datasets with epi > 400.")
        return

    rows, cols = _auto_grid(n)
    fig, axes = plt.subplots(rows, cols, figsize=(cols*figsize_scale, rows*figsize_scale))
    axes = np.atleast_2d(axes).reshape(rows, cols)

    for i, (csv_path, _epi_cnt) in enumerate(selected):
        r, c = divmod(i, cols)
        tsuboyama_epi_single(
            axes[r, c],
            csv_path,
            title=convert_name_tsuboyama(csv_path),
            hist=hist,
            bins=bins,
            show_modes=show_modes,
            fit=fit,
            density=density
        )

    for k in range(n, rows*cols):
        r, c = divmod(k, cols)
        axes[r, c].axis("off")

    plt.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_png} ({n} panels)")

def plot_epi_hist_200to400_onepage(
    input_dir,
    out_png,
    hist=True,
    bins=70,
    show_modes=True,
    fit=None,
    density=True,
    figsize_scale=6
):
    selected = _collect_files_by_epi(input_dir, min_epi=201, max_epi=400)
    n = len(selected)
    if n == 0:
        print("No datasets with 200 < epi ≤ 400.")
        return

    rows, cols = _auto_grid(n)
    fig, axes = plt.subplots(rows, cols, figsize=(cols*figsize_scale, rows*figsize_scale))
    axes = np.atleast_2d(axes).reshape(rows, cols)

    for i, (csv_path, _epi_cnt) in enumerate(selected):
        r, c = divmod(i, cols)
        tsuboyama_epi_single(
            axes[r, c],
            csv_path,
            title=convert_name_tsuboyama(csv_path),
            hist=hist,
            bins=bins,
            show_modes=show_modes,
            fit=fit,
            density=density,
            right_limit=3
        )

    for k in range(n, rows*cols):
        r, c = divmod(k, cols)
        axes[r, c].axis("off")

    plt.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_png} ({n} panels)")
    

def plot_epi_hist_100to200_onepage(
    input_dir,
    out_png,
    hist=True,
    bins=70,
    show_modes=True,
    fit=None,
    density=True,
    figsize_scale=6
):
    """Все датасеты с 200 < epi ≤ 400 в одну фигуру (PNG)."""
    selected = _collect_files_by_epi(input_dir, min_epi=101, max_epi=200)
    n = len(selected)
    if n == 0:
        print("No datasets with 200 < epi ≤ 400.")
        return

    rows, cols = _auto_grid(n)
    fig, axes = plt.subplots(rows, cols, figsize=(cols*figsize_scale, rows*figsize_scale))
    axes = np.atleast_2d(axes).reshape(rows, cols)

    for i, (csv_path, _epi_cnt) in enumerate(selected):
        r, c = divmod(i, cols)
        tsuboyama_epi_single(
            axes[r, c],
            csv_path,
            title=convert_name_tsuboyama(csv_path),
            hist=hist,
            bins=bins,
            show_modes=show_modes,
            fit=fit,
            density=density,
            right_limit=3
        )

    for k in range(n, rows*cols):
        r, c = divmod(k, cols)
        axes[r, c].axis("off")

    plt.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_png} ({n} panels)")