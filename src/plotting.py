import os
from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from typing import Union, Optional, Tuple, List
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
    ax.set_xlabel("Number of mutations", fontsize=13, labelpad=6)
    ax.set_ylabel("Frequency", fontsize=13, labelpad=6)
    ax.tick_params(axis="both", labelsize=12)

    name = file.name.split(".")[0]
    name = convert_name_to_gfp(name)
    ax.set_title(f"Distribution of the number of mutations for {name}", fontsize=13)
    plt.tight_layout()
    # annotate how many variants are beyond max_k
    ax.text(
        0.98, 0.92,
        f"{tail_n} variants with >{max_k} mutations",
        ha="right", va="top", transform=ax.transAxes, fontsize=13
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
    ax.set_xlabel("Dataset", fontsize=13)
    ax.set_ylabel("Frequency", fontsize=13)
    ax.set_title("Distribution of mutations per Tsuboyama dataset", fontsize=13)
    ax.legend(title="Mutations", title_fontsize=13, fontsize=13)
    ax.tick_params(axis="both", labelsize=12)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
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
     
     
def val_violin_somermeyer(somermeyer_data_dir, save_path=None):
    data = []
    labels = []

    for file in Path(somermeyer_data_dir).glob("*.csv"):
        df = pd.read_csv(file)
        name = convert_name_to_gfp(file.stem)

        vals = df["DMS_score"].dropna().values
        if len(vals) == 0:
            continue

        data.append(vals)
        labels.append(name)

    plt.figure(figsize=(6, 6))
    parts = plt.violinplot(data, showmeans=False, showmedians=True)

    # Add x-axis labels
    plt.xticks(
        ticks=range(1, len(labels) + 1),
        labels=labels,
        fontsize=12
    )

    plt.ylabel("DMS_score", fontsize=13)
    plt.title("Distribution of DMS_score per GFP dataset", fontsize=13)
    plt.grid(alpha=0.4)
    plt.tight_layout()

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
    
    
def val_violin_tsuboyama(tsuboyama_data_dir, save_path=None):
    dict_names = {}

    for file in Path(tsuboyama_data_dir).glob("*.csv"):
        # dataset name = first two parts of filename
        name = "_".join(file.stem.split("_")[:2])
        df = pd.read_csv(file)
        dict_names[name] = df["DMS_score"].dropna().values

    plt.figure(figsize=(12, 6))

    # violin plot requires list of arrays in order
    data = list(dict_names.values())
    labels = list(dict_names.keys())

    parts = plt.violinplot(
        dataset=data,
        showmeans=False,
        showmedians=True,
        showextrema=False
    )

    # color customization
    for body in parts['bodies']:
        body.set_facecolor("lightblue")
        body.set_edgecolor("black")
        body.set_alpha(0.7)

    # median line color
    if "cmedians" in parts:
        parts["cmedians"].set_color("black")
        parts["cmedians"].set_linewidth(2)

    plt.xticks(
        ticks=range(1, len(labels) + 1),
        labels=labels,
        ha="right",
        rotation=90,
        fontsize=12
    )

    plt.ylabel("DMS_score", fontsize=13)
    plt.title("Distribution of DMS_score across Tsuboyama datasets", fontsize=13)
    plt.grid(axis="y", linestyle='--', alpha=0.7)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
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
        counts = [len(epi_x), len(x)]
        bar_labels = [len(epi_x), len(x)]
        bar_colors = [cmap(1), cmap(0)]
        # Position the bar plot in the right bottom corner
        inset_ax = ax.inset_axes([0.75, 0.05, 0.2, 0.2])
        inset_ax.bar(range(len(bar_labels)), counts, color=bar_colors, alpha=0.8)
        inset_ax.set_xticks(range(len(bar_labels)))
        inset_ax.set_xticklabels(bar_labels, fontsize=14)
        inset_ax.get_yaxis().set_visible(False)
        inset_ax.set_title('Sequence count', fontsize=16)
        inset_ax.tick_params(axis='both', which='major', labelsize=14)
        ax.legend(loc="lower center", fontsize=16)

    if title == "amacGFP":
        title = r"GFP from $\mathit{A.\ macrodactyla}$ (amacGFP)"
    if title == "cgreGFP":
        title = r"GFP from $\mathit{C.\ gregaria}$ (cgreGFP)"
    if title == "ppluGFP":
        title = r"GFP from $\mathit{P.\ plumata}$ (ppluGFP)"
    ax.set_title(title, fontsize=18)
    ax.tick_params(axis='both', which='both', labelsize=14)
    ax.set_xlabel("Brightness value of a multi mutant", fontsize=18)
    ax.set_ylabel("Sum of brightness values of single mutants", fontsize=18)
    if not hist:
        ax.legend(fontsize=16)
    ax.grid(True)
    plt.tight_layout()
    

def dotplot_triplet(directory, save_path=None, hist=True, pattern="*GFP*.csv", axes=None, title=None):
    directory = Path(directory)
    files = sorted(directory.glob(pattern))[:3]
    
    created_fig = None
    if axes is None:
        created_fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(27, 9))

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

    if title == "amacGFP":
        title = r"GFP from $\mathit{A.\ macrodactyla}$ (amacGFP)"
    if title == "cgreGFP":
        title = r"GFP from $\mathit{C.\ gregaria}$ (cgreGFP)"
    if title == "ppluGFP":
        title = r"GFP from $\mathit{P.\ plumata}$ (ppluGFP)"
        
    ax.tick_params(axis='both', which='both', labelsize=14)
    ax.set_title(title, fontsize=18)
    ax.set_xlabel("Standard deviation of brightness", fontsize=18)
    ax.set_ylabel("Density", fontsize=18)    
    ax.legend(fontsize=16)
    plt.tight_layout()


def std_distribution_triplet(directory, save_path=None, bins=30):
    directory = Path(directory)
    file_paths = sorted(directory.glob("*GFP*.csv"))[:3]

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(27, 9))

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
        inset_ax.set_title("Sequence count", fontsize=16)
        inset_ax.tick_params(axis="both", which="major", labelsize=14)
        # local legend placement same as before when hist=True
        ax.legend(loc="lower center", fontsize=16)

    ax.tick_params(axis='both', which='both', labelsize=14)
    ax.set_title(title or convert_name_tsuboyama(csv_path), fontsize=18)
    ax.set_xlabel("ΔG of a double mutant", fontsize=18)
    ax.set_ylabel("Reconstructed ΔG of a double mutant", fontsize=18)
    if not hist:
        ax.legend(fontsize=16)
    ax.grid(True)
    plt.tight_layout()
    

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
        fig.savefig(out_dir / f"part_{page_idx}.png", dpi=300, bbox_inches="tight")
        fig.savefig(out_dir / f"part_{page_idx}.pdf",bbox_inches="tight")
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
        #plt.savefig(output_dir / f'2_combined_plot_{dataset_name}.png', bbox_inches='tight', dpi=300)
        plt.savefig(output_dir / f'3_combined_plot_{dataset_name}.pdf', bbox_inches='tight')
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
    
    
def gfp_rank_heatmap(
    models_eval_dir: Union[str, Path],
    source: str = "best",                # "best" -> somermeyer_best_models.csv, "all" -> somermeyer_all_models.csv
    subset: str = "epistatic",           # "epistatic" or "all"
    out_png: Optional[Union[str, Path]] = None,
    out_pdf: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (18, 12),
    annot: bool = True,                  # show rank numbers in cells
    fontsize: int = 18,
    dataset_order: Optional[List[str]] = None  # e.g. ["amacGFP", "cgreGFP", "ppluGFP"]
):
    """
    Build per-dataset rank table (1 = best Spearman) for the GFP datasets,
    plus a 'Mean' rank column. Heatmap colors: green = better (lower rank), red = worse.
    No statistical tests are performed.

    Returns:
      ranks_df (models × [datasets + 'Mean']), winners (Series with Top-1 counts)
    """
    models_eval_dir = Path(models_eval_dir)
    csv_name = "somermeyer_best_models.csv" if source == "best" else "somermeyer_all_models.csv"
    df = pd.read_csv(models_eval_dir / csv_name, index_col=0)

    # pick columns
    suffix = "_epistatic" if subset == "epistatic" else "_all"
    cols = [c for c in df.columns if c.endswith(suffix)]

    if not cols:
        raise ValueError(f"No columns with suffix '{suffix}' found in {csv_name}.")

    # Define your suffix once (edit if needed)
    suffix = "_epistatic"

    # --- Robust column detection + mapping ---
    # 1) Try to find columns that end with the suffix (case-insensitive)
    cand = [c for c in df.columns if c.lower().endswith(suffix.lower())]

    if cand:
        # Map "pretty label" -> actual column with suffix
        mapping = {}
        for c in cand:
            raw = c[: -len(suffix)]
            try:
                pretty = convert_name_to_gfp(raw)
            except NameError:
                pretty = raw
            mapping[pretty] = c
    else:
        # 2) Fallback: no suffix columns found. Try to pick dataset-like columns directly.
        #    Heuristic: columns that contain 'GFP' and are not the '_all' ones.
        guess = [c for c in df.columns if "GFP" in c and not c.lower().endswith("_all")]
        # If user specified order, keep only those that actually exist
        if 'dataset_order' in locals() and dataset_order is not None:
            guess = [c for c in dataset_order if c in df.columns] or guess

        mapping = {}
        for c in guess:
            try:
                pretty = convert_name_to_gfp(c)
            except NameError:
                pretty = c
            mapping[pretty] = c

    # 3) Final dataset list honoring optional user order
    if 'dataset_order' in locals() and dataset_order is not None:
        datasets = [d for d in dataset_order if d in mapping]
    else:
        datasets = sorted(mapping.keys())

    if not datasets:
        raise KeyError("Could not find any dataset columns matching the expected pattern.")

    # 4) Assemble Spearman matrix and rename columns to pretty labels
    spearman = df[[mapping[d] for d in datasets]].apply(pd.to_numeric, errors="coerce")
    spearman.columns = datasets

    # Rank within each dataset (descending Spearman => ascending rank)
    ranks = spearman.rank(axis=0, ascending=False, method="average")

    # Mean rank (lower is better)
    ranks["Mean"] = ranks.mean(axis=1, skipna=True)

    # Sort models by Mean rank (asc), tie-break alphabetically
    order = sorted(ranks.index.tolist(), key=lambda m: (ranks.loc[m, "Mean"], m.lower()))
    ranks = ranks.loc[order]

    # Count how many times each model is rank-1 (top) across datasets
    winners = (ranks[datasets].apply(lambda col: col == col.min(), axis=0)).sum(axis=1)
    winners = winners.loc[order].astype(int)

    # Plot heatmap (green = good/low rank; red = bad/high rank)
    plt.figure(figsize=figsize)
    annot_kws = {"fontsize": fontsize}

    ax = sns.heatmap(
        ranks,
        cmap="RdYlGn",
        vmin=1, vmax=float(ranks[datasets].max().max()),
        annot=annot,
        annot_kws=annot_kws,              # <- make cell numbers match your font size/family
        fmt=".1f" if annot else "",
        cbar_kws={"label": "Rank (lower is better)"},
        linewidths=0.5, linecolor="white", square=False
    )
    
    ax.set_title(f"GFP model ranking by {subset} Spearman (per dataset)", fontsize=fontsize+2)
    ax.set_xlabel("Dataset", fontsize=fontsize)
    ax.set_ylabel("Model", fontsize=fontsize)
    ax.tick_params(axis="x", labelsize=fontsize)
    ax.tick_params(axis="y", labelsize=fontsize)

    # Colorbar fonts
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=fontsize)       # tick font size
    cbar.set_label("Rank (lower is better)", fontsize=fontsize)  # label size

    plt.tight_layout()
    if out_png is not None:
        Path(out_png).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_png, dpi=300, bbox_inches="tight")
    elif out_pdf is not None:
        Path(out_pdf).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_pdf, bbox_inches="tight")
    else:
        plt.show()

    # Small printed summary
    #print("Top-1 counts across datasets (ties count for all tied best):")
    #print(winners.sort_values(ascending=False))

    return ranks, winners    
    

def models_dotplot_single(
    ax,
    csv_path,
    dataset_key,
    title=None,
    abs_vals: bool = False,
    xlim=(-1.0, 1.0),
    ylim=(-1.0, 1.0),
    point_size=80,
    alpha=0.75,
    top_k: int = 5,
    out_topk_csv=None,         # path to save Top-K + baselines (optional)
    list_topk_in_legend=False, # if True: list each Top-K model by name in the legend
    legend_fontsize=14,
    exclude_baselines_in_topk: bool = True,  # NEW: MLP/LinReg don't count toward Top-K
):
    """
    Scatter for a single dataset across models:
      x = Spearman('<dataset_key>_all'), y = Spearman('<dataset_key>_epistatic')

    Highlights:
      - Top-K by epistatic (y) in black (optionally excluding baselines from the K)
      - 'MLP' and 'Linear_regression' in red
    If list_topk_in_legend=True, the legend shows each Top-K model name as a separate black point.
    Saves CSV with Top-K plus baselines if out_topk_csv is provided.
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from pathlib import Path

    df = pd.read_csv(csv_path, index_col=0)

    def _resolve(col_suffix: str) -> str:
        exact = f"{dataset_key}{col_suffix}"
        if exact in df.columns:
            return exact
        cand = [c for c in df.columns
                if c.lower().endswith(col_suffix.lower())
                and dataset_key.lower() in c.lower()]
        if len(cand) == 1:
            return cand[0]
        if len(cand) == 0:
            raise KeyError(
                f"No column matching '{dataset_key}{col_suffix}' "
                f"or containing '{dataset_key}' with suffix '{col_suffix}'."
            )
        raise KeyError(f"Multiple matches for '{dataset_key}{col_suffix}': {cand}")

    col_all = _resolve("_all")
    col_epi = _resolve("_epistatic")

    x_raw = pd.to_numeric(df[col_all], errors="coerce").to_numpy()
    y_raw = pd.to_numeric(df[col_epi],  errors="coerce").to_numpy()
    names_raw = df.index.to_numpy(dtype=object)

    x = x_raw.copy(); y = y_raw.copy(); names = names_raw.copy()
    if abs_vals:
        x = np.abs(x); y = np.abs(y)
        x_raw = np.abs(x_raw); y_raw = np.abs(y_raw)

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]; y = y[mask]; names = names[mask]

    cmap = plt.get_cmap("Paired")
    ax.scatter(x, y, s=point_size, color=cmap(1), alpha=alpha, edgecolors="none", zorder=1)

    if x.size: ax.axvline(float(np.median(x)), color="gray", ls="--", lw=1.2, alpha=0.9, zorder=0)
    if y.size: ax.axhline(float(np.median(y)), color="gray", ls="--", lw=1.2, alpha=0.9, zorder=0)

    # Top-K by epistatic (desc), optionally excluding baselines from K
    if y.size:
        order = np.argsort(-y)  # descending
        if exclude_baselines_in_topk:
            order = [i for i in order if names[i] not in {"MLP", "Linear_regression"}]
        k = min(top_k, len(order))
        top_idx = np.array(order[:k], dtype=int) if k > 0 else np.array([], dtype=int)
    else:
        top_idx = np.array([], dtype=int)

    top_mask = np.zeros_like(y, dtype=bool)
    if top_idx.size:
        top_mask[top_idx] = True

    red_mask = (names == "MLP") | (names == "Linear_regression")

    if top_mask.any():
        ax.scatter(x[top_mask], y[top_mask],
                   s=point_size, color="black", alpha=0.95, zorder=3)
    if red_mask.any():
        ax.scatter(x[red_mask], y[red_mask],
                   s=point_size, color="#E74C3C", alpha=0.95, zorder=4)

    if out_topk_csv is not None:
        rows = []
        if top_mask.any():
            for xi, yi, ni in zip(x[top_mask], y[top_mask], names[top_mask]):
                rows.append({"model": ni, "spearman_all": xi, "spearman_epistatic": yi, "group": "top_k"})
        for base in ["MLP", "Linear_regression"]:
            if base in names_raw:
                b_idx = np.where(names_raw == base)[0][0]
                rows.append({
                    "model": base,
                    "spearman_all": x_raw[b_idx],
                    "spearman_epistatic": y_raw[b_idx],
                    "group": "baseline"
                })
        top_df = pd.DataFrame(rows)
        if not top_df.empty:
            top_df = top_df.drop_duplicates(subset=["model"], keep="first")
            Path(out_topk_csv).parent.mkdir(parents=True, exist_ok=True)
            top_df.to_csv(out_topk_csv, index=False)

    if list_topk_in_legend and top_mask.any():
        handles = [Line2D([0], [0], linestyle="--", color="gray", lw=1.6, label="median Spearman")]
        if red_mask.any():
            handles.append(Line2D([0], [0], marker="o", linestyle="None", markersize=8,
                                  color="#E74C3C", label="baseline models"))
        for ni in names[top_mask]:
            handles.append(Line2D([0], [0], marker="o", linestyle="None", markersize=8,
                                  color="black", label=str(ni)))
        ax.legend(handles=handles, loc="upper left", frameon=True, framealpha=0.9,
                  fontsize=legend_fontsize, scatterpoints=1)
    else:
        legend_handles = [
            Line2D([0], [0], linestyle="--", color="gray", lw=1.6, label="median Spearman"),
            Line2D([0], [0], marker="o", linestyle="None", markersize=8, color="black", label=f"top-{top_k} epistatic"),
            Line2D([0], [0], marker="o", linestyle="None", markersize=8, color="#E74C3C", label="baseline models"),
        ]
        ax.legend(handles=legend_handles, loc="upper left", frameon=True, framealpha=0.9,
                  fontsize=legend_fontsize, scatterpoints=1)

    if title is None:
        try:
            from .utils import convert_name_to_gfp
            title = f"{convert_name_to_gfp(dataset_key)}"
        except Exception:
            title = dataset_key

    if title == "amacGFP":
        title = r"GFP from $\mathit{A.\ macrodactyla}$ (amacGFP)"
    if title == "cgreGFP":
        title = r"GFP from $\mathit{C.\ gregaria}$ (cgreGFP)"
    if title == "ppluGFP":
        title = r"GFP from $\mathit{P.\ plumata}$ (ppluGFP)"

    ax.set_title(title, fontsize=20)
    ax.set_xlabel(r"Spearman $\rho$ (all genotypes)", fontsize=18)
    ax.set_ylabel(r"Spearman $\rho$ (epistatic genotypes)", fontsize=18)
    ax.set_xlim(xlim); ax.set_ylim(ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.tick_params(axis='both', which='both', labelsize=18)
    ax.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    

def _resolve_cols(df: pd.DataFrame, dataset_key: str) -> tuple[str, str]:
    """Return('<dataset_key>_all','<dataset_key>_epistatic') with robust matching."""
    def _resolve(col_suffix: str) -> str:
        exact = f"{dataset_key}{col_suffix}"
        if exact in df.columns:
            return exact
        cand = [c for c in df.columns
                if c.lower().endswith(col_suffix.lower())
                and dataset_key.lower() in c.lower()]
        if len(cand) == 1:
            return cand[0]
        if len(cand) == 0:
            raise KeyError(
                f"No column matching '{dataset_key}{col_suffix}' "
                f"or containing '{dataset_key}' with suffix '{col_suffix}'."
            )
        raise KeyError(f"Multiple matches for '{dataset_key}{col_suffix}': {cand}")
    return _resolve("_all"), _resolve("_epistatic")


def collect_topk_and_baselines(
    df: pd.DataFrame,
    dataset_key: str,
    *,
    abs_vals: bool = False,
    top_k: int = 5,
    exclude_baselines_in_topk: bool = True,
) -> pd.DataFrame:
    """
    Return a table with rows:
      - Top-K (by epistatic Spearman) non-baseline models
      - Baselines: MLP, Linear_regression (included even if NaN)
    Columns: dataset_key, dataset_pretty, model, group, rank_in_topk, spearman_all, spearman_epistatic
    """
    col_all, col_epi = _resolve_cols(df, dataset_key)

    x = pd.to_numeric(df[col_all], errors="coerce").to_numpy()
    y = pd.to_numeric(df[col_epi], errors="coerce").to_numpy()
    names = df.index.to_numpy(object)

    if abs_vals:
        x = np.abs(x); y = np.abs(y)

    valid = np.isfinite(x) & np.isfinite(y)
    x_ok, y_ok, names_ok = x[valid], y[valid], names[valid]

    order = np.argsort(-y_ok)  # descending epistatic Spearman
    if exclude_baselines_in_topk:
        order = [i for i in order if names_ok[i] not in {"MLP", "Linear_regression"}]

    pick = order[:min(top_k, len(order))]

    rows = []
    # Top-K non-baselines
    for rank, idx in enumerate(pick, start=1):
        rows.append({
            "dataset_key": dataset_key,
            "model": names_ok[idx],
            "group": "top_k",
            "rank_in_topk": rank,
            "spearman_all": float(x_ok[idx]),
            "spearman_epistatic": float(y_ok[idx]),
        })

    # Baselines (use original arrays to include even if NaN)
    for base in ["MLP", "Linear_regression"]:
        if base in names:
            bidx = int(np.where(names == base)[0][0])
            rows.append({
                "dataset_key": dataset_key,
                "model": base,
                "group": "baseline",
                "rank_in_topk": np.nan,
                "spearman_all": float(x[bidx]) if np.isfinite(x[bidx]) else np.nan,
                "spearman_epistatic": float(y[bidx]) if np.isfinite(y[bidx]) else np.nan,
            })

    out = pd.DataFrame(rows)

    # add pretty names
    try:
        from .utils import convert_name_tsuboyama, convert_name_to_gfp
        if "_Tsuboyama_" in dataset_key:
            out["dataset_pretty"] = convert_name_tsuboyama(dataset_key)
        else:
            out["dataset_pretty"] = convert_name_to_gfp(dataset_key)
    except Exception:
        out["dataset_pretty"] = dataset_key

    return out
    
    
from pathlib import Path
from typing import Union, Optional, Tuple, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# assumes models_dotplot_single and convert_name_to_gfp are already imported in this module

def models_dotplot_triplet(
    csv_path: Union[str, Path],
    dataset_keys: Optional[List[str]] = None,   # e.g. ["D7PM05_CLYGR_Somermeyer_2022", ...]
    out_png: Optional[Union[str, Path]] = None,
    out_pdf: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (18, 6),
    abs_vals: bool = False,
    share_limits: bool = True,
    point_size: int = 80,
    alpha: float = 0.75,
    # --- NEW: combined table options ---
    out_table_csv: Optional[Union[str, Path]] = None,  # save one combined table for the 3 datasets
    top_k: int = 5,
    exclude_baselines_in_topk: bool = True,
):
    """
    Build a 1×3 figure of dotplots for three datasets using column bases as-is.
    Titles are pretty-printed via convert_name_to_gfp(base).

    NEW: if out_table_csv is provided, also save a combined table of Top-K (by epistatic)
         and baselines (MLP, Linear_regression) for the three datasets.
    """
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path, index_col=0)

    bases_all = {c[:-4] for c in df.columns if c.endswith("_all")}
    bases_epi = {c[:-10] for c in df.columns if c.endswith("_epistatic")}
    available = sorted(bases_all & bases_epi)
    if len(available) < 3:
        raise ValueError(f"Need at least 3 dataset bases with both suffixes; found {len(available)}: {available}")

    if dataset_keys is None:
        use = available[:3]
    else:
        missing = [b for b in dataset_keys if b not in available]
        if missing:
            raise KeyError(f"Requested bases not found (need both columns): {missing}\nAvailable: {available}")
        if len(dataset_keys) != 3:
            raise ValueError(f"Provide exactly 3 dataset_keys; got {len(dataset_keys)}.")
        use = dataset_keys

    # collect per-panel data for shared limits and for the combined table
    x_list, y_list = [], []
    panel_arrays = {}  # base -> (x, y, names)
    for base in use:
        x = pd.to_numeric(df[f"{base}_all"], errors="coerce").to_numpy()
        y = pd.to_numeric(df[f"{base}_epistatic"], errors="coerce").to_numpy()
        if abs_vals:
            x = np.abs(x); y = np.abs(y)
        m = np.isfinite(x) & np.isfinite(y)
        x, y = x[m], y[m]
        names = df.index.to_numpy(dtype=object)[m]
        x_list.append(x); y_list.append(y)
        panel_arrays[base] = (x, y, names)

    if share_limits:
        if abs_vals:
            lo, hi = 0.0, 1.0
        else:
            vals = [v for arr in (x_list + y_list) for v in (arr if arr.size else [])]
            lo = max(-1.0, (min(vals) if len(vals) else -1.0))
            hi = min( 1.0, (max(vals) if len(vals) else  1.0))
            pad = 0.02 * (hi - lo if hi > lo else 1.0)
            lo, hi = lo - pad, hi + pad
        xlim = ylim = (lo, hi)
    else:
        xlim = ylim = None  # models_dotplot_single expects tuples; keep share_limits=True for safety

    fig, axes = plt.subplots(1, 3, figsize=figsize)
    for ax, base in zip(axes, use):
        pretty_title = convert_name_to_gfp(base)
        # keep plot behavior unchanged
        models_dotplot_single(
            ax=ax,
            csv_path=csv_path,
            dataset_key=base,          # lookup by raw base
            title=pretty_title,        # title uses convert_name_to_gfp
            abs_vals=abs_vals,
            xlim=xlim,
            ylim=ylim,
            point_size=point_size,
            alpha=alpha,
            out_topk_csv=f"{base}.csv"  # per-panel CSV, unchanged
        )

    plt.tight_layout()
    if out_png is not None:
        Path(out_png).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_png, dpi=300, bbox_inches="tight")
        plt.close(fig)
    elif out_pdf is not None:
        Path(out_pdf).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_pdf, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()

    # --- NEW: build one combined table (Top-K + baselines) across the 3 datasets ---
    if out_table_csv is not None:
        rows = []
        for base in use:
            x, y, names = panel_arrays[base]
            # order by epistatic desc
            order = np.argsort(-y)
            if exclude_baselines_in_topk:
                order = [i for i in order if names[i] not in {"MLP", "Linear_regression"}]
            k = min(top_k, len(order))
            top_idx = np.array(order[:k], dtype=int) if k > 0 else np.array([], dtype=int)

            # Top-K entries with rank among non-baselines
            for rank, idx in enumerate(top_idx, start=1):
                rows.append({
                    "dataset_key": base,
                    "dataset_pretty": convert_name_to_gfp(base) if 'convert_name_to_gfp' in globals() else base,
                    "model": names[idx],
                    "spearman_all": x[idx],
                    "spearman_epistatic": y[idx],
                    "group": "top_k",
                    "rank_epistatic": rank,
                })

            # Baselines (include even if NaN was dropped earlier — read from full df)
            for base_model in ["MLP", "Linear_regression"]:
                if base_model in df.index:
                    xa = pd.to_numeric(df.loc[base_model, f"{base}_all"], errors="coerce")
                    ya = pd.to_numeric(df.loc[base_model, f"{base}_epistatic"], errors="coerce")
                    rows.append({
                        "dataset_key": base,
                        "dataset_pretty": convert_name_to_gfp(base) if 'convert_name_to_gfp' in globals() else base,
                        "model": base_model,
                        "spearman_all": float(np.abs(xa) if abs_vals and np.isfinite(xa) else xa),
                        "spearman_epistatic": float(np.abs(ya) if abs_vals and np.isfinite(ya) else ya),
                        "group": "baseline",
                        "rank_epistatic": np.nan,
                    })

        table = pd.DataFrame(rows)
        Path(out_table_csv).parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(out_table_csv, index=False)

    return fig, axes, use


def models_dotplot_tsuboyama(
    csv_path: Union[str, Path],
    out_png: Optional[Union[str, Path]] = None,
    out_pdf: Optional[Union[str, Path]] = None,
    abs_vals: bool = False,
    rows: int = 7,
    cols: int = 7,
    point_size: int = 50,
    alpha: float = 0.75,
    counts_csv: Optional[Union[str, Path]] = None,   # 2-row CSV you showed (header=dataset keys, row 'counts')
    thr_mid: int = 200,                               # >200 -> light yellow
    thr_hi: int = 400,                                # >400 -> light green
    color_mid: str = "#fff7cc",                       # light yellow
    color_hi: str = "#eaffea",                        # light green
    list_topk_in_legend: bool = True,
    # --- NEW: combined table options ---
    out_table_csv: Optional[Union[str, Path]] = None, # save Top-K+baselines for all panels
    top_k: int = 5,
    exclude_baselines_in_topk: bool = True,
):
    """
    7×7 grid of scatter plots:
      x = Spearman rho (…_all), y = Spearman rho (…_epistatic)
    Titles unchanged in size. Axis labels only on left column (y) and bottom row (x).
    Global axis limits shared across all panels.

    Background coloring (if counts_csv provided):
      count > thr_hi  -> color_hi
      count > thr_mid -> color_mid

    Also writes one combined table of Top-K (by epistatic) and baselines per dataset if out_table_csv is set.
    """

    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path, index_col=0)

    # datasets that have both columns
    bases_all = {c[:-4]  for c in df.columns if c.endswith("_all")}
    bases_epi = {c[:-10] for c in df.columns if c.endswith("_epistatic")}
    bases = sorted(bases_all & bases_epi)

    need = rows * cols
    if len(bases) < need:
        raise ValueError(f"Need at least {need} datasets with both _all and _epistatic; found {len(bases)}.")
    bases = bases[:need]

    # counts for backgrounds
    counts_map = {}
    if counts_csv is not None:
        cc = pd.read_csv(counts_csv, index_col=0)
        if "counts" in cc.index:
            s = cc.loc["counts"]
        else:
            s = cc.iloc[0]
        s = pd.to_numeric(s, errors="coerce")
        counts_map = s.to_dict()  # keys must match 'base' strings

    # global axis limits
    xs, ys = [], []
    for base in bases:
        xv = pd.to_numeric(df[f"{base}_all"], errors="coerce").to_numpy()
        yv = pd.to_numeric(df[f"{base}_epistatic"], errors="coerce").to_numpy()
        if abs_vals:
            xv, yv = np.abs(xv), np.abs(yv)
        m = np.isfinite(xv) & np.isfinite(yv)
        xs.append(xv[m]); ys.append(yv[m])

    if abs_vals:
        xlim = ylim = (0.0, 1.0)
    else:
        allv = np.concatenate([*xs, *ys]) if (xs and ys) else np.array([])
        if allv.size:
            lo = max(-1.0, float(np.nanmin(allv)))
            hi = min( 1.0, float(np.nanmax(allv)))
            pad = 0.02 * (hi - lo if hi > lo else 1.0)
        else:
            lo, hi, pad = -1.0, 1.0, 0.0
        xlim = ylim = (lo - pad, hi + pad)

    fig, axes = plt.subplots(rows, cols, figsize=(cols*6, rows*6))
    axes = np.asarray(axes).reshape(rows, cols)

    # stash arrays for table
    panel_arrays = {}

    for k, base in enumerate(bases):
        r, c = divmod(k, cols)
        ax = axes[r, c]

        # background
        cnt = counts_map.get(base, np.nan)
        if np.isfinite(cnt):
            if cnt > thr_hi:
                ax.set_facecolor(color_hi)
            elif cnt > thr_mid:
                ax.set_facecolor(color_mid)

        title = convert_name_tsuboyama(base)

        # draw single panel
        models_dotplot_single(
            ax=ax,
            csv_path=csv_path,
            dataset_key=base,
            title=title,
            abs_vals=abs_vals,
            xlim=xlim,
            ylim=ylim,
            point_size=point_size,
            alpha=alpha,
            top_k=top_k,
            list_topk_in_legend=list_topk_in_legend,
        )

        # keep arrays for table
        x_all = pd.to_numeric(df[f"{base}_all"], errors="coerce").to_numpy()
        y_epi = pd.to_numeric(df[f"{base}_epistatic"], errors="coerce").to_numpy()
        if abs_vals:
            x_all = np.abs(x_all); y_epi = np.abs(y_epi)
        names = df.index.to_numpy(dtype=object)
        m = np.isfinite(x_all) & np.isfinite(y_epi)
        panel_arrays[base] = (x_all[m], y_epi[m], names[m])

        # edge labels only
        if c != 0:
            ax.set_ylabel("")
        if r != rows - 1:
            ax.set_xlabel("")

    # hide unused axes
    for k in range(len(bases), rows*cols):
        r, c = divmod(k, cols)
        axes[r, c].axis("off")

    plt.tight_layout()
    if out_png:
        Path(out_png).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_png, dpi=300, bbox_inches="tight")
    if out_pdf:
        Path(out_pdf).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_pdf, bbox_inches="tight")

    # build combined table (Top-K + baselines) if requested
    if out_table_csv is not None:
        rows_out = []
        for base in bases:
            x, y, names = panel_arrays[base]
            # rank by epistatic descending
            order = np.argsort(-y)
            if exclude_baselines_in_topk:
                order = [i for i in order if names[i] not in {"MLP", "Linear_regression"}]
            ksel = min(top_k, len(order))
            top_idx = np.array(order[:ksel], dtype=int) if ksel > 0 else np.array([], dtype=int)

            for rank, idx in enumerate(top_idx, start=1):
                rows_out.append({
                    "dataset_key": base,
                    "dataset_pretty": convert_name_tsuboyama(base),
                    "model": names[idx],
                    "spearman_all": x[idx],
                    "spearman_epistatic": y[idx],
                    "group": "top_k",
                    "rank_epistatic": rank,
                })

            # baselines from full df (even if NaN originally)
            for base_model in ["MLP", "Linear_regression"]:
                if base_model in df.index:
                    xa = pd.to_numeric(df.loc[base_model, f"{base}_all"], errors="coerce")
                    ya = pd.to_numeric(df.loc[base_model, f"{base}_epistatic"], errors="coerce")
                    if abs_vals and np.isfinite(xa): xa = float(abs(xa))
                    if abs_vals and np.isfinite(ya): ya = float(abs(ya))
                    rows_out.append({
                        "dataset_key": base,
                        "dataset_pretty": convert_name_tsuboyama(base),
                        "model": base_model,
                        "spearman_all": xa,
                        "spearman_epistatic": ya,
                        "group": "baseline",
                        "rank_epistatic": np.nan,
                    })

        table = pd.DataFrame(rows_out)
        Path(out_table_csv).parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(out_table_csv, index=False)

    return fig, axes, bases