import os
from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from .utils import convert_name_to_gfp

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
    plt.figure(figsize=(8, 6))

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
