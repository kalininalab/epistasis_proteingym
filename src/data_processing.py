import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import FunctionTransformer
from Bio import SeqIO

# Calculate weighted mean and weighted stdev
def weighted_stats(group):
    weights = group["total_cell_count"]
    brightness = group["brightness"]
    stdev = group["brightness_stdev"]

    # Weighted mean
    weighted_mean = np.average(brightness, weights=weights)

    # Weighted standard deviation of the mean:
    # Combine individual standard deviations using weights
    weighted_var = np.sum((stdev ** 2) * (weights ** 2)) / (weights.sum() ** 2)
    weighted_stdev = np.sqrt(weighted_var)
    return pd.Series({
        "aa_genotype": group.name,
        "brightness": weighted_mean,
        "brightness_stdev": weighted_stdev
    })

def table_parser(df):
    # Group by aa_genotype
    grouped = df.groupby("aa_genotype")
    result = grouped.apply(weighted_stats, include_groups=False)
    result.reset_index(drop=True, inplace=True)

    # Calculate delta from wt and remove it
    wt_row = result.loc[result["aa_genotype"] == "wt"]

    if wt_row.empty:
        print("No wild type (wt) genotype found in the data.")
        return {}
    wt_brightness = wt_row["brightness"].values[0]
    wt_brightness_stdev = wt_row["brightness_stdev"].values[0]

    result["brightness"] -= wt_brightness
    result["brightness_stdev"] = np.sqrt(result["brightness_stdev"] ** 2 + wt_brightness_stdev ** 2)

    # Remove rows that are wt or contain 'wt' in the label
    result = result[~result["aa_genotype"].str.contains("wt", case=False, na=False)].reset_index(drop=True)

    # Count number of mutations
    result["num_mutations"] = result["aa_genotype"].apply(lambda x: len(x.split(":")))
    return result


# Recreate mutated seq from wt_seq + aaMutations
def genotype_to_seq(df, wt_seqs, gene_name, verbose=False):
    new_seqs = []
    # genotype to seq
    for index, row in df.iterrows():

        wt_seq=wt_seqs[gene_name]
        new_seq = str(wt_seq)

        if row["aa_genotype"]=="wt":
            aaMutations=[]
        else:
            aaMutations = [(elt[0], int(elt[1:-1]), elt[-1]) for elt in str(row["aa_genotype"]).split(":")]

        if verbose:
            print(new_seq)
            print(len(new_seq))
            print(aaMutations)

        for elt in aaMutations:
            new_seq = new_seq[0:elt[1]]+elt[2]+new_seq[elt[1]+1:]
            if verbose: print("*"*elt[1])

        if verbose:
            print(new_seq)
            print(len(new_seq))
            print("\n\n\n")

        new_seqs.append(new_seq)

    new_seqs = pd.DataFrame(new_seqs,
                            columns=["sequence"],
                            index=df.index)

    return(new_seqs)


def add_dg(df, name, table):
    """
    Filters and processes a DataFrame based on mutation data and thermodynamic parameters.

    Args:
        df (pd.DataFrame): The input DataFrame containing mutation data.
        name (str): The name of the dataset to filter from the table.
        table (pd.DataFrame): The table containing thermodynamic parameters for mutations.

    Returns:
        pd.DataFrame: A DataFrame with additional columns for the number of mutations, 
                      deltaG values, associated CI, and pair_name for double mutants.
    """
    # Create a copy of the input DataFrame to avoid modifying the original
    df = df.copy()

    # Add a column to count the number of mutations in each row
    df["num_mutations"] = df["mutant"].apply(lambda x: len(x.split(":")))

    # Initialize new columns for deltaG and error values
    df["dG"] = None
    df["deltaG_95CI_low"] = None
    df["deltaG_95CI_high"] = None
    df["pair_name"] = None  # New column to store pair_name for double mutants

    # Filter the table for rows matching the given name
    table_filtered = table[table["name"].str.contains(name, case=False, na=False)].copy()

    # Dictionaries to store deltaG values for single and double mutations
    single_mut_map = {}
    double_mut_map = {}

    # Populate the dictionaries with data from the filtered table
    for _, row in table_filtered.iterrows():
        mut_type = row.get("mut_type")
        deltaG = row.get("deltaG")
        deltaG_95CI_low = row.get("deltaG_95CI_low")
        deltaG_95CI_high = row.get("deltaG_95CI_high")
        pair_name = row.get("pair_name")  # Get pair_name if available

        # Handle double mutations
        if isinstance(mut_type, str) and len(mut_type.split(":")) == 2:
            muts = mut_type.split(":")
            if len(muts) == 2:
                key = tuple(sorted(muts))  # Sort to ensure consistent key ordering
                double_mut_map[key] = (deltaG, deltaG_95CI_low, deltaG_95CI_high, pair_name)

        # Handle single mutations
        elif isinstance(mut_type, str) and len(mut_type.split(":")) == 1:
            single_mut_map[mut_type] = (deltaG, deltaG_95CI_low, deltaG_95CI_high)

    # Define a lookup function to assign deltaG and error values to rows in the DataFrame
    def lookup(row):
        if row["num_mutations"] == 2:
            # For double mutations, look up the deltaG value using the sorted key
            muts = row["mutant"].split(":")
            key = tuple(sorted(muts))
            val = double_mut_map.get(key)
            if val:
                # Assign deltaG, error values, and pair_name to the corresponding columns
                df.at[row.name, "deltaG_95CI_low"] = val[1]
                df.at[row.name, "deltaG_95CI_high"] = val[2]
                df.at[row.name, "pair_name"] = val[3]
                return val[0]
        elif row["num_mutations"] == 1:
            # For single mutations, look up deltaG and error values
            mut = row["mutant"]
            val = single_mut_map.get(mut)
            if val:
                # Assign error values to the corresponding columns
                df.at[row.name, "deltaG_95CI_low"] = val[1]
                df.at[row.name, "deltaG_95CI_high"] = val[2]
                return val[0]
        return None

    # Apply the lookup function to calculate deltaG for each row
    df["dG"] = df.apply(lookup, axis=1)

    return df