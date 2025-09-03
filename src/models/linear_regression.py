import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
from src.constants import set_seed
from src.utils import convert_name_to_gfp, shift_mutation_positions_up

# Set random seed for reproducibility
set_seed()

def linear_regression(result_all, model_dir, epi_dir, dataset_name, seeds=5):
    input_dir = model_dir / dataset_name
    for file_path in input_dir.glob("*.csv"): # iterate over datasets
        df = pd.read_csv(file_path)
        df["num_mutations"] = df["mutant"].apply(lambda x: len(x.split(":")))
        singles = df[df['num_mutations'] == 1].copy()
        multis = df[df['num_mutations'] > 1].copy()
        dataset = file_path.stem
                
        if dataset_name == "somermeyer":
            selected = pd.read_csv(epi_dir / f"{convert_name_to_gfp(dataset)}.csv")
            selected.rename(columns={"aa_genotype": "mutant"}, inplace=True)
            selected['mutant'] = selected['mutant'].apply(shift_mutation_positions_up)
        else:
            selected = pd.read_csv(epi_dir / file_path.name)
        
        selected = selected[['mutant', 'epistatic']]
        multis = multis.merge(selected, on='mutant', how='left')
        epistatic = multis[multis["epistatic"].notna() & multis["epistatic"]]
        
        if len(epistatic) == 0: # no epistasis detected for the dataset
            continue

        seqs_train = singles['mutated_sequence'].apply(lambda x: x.rstrip('*')).tolist()
        y_train = np.array(singles['DMS_score'])

        seqs_epistatic = epistatic['mutated_sequence'].apply(lambda x: x.rstrip('*')).tolist()
        y_test_epistatic = np.array(epistatic['DMS_score'])

        spearman_all = []
        spearman_epistatic = []
        for seed in range(seeds):
            # sample as the same size as epistatic points
            sample = multis.sample(n=len(epistatic), random_state=seed)
            seqs_all = sample['mutated_sequence'].apply(lambda x: x.rstrip('*')).tolist()
            y_test_all = np.array(sample['DMS_score'])

            enc = preprocessing.OneHotEncoder(handle_unknown="ignore")
            max_length = max(max(len(seq) for seq in seqs_train),
                        max(len(seq) for seq in seqs_epistatic),
                        max(len(seq) for seq in seqs_all))
            seqs_train = [list(seq) for seq in seqs_train]
            padded_sequences = [seq + [''] * (max_length - len(seq)) for seq in seqs_train]
            padded_array = np.array(padded_sequences)
            enc.fit(padded_array)
            x_train = enc.transform(padded_array).toarray()

            test_sequences = [list(seq) for seq in seqs_all]
            padded_sequences = [seq + [''] * (max_length - len(seq)) for seq in test_sequences]
            padded_array = np.array(padded_sequences)
            genotype_matrix = enc.transform(padded_array)
            x_test_all = genotype_matrix.toarray()

            test_sequences = [list(seq) for seq in seqs_epistatic]
            padded_sequences = [seq + [''] * (max_length - len(seq)) for seq in test_sequences]
            padded_array = np.array(padded_sequences)
            genotype_matrix = enc.transform(padded_array)
            x_test_epistatic = genotype_matrix.toarray() 

            model = LinearRegression()
            if dataset_name == "somermeyer":
                y_train = np.log1p(y_train)  # Log-transform the target variable
            model.fit(x_train, y_train)

            y_pred_all = model.predict(x_test_all).flatten()
            if dataset_name == "somermeyer":
                y_pred_all = np.expm1(y_pred_all)
            spearman_all.append(spearmanr(y_pred_all, y_test_all)[0])

            y_pred_epistatic = model.predict(x_test_epistatic).flatten()
            if dataset_name == "somermeyer":
                y_pred_epistatic = np.expm1(y_pred_epistatic)
            spearman_epistatic.append(spearmanr(y_pred_epistatic, y_test_epistatic)[0])

        result_all.loc["Linear_regression", dataset + '_all'] = f"{np.mean(spearman_all):.2f}"
        result_all.loc["Linear_regression", dataset + '_epistatic'] = f"{np.mean(spearman_epistatic):.2f}"
        
    return result_all
