import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import os
from collections import Counter
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn import metrics
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression, Ridge

suitable_datasets = []
cols = ['model']
with open('/home/ank24/epistasis_prediction_review/selected_datasets.txt', 'r') as file:
    for line in file:
        dataset = line.strip()
        suitable_datasets.append(dataset)
        cols.append(dataset + '_all')
        cols.append(dataset + '_1std')
        cols.append(dataset + '_2std')
result_all = pd.DataFrame(columns=cols)

dfs = {}
selected_mutations = {}
directory = 'data/ProteinGym_subs_apr_2024'

for filename in os.listdir(directory): # iterate over datasets
    file_path = os.path.join(directory, filename)
    dataset = filename.split('.')[0]
    if os.path.isfile(file_path) and dataset in suitable_datasets:
        df = pd.read_csv(file_path)
        indices = {}
        for ind, row in df.iterrows():
            mutations = row['mutant'].split(':')
            if len(mutations) == 2: # find 2 mutations
                indices[(mutations[0], mutations[1], ind)] = [-1, -1] # indices[mutation 1, mutation 2, index of the pair], and inside we store indices of individual mutations

        for ind, row in df.iterrows():
            mutations = row['mutant'].split(':')
            if len(mutations) == 1: # match indices of the 2 mutations to a pair
                for key, value in indices.items():
                    if key[0] == mutations[0]:
                        indices[key][0] = ind
                    if key[1] == mutations[0]:
                        indices[key][1] = ind

        dic_final = {}
        for key, value in indices.items():
            if value[0] != -1 and value[1] != -1:
                dic_final[key] = value

        scaler = MinMaxScaler()
        df['DMS_score_scaled'] = scaler.fit_transform(df['DMS_score'].values.reshape(-1, 1)) # minmax scale values

        x = []
        y = []
        selected = []
        for key, value in dic_final.items():
            ind = key[2]
            x.append(df.loc[ind, 'DMS_score_scaled']) # score for a pair
            selected.append(key[0] + ':' + key[1])
            ind_first = value[0]
            ind_second = value[1]
            y.append(df.loc[ind_first, 'DMS_score_scaled'] * df.loc[ind_second, 'DMS_score_scaled']) # multiplied individuals

        # Calculate distances from the line y = x
        distances = np.abs(np.array(y) - np.array(x)) / np.sqrt(2)
        # Calculate mean and standard deviation of the distances
        mean_distance = np.mean(distances)
        std_distance = np.std(distances)
        # Define a threshold for "distinct" points (e.g., 2 standard deviations from the mean)
        threshold = mean_distance + 1 * std_distance
        # Identify points that are "distinct"
        distinct_points = distances > threshold
        selected_muts_1 = [element for element, mask in zip(selected, distinct_points.tolist()) if mask] # for 1 std

        threshold = mean_distance + 2 * std_distance
        distinct_points = distances > threshold
        selected_muts_2 = [element for element, mask in zip(selected, distinct_points.tolist()) if mask] # for 2 std
        
        selected_mutations[dataset] = (selected_muts_1, selected_muts_2)
        dfs[dataset] = df# Iterate over all model prediction folders in the directory

def calculate_spearman(result_all, entry_path, model):
    score_columns = []
    score_indices = {}
    for filename in os.listdir(entry_path): # let's just open 1 file and check if there're several rows with predictions (different num of parameters)
        file_path = os.path.join(entry_path, filename)
        dataset = filename.split('.')[0]
        if dataset in suitable_datasets: 
            df = dfs[dataset]
            df_pred = pd.read_csv(file_path)
            for col in df_pred.columns:
                if ('mut' not in col) and ('seq' not in col) and ('DMS_score' not in col):
                    score_columns.append(col)
            break
    for score_column in score_columns:
        new_row = pd.DataFrame(dict(zip(list(result_all.columns), [model + '_' + score_column] + [None]*(result_all.shape[1] - 1))), index=[0])
        result_all = pd.concat([result_all, new_row], ignore_index=True) # SAVE INDICES + SCORE NAMES IN A DICT TO APPEND AFTERWARDS
        score_indices[score_column] = result_all.index[-1]
    for filename in os.listdir(entry_path): # iterate over datasets
        file_path = os.path.join(entry_path, filename)
        dataset = filename.split('.')[0]
        if dataset in suitable_datasets: 
            df = dfs[dataset]
            df_pred = pd.read_csv(file_path)
            if 'mutant' in df_pred.columns:
                if 'DMS_score' in df_pred.columns:
                    merged_df = df.merge(df_pred, left_on=['mutant', 'DMS_score'], right_on=['mutant', 'DMS_score'])
                else:
                    merged_df = df.merge(df_pred, left_on='mutant', right_on='mutant')
            else:
                if 'DMS_score' in df_pred.columns:
                    merged_df = df.merge(df_pred, left_on=['mutated_sequence', 'DMS_score'], right_on=['mutated_sequence', 'DMS_score'])
                else:
                    merged_df = df.merge(df_pred, left_on='mutated_sequence', right_on='mutated_sequence')

            for score_column in score_columns: # calculate for all predicted scores
                spearman_all = spearmanr(merged_df['DMS_score'], merged_df[score_column])[0]

                merged_df['num_mutations'] = merged_df['mutant'].str.split(':').apply(len)

                merged_df_doubles = merged_df[merged_df['num_mutations'] == 2]
                spearman_doubles = spearmanr(merged_df_doubles['DMS_score'], merged_df_doubles[score_column])[0]

                merged_df_multi = merged_df[merged_df['num_mutations'] >= 2]
                spearman_multi = spearmanr(merged_df_multi['DMS_score'], merged_df_multi[score_column])[0]
                
                selected_mutations_1, selected_mutations_2 = selected_mutations[dataset]
                merged_df_selected_1 = merged_df[merged_df['mutant'].isin(selected_mutations_1)]
                spearman_distinct_1 = spearmanr(merged_df_selected_1['DMS_score'], merged_df_selected_1[score_column])[0]
                merged_df_selected_2 = merged_df[merged_df['mutant'].isin(selected_mutations_2)]
                spearman_distinct_2 = spearmanr(merged_df_selected_2['DMS_score'], merged_df_selected_2[score_column])[0]

                result_all.loc[score_indices[score_column], dataset + '_all'] = f"{spearman_all:.2f}"
                #result_all.loc[score_indices[score_column], dataset + '_Spearman_doubles'] = f"{spearman_doubles:.2f}"
                #result_all.loc[score_indices[score_column], dataset + '_Spearman_multi'] = f"{spearman_multi:.2f}"
                result_all.loc[score_indices[score_column], dataset + '_1std'] = f"{spearman_distinct_1:.2f}"
                result_all.loc[score_indices[score_column], dataset + '_2std'] = f"{spearman_distinct_2:.2f}"
    return result_all

directory = 'data/Predictions'
for model in os.listdir(directory): # iterate over models
    entry_path = os.path.join(directory, model)
    if os.path.isdir(entry_path) and any(os.path.isdir(os.path.join(entry_path, entry)) for entry in os.listdir(entry_path)): # case when there are subfolders inside a model (different parameter number)
        for folder_name in os.listdir(entry_path): # iterate over submodels
            folder_path = os.path.join(entry_path, folder_name)
            result_all = calculate_spearman(result_all, folder_path, model + '_' + folder_name)
    elif os.path.isdir(entry_path): # standart case
        result_all = calculate_spearman(result_all, entry_path, model)
result_all.to_csv('models_all_Spearman_values.csv')