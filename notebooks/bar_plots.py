import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import seaborn as sns
import os

suitable_datasets = []
with open('selected_datasets.txt', 'r') as file:
    for line in file:
        dataset = line.strip()
        suitable_datasets.append(dataset)

df = pd.read_csv('all_models.csv', index_col=1)
df.drop(columns='Unnamed: 0', inplace=True)
for dataset in suitable_datasets:
    all = np.abs(df[dataset + '_Spearman_all'])
    std1 = np.abs(df[dataset + '_Spearman_distinct_1std'])
    std2 = np.abs(df[dataset + '_Spearman_distinct_2std'])
    cmap = plt.get_cmap('Paired')
    #plt.figure(figsize=[34,38])
    categories = df.index  # Coordinates on the x-axis
    x = np.arange(len(categories))  # Positions for the groups
    width = 0.2  # Width of the bars
    fig, ax = plt.subplots(figsize=(28, 8))
    # Plot each set of bars with a specific x offset
    ax.bar(x - width, all, width, color='midnightblue', label='All points')
    ax.bar(x, std1, width, color=cmap(1), label='Distant points (1 std)')
    ax.bar(x + width, std2, width, color=cmap(0), label='Distant points (2 std)')
    # Add labels, title, and legend
    ax.set_xlabel('Model', fontsize=14)
    ax.set_ylabel('Spearman correlation', fontsize=14)
    ax.set_title(f'Spearman correlations for {dataset}', fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.grid(True, axis='y')
    plt.xticks(rotation=90)
    plt.legend(prop={'size': 14})
    plt.savefig(f'../figures/plots_models/{dataset}.svg', bbox_inches='tight')
    plt.close()