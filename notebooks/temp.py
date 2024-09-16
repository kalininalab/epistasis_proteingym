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

directory = '/home/ank24/epistasis_prediction_review/data/ProteinGym_subs_apr_2024'

suitable_datasets = []
with open('/home/ank24/epistasis_prediction_review/selected_datasets.txt', 'r') as file:
    for line in file:
        dataset = line.strip()
        suitable_datasets.append(dataset)

doubles = dict(zip(suitable_datasets, [0 for i in range(len(suitable_datasets))]))
triplets = dict(zip(suitable_datasets, [0 for i in range(len(suitable_datasets))]))
quads = dict(zip(suitable_datasets, [0 for i in range(len(suitable_datasets))]))

for filename in os.listdir(directory): # iterate over datasets
    file_path = os.path.join(directory, filename)
    dataset = filename.split('.')[0]
    if dataset in suitable_datasets:
        df = pd.read_csv(file_path)
        df['split_count'] = df['mutant'].str.split(':').apply(len)
        for ind, row in df.iterrows():
            mutations = row['mutant'].split(':')
            if len(mutations) == 2: # find 2 mutations
                doubles[dataset] += 1
            if len(mutations) == 3: # find 3 mutations
                triplets[dataset] += 1
            if len(mutations) == 4: # find 4 mutations
                quads[dataset] += 1
            if len(mutations) == 10: # find 4 mutations
                print('nu kek')
