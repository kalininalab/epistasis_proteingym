import pandas as pd

df = pd.read_csv('all_models.csv')
df['mean_all'] = df.filter(like='all').abs().mean(axis=1)
models = ['CARP', 'DeepSequence', 'ESM-IF1', 'ESM1b', 'ESM1v', 'ESM2', 'EVE', 'EVmutation', 'GEMME', 'MIF', 'MSA', 'Progen2', 'ProtGPT2', 'ProteinMPNN', 'RITA', 'TranceptEVE', 'Tranception', 'UniRep', 'VESPA', 'Wavenet', 'ProtSSN', 'LinearRegression']
for model in models:
    model_rows = df[df['model'].str.startswith(model)]
    if not model_rows.empty:
        max_mean_row = model_rows.loc[model_rows['mean_all'].idxmax()]
        df = df.drop(model_rows.index)
        df = pd.concat([df, max_mean_row.to_frame().T], ignore_index=True)
df = df.sort_values(by='model').reset_index(drop=True)
df.to_csv('best_models.csv')

# df = pd.read_csv('all_Spearman_values.csv')
# df = df.drop(columns=[col for col in df.columns if col.endswith('doubles') or col.endswith('multi')])
# df.rename(columns=lambda x: x.replace('Spearman_distinct_1std', '1std') if x.endswith('Spearman_distinct_1std') else x, inplace=True)
# df.rename(columns=lambda x: x.replace('Spearman_distinct_2std', '2std') if x.endswith('Spearman_distinct_2std') else x, inplace=True)
# df.rename(columns=lambda x: x.replace('Spearman_all', 'all') if x.endswith('Spearman_all') else x, inplace=True)
# df = df.sort_values(by='model').reset_index(drop=True)
# df.to_csv('all_models.csv')