#!/usr/bin/env python
"""Plot PLL(AB) distributions for every Tsuboyama assay and selected ESM model."""
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path('/data/users/akolchina/epistasis_proteingym')
BASE = ROOT / 'results/reanalysis/tsuboyama'
OUT = BASE / 'multimodel_mm_pll/summary'
MODELS = {
    'ESM1b': BASE / 'multimodel_mm_pll/scores/ESM1b',
    'ESM1v ensemble': BASE / 'multimodel_mm_pll/scores/ESM1v_ensemble',
    'ESM2-150M': BASE / 'multimodel_mm_pll/scores/ESM2_150M',
    'ESM2-650M': BASE / 'pll_esm2_650m/scores',
}
COLORS = ['#466bb0', '#d98236', '#8959a8', '#c0527d']

frames = []
for model, directory in MODELS.items():
    for path in sorted(directory.glob('*/scores.csv')):
        d = pd.read_csv(path, usecols=['dataset', 'mutated_sequence', 'PLL_AB'])
        d['model'] = model
        d['sequence_length'] = d.mutated_sequence.str.len()
        d['PLL_per_residue'] = d.PLL_AB / d.sequence_length
        frames.append(d.drop(columns='mutated_sequence'))
d = pd.concat(frames, ignore_index=True)
if d.groupby('model').dataset.nunique().ne(50).any():
    raise RuntimeError(f'incomplete assays: {d.groupby("model").dataset.nunique().to_dict()}')

summary = (d.groupby(['model', 'dataset', 'sequence_length'])
           .agg(n=('PLL_AB','size'), mean=('PLL_AB','mean'), sd=('PLL_AB','std'),
                q05=('PLL_AB',lambda x:x.quantile(.05)), q25=('PLL_AB',lambda x:x.quantile(.25)),
                median=('PLL_AB','median'), q75=('PLL_AB',lambda x:x.quantile(.75)),
                q95=('PLL_AB',lambda x:x.quantile(.95)),
                normalized_mean=('PLL_per_residue','mean'),
                normalized_sd=('PLL_per_residue','std'),
                normalized_q05=('PLL_per_residue',lambda x:x.quantile(.05)),
                normalized_q25=('PLL_per_residue',lambda x:x.quantile(.25)),
                normalized_median=('PLL_per_residue','median'),
                normalized_q75=('PLL_per_residue',lambda x:x.quantile(.75)),
                normalized_q95=('PLL_per_residue',lambda x:x.quantile(.95)))
           .reset_index())
OUT.mkdir(parents=True, exist_ok=True)
summary.to_csv(OUT / 'pll_distribution_quantiles.csv', index=False)

datasets = sorted(d.dataset.unique())
short = [x.split('_Tsuboyama')[0] for x in datasets]
for value, suffix, ylabel in [
    ('PLL_AB', 'raw', 'PLL(AB)'),
    ('PLL_per_residue', 'per_residue', 'PLL(AB) / sequence length'),
]:
    fig, axes = plt.subplots(4, 1, figsize=(18, 14), sharex=True, constrained_layout=True)
    for ax, (model, color) in zip(axes, zip(MODELS, COLORS)):
        groups = [d.loc[(d.model == model) & (d.dataset == assay), value].dropna() for assay in datasets]
        bp = ax.boxplot(groups, showfliers=False, patch_artist=True, widths=.7,
                        medianprops={'color':'#172033','linewidth':1.1},
                        whiskerprops={'color':'#697386'}, capprops={'color':'#697386'})
        for box in bp['boxes']:
            box.set(facecolor=color, edgecolor=color, alpha=.75)
        ax.set_title(model, loc='left', weight='bold', color=color)
        ax.set_ylabel(ylabel)
        ax.grid(axis='y', color='#d9e0ea', linewidth=.7, alpha=.8)
        ax.spines[['top','right']].set_visible(False)
    axes[-1].set_xticks(range(1, len(datasets)+1), short, rotation=90, fontsize=7)
    axes[-1].set_xlabel('Tsuboyama assay')
    fig.suptitle('Double-mutant PLL distributions across all Tsuboyama assays', fontsize=18, weight='bold')
    for ext in ['png', 'pdf']:
        fig.savefig(OUT / f'pll_distributions_all_assays_{suffix}.{ext}', dpi=220, bbox_inches='tight')
    plt.close(fig)

print(OUT)
