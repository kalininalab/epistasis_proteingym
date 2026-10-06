#!/usr/bin/env python
"""Reproduce the ESM2-650M three-score assay barplots for every new model."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path('/data/users/akolchina/epistasis_proteingym')
BASE = ROOT / 'results/reanalysis/tsuboyama/multimodel_mm_pll/summary'
OUT = BASE / 'three_score_barplots'
MODELS = ['ESM1b', 'ESM1v_ensemble', 'ESM2_150M', 'ESM2_650M']
METHODS = [('default_mm','Default MM','#4267AC'),
           ('conditional_mm','Conditional MM','#BE577C'),
           ('pll','PLL','#D67F32')]

def short_name(s):
    return s.replace('_Tsuboyama_2023_', '\n').replace('_Tsuboyama_2023','')

def draw(data, model, metric, title_text, stem):
    datasets=sorted(data.dataset.unique()); x=np.arange(len(datasets)); width=.25
    fig,axes=plt.subplots(2,1,figsize=(20,11.5),sharex=True,sharey=True)
    for ax,subset in zip(axes,['all_doubles','epistatic']):
        sub=data[(data.metric==metric)&(data.subset==subset)]
        for offset,(method,label,color) in zip([-1,0,1],METHODS):
            values=sub[sub.method==method].set_index('dataset').reindex(datasets).rho
            ax.bar(x+offset*width,values,width,color=color,label=label)
        ax.axhline(0,color='#64748B',lw=.85); ax.grid(axis='y',alpha=.22); ax.set_axisbelow(True)
        ax.set_ylabel('Spearman ρ')
        ax.set_title('All double mutants' if subset=='all_doubles' else 'Predefined epistatic subset',loc='left',weight='bold')
    axes[0].legend(frameon=False,ncol=3,loc='upper right')
    axes[1].set_xticks(x,[short_name(v) for v in datasets],rotation=65,ha='right',fontsize=8)
    fig.suptitle(f'{title_text}: {model.replace("_"," ")}',fontsize=16,weight='bold',y=.995)
    fig.tight_layout(); target=OUT/model; target.mkdir(parents=True,exist_ok=True)
    fig.savefig(target/f'{stem}.png',dpi=240,bbox_inches='tight')
    fig.savefig(target/f'{stem}.pdf',bbox_inches='tight'); plt.close(fig)

def main():
    d=pd.read_csv(BASE/'per_assay_correlations.csv')
    for model in MODELS:
        sub=d[d.model==model]
        if len(sub)!=600: raise RuntimeError(f'{model}: expected 600 rows, found {len(sub)}')
        draw(sub,model,'raw_fitness','Raw sequence scores vs experimental double-mutant stability','raw_fitness_three_scores')
        draw(sub,model,'fitted_epistasis','Fitted score residuals vs experimental thermodynamic coupling','fitted_epistasis_three_scores')
    print(OUT)

if __name__=='__main__': main()
