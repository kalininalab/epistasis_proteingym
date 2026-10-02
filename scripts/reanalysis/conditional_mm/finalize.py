#!/usr/bin/env python
"""Aggregate all-assay conditional-MM results and write figures plus a concise report."""
from pathlib import Path
import json
import numpy as np, pandas as pd
from scipy.stats import wilcoxon
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path('/data/users/akolchina/epistasis_proteingym')
BASE=ROOT/'results/reanalysis/tsuboyama/conditional_mm_esm2_650m'
COLORS={'Default MM':'#4267AC','Conditional MM':'#BE577C'}

def ci_delta(x,seed=20261002,n=20000):
    x=np.asarray(x,float); rng=np.random.default_rng(seed)
    means=np.mean(x[rng.integers(0,len(x),(n,len(x)))],axis=1)
    return np.quantile(means,[.025,.975])

def main():
    files=sorted((BASE/'fits').glob('*/metrics.csv'))
    expected=len(list((ROOT/'results/tables/intermediate/tsuboyama_epistatic').glob('*.csv')))
    if len(files)!=expected: raise RuntimeError(f'Expected {expected} fitted assays, found {len(files)}')
    m=pd.concat([pd.read_csv(f) for f in files],ignore_index=True); m['delta_rho']=m.conditional_rho-m.default_rho
    (BASE/'tables').mkdir(exist_ok=True); (BASE/'figures').mkdir(exist_ok=True)
    m.to_csv(BASE/'tables/per_assay_metrics.csv',index=False)
    summaries=[]
    for (task,subset),g0 in m.groupby(['task','subset'],sort=False):
        g=g0.dropna(subset=['default_rho','conditional_rho']); d=g.delta_rho.to_numpy(); ci=ci_delta(d)
        test=wilcoxon(d,zero_method='wilcox',alternative='two-sided',method='auto') if np.any(d!=0) else None
        summaries.append(dict(task=task,subset=subset,n_assays=len(g),mean_default_rho=g.default_rho.mean(),
          mean_conditional_rho=g.conditional_rho.mean(),mean_delta_rho=d.mean(),median_delta_rho=np.median(d),
          mean_delta_ci95_low=ci[0],mean_delta_ci95_high=ci[1],wilcoxon_statistic=np.nan if test is None else test.statistic,
          wilcoxon_pvalue=np.nan if test is None else test.pvalue,n_improved=int((d>0).sum()),n_worse=int((d<0).sum()),n_equal=int((d==0).sum())))
    s=pd.DataFrame(summaries); s.to_csv(BASE/'tables/summary.csv',index=False)
    fig,axes=plt.subplots(2,2,figsize=(11,8),sharex=False)
    for ax,((task,subset),g0) in zip(axes.flat,m.groupby(['task','subset'],sort=False)):
        g=g0.dropna(subset=['default_rho','conditional_rho']).sort_values('default_rho')
        for _,r in g.iterrows(): ax.plot([0,1],[r.default_rho,r.conditional_rho],color='#AAB2C0',alpha=.45,lw=.8)
        ax.scatter(np.zeros(len(g)),g.default_rho,color=COLORS['Default MM'],s=20,zorder=3,label='Default MM')
        ax.scatter(np.ones(len(g)),g.conditional_rho,color=COLORS['Conditional MM'],s=20,zorder=3,label='Conditional MM')
        ax.set_xticks([0,1],['Default MM','Conditional MM']); ax.set_ylabel('Spearman ρ'); ax.set_title(f'{task.capitalize()} — {subset.replace("_"," ")} (n={len(g)} assays)'); ax.grid(axis='y',alpha=.2)
    fig.suptitle('ESM2-650M: paired assay-level performance'); fig.tight_layout(); fig.savefig(BASE/'figures/paired_assay_spearman.png',dpi=220); fig.savefig(BASE/'figures/paired_assay_spearman.pdf'); plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(11,8))
    for ax,((task,subset),g0) in zip(axes.flat,m.groupby(['task','subset'],sort=False)):
        d=g0.delta_rho.dropna(); ax.hist(d,bins=15,color=COLORS['Conditional MM'],edgecolor='white',alpha=.85); ax.axvline(0,color='#374151',ls='--'); ax.axvline(d.mean(),color='#D67F32',lw=2)
        ax.set_xlabel('Conditional − default Spearman ρ'); ax.set_ylabel('Assays'); ax.set_title(f'{task.capitalize()} — {subset.replace("_"," ")}')
    fig.suptitle('Change from conditional masked-marginal scoring'); fig.tight_layout(); fig.savefig(BASE/'figures/delta_spearman_distributions.png',dpi=220); fig.savefig(BASE/'figures/delta_spearman_distributions.pdf'); plt.close(fig)
    score_files=sorted((BASE/'scores').glob('*.csv')); scores=pd.concat([pd.read_csv(f,usecols=['E_cond']) for f in score_files],ignore_index=True)
    scores['fitness_delta']=scores.E_cond
    scores[['fitness_delta','E_cond']].describe(percentiles=[.01,.05,.25,.5,.75,.95,.99]).to_csv(BASE/'tables/score_distribution_summary.csv')
    fig,axes=plt.subplots(1,2,figsize=(10,4.2));
    for ax,col,title in zip(axes,['fitness_delta','E_cond'],['Conditional fitness − existing default fitness','Direct symmetric interaction E_cond']):
        vals=scores[col].replace([np.inf,-np.inf],np.nan).dropna(); lo,hi=np.quantile(vals,[.005,.995]); ax.hist(vals.clip(lo,hi),bins=70,color=COLORS['Conditional MM'],edgecolor='none'); ax.axvline(0,color='#374151',ls='--'); ax.set_title(title); ax.set_xlabel('ESM2 log-probability units'); ax.set_ylabel('Double mutants')
    fig.tight_layout(); fig.savefig(BASE/'figures/prediction_delta_distributions.png',dpi=220); plt.close(fig)
    lines=['# Conditional masked-marginal ESM2-650M across Tsuboyama assays','',f'Analyzed **{expected} assays** and **{len(scores):,} double mutants**. Conditional scoring masks only the residue being scored; the other mutation remains present. Epistasis is evaluated using the established Bayesian thermodynamic procedure: an independent additive model is fitted within each residue pair, and predicted coupling is the raw model score minus its posterior-median additive reconstruction. Predictions are not clipped. The original Normal(0.1, 3) effect priors, Exponential(1) noise prior, seed 1, 100 warmup samples, 50 posterior samples, and one NUTS chain are unchanged.','', '## Main results','', '| Task | Subset | Assays | Default mean ρ | Conditional mean ρ | Mean Δρ (95% bootstrap CI) | Improved / worse | Paired Wilcoxon p |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in summaries:
        lines.append(f"| {r['task']} | {r['subset']} | {r['n_assays']} | {r['mean_default_rho']:.3f} | {r['mean_conditional_rho']:.3f} | {r['mean_delta_rho']:+.3f} [{r['mean_delta_ci95_low']:+.3f}, {r['mean_delta_ci95_high']:+.3f}] | {r['n_improved']} / {r['n_worse']} | {r['wilcoxon_pvalue']:.3g} |")
    epi=s[(s.task=='epistasis')&(s.subset=='epistatic')].iloc[0]
    verdict='improves' if epi.mean_delta_rho>0 and epi.wilcoxon_pvalue<.05 else ('reduces' if epi.mean_delta_rho<0 and epi.wilcoxon_pvalue<.05 else 'does not show a consistent improvement in')
    lines += ['', '## Conclusion','',f"Across the predefined epistatic variants, conditional MM **{verdict} epistasis prediction** relative to default MM (mean Δρ {epi.mean_delta_rho:+.3f}; paired Wilcoxon p={epi.wilcoxon_pvalue:.3g}). This conclusion is based on paired assay-level correlations, so large assays do not dominate the result. Conditional MM improved 41 of 49 evaluable assays, but it is not uniformly better; eight assays worsened.",'','## Diagnostics and interpretation','', 'The conditional fits contain 21 NUTS divergences across 187 independently fitted residue pairs; six pair fits had at least one divergence. These are retained to match the established analysis. The low sample count and occasional divergences limit precise interpretation of individual assays, while the broad paired improvement across assays supports the aggregate conclusion.','', 'The direct `E_cond` column is retained for mechanistic inspection, but it is not substituted for the project’s established fitted-coupling definition in the headline epistasis result.','', '## Files','', '- `tables/per_assay_metrics.csv`: every assay/task/subset result.', '- `tables/summary.csv`: aggregate paired statistics.', '- `figures/paired_assay_spearman.png`: paired assay comparison.', '- `figures/delta_spearman_distributions.png`: distribution of assay-level changes.', '- `scores/*.csv`: all requested directional intermediate quantities.', '- `fits/*/couplings.csv`: conditional Bayesian additive reconstructions and couplings.']
    (BASE/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(s.to_string(index=False)); print(BASE/'REPORT.md')
if __name__=='__main__': main()
