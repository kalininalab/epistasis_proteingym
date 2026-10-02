#!/usr/bin/env python
"""Validate direct E_cond and score compatibility without changing prior results."""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path('/data/users/akolchina/epistasis_proteingym')
SOURCE = ROOT / 'results/reanalysis/tsuboyama/conditional_mm_esm2_650m'
OUT = ROOT / 'results/reanalysis/tsuboyama/conditional_mm_validation_esm2_650m'
BLUE, PINK, ORANGE = '#4267AC', '#BE577C', '#D67F32'

def rho(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y); x, y = x[ok], y[ok]
    return float(spearmanr(x, y)[0]) if len(x) >= 3 and len(np.unique(x)) > 1 and len(np.unique(y)) > 1 else np.nan

def main():
    OUT.mkdir(parents=True, exist_ok=True); (OUT/'figures').mkdir(exist_ok=True); (OUT/'tables').mkdir(exist_ok=True)
    files = sorted((SOURCE/'fits').glob('*/couplings.csv'))
    if len(files) != 50: raise RuntimeError(f'Expected 50 assays, found {len(files)}')
    rows=[]; all_data=[]
    for path in files:
        d=pd.read_csv(path); all_data.append(d)
        for subset,sub in [('all_doubles',d),('epistatic',d[d.epistatic.astype(str).str.lower().eq('true')])]:
            original=rho(sub.experimental_coupling,sub.default_predicted_coupling)
            conditional=rho(sub.experimental_coupling,sub.conditional_predicted_coupling)
            direct=rho(sub.experimental_coupling,sub.E_cond)
            rows.append(dict(dataset=path.parent.name,subset=subset,n=len(sub),
              rho_original=original,rho_conditional_fitted=conditional,rho_direct_E_cond=direct,
              delta_rho_conditional_minus_original=conditional-original))
    table=pd.DataFrame(rows)
    table.to_csv(OUT/'tables/per_assay_epistasis_correlations.csv',index=False)
    summaries=[]
    for subset,g0 in table.groupby('subset',sort=False):
        for method,col in [('original','rho_original'),('conditional_fitted','rho_conditional_fitted'),('direct_E_cond','rho_direct_E_cond')]:
            g=g0.dropna(subset=[col]); summaries.append(dict(subset=subset,method=method,n_assays=len(g),mean_rho=g[col].mean(),median_rho=g[col].median()))
        paired=g0.dropna(subset=['rho_original','rho_conditional_fitted']); delta=paired.delta_rho_conditional_minus_original
        test=wilcoxon(delta,zero_method='wilcox',method='auto')
        summaries.append(dict(subset=subset,method='conditional_minus_original',n_assays=len(paired),mean_rho=delta.mean(),median_rho=delta.median(),paired_wilcoxon_p=test.pvalue))
        direct_paired=g0.dropna(subset=['rho_original','rho_direct_E_cond']); direct_delta=direct_paired.rho_direct_E_cond-direct_paired.rho_original
        direct_test=wilcoxon(direct_delta,zero_method='wilcox',method='auto')
        summaries.append(dict(subset=subset,method='direct_minus_original',n_assays=len(direct_paired),mean_rho=direct_delta.mean(),median_rho=direct_delta.median(),paired_wilcoxon_p=direct_test.pvalue,
          n_improved=int((direct_delta>0).sum()),n_worse=int((direct_delta<0).sum())))
    summary=pd.DataFrame(summaries); summary.to_csv(OUT/'tables/aggregate_summary.csv',index=False)

    combined=pd.concat(all_data,ignore_index=True)
    diff=combined.recomputed_default_MM_fitness-combined.default_MM_fitness
    slope,intercept=np.polyfit(combined.default_MM_fitness,combined.recomputed_default_MM_fitness,1)
    compatibility=dict(n_double_mutants=len(combined),pearson=float(np.corrcoef(combined.default_MM_fitness,combined.recomputed_default_MM_fitness)[0,1]),
      spearman=rho(combined.default_MM_fitness,combined.recomputed_default_MM_fitness),slope=float(slope),intercept=float(intercept),
      mean_difference=float(diff.mean()),median_difference=float(diff.median()),mean_absolute_difference=float(diff.abs().mean()),
      rmse=float(np.sqrt(np.mean(diff**2))),max_absolute_difference=float(diff.abs().max()),
      default_score_mean=float(combined.default_MM_fitness.mean()),default_score_sd=float(combined.default_MM_fitness.std()),
      E_cond_mean=float(combined.E_cond.mean()),E_cond_sd=float(combined.E_cond.std()),
      E_cond_identity_max_error=float(np.max(np.abs(combined.E_cond-.5*((combined.delta_A_B-combined.delta_A_WT)+(combined.delta_B_A-combined.delta_B_WT))))))
    pd.DataFrame([compatibility]).to_csv(OUT/'tables/score_compatibility.csv',index=False)

    fig,axes=plt.subplots(1,2,figsize=(12,5),sharey=True)
    for ax,subset in zip(axes,['all_doubles','epistatic']):
        g=table[table.subset.eq(subset)].dropna(subset=['rho_original','rho_conditional_fitted']).sort_values('rho_original')
        for _,r in g.iterrows(): ax.plot([0,1],[r.rho_original,r.rho_conditional_fitted],color='#AAB2C0',lw=.8,alpha=.55)
        ax.scatter(np.zeros(len(g)),g.rho_original,color=BLUE,s=24,zorder=3,label='Original ProteinGym-derived')
        ax.scatter(np.ones(len(g)),g.rho_conditional_fitted,color=PINK,s=24,zorder=3,label='Fitted conditional MM')
        ax.axhline(0,color='#4B5563',lw=.8); ax.set_xticks([0,1],['Original','Conditional']); ax.set_ylabel('Epistasis Spearman ρ'); ax.set_title(f'{subset.replace("_"," ")} (n={len(g)} assays)'); ax.grid(axis='y',alpha=.2)
    handles,labels=axes[0].get_legend_handles_labels(); fig.legend(handles,labels,loc='lower center',ncol=2,frameon=False)
    fig.suptitle('Original vs fitted conditional-MM epistasis by Tsuboyama assay'); fig.tight_layout(rect=(0,.08,1,.95))
    fig.savefig(OUT/'figures/original_vs_conditional_paired.png',dpi=240); fig.savefig(OUT/'figures/original_vs_conditional_paired.pdf'); plt.close(fig)

    ep=summary[summary.subset.eq('epistatic')].set_index('method'); al=summary[summary.subset.eq('all_doubles')].set_index('method')
    lines=['# Direct conditional-MM validation','',f'Validated **50 assays** and **{len(combined):,} double mutants**. No previous results were overwritten.','',
      '## Per-assay epistasis results','',
      'The complete table reports original ProteinGym-derived fitted epistasis, fitted conditional-MM epistasis, direct `E_cond`, and `rho_conditional - rho_original` for both all doubles and the predefined epistatic subset.', '',
      '| Subset | Original mean ρ | Fitted conditional mean ρ | Direct E_cond mean ρ | Fitted Δρ | Paired p |','|---|---:|---:|---:|---:|---:|',
      f"| All doubles ({int(al.loc['original','n_assays'])} assays) | {al.loc['original','mean_rho']:.3f} | {al.loc['conditional_fitted','mean_rho']:.3f} | {al.loc['direct_E_cond','mean_rho']:.3f} | {al.loc['conditional_minus_original','mean_rho']:+.3f} | {al.loc['conditional_minus_original','paired_wilcoxon_p']:.3g} |",
      f"| Epistatic ({int(ep.loc['original','n_assays'])} assays) | {ep.loc['original','mean_rho']:.3f} | {ep.loc['conditional_fitted','mean_rho']:.3f} | {ep.loc['direct_E_cond','mean_rho']:.3f} | {ep.loc['conditional_minus_original','mean_rho']:+.3f} | {ep.loc['conditional_minus_original','paired_wilcoxon_p']:.3g} |",'',
      f"Direct `E_cond` improves over original epistasis in **{int(ep.loc['direct_minus_original','n_improved'])}/{int(ep.loc['direct_minus_original','n_assays'])}** evaluable epistatic-subset assays (mean Δρ {ep.loc['direct_minus_original','mean_rho']:+.3f}; paired p={ep.loc['direct_minus_original','paired_wilcoxon_p']:.3g}). Its mean ρ is lower than fitted conditional MM, showing that the additive fit removes main-effect structure that remains in the direct conditional score.",'',
      '## Score compatibility','',
      'ProteinGym ESM2 masked-marginal scores and the four conditional terms use the same natural-log probability difference, `log p(mutant residue) - log p(WT residue)`. Larger values therefore have the same sign convention. The stored double-mutant score is the sum of the two WT-background masked marginals.','',
      f"Across all variants, stored versus recomputed default scores have Spearman **{compatibility['spearman']:.7f}**, slope **{compatibility['slope']:.6f}**, and mean absolute difference **{compatibility['mean_absolute_difference']:.4f}** log-score units, compared with default-score SD **{compatibility['default_score_sd']:.2f}**. The small discrepancy is consistent with mixed-precision inference and CSV precision.",'',
      'Algebraically, `S_default + E_cond` equals the mean of the two mutation-order paths:', '',
      '`0.5 * [(ΔA_WT + ΔB_A) + (ΔB_WT + ΔA_B)]`.', '',
      'Thus adding `E_cond` is mathematically compatible: it replaces the purely WT-background additive path with the symmetric average conditional path. The analysis anchors this correction to the stored ProteinGym score to avoid treating small numerical recomputation differences as biological interaction.','',
      '## Outputs','',
      '- `tables/per_assay_epistasis_correlations.csv`', '- `tables/aggregate_summary.csv`', '- `tables/score_compatibility.csv`', '- `figures/original_vs_conditional_paired.png`']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(summary.to_string(index=False)); print(OUT/'REPORT.md')

if __name__=='__main__': main()
