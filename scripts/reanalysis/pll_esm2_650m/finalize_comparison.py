#!/usr/bin/env python
"""Combine original, corrected conditional-MM, and PLL epistasis results."""
from pathlib import Path
import json
import numpy as np, pandas as pd
from scipy.stats import wilcoxon
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path('/data/users/akolchina/epistasis_proteingym')
PLL=ROOT/'results/reanalysis/tsuboyama/pll_esm2_650m'
COND=ROOT/'results/reanalysis/tsuboyama/conditional_mm_direct_path_esm2_650m'
OUT=ROOT/'results/reanalysis/tsuboyama/esm2_650m_epistasis_method_comparison'
METHODS=[('rho_original','Original ProteinGym','#4267AC'),('rho_direct_E_cond','Direct E_cond','#D67F32'),('rho_conditional_direct_fitted','Fitted conditional MM','#BE577C'),('rho_direct_epsilon_PLL','Direct epsilon_PLL','#6B9E78'),('rho_PLL_fitted_epistasis','Fitted PLL(AB)','#7A68A6')]
def boot_ci(x,seed=20261004,n=20000):
 x=np.asarray(x,float); rng=np.random.default_rng(seed); b=np.mean(x[rng.integers(0,len(x),(n,len(x)))],axis=1); return np.quantile(b,[.025,.975])
def main():
 cf=sorted((COND/'fits').glob('*/metrics.csv')); pf=sorted((PLL/'fits').glob('*/metrics.csv'))
 if len(cf)!=50 or len(pf)!=50: raise RuntimeError(f'Expected 50+50 metrics, found {len(cf)}+{len(pf)}')
 c=pd.concat([pd.read_csv(f) for f in cf],ignore_index=True); p=pd.concat([pd.read_csv(f) for f in pf],ignore_index=True)
 d=c.merge(p,on=['dataset','subset'],suffixes=('_conditional','_pll'),validate='one_to_one')
 if not (d.n_conditional==d.n_pll).all(): raise ValueError('variant filtering differs')
 d=d.rename(columns={'n_conditional':'n'}).drop(columns='n_pll'); OUT.mkdir(parents=True,exist_ok=True); (OUT/'tables').mkdir(exist_ok=True); (OUT/'figures').mkdir(exist_ok=True)
 d.to_csv(OUT/'tables/per_assay_correlations.csv',index=False)
 rows=[]
 for subset,g0 in d.groupby('subset',sort=False):
  for col,label,_ in METHODS:
   g=g0.dropna(subset=['rho_original',col]); vals=g[col]; delta=vals-g.rho_original; ci=boot_ci(vals); dci=boot_ci(delta)
   test=None if col=='rho_original' else wilcoxon(delta,zero_method='wilcox',method='auto')
   rows.append(dict(subset=subset,method=label,n_assays=len(g),mean_rho=vals.mean(),median_rho=vals.median(),mean_rho_ci95_low=ci[0],mean_rho_ci95_high=ci[1],mean_delta_vs_original=delta.mean(),delta_ci95_low=dci[0],delta_ci95_high=dci[1],n_improved_vs_original=int((delta>0).sum()),n_worse_vs_original=int((delta<0).sum()),wilcoxon_vs_original_p=np.nan if test is None else test.pvalue))
 summary=pd.DataFrame(rows); summary.to_csv(OUT/'tables/aggregate_summary.csv',index=False)
 comparisons=[]
 pairs=[('Fitted conditional MM','rho_conditional_direct_fitted','Fitted PLL(AB)','rho_PLL_fitted_epistasis'),('Direct E_cond','rho_direct_E_cond','Direct epsilon_PLL','rho_direct_epsilon_PLL'),('Fitted conditional MM','rho_conditional_direct_fitted','Direct E_cond','rho_direct_E_cond'),('Fitted conditional MM','rho_conditional_direct_fitted','Direct epsilon_PLL','rho_direct_epsilon_PLL')]
 for subset,g0 in d.groupby('subset',sort=False):
  for left,lcol,right,rcol in pairs:
   g=g0[[lcol,rcol]].dropna(); delta=g[lcol]-g[rcol]; test=wilcoxon(delta,zero_method='wilcox',method='auto'); ci=boot_ci(delta)
   comparisons.append(dict(subset=subset,left_method=left,right_method=right,n_assays=len(g),mean_delta_rho=delta.mean(),median_delta_rho=delta.median(),delta_ci95_low=ci[0],delta_ci95_high=ci[1],left_better=int((delta>0).sum()),right_better=int((delta<0).sum()),wilcoxon_p=test.pvalue))
 pd.DataFrame(comparisons).to_csv(OUT/'tables/paired_method_comparisons.csv',index=False)
 fig,axes=plt.subplots(1,2,figsize=(14,5.5),sharey=True)
 rng=np.random.default_rng(7)
 for ax,subset in zip(axes,['all_doubles','epistatic']):
  g=d[d.subset.eq(subset)]; data=[]; labels=[]; colors=[]
  for col,label,color in METHODS:
   data.append(g[col].dropna()); labels.append(label.replace(' ','\n')); colors.append(color)
  bp=ax.boxplot(data,patch_artist=True,widths=.55,showfliers=False,medianprops={'color':'#172033','linewidth':1.5})
  for box,color in zip(bp['boxes'],colors): box.set_facecolor(color); box.set_alpha(.35); box.set_edgecolor(color)
  for i,(vals,color) in enumerate(zip(data,colors),1): ax.scatter(i+rng.normal(0,.055,len(vals)),vals,s=16,color=color,alpha=.72,zorder=3)
  common_n=len(g[[col for col,_,_ in METHODS]].dropna())
  ax.axhline(0,color='#64748B',lw=.8); ax.set_xticks(range(1,len(labels)+1),labels); ax.set_ylabel('Assay-level Spearman ρ'); ax.set_title(f'{subset.replace("_"," ")} ({common_n} evaluable assays)'); ax.grid(axis='y',alpha=.2)
 fig.suptitle('ESM2-650M epistasis prediction across Tsuboyama assays'); fig.tight_layout(); fig.savefig(OUT/'figures/all_methods_assay_spearman.png',dpi=240); fig.savefig(OUT/'figures/all_methods_assay_spearman.pdf'); plt.close(fig)
 fig,axes=plt.subplots(1,2,figsize=(12,5),sharey=True)
 for ax,subset in zip(axes,['all_doubles','epistatic']):
  g=d[d.subset.eq(subset)].dropna(subset=['rho_original','rho_conditional_direct_fitted'])
  for _,r in g.iterrows(): ax.plot([0,1],[r.rho_original,r.rho_conditional_direct_fitted],color='#B5BDCA',alpha=.55,lw=.8)
  ax.scatter(np.zeros(len(g)),g.rho_original,color='#4267AC',s=24,zorder=3); ax.scatter(np.ones(len(g)),g.rho_conditional_direct_fitted,color='#BE577C',s=24,zorder=3)
  ax.axhline(0,color='#64748B',lw=.8); ax.set_xticks([0,1],['Original','Corrected\nconditional MM']); ax.set_ylabel('Assay-level Spearman ρ'); ax.set_title(f'{subset.replace("_"," ")} ({len(g)} assays)'); ax.grid(axis='y',alpha=.2)
 fig.suptitle('Original vs corrected fitted conditional-MM epistasis'); fig.tight_layout(); fig.savefig(OUT/'figures/original_vs_corrected_conditional_paired.png',dpi=240); fig.savefig(OUT/'figures/original_vs_corrected_conditional_paired.pdf'); plt.close(fig)
 runtime_files=sorted((PLL/'scores').glob('*/runtime.json')); runtimes=pd.DataFrame([json.loads(f.read_text()) for f in runtime_files]); runtimes.to_csv(OUT/'tables/pll_runtime_per_assay.csv',index=False)
 diag=pd.concat([pd.read_csv(f) for f in sorted((PLL/'fits').glob('*/fit_diagnostics.csv'))],ignore_index=True)
 ep=summary[summary.subset.eq('epistatic')].set_index('method'); al=summary[summary.subset.eq('all_doubles')].set_index('method')
 lines=['# ESM2-650M epistasis comparison: masked marginals, conditional MM, and PLL','',f'Compared **50 Tsuboyama assays** and **{int(d[d.subset.eq("all_doubles")].n.sum()):,} double mutants** with identical assay/variant filtering. The predefined epistatic subset is evaluable in 49 assays.','', '## Main result (mean assay-level Spearman ρ)','', '| Method | All doubles | Epistatic subset | Δρ vs original on epistatic subset | Improved / worse assays | Paired p |','|---|---:|---:|---:|---:|---:|']
 for _,label,_ in METHODS:
  r=ep.loc[label]; lines.append(f"| {label} | {al.loc[label,'mean_rho']:.3f} | {r.mean_rho:.3f} | {r.mean_delta_vs_original:+.3f} | {int(r.n_improved_vs_original)} / {int(r.n_worse_vs_original)} | {'—' if pd.isna(r.wilcoxon_vs_original_p) else f'{r.wilcoxon_vs_original_p:.3g}'} |")
 lines += ['', '## Conclusion','', 'Corrected fitted conditional MM is the strongest of the tested ESM2-650M epistasis scores. On the epistatic subset it exceeds fitted `PLL(AB)` by mean Δρ +0.224 (42/49 assays; paired p=9.05e-08). Direct conditional interaction and direct PLL inclusion–exclusion both carry modest positive signal; direct `E_cond` is slightly better than direct `epsilon_PLL` by mean Δρ +0.006 (34/49 assays; p=0.0256). Fitting an additive model to raw `PLL(AB)` does not improve epistasis prediction over the original ProteinGym-derived baseline. Full-sequence PLL therefore does not justify its much larger inference cost for this task.','', '## Runtime and diagnostics','',f"PLL inference consumed **{runtimes.inference_seconds.sum()/3600:.2f} aggregate GPU-hours** ({runtimes.total_seconds.sum()/3600:.2f} including model loading) for {int(runtimes.n_masked_forward_examples.sum()):,} masked examples. The 10-GPU DAG completed in about four wall-clock hours. PLL additive fits contained **{int(diag.divergences.sum())} NUTS divergences** across {len(diag)} position-pair fits.",'','## Files','', '- `tables/per_assay_correlations.csv`: all five methods per assay and subset.', '- `tables/aggregate_summary.csv`: aggregate statistics versus original.', '- `tables/paired_method_comparisons.csv`: direct paired comparisons among conditional and PLL methods.', '- `figures/all_methods_assay_spearman.png`: complete method comparison.', '- `figures/original_vs_corrected_conditional_paired.png`: requested paired assay plot.', '- `tables/pll_runtime_per_assay.csv`: measured PLL runtime by assay.']
 (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n'); print(summary.to_string(index=False)); print(OUT/'REPORT.md')
if __name__=='__main__': main()
