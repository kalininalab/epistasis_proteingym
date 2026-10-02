#!/usr/bin/env python
"""Fit the established Bayesian additive thermodynamic model to conditional MM fitness."""
import argparse, importlib.util
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr
ROOT=Path('/data/users/akolchina/epistasis_proteingym')
BASE=ROOT/'results/reanalysis/tsuboyama/conditional_mm_esm2_650m'
def rho(x,y):
    ok=np.isfinite(x)&np.isfinite(y); x=np.asarray(x)[ok]; y=np.asarray(y)[ok]
    return spearmanr(x,y)[0] if len(x)>=3 and len(set(x))>1 and len(set(y))>1 else np.nan
def main():
    p=argparse.ArgumentParser(); p.add_argument('--dataset',required=True); a=p.parse_args()
    dest=BASE/'fits'/a.dataset; dest.mkdir(parents=True,exist_ok=True)
    if (dest/'metrics.csv').exists(): return
    spec=importlib.util.spec_from_file_location('conditional_thermodynamic',Path(__file__).with_name('thermodynamic_model.py')); mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    d=pd.read_csv(BASE/'scores'/f'{a.dataset}.csv')
    if (dest/'couplings.csv').exists():
        fit=pd.read_csv(dest/'couplings.csv')
    else:
        fit=d.copy(); fit['num_mutations']=2
        # score_all.py defines this directly as the symmetric mean of the two
        # mutation-order paths. Never replace it with stored_default + E_cond.
        fit['conditional_path_MM_fitness']=fit.conditional_MM_fitness
        fit['experimental_fitness']=fit.dG; fit['experimental_coupling']=fit.thermodynamic_coupling; fit['dG']=fit.conditional_MM_fitness
        fit=mod.fill_recon_columns_full(fit,clip_predictions=False)
        diag=pd.DataFrame(fit.attrs.get('fit_diagnostics',[])); diag.to_csv(dest/'fit_diagnostics.csv',index=False)
        fit=fit.rename(columns={'thermodynamic_coupling':'conditional_predicted_coupling','recon_dg':'conditional_additive_fitness'})
        fit.to_csv(dest/'couplings.csv',index=False)
    rows=[]
    default_path=ROOT/'results/reanalysis/tsuboyama/analysis2/raw_unclipped'/a.dataset/'ESM2_650M'/'couplings.csv'
    if default_path.exists():
        default=pd.read_csv(default_path)[['mutant','predicted_coupling']].rename(columns={'predicted_coupling':'default_predicted_coupling'})
    else:
        base=d.copy(); base['num_mutations']=2; base['dG']=base.default_MM_fitness
        base=mod.fill_recon_columns_full(base,clip_predictions=False)
        pd.DataFrame(base.attrs.get('fit_diagnostics',[])).to_csv(dest/'default_fit_diagnostics.csv',index=False)
        default=base[['mutant','thermodynamic_coupling']].rename(columns={'thermodynamic_coupling':'default_predicted_coupling'})
    fit=fit.merge(default,on='mutant',validate='one_to_one')
    for subset,sub in [('all_doubles',fit),('epistatic',fit[fit.epistatic.astype(str).str.lower().eq('true')])]:
      for task,truth,old,new in [('fitness','experimental_fitness','default_MM_fitness','conditional_MM_fitness'),('epistasis','experimental_coupling','default_predicted_coupling','conditional_predicted_coupling')]:
       rows.append(dict(dataset=a.dataset,subset=subset,task=task,n=len(sub),default_rho=rho(sub[truth],sub[old]),conditional_rho=rho(sub[truth],sub[new])))
    tmp=dest/'metrics.tmp.csv'; pd.DataFrame(rows).to_csv(tmp,index=False); tmp.replace(dest/'metrics.csv')
if __name__=='__main__': main()
