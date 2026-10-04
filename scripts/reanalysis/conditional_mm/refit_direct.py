#!/usr/bin/env python
"""Fit the additive model to the corrected direct-path conditional-MM score."""
import argparse, importlib.util, json, time
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr
ROOT=Path('/data/users/akolchina/epistasis_proteingym')
SCORES=ROOT/'results/reanalysis/tsuboyama/conditional_mm_esm2_650m/scores'
OUT=ROOT/'results/reanalysis/tsuboyama/conditional_mm_direct_path_esm2_650m/fits'
MODEL=ROOT/'scripts/reanalysis/conditional_mm/thermodynamic_model.py'
def rho(x,y):
 x,y=np.asarray(x,float),np.asarray(y,float); ok=np.isfinite(x)&np.isfinite(y); x,y=x[ok],y[ok]
 return float(spearmanr(x,y)[0]) if len(x)>=3 and len(set(x))>1 and len(set(y))>1 else np.nan
def main():
 p=argparse.ArgumentParser(); p.add_argument('--dataset',required=True); a=p.parse_args(); dest=OUT/a.dataset; dest.mkdir(parents=True,exist_ok=True)
 if (dest/'metrics.csv').exists(): return
 started=time.perf_counter(); spec=importlib.util.spec_from_file_location('thermo',MODEL); mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
 d=pd.read_csv(SCORES/f'{a.dataset}.csv'); d['S_cond_direct']=.5*((d.delta_A_WT+d.delta_B_A)+(d.delta_B_WT+d.delta_A_B))
 fit=d.copy(); fit['num_mutations']=2; fit['experimental_coupling']=fit.thermodynamic_coupling; fit['dG']=fit.S_cond_direct
 fit=mod.fill_recon_columns_full(fit,clip_predictions=False); diag=pd.DataFrame(fit.attrs.get('fit_diagnostics',[])); diag.to_csv(dest/'fit_diagnostics.csv',index=False)
 fit=fit.rename(columns={'thermodynamic_coupling':'predicted_conditional_direct_epistasis','recon_dg':'fitted_additive_S_cond_direct'})
 fit.to_csv(dest/'couplings.csv',index=False)
 old=pd.read_csv(ROOT/'results/reanalysis/tsuboyama/analysis2/raw_unclipped'/a.dataset/'ESM2_650M/couplings.csv',usecols=['mutant','predicted_coupling']).rename(columns={'predicted_coupling':'original_predicted_epistasis'})
 fit=fit.merge(old,on='mutant',validate='one_to_one'); rows=[]
 for subset,sub in [('all_doubles',fit),('epistatic',fit[fit.epistatic.astype(str).str.lower().eq('true')])]:
  rows.append(dict(dataset=a.dataset,subset=subset,n=len(sub),rho_original=rho(sub.experimental_coupling,sub.original_predicted_epistasis),rho_conditional_direct_fitted=rho(sub.experimental_coupling,sub.predicted_conditional_direct_epistasis),rho_direct_E_cond=rho(sub.experimental_coupling,sub.E_cond)))
 tmp=dest/'metrics.tmp.csv'; pd.DataFrame(rows).to_csv(tmp,index=False); tmp.replace(dest/'metrics.csv')
 (dest/'runtime.json').write_text(json.dumps({'fit_seconds':time.perf_counter()-started,'n_pairs':int(fit.pair_name.nunique()),'divergences':int(diag.divergences.sum())},indent=2)+'\n')
if __name__=='__main__': main()
