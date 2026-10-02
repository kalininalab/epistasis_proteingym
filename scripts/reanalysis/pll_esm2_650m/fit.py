#!/usr/bin/env python
"""Fit the established Bayesian additive model to saved PLL(AB) scores."""
import argparse, importlib.util, json, time
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr

ROOT=Path('/data/users/akolchina/epistasis_proteingym')
MODEL_FILE=ROOT/'scripts/reanalysis/conditional_mm/thermodynamic_model.py'
def rho(x,y):
    x,y=np.asarray(x,float),np.asarray(y,float); ok=np.isfinite(x)&np.isfinite(y); x,y=x[ok],y[ok]
    return float(spearmanr(x,y)[0]) if len(x)>=3 and len(set(x))>1 and len(set(y))>1 else np.nan
def main():
    p=argparse.ArgumentParser(); p.add_argument('--scores',type=Path,required=True); p.add_argument('--output',type=Path,required=True); a=p.parse_args()
    started=time.perf_counter(); a.output.mkdir(parents=True,exist_ok=True)
    spec=importlib.util.spec_from_file_location('thermo',MODEL_FILE); mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    d=pd.read_csv(a.scores); fit=d.copy(); fit['num_mutations']=2; fit['dG']=fit.PLL_AB
    fit=mod.fill_recon_columns_full(fit,clip_predictions=False)
    diagnostics=pd.DataFrame(fit.attrs.get('fit_diagnostics',[])); diagnostics.to_csv(a.output/'fit_diagnostics.csv',index=False)
    fit=fit.rename(columns={'recon_dg':'fitted_additive_PLL','thermodynamic_coupling':'predicted_PLL_epistasis'})
    fit.to_csv(a.output/'couplings.csv',index=False)
    rows=[]
    for subset,sub in [('all_doubles',fit),('epistatic',fit[fit.epistatic.astype(str).str.lower().eq('true')])]:
        rows.append(dict(dataset=str(fit.dataset.iloc[0]),subset=subset,n=len(sub),rho_PLL_fitted_epistasis=rho(sub.experimental_coupling,sub.predicted_PLL_epistasis),rho_direct_epsilon_PLL=rho(sub.experimental_coupling,sub.epsilon_PLL)))
    pd.DataFrame(rows).to_csv(a.output/'metrics.csv',index=False)
    runtime={'fit_seconds':time.perf_counter()-started,'n_double_mutants':len(fit),'n_position_pairs':int(fit.pair_name.nunique()),'divergences':int(diagnostics.divergences.sum()) if len(diagnostics) else 0}
    (a.output/'runtime.json').write_text(json.dumps(runtime,indent=2)+'\n'); print(json.dumps(runtime,indent=2))
if __name__=='__main__': main()
