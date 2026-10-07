#!/usr/bin/env python
"""Fit the established additive model to conditional-MM and PLL scores."""
import argparse, importlib.util, json, time
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr
ROOT=Path('/data/users/akolchina/epistasis_proteingym'); MODEL_FILE=ROOT/'scripts/reanalysis/conditional_mm/thermodynamic_model.py'
def rho(x,y):
    x,y=np.asarray(x,float),np.asarray(y,float); ok=np.isfinite(x)&np.isfinite(y); x,y=x[ok],y[ok]
    return float(spearmanr(x,y)[0]) if len(x)>=3 and len(set(x))>1 and len(set(y))>1 else np.nan
def main():
    p=argparse.ArgumentParser(); p.add_argument('--scores',type=Path,required=True); p.add_argument('--output',type=Path,required=True); a=p.parse_args(); started=time.perf_counter(); a.output.mkdir(parents=True,exist_ok=True)
    if (a.output/'metrics.csv').exists() and (a.output/'couplings.csv').exists(): print(f'complete: {a.output}',flush=True); return
    spec=importlib.util.spec_from_file_location('thermo',MODEL_FILE); mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); d=pd.read_csv(a.scores); results=d.copy(); diagnostics=[]
    for score,additive,coupling in [('S_cond_direct','fitted_additive_conditional','predicted_conditional_epistasis'),('PLL_AB','fitted_additive_PLL','predicted_PLL_epistasis')]:
        fit=d.copy(); fit['num_mutations']=2; fit['dG']=fit[score]; fit=mod.fill_recon_columns_full(fit,clip_predictions=False)
        diag=pd.DataFrame(fit.attrs.get('fit_diagnostics',[])); diag['score']=score; diagnostics.append(diag)
        results[additive]=fit.recon_dg.to_numpy(); results[coupling]=fit.thermodynamic_coupling.to_numpy()
    diagnostics=pd.concat(diagnostics,ignore_index=True); diagnostics.to_csv(a.output/'fit_diagnostics.csv',index=False); results.to_csv(a.output/'couplings.csv',index=False)
    rows=[]
    for subset,sub in [('all_doubles',results),('epistatic',results[results.epistatic.astype(str).str.lower().eq('true')])]:
        rows.append(dict(dataset=str(results.dataset.iloc[0]),model_key=str(results.model_key.iloc[0]),subset=subset,n=len(sub),rho_default_fitness=rho(sub.ProteinGym_default,sub.experimental_dG),rho_conditional_fitness=rho(sub.S_cond_direct,sub.experimental_dG),rho_PLL_fitness=rho(sub.PLL_AB,sub.experimental_dG),rho_conditional_fitted_epistasis=rho(sub.predicted_conditional_epistasis,sub.experimental_coupling),rho_PLL_fitted_epistasis=rho(sub.predicted_PLL_epistasis,sub.experimental_coupling),rho_direct_E_cond=rho(sub.E_cond,sub.experimental_coupling),rho_direct_epsilon_PLL=rho(sub.epsilon_PLL,sub.experimental_coupling)))
    pd.DataFrame(rows).to_csv(a.output/'metrics.csv',index=False); runtime={'fit_seconds':time.perf_counter()-started,'n_double_mutants':len(results),'n_position_pairs':int(results.pair_name.nunique()),'divergences':int(diagnostics.divergences.sum())}
    (a.output/'runtime.json').write_text(json.dumps(runtime,indent=2)+'\n'); print(json.dumps(runtime,indent=2),flush=True)
if __name__=='__main__': main()
