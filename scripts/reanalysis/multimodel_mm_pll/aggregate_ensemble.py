#!/usr/bin/env python
"""Average checkpoint-level ESM1v scores into the ProteinGym ensemble."""
import argparse
from pathlib import Path
import numpy as np, pandas as pd
AVERAGE=['ProteinGym_default','delta_A_WT','delta_A_B','delta_B_WT','delta_B_A','E_B_to_A','E_A_to_B','E_cond','recomputed_default_MM','S_cond_direct','PLL_WT','PLL_A','PLL_B','PLL_AB','epsilon_PLL']
KEY=['dataset','mutant','mutated_sequence','pair_name','experimental_dG','experimental_coupling','epistatic']
def main():
    p=argparse.ArgumentParser(); p.add_argument('--input',type=Path,action='append',required=True); p.add_argument('--output',type=Path,required=True); p.add_argument('--model-key',required=True); a=p.parse_args(); a.output.parent.mkdir(parents=True,exist_ok=True)
    frames=[pd.read_csv(x).sort_values('mutant').reset_index(drop=True) for x in a.input]; base=frames[0][KEY].copy()
    for other in frames[1:]:
        if not base.equals(other[KEY]): raise ValueError('checkpoint outputs do not contain identical variants')
    for col in AVERAGE: base[col]=np.mean([x[col].to_numpy(float) for x in frames],axis=0)
    base['model_key']=a.model_key; base['checkpoint']='mean('+','.join(str(x) for x in a.input)+')'; tmp=a.output.with_suffix('.tmp.csv'); base.to_csv(tmp,index=False); tmp.replace(a.output); print(a.output,len(base),flush=True)
if __name__=='__main__': main()
