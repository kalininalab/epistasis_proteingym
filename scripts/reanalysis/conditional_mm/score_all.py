#!/usr/bin/env python
"""Score Tsuboyama double mutants with symmetric conditional ESM2 masked marginals."""
from __future__ import annotations
import argparse, re
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

ROOT = Path('/data/users/akolchina/epistasis_proteingym')
IN = ROOT / 'results/tables/intermediate/tsuboyama_epistatic'
OUT = ROOT / 'results/reanalysis/tsuboyama/conditional_mm_esm2_650m/scores'
MODEL = 'facebook/esm2_t33_650M_UR50D'
AA = 'ACDEFGHIKLMNPQRSTVWY'
RX = re.compile(r'^([A-Z])(\d+)([A-Z])$')

def parse(text):
    parts = str(text).split(':')
    if len(parts) != 2: raise ValueError(text)
    ans=[]
    for p in parts:
        m=RX.fullmatch(p)
        if not m: raise ValueError(text)
        old,pos,new=m.groups(); ans.append((old,int(pos)-1,new))
    if ans[0][1] == ans[1][1]: raise ValueError(text)
    return ans

def infer_wt(df):
    candidates=[]
    for row in df.itertuples():
        try: muts=parse(row.mutant)
        except ValueError: continue
        seq=list(row.mutated_sequence)
        for old,pos,new in muts:
            if not (0 <= pos < len(seq) and seq[pos] == new): raise ValueError(f'sequence mismatch: {row.mutant}')
            seq[pos]=old
        candidates.append(''.join(seq))
    if not candidates or len(set(candidates)) != 1: raise ValueError(f'could not infer unique WT ({len(set(candidates))})')
    return candidates[0]

@torch.inference_mode()
def predict(requests, tok, model, device, batch_size):
    unique=list(dict.fromkeys(requests)); result={}
    aa_ids=torch.tensor([tok.convert_tokens_to_ids(x) for x in AA],device=device)
    for start in range(0,len(unique),batch_size):
        batch=unique[start:start+batch_size]
        enc=tok([x[0] for x in batch],return_tensors='pt',padding=True,add_special_tokens=True).to(device)
        for k,(_,pos) in enumerate(batch): enc.input_ids[k,pos+1]=tok.mask_token_id
        with torch.autocast(device_type='cuda',dtype=torch.float16,enabled=device.type=='cuda'):
            logits=model(**enc).logits
        for k,key in enumerate(batch):
            result[key]=torch.log_softmax(logits[k,key[1]+1,aa_ids].float(),-1).cpu().numpy()
    return result

def score(path,tok,model,device,batch_size):
    dest=OUT/(path.stem+'.csv')
    if dest.exists(): print('skip',path.stem,flush=True); return
    df=pd.read_csv(path,low_memory=False)
    doubles=df.loc[df.num_mutations.eq(2)].copy()
    wt=infer_wt(doubles)
    rows=[]; requests=[]; keys=[]
    for row in doubles.itertuples():
        a,b=parse(row.mutant)
        for old,pos,new in (a,b):
            if wt[pos] != old: raise ValueError(f'WT mismatch {row.mutant}')
        bg_b=list(wt); bg_b[b[1]]=b[2]
        bg_a=list(wt); bg_a[a[1]]=a[2]
        kk=((wt,a[1]),(''.join(bg_b),a[1]),(wt,b[1]),(''.join(bg_a),b[1]))
        requests.extend(kk); keys.append(kk); rows.append(row)
    lp=predict(requests,tok,model,device,batch_size); ai={x:i for i,x in enumerate(AA)}
    out=[]
    for row,kk in zip(rows,keys):
        a,b=parse(row.mutant)
        delta=lambda key,m: float(lp[key][ai[m[2]]]-lp[key][ai[m[0]]])
        da0,da_b,db0,db_a=delta(kk[0],a),delta(kk[1],a),delta(kk[2],b),delta(kk[3],b)
        eba=da_b-da0; eab=db_a-db0; ec=.5*(eba+eab)
        # Define the conditional score directly as the symmetric mean of the
        # two mutation-order paths; do not depend on a stored default score.
        s_cond=.5*((da0+db_a)+(db0+da_b))
        out.append(dict(dataset=path.stem,mutant=row.mutant,mutated_sequence=row.mutated_sequence,
          pair_name=row.pair_name,dG=row.dG,thermodynamic_coupling=row.thermodynamic_coupling,
          epistatic=row.epistatic,default_MM_fitness=row.ESM2_650M,
          mutation_A=f'{a[0]}{a[1]+1}{a[2]}',mutation_B=f'{b[0]}{b[1]+1}{b[2]}',
          delta_A_WT=da0,delta_A_B=da_b,delta_B_WT=db0,delta_B_A=db_a,
          E_B_to_A=eba,E_A_to_B=eab,E_cond=ec,
          recomputed_default_MM_fitness=da0+db0,conditional_MM_fitness=s_cond))
    OUT.mkdir(parents=True,exist_ok=True); tmp=dest.with_suffix('.tmp.csv')
    pd.DataFrame(out).to_csv(tmp,index=False); tmp.replace(dest)
    print('wrote',path.stem,len(out),flush=True)

def main():
    p=argparse.ArgumentParser(); p.add_argument('--shard',type=int,required=True); p.add_argument('--n-shards',type=int,default=10); p.add_argument('--batch-size',type=int,default=16); a=p.parse_args()
    paths=sorted(IN.glob('*.csv'))[a.shard::a.n_shards]
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type!='cuda': raise RuntimeError('GPU required')
    tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
    model=AutoModelForMaskedLM.from_pretrained(MODEL,local_files_only=True).to(device).eval()
    for path in paths: score(path,tok,model,device,a.batch_size)
if __name__=='__main__': main()
