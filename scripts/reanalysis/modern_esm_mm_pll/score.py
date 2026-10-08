#!/usr/bin/env python
"""Conditional masked marginals and full PLL for ESMC-300M or ESM3-open-small."""
from __future__ import annotations
import argparse,json,sys,time
from pathlib import Path
import numpy as np,pandas as pd,torch
ROOT=Path('/data/users/akolchina/epistasis_proteingym'); INPUT=ROOT/'results/tables/intermediate/tsuboyama_epistatic'
sys.path.insert(0,str(ROOT/'scripts/reanalysis/multimodel_mm_pll'))
from score import AA,atomic_csv,infer_wt,mutate,parse_double
from esm.models.esm3 import ESM3
from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein
MODEL_CONFIG={'ESMC-300M':(ESMC,'esmc_300m'),'ESM3':(ESM3,'esm3_sm_open_v1')}

def tok(model): return model.tokenizer if isinstance(model,ESMC) else model.tokenizers.sequence

def forward_logits(model,x):
    # Match the official logits() wrapper: ESM3 creates floating auxiliary tracks
    # internally, so CUDA autocast is required when checkpoint weights are bfloat16.
    with torch.autocast(device_type='cuda',dtype=torch.bfloat16):
        return model.forward(sequence_tokens=x).sequence_logits.float().cpu()
def encode(model,seq):
    x=model.encode(ESMProtein(sequence=seq)).sequence
    assert x is not None and len(x)==len(seq)+2
    return x

@torch.inference_mode()
def masked_logits(requests,model,batch):
    unique=list(dict.fromkeys(requests)); tokenizer=tok(model); base={s:encode(model,s) for s,_ in unique}; result={}
    for start in range(0,len(unique),batch):
        chunk=unique[start:start+batch]; x=torch.stack([base[s].clone() for s,_ in chunk])
        for row,(_,pos) in enumerate(chunk): x[row,pos+1]=tokenizer.mask_token_id
        logits=forward_logits(model,x)
        for row,key in enumerate(chunk): result[key]=logits[row,key[1]+1].numpy()
    return result

@torch.inference_mode()
def pll(seqs,model,batch):
    tokenizer=tok(model); aa_id={a:tokenizer.encode(a,add_special_tokens=False)[0] for a in AA}; base=[encode(model,s) for s in seqs]
    requests=[(row,pos) for row,s in enumerate(seqs) for pos in range(len(s))]; totals=np.zeros(len(seqs))
    for start in range(0,len(requests),batch):
        chunk=requests[start:start+batch]; x=torch.stack([base[row].clone() for row,_ in chunk])
        for br,(_,pos) in enumerate(chunk): x[br,pos+1]=tokenizer.mask_token_id
        logp=torch.log_softmax(forward_logits(model,x),-1)
        for br,(row,pos) in enumerate(chunk): totals[row]+=float(logp[br,pos+1,aa_id[seqs[row][pos]]])
    return totals

def default_col(df,key):
    for name in ({'ESMC-300M':['ESMC-300M','ESMC_300M'],'ESM3':['ESM3']}[key]):
        if name in df.columns:return name
    raise KeyError(f'No baseline column for {key}')

def main():
    p=argparse.ArgumentParser(); p.add_argument('--dataset',required=True);p.add_argument('--model',choices=MODEL_CONFIG,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--batch-size',type=int,default=8);p.add_argument('--max-doubles',type=int);a=p.parse_args();started=time.perf_counter();a.output.mkdir(parents=True,exist_ok=True)
    if (a.output/'scores.csv').exists() and (a.output/'runtime.json').exists():return
    d=pd.read_csv(INPUT/f'{a.dataset}.csv',low_memory=False);d=d[d.num_mutations.eq(2)].copy()
    if a.max_doubles and len(d)>a.max_doubles:d=d.iloc[np.sort(np.random.default_rng(20261008).choice(len(d),a.max_doubles,False))]
    baseline=default_col(d,a.model);wt=infer_wt(d);records=[];requests=[];keys=[];unique={wt:None}
    for row in d.itertuples():
        ma,mb=parse_double(row.mutant);sa,sb=mutate(wt,ma),mutate(wt,mb);sab=mutate(sa,mb)
        if sab!=row.mutated_sequence:raise ValueError(row.mutant)
        unique.update({sa:None,sb:None,sab:None}); k=((wt,ma[1]),(sb,ma[1]),(wt,mb[1]),(sa,mb[1]));requests+=list(k);keys.append(k);records.append((row,ma,mb,sa,sb,sab))
    cls,name=MODEL_CONFIG[a.model];model=cls.from_pretrained(name,device=torch.device('cuda')).eval();tokenizer=tok(model);aa_id={x:tokenizer.encode(x,add_special_tokens=False)[0] for x in AA};z=masked_logits(requests,model,a.batch_size);rows=[]
    for rec,k in zip(records,keys):
        row,ma,mb,sa,sb,sab=rec;delta=lambda q,m:float(z[q][aa_id[m[2]]]-z[q][aa_id[m[0]]]);da0,da_b,db0,db_a=delta(k[0],ma),delta(k[1],ma),delta(k[2],mb),delta(k[3],mb);eba,eab=da_b-da0,db_a-db0
        rows.append(dict(dataset=a.dataset,model_key=a.model,mutant=row.mutant,mutated_sequence=row.mutated_sequence,pair_name=row.pair_name,experimental_dG=row.dG,experimental_coupling=row.thermodynamic_coupling,epistatic=row.epistatic,ProteinGym_default=d.loc[row.Index,baseline],delta_A_WT=da0,delta_A_B=da_b,delta_B_WT=db0,delta_B_A=db_a,E_B_to_A=eba,E_A_to_B=eab,E_cond=.5*(eba+eab),recomputed_default_MM=da0+db0,S_cond_direct=.5*((da0+db_a)+(db0+da_b)),sequence_A=sa,sequence_B=sb))
    out=pd.DataFrame(rows);atomic_csv(out,a.output/'conditional_scores.csv');cache={};cp=a.output/'sequence_pll_cache.csv'
    if cp.exists():saved=pd.read_csv(cp);cache=dict(zip(saved.sequence,saved.PLL))
    missing=[s for s in unique if s not in cache]
    for start in range(0,len(missing),32):
        chunk=missing[start:start+32];cache.update(zip(chunk,pll(chunk,model,a.batch_size)));atomic_csv(pd.DataFrame({'sequence':list(cache),'PLL':[cache[x] for x in cache]}),cp)
    out['PLL_WT']=[cache[wt]]*len(out);out['PLL_A']=[cache[x[3]] for x in records];out['PLL_B']=[cache[x[4]] for x in records];out['PLL_AB']=[cache[x[5]] for x in records];out['epsilon_PLL']=out.PLL_AB-out.PLL_A-out.PLL_B+out.PLL_WT;out=out.drop(columns=['sequence_A','sequence_B']);atomic_csv(out,a.output/'scores.csv');(a.output/'runtime.json').write_text(json.dumps({'dataset':a.dataset,'model_key':a.model,'pretrained_name':name,'esm_package':'3.2.1','n_double_mutants':len(out),'n_unique_sequences':len(unique),'total_seconds':time.perf_counter()-started},indent=2)+'\n')
if __name__=='__main__':main()
