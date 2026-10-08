#!/usr/bin/env python
"""Conditional masked marginals and PLL for CARP-76M."""
from __future__ import annotations
import argparse,json,sys,time
from pathlib import Path
import numpy as np,pandas as pd,torch
ROOT=Path('/data/users/akolchina/epistasis_proteingym'); INPUT=ROOT/'results/tables/intermediate/tsuboyama_epistatic'
sys.path.insert(0,str(ROOT/'scripts/reanalysis/multimodel_mm_pll'))
from score import AA,infer_wt,mutate,parse_double,atomic_csv
sys.path.insert(0,'/data/users/akolchina/software/protein-sequence-models')
from sequence_models.constants import PROTEIN_ALPHABET
from sequence_models.pretrained import load_model_and_alphabet
AA_ID={a:PROTEIN_ALPHABET.index(a) for a in AA}

def masked_batch(items,collater,device):
    seqs=[]
    for seq,pos in items:
        x=list(seq);x[pos]='#';seqs.append([''.join(x)])
    return collater(seqs)[0].to(device)

@torch.inference_mode()
def masked_logits(requests,model,collater,device,batch):
    unique=list(dict.fromkeys(requests));out={}
    for s in range(0,len(unique),batch):
        chunk=unique[s:s+batch];x=masked_batch(chunk,collater,device)
        with torch.autocast('cuda',dtype=torch.float16): z=model(x,repr_layers=[],logits=True)['logits']
        for r,k in enumerate(chunk):out[k]=z[r,k[1]].float().cpu().numpy()
    return out

@torch.inference_mode()
def pll(seqs,model,collater,device,batch):
    total=np.zeros(len(seqs));req=[(k,p) for k,x in enumerate(seqs) for p in range(len(x))]
    for s in range(0,len(req),batch):
        chunk=req[s:s+batch];items=[(seqs[k],p) for k,p in chunk];x=masked_batch(items,collater,device)
        with torch.autocast('cuda',dtype=torch.float16):lp=torch.log_softmax(model(x,repr_layers=[],logits=True)['logits'].float(),-1)
        for r,(k,p) in enumerate(chunk):total[k]+=float(lp[r,p,AA_ID[seqs[k][p]]].cpu())
    return total

def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--weights',type=Path,required=True);p.add_argument('--batch-size',type=int,default=16);p.add_argument('--max-doubles',type=int);a=p.parse_args();t0=time.perf_counter();a.output.mkdir(parents=True,exist_ok=True)
    if (a.output/'scores.csv').exists() and (a.output/'runtime.json').exists():return
    d=pd.read_csv(INPUT/f'{a.dataset}.csv',low_memory=False);d=d[d.num_mutations.eq(2)].copy()
    if a.max_doubles and len(d)>a.max_doubles:d=d.iloc[np.sort(np.random.default_rng(20261008).choice(len(d),a.max_doubles,False))]
    wt=infer_wt(d);records=[];unique={wt:None};requests=[];keys=[]
    for row in d.itertuples():
        ma,mb=parse_double(row.mutant);sa,sb=mutate(wt,ma),mutate(wt,mb);sab=mutate(sa,mb)
        if sab!=row.mutated_sequence:raise ValueError(row.mutant)
        unique.update({sa:None,sb:None,sab:None});k=((wt,ma[1]),(sb,ma[1]),(wt,mb[1]),(sa,mb[1]));requests+=list(k);keys.append(k);records.append((row,ma,mb,sa,sb,sab))
    dev=torch.device('cuda');model,collater=load_model_and_alphabet(str(a.weights));model=model.to(dev).eval();z=masked_logits(requests,model,collater,dev,a.batch_size);rows=[]
    for (row,ma,mb,sa,sb,sab),k,default in zip(records,keys,d.CARP_76M.to_numpy()):
        delta=lambda q,m:float(z[q][AA_ID[m[2]]]-z[q][AA_ID[m[0]]]);da0,da_b,db0,db_a=delta(k[0],ma),delta(k[1],ma),delta(k[2],mb),delta(k[3],mb);eba,eab=da_b-da0,db_a-db0
        rows.append(dict(dataset=a.dataset,model_key='CARP_76M',mutant=row.mutant,mutated_sequence=row.mutated_sequence,pair_name=row.pair_name,experimental_dG=row.dG,experimental_coupling=row.thermodynamic_coupling,epistatic=row.epistatic,ProteinGym_default=default,delta_A_WT=da0,delta_A_B=da_b,delta_B_WT=db0,delta_B_A=db_a,E_B_to_A=eba,E_A_to_B=eab,E_cond=.5*(eba+eab),recomputed_default_MM=da0+db0,S_cond_direct=.5*((da0+db_a)+(db0+da_b)),sequence_A=sa,sequence_B=sb))
    out=pd.DataFrame(rows);atomic_csv(out,a.output/'conditional_scores.csv');cache={};cp=a.output/'sequence_pll_cache.csv'
    if cp.exists():q=pd.read_csv(cp);cache=dict(zip(q.sequence,q.PLL))
    missing=[x for x in unique if x not in cache]
    for s in range(0,len(missing),64):
        c=missing[s:s+64];cache.update(zip(c,pll(c,model,collater,dev,a.batch_size)));atomic_csv(pd.DataFrame({'sequence':list(cache),'PLL':[cache[x] for x in cache]}),cp)
    out['PLL_WT']=[cache[wt]]*len(out);out['PLL_A']=[cache[x[3]] for x in records];out['PLL_B']=[cache[x[4]] for x in records];out['PLL_AB']=[cache[x[5]] for x in records];out['epsilon_PLL']=out.PLL_AB-out.PLL_A-out.PLL_B+out.PLL_WT;out=out.drop(columns=['sequence_A','sequence_B']);atomic_csv(out,a.output/'scores.csv');(a.output/'runtime.json').write_text(json.dumps({'dataset':a.dataset,'model_key':'CARP_76M','n_double_mutants':len(out),'n_unique_sequences':len(unique),'total_seconds':time.perf_counter()-t0},indent=2)+'\n')
if __name__=='__main__':main()
