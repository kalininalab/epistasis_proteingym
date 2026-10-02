#!/usr/bin/env python
"""Compute cached ESM2-650M pseudo-log-likelihoods for Tsuboyama double mutants."""
from __future__ import annotations
import argparse, hashlib, json, re, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

ROOT=Path('/data/users/akolchina/epistasis_proteingym')
INPUT=ROOT/'results/tables/intermediate/tsuboyama_epistatic'
MODEL='facebook/esm2_t33_650M_UR50D'
AA=set('ACDEFGHIKLMNPQRSTVWY'); RX=re.compile(r'^([A-Z])(\d+)([A-Z])$')

def parse_double(text):
    ans=[]
    for item in str(text).split(':'):
        m=RX.fullmatch(item)
        if not m: raise ValueError(text)
        old,pos,new=m.groups(); ans.append((old,int(pos)-1,new))
    if len(ans)!=2 or ans[0][1]==ans[1][1]: raise ValueError(text)
    return ans

def infer_wt(doubles):
    candidates=[]
    for row in doubles.itertuples():
        muts=parse_double(row.mutant); seq=list(row.mutated_sequence)
        for old,pos,new in muts:
            if seq[pos]!=new: raise ValueError(f'mutated sequence mismatch: {row.mutant}')
            seq[pos]=old
        candidates.append(''.join(seq))
    if not candidates or len(set(candidates))!=1: raise ValueError('WT sequence is not unique')
    wt=candidates[0]
    if set(wt)-AA: raise ValueError('noncanonical WT sequence')
    return wt

def single_sequence(wt, mutation):
    old,pos,new=mutation
    if wt[pos]!=old: raise ValueError(f'WT mismatch at {pos+1}')
    seq=list(wt); seq[pos]=new; return ''.join(seq)

@torch.inference_mode()
def score_missing(sequences,tokenizer,model,device,batch_size):
    """Return total PLL for each sequence, masking exactly one residue per request."""
    totals=np.zeros(len(sequences),dtype=np.float64)
    requests=[(k,pos) for k,seq in enumerate(sequences) for pos in range(len(seq))]
    for start in range(0,len(requests),batch_size):
        batch=requests[start:start+batch_size]
        texts=[sequences[k] for k,_ in batch]
        enc=tokenizer(texts,return_tensors='pt',padding=True,add_special_tokens=True).to(device)
        targets=[]
        for row,(k,pos) in enumerate(batch):
            targets.append(int(enc.input_ids[row,pos+1]))
            enc.input_ids[row,pos+1]=tokenizer.mask_token_id
        with torch.autocast(device_type='cuda',dtype=torch.float16,enabled=device.type=='cuda'):
            logits=model(**enc).logits
        logp=torch.log_softmax(logits.float(),dim=-1)
        for row,(k,pos) in enumerate(batch): totals[k]+=float(logp[row,pos+1,targets[row]].cpu())
    return totals

def main():
    p=argparse.ArgumentParser(); p.add_argument('--dataset',required=True); p.add_argument('--output',type=Path,required=True)
    p.add_argument('--max-doubles',type=int); p.add_argument('--seed',type=int,default=20261002); p.add_argument('--batch-size',type=int,default=32)
    a=p.parse_args(); started=time.perf_counter(); a.output.mkdir(parents=True,exist_ok=True)
    source=INPUT/f'{a.dataset}.csv'; df=pd.read_csv(source,low_memory=False); doubles=df[df.num_mutations.eq(2)].copy(); wt=infer_wt(doubles)
    if a.max_doubles and len(doubles)>a.max_doubles:
        rng=np.random.default_rng(a.seed); chosen=np.sort(rng.choice(len(doubles),a.max_doubles,replace=False)); doubles=doubles.iloc[chosen].copy()
    records=[]; unique={wt:None}
    for row in doubles.itertuples():
        ma,mb=parse_double(row.mutant); sa=single_sequence(wt,ma); sb=single_sequence(wt,mb); sab=single_sequence(sa,mb)
        if sab!=row.mutated_sequence: raise ValueError(f'AB mismatch: {row.mutant}')
        unique.update({sa:None,sb:None,sab:None}); records.append((row,sa,sb,sab))
    scores_path=a.output/'scores.csv'
    if scores_path.exists() and (a.output/'runtime.json').exists():
        print(f'complete: {a.dataset}',flush=True); return
    cache_path=a.output/'sequence_pll_cache.csv'; cache={}
    if cache_path.exists():
        prior=pd.read_csv(cache_path)
        cache=dict(zip(prior.sequence,prior.PLL))
    missing=[s for s in unique if s not in cache]
    load_started=time.perf_counter(); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type!='cuda': raise RuntimeError('GPU required')
    tokenizer=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
    model=AutoModelForMaskedLM.from_pretrained(MODEL,local_files_only=True).to(device).eval(); load_seconds=time.perf_counter()-load_started
    inference_started=time.perf_counter(); values=score_missing(missing,tokenizer,model,device,a.batch_size); inference_seconds=time.perf_counter()-inference_started
    cache.update(dict(zip(missing,values)))
    cache_df=pd.DataFrame({'sequence':list(cache),'PLL':[cache[s] for s in cache]}); tmp=cache_path.with_suffix('.tmp.csv'); cache_df.to_csv(tmp,index=False); tmp.replace(cache_path)
    out=[]
    for row,sa,sb,sab in records:
        pll_wt,pll_a,pll_b,pll_ab=cache[wt],cache[sa],cache[sb],cache[sab]
        out.append(dict(dataset=a.dataset,mutant=row.mutant,mutated_sequence=row.mutated_sequence,pair_name=row.pair_name,
          experimental_dG=row.dG,experimental_coupling=row.thermodynamic_coupling,epistatic=row.epistatic,
          PLL_WT=pll_wt,PLL_A=pll_a,PLL_B=pll_b,PLL_AB=pll_ab,
          epsilon_PLL=pll_ab-pll_a-pll_b+pll_wt))
    result=pd.DataFrame(out); result_tmp=a.output/'scores.tmp.csv'; result.to_csv(result_tmp,index=False); result_tmp.replace(scores_path)
    runtime=dict(dataset=a.dataset,model=MODEL,n_double_mutants=len(result),n_unique_sequences=len(unique),n_cached_before=len(unique)-len(missing),
      n_scored_sequences=len(missing),n_masked_forward_examples=sum(map(len,missing)),batch_size=a.batch_size,model_load_seconds=load_seconds,
      inference_seconds=inference_seconds,total_seconds=time.perf_counter()-started,sequences_per_second=len(missing)/inference_seconds if inference_seconds else None,
      masked_examples_per_second=sum(map(len,missing))/inference_seconds if inference_seconds else None,device=str(device),dtype='float16 autocast')
    (a.output/'runtime.json').write_text(json.dumps(runtime,indent=2)+'\n'); print(json.dumps(runtime,indent=2),flush=True)
if __name__=='__main__': main()
