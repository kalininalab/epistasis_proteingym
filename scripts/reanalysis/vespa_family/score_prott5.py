#!/usr/bin/env python
"""Compute ProtT5 reconstruction MM, conditional MM, and PLL for one assay."""
from __future__ import annotations
import argparse, json, re, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

ROOT=Path('/data/users/akolchina/epistasis_proteingym')
INPUT=ROOT/'results/tables/intermediate/tsuboyama_epistatic'
AA='ACDEFGHIKLMNPQRSTVWY'; AA_SET=set(AA); RX=re.compile(r'^([A-Z])(\d+)([A-Z])$')

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
        seq=list(row.mutated_sequence)
        for old,pos,new in parse_double(row.mutant):
            if not 0<=pos<len(seq) or seq[pos]!=new: raise ValueError(f'mutated sequence mismatch: {row.mutant}')
            seq[pos]=old
        candidates.append(''.join(seq))
    if not candidates or len(set(candidates))!=1: raise ValueError(f'WT sequence is not unique: {len(set(candidates))}')
    wt=candidates[0]
    if set(wt)-AA_SET: raise ValueError('noncanonical WT sequence')
    return wt

def mutate(wt,mutation):
    old,pos,new=mutation
    if wt[pos]!=old: raise ValueError(f'WT mismatch at {pos+1}: expected {old}, observed {wt[pos]}')
    seq=list(wt); seq[pos]=new; return ''.join(seq)

def load_model(checkpoint,revision):
    return T5ForConditionalGeneration.from_pretrained(checkpoint,revision=revision,local_files_only=True,torch_dtype=torch.float16)

def masked_batch(batch,tokenizer,device):
    texts=[]
    for sequence,pos in batch:
        tokens=list(sequence); tokens[pos]='<extra_id_0>'; texts.append(' '.join(tokens))
    return tokenizer(texts,return_tensors='pt',padding=True,add_special_tokens=True).to(device)

@torch.inference_mode()
def masked_distributions(requests,tokenizer,model,device,batch_size):
    unique=list(dict.fromkeys(requests)); result={}; aa_ids=torch.tensor([tokenizer.get_vocab()['▁'+x] for x in AA],device=device)
    for start in range(0,len(unique),batch_size):
        batch=unique[start:start+batch_size]; enc=masked_batch(batch,tokenizer,device)
        with torch.autocast(device_type='cuda',dtype=torch.float16): logits=model(**enc,labels=enc.input_ids).logits
        for row,key in enumerate(batch): result[key]=torch.log_softmax(logits[row,key[1],aa_ids].float(),-1).cpu().numpy()
    return result

@torch.inference_mode()
def pll_sequences(sequences,tokenizer,model,device,batch_size):
    totals=np.zeros(len(sequences),dtype=np.float64); requests=[(k,pos) for k,s in enumerate(sequences) for pos in range(len(s))]
    for start in range(0,len(requests),batch_size):
        batch=requests[start:start+batch_size]; enc=masked_batch([(sequences[k],pos) for k,pos in batch],tokenizer,device)
        targets=[tokenizer.get_vocab()['▁'+sequences[k][pos]] for k,pos in batch]
        with torch.autocast(device_type='cuda',dtype=torch.float16): logits=model(**enc,labels=enc.input_ids).logits
        logp=torch.log_softmax(logits.float(),dim=-1)
        for row,(k,pos) in enumerate(batch): totals[k]+=float(logp[row,pos,targets[row]].cpu())
    return totals

def atomic_csv(frame,path):
    tmp=path.with_suffix('.tmp.csv'); frame.to_csv(tmp,index=False); tmp.replace(path)

def main():
    p=argparse.ArgumentParser(); p.add_argument('--checkpoint',required=True); p.add_argument('--revision',required=True); p.add_argument('--model-key',required=True); p.add_argument('--default-column',required=True)
    p.add_argument('--dataset',required=True); p.add_argument('--output',type=Path,required=True); p.add_argument('--batch-size',type=int,default=16)
    p.add_argument('--pll-sequence-chunk',type=int,default=128); p.add_argument('--max-doubles',type=int); p.add_argument('--seed',type=int,default=20261004)
    a=p.parse_args(); started=time.perf_counter(); a.output.mkdir(parents=True,exist_ok=True)
    scores_path=a.output/'scores.csv'; runtime_path=a.output/'runtime.json'
    if scores_path.exists() and runtime_path.exists(): print(f'complete: {a.model_key}/{a.dataset}',flush=True); return
    df=pd.read_csv(INPUT/f'{a.dataset}.csv',low_memory=False); doubles=df[df.num_mutations.eq(2)].copy()
    if a.max_doubles and len(doubles)>a.max_doubles:
        rng=np.random.default_rng(a.seed); doubles=doubles.iloc[np.sort(rng.choice(len(doubles),a.max_doubles,replace=False))].copy()
    wt=infer_wt(doubles); records=[]; unique={wt:None}; requests=[]; request_keys=[]
    for row in doubles.itertuples():
        ma,mb=parse_double(row.mutant); sa=mutate(wt,ma); sb=mutate(wt,mb); sab=mutate(sa,mb)
        if sab!=row.mutated_sequence: raise ValueError(f'AB mismatch: {row.mutant}')
        unique.update({sa:None,sb:None,sab:None}); keys=((wt,ma[1]),(sb,ma[1]),(wt,mb[1]),(sa,mb[1])); requests.extend(keys); request_keys.append(keys); records.append((row,ma,mb,sa,sb,sab))
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type!='cuda': raise RuntimeError('GPU required')
    load_start=time.perf_counter(); tokenizer=T5Tokenizer.from_pretrained(a.checkpoint,revision=a.revision,local_files_only=True,do_lower_case=False); model=load_model(a.checkpoint,a.revision).to(device).eval(); load_seconds=time.perf_counter()-load_start
    cond_start=time.perf_counter(); distributions=masked_distributions(requests,tokenizer,model,device,a.batch_size); cond_seconds=time.perf_counter()-cond_start; aa_i={x:i for i,x in enumerate(AA)}
    conditional=[]
    for (row,ma,mb,sa,sb,sab),keys in zip(records,request_keys):
        delta=lambda key,m: float(distributions[key][aa_i[m[2]]]-distributions[key][aa_i[m[0]]])
        da0,da_b,db0,db_a=delta(keys[0],ma),delta(keys[1],ma),delta(keys[2],mb),delta(keys[3],mb); eba=da_b-da0; eab=db_a-db0
        conditional.append(dict(dataset=a.dataset,model_key=a.model_key,checkpoint=a.checkpoint,mutant=row.mutant,mutated_sequence=row.mutated_sequence,pair_name=row.pair_name,experimental_dG=row.dG,experimental_coupling=row.thermodynamic_coupling,epistatic=row.epistatic,ProteinGym_default=da0+db0,delta_A_WT=da0,delta_A_B=da_b,delta_B_WT=db0,delta_B_A=db_a,E_B_to_A=eba,E_A_to_B=eab,E_cond=.5*(eba+eab),recomputed_default_MM=da0+db0,S_cond_direct=.5*((da0+db_a)+(db0+da_b)),sequence_A=sa,sequence_B=sb))
    conditional=pd.DataFrame(conditional); atomic_csv(conditional,a.output/'conditional_scores.csv')
    cache_path=a.output/'sequence_pll_cache.csv'; cache={}
    if cache_path.exists():
        prior=pd.read_csv(cache_path); cache=dict(zip(prior.sequence,prior.PLL))
    missing=[s for s in unique if s not in cache]; pll_start=time.perf_counter(); masked_examples=0
    for start in range(0,len(missing),a.pll_sequence_chunk):
        chunk=missing[start:start+a.pll_sequence_chunk]; vals=pll_sequences(chunk,tokenizer,model,device,a.batch_size); cache.update(zip(chunk,vals)); masked_examples+=sum(map(len,chunk))
        atomic_csv(pd.DataFrame({'sequence':list(cache),'PLL':[cache[s] for s in cache]}),cache_path); print(f'{a.model_key}/{a.dataset}: PLL sequences {min(start+len(chunk),len(missing))}/{len(missing)}',flush=True)
    pll_seconds=time.perf_counter()-pll_start; out=conditional.copy(); pll_wt=[]; pll_a=[]; pll_b=[]; pll_ab=[]
    for _,_,_,sa,sb,sab in records: pll_wt.append(cache[wt]); pll_a.append(cache[sa]); pll_b.append(cache[sb]); pll_ab.append(cache[sab])
    out['PLL_WT']=pll_wt; out['PLL_A']=pll_a; out['PLL_B']=pll_b; out['PLL_AB']=pll_ab; out['epsilon_PLL']=out.PLL_AB-out.PLL_A-out.PLL_B+out.PLL_WT
    out=out.drop(columns=['sequence_A','sequence_B']); atomic_csv(out,scores_path)
    runtime=dict(dataset=a.dataset,model_key=a.model_key,checkpoint=a.checkpoint,revision=a.revision,n_double_mutants=len(out),n_unique_sequences=len(unique),n_cached_before=len(unique)-len(missing),n_scored_sequences=len(missing),n_conditional_requests=len(set(requests)),n_pll_masked_examples=masked_examples,batch_size=a.batch_size,model_load_seconds=load_seconds,conditional_seconds=cond_seconds,pll_seconds=pll_seconds,total_seconds=time.perf_counter()-started,device=str(device),dtype='float16 autocast')
    runtime_path.write_text(json.dumps(runtime,indent=2)+'\n'); print(json.dumps(runtime,indent=2),flush=True)

if __name__=='__main__': main()
