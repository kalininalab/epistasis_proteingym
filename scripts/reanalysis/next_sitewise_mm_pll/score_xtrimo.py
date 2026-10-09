#!/usr/bin/env python
"""Conditional masked marginals and full PLL for xTrimoPGLM-10B-MLM."""
from __future__ import annotations
import argparse,json,time,sys
from pathlib import Path
import numpy as np,pandas as pd,torch
from transformers import AutoConfig,AutoModelForMaskedLM,AutoTokenizer
ROOT=Path('/data/users/akolchina/epistasis_proteingym');INPUT=ROOT/'results/tables/intermediate/tsuboyama_epistatic';CHECKPOINT=Path('/data/users/akolchina/model_weights/hf/proteinglm-10b-mlm')
sys.path.insert(0,str(ROOT/'scripts/reanalysis/multimodel_mm_pll'));from score import AA,atomic_csv,infer_wt,mutate,parse_double
class Backend:
 def __init__(self):
  self.tok=AutoTokenizer.from_pretrained(CHECKPOINT,trust_remote_code=True,use_fast=True);cfg=AutoConfig.from_pretrained(CHECKPOINT,trust_remote_code=True,torch_dtype=torch.bfloat16);cfg.is_causal=False;cfg.post_layer_norm=True
  self.model=AutoModelForMaskedLM.from_pretrained(CHECKPOINT,config=cfg,trust_remote_code=True,torch_dtype=torch.bfloat16,low_cpu_mem_usage=True).cuda().eval();self.ids={a:self.tok.encode(a,add_special_tokens=False)[0] for a in AA}
 def batch(self,req):
  seqs=[s for s,_ in req];x=self.tok(seqs,return_tensors='pt',padding=True).to('cuda')
  for r,(s,p) in enumerate(req):
   ids=self.tok.encode(s,add_special_tokens=True);aa=self.ids[s[p]];q=p+1
   if q>=len(ids) or ids[q]!=aa:raise ValueError(f'token/sequence position mismatch at {p}')
   x['input_ids'][r,q]=self.tok.mask_token_id
  with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):z=torch.log_softmax(self.model(**x).logits.float(),-1)
  return z[:,1:-1].cpu()
 def logits(self,requests,batch):
  unique=list(dict.fromkeys(requests));out={}
  for i in range(0,len(unique),batch):
   cur=unique[i:i+batch];z=self.batch(cur)
   for r,key in enumerate(cur):out[key]=z[r,key[1]].numpy()
  return out
 def pll(self,seqs,batch):
  todo=[(r,p) for r,s in enumerate(seqs) for p in range(len(s))];total=np.zeros(len(seqs))
  for i in range(0,len(todo),batch):
   cur=todo[i:i+batch];z=self.batch([(seqs[r],p) for r,p in cur])
   for q,(r,p) in enumerate(cur):total[r]+=float(z[q,p,self.ids[seqs[r][p]]])
  return total
def main():
 p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--batch-size',type=int,default=2);p.add_argument('--max-doubles',type=int);a=p.parse_args();t=time.perf_counter();a.output.mkdir(parents=True,exist_ok=True)
 if (a.output/'scores.csv').exists() and (a.output/'runtime.json').exists():return
 d=pd.read_csv(INPUT/f'{a.dataset}.csv');d=d[d.num_mutations.eq(2)].copy().reset_index(drop=True)
 if a.max_doubles and len(d)>a.max_doubles:d=d.iloc[np.sort(np.random.default_rng(20261009).choice(len(d),a.max_doubles,False))].reset_index(drop=True)
 wt=infer_wt(d);b=Backend();recs=[];req=[];keys=[];unique={wt:None}
 for row in d.itertuples():
  ma,mb=parse_double(row.mutant);sa,sb=mutate(wt,ma),mutate(wt,mb);sab=mutate(sa,mb);assert sab==row.mutated_sequence;k=((wt,ma[1]),(sb,ma[1]),(wt,mb[1]),(sa,mb[1]));req+=list(k);keys.append(k);recs.append((row,ma,mb,sa,sb,sab));unique.update({sa:None,sb:None,sab:None})
 z=b.logits(req,a.batch_size);rows=[]
 for rec,k in zip(recs,keys):
  row,ma,mb,sa,sb,sab=rec;delta=lambda q,m:float(z[q][b.ids[m[2]]]-z[q][b.ids[m[0]]]);da0,da_b,db0,db_a=delta(k[0],ma),delta(k[1],ma),delta(k[2],mb),delta(k[3],mb);eba,eab=da_b-da0,db_a-db0
  rows.append(dict(dataset=a.dataset,model_key='xTrimoPGLM-10B-MLM',mutant=row.mutant,mutated_sequence=row.mutated_sequence,pair_name=row.pair_name,experimental_dG=row.dG,experimental_coupling=row.thermodynamic_coupling,epistatic=row.epistatic,ProteinGym_default=getattr(row,'_'+str(list(d.columns).index('xTrimoPGLM-10B-MLM')+1),np.nan),delta_A_WT=da0,delta_A_B=da_b,delta_B_WT=db0,delta_B_A=db_a,E_B_to_A=eba,E_A_to_B=eab,E_cond=.5*(eba+eab),recomputed_default_MM=da0+db0,S_cond_direct=.5*((da0+db_a)+(db0+da_b))))
 out=pd.DataFrame(rows);out['ProteinGym_default']=d['xTrimoPGLM-10B-MLM'].to_numpy();atomic_csv(out,a.output/'conditional_scores.csv');cache={};cp=a.output/'sequence_pll_cache.csv'
 if cp.exists():q=pd.read_csv(cp);cache=dict(zip(q.sequence,q.PLL))
 miss=[x for x in unique if x not in cache]
 for i in range(0,len(miss),16):cur=miss[i:i+16];cache.update(zip(cur,b.pll(cur,a.batch_size)));atomic_csv(pd.DataFrame({'sequence':list(cache),'PLL':[cache[x] for x in cache]}),cp)
 out['PLL_WT']=cache[wt];out['PLL_A']=[cache[x[3]] for x in recs];out['PLL_B']=[cache[x[4]] for x in recs];out['PLL_AB']=[cache[x[5]] for x in recs];out['epsilon_PLL']=out.PLL_AB-out.PLL_A-out.PLL_B+out.PLL_WT;atomic_csv(out,a.output/'scores.csv');(a.output/'runtime.json').write_text(json.dumps({'dataset':a.dataset,'model_key':'xTrimoPGLM-10B-MLM','n_double_mutants':len(out),'seconds':time.perf_counter()-t},indent=2)+'\n')
if __name__=='__main__':main()
