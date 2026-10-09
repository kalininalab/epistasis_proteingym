#!/usr/bin/env python
"""Conditional mutation scoring and PLL for the official MULAN-small model."""
from __future__ import annotations
import argparse,json,sys,tempfile,time,os
from pathlib import Path
import numpy as np,pandas as pd,torch
from torch.utils.data import DataLoader
ROOT=Path('/data/users/akolchina/epistasis_proteingym');PG=Path('/data/users/akolchina/software/ProteinGym');ASSET=Path('/data/users/akolchina/proteingym_assets');INPUT=ROOT/'results/tables/intermediate/tsuboyama_epistatic';CKPT=Path('/data/users/akolchina/model_weights/hf/MULAN-small')
sys.path[:0]=[str(PG/'proteingym/baselines/mulan'),'/data/users/akolchina/software/mulan_deps',str(ROOT/'scripts/reanalysis/multimodel_mm_pll')]
from mulan.dataset import ProteinDataset,data_collate_fn_dynamic
from mulan.model import StructEsmForMaskedLM
from mulan.model_utils import auto_detect_base_tokenizer
from score import AA,atomic_csv,infer_wt,mutate,parse_double
def pdb_for(ds):
 r=pd.read_csv(ASSET/'v1.3/DMS_substitutions.csv');name=str(r[r.DMS_id.eq(ds)].iloc[0].pdb_file).split('|')[0];return next((ASSET/'v1.3/structures').rglob(name))
class Backend:
 def __init__(self,ds,wt):
  self.model=StructEsmForMaskedLM.from_pretrained(CKPT).cuda().eval();self.tok=auto_detect_base_tokenizer(self.model.config,False);self.ids={a:self.tok.get_vocab()[a] for a in AA}
  self.tmp=tempfile.TemporaryDirectory(prefix='mulan_');base=Path(self.tmp.name);src=pdb_for(ds);(base/src.name).symlink_to(src)
  data=ProteinDataset(str(base),saved_dataset_path=str(base/'prepared'),use_foldseek_sequences=False,is_experimental_structure=False,extract_foldseek_in_tokenizer=False)
  def collate(x):return data_collate_fn_dynamic(x,esm_tokenizer=self.tok,nan_value=np.deg2rad(data.tokenizer.nan_fill_value),mask_inputs=False,all_amino_acids=data.tokenizer.one_letter_aas,use_foldseek_sequences=False)
  batch=next(iter(DataLoader(data,batch_size=1,shuffle=False,collate_fn=collate)));self.base_ids=batch['input_ids'];self.base_attention=batch['attention_mask'];self.base_struct=[x for x in batch['struct_inputs']]
  decoded=''.join(self.tok.convert_ids_to_tokens(self.base_ids[0,1:-1].tolist())).replace(' ','')
  if decoded!=wt:raise ValueError(f'MULAN structure sequence differs from WT for {ds}: {decoded} != {wt}')
 def batch(self,req):
  n=len(req);ids=self.base_ids.expand(n,-1).clone();att=self.base_attention.expand(n,-1).clone();struct=[x.expand(n,*x.shape[1:]).clone() for x in self.base_struct]
  for r,(seq,p) in enumerate(req):
   ids[r,1:len(seq)+1]=torch.tensor([self.ids[a] for a in seq]);ids[r,p+1]=self.tok.mask_token_id
   for x in struct:x[r,p+1]=-4.
  struct=[x.cuda() for x in struct]
  with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):z=torch.log_softmax(self.model(input_ids=ids.cuda(),attention_mask=att.cuda(),struct_inputs=struct,output_hidden_states=False)['logits']['scores'].float(),-1)
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
 p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--batch-size',type=int,default=16);p.add_argument('--max-doubles',type=int);a=p.parse_args();t=time.perf_counter();a.output.mkdir(parents=True,exist_ok=True)
 if (a.output/'scores.csv').exists() and (a.output/'runtime.json').exists():return
 d=pd.read_csv(INPUT/f'{a.dataset}.csv');d=d[d.num_mutations.eq(2)].copy().reset_index(drop=True)
 if a.max_doubles and len(d)>a.max_doubles:d=d.iloc[np.sort(np.random.default_rng(20261009).choice(len(d),a.max_doubles,False))].reset_index(drop=True)
 wt=infer_wt(d);b=Backend(a.dataset,wt);recs=[];req=[];keys=[];unique={wt:None}
 for row in d.itertuples():
  ma,mb=parse_double(row.mutant);sa,sb=mutate(wt,ma),mutate(wt,mb);sab=mutate(sa,mb);assert sab==row.mutated_sequence;k=((wt,ma[1]),(sb,ma[1]),(wt,mb[1]),(sa,mb[1]));req+=list(k);keys.append(k);recs.append((row,ma,mb,sa,sb,sab));unique.update({sa:None,sb:None,sab:None})
 z=b.logits(req,a.batch_size);rows=[]
 for rec,k in zip(recs,keys):
  row,ma,mb,sa,sb,sab=rec;delta=lambda q,m:float(z[q][b.ids[m[2]]]-z[q][b.ids[m[0]]]);da0,da_b,db0,db_a=delta(k[0],ma),delta(k[1],ma),delta(k[2],mb),delta(k[3],mb);eba,eab=da_b-da0,db_a-db0
  rows.append(dict(dataset=a.dataset,model_key='MULAN_small',mutant=row.mutant,mutated_sequence=row.mutated_sequence,pair_name=row.pair_name,experimental_dG=row.dG,experimental_coupling=row.thermodynamic_coupling,epistatic=row.epistatic,ProteinGym_default=d.loc[row.Index,'MULAN_small'],delta_A_WT=da0,delta_A_B=da_b,delta_B_WT=db0,delta_B_A=db_a,E_B_to_A=eba,E_A_to_B=eab,E_cond=.5*(eba+eab),recomputed_default_MM=da0+db0,S_cond_direct=.5*((da0+db_a)+(db0+da_b))))
 out=pd.DataFrame(rows);atomic_csv(out,a.output/'conditional_scores.csv');cache={};cp=a.output/'sequence_pll_cache.csv'
 if cp.exists():q=pd.read_csv(cp);cache=dict(zip(q.sequence,q.PLL))
 miss=[x for x in unique if x not in cache]
 for i in range(0,len(miss),32):cur=miss[i:i+32];cache.update(zip(cur,b.pll(cur,a.batch_size)));atomic_csv(pd.DataFrame({'sequence':list(cache),'PLL':[cache[x] for x in cache]}),cp)
 out['PLL_WT']=cache[wt];out['PLL_A']=[cache[x[3]] for x in recs];out['PLL_B']=[cache[x[4]] for x in recs];out['PLL_AB']=[cache[x[5]] for x in recs];out['epsilon_PLL']=out.PLL_AB-out.PLL_A-out.PLL_B+out.PLL_WT;atomic_csv(out,a.output/'scores.csv');(a.output/'runtime.json').write_text(json.dumps({'dataset':a.dataset,'model_key':'MULAN_small','n_double_mutants':len(out),'seconds':time.perf_counter()-t},indent=2)+'\n')
if __name__=='__main__':main()
