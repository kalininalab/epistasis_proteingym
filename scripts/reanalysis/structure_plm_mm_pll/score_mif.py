#!/usr/bin/env python
"""Conditional masked-marginals and full PLL for MIF and MIFST with fixed AF2 backbones."""
from __future__ import annotations
import argparse,json,sys,time
from pathlib import Path
import numpy as np,pandas as pd,torch
ROOT=Path('/data/users/akolchina/epistasis_proteingym');INPUT=ROOT/'results/tables/intermediate/tsuboyama_epistatic';ASSET=Path('/data/users/akolchina/proteingym_assets/v1.3')
sys.path.insert(0,str(ROOT/'scripts/reanalysis/multimodel_mm_pll'));from score import AA,atomic_csv,infer_wt,mutate,parse_double
sys.path.insert(0,'/data/users/akolchina/software/protein-sequence-models')
from sequence_models.constants import PROTEIN_ALPHABET
from sequence_models.pdb_utils import parse_PDB,process_coords
from sequence_models.pretrained import load_model_and_alphabet
AAID={x:PROTEIN_ALPHABET.index(x) for x in AA};MASK=PROTEIN_ALPHABET.index('#')
def pdb_for(ds):
 r=pd.read_csv(ASSET/'DMS_substitutions.csv');name=str(r[r.DMS_id.eq(ds)].iloc[0].pdb_file).split('|')[0];return next((ASSET/'structures').rglob(name))
class Backend:
 def __init__(self,key,ds,wt,weights):
  self.model,self.collater=load_model_and_alphabet(str(weights));self.model=self.model.cuda().eval();coords,pdbseq,_=parse_PDB(str(pdb_for(ds)));assert len(coords)==len(wt),(len(coords),len(wt));c={'N':coords[:,0],'CA':coords[:,1],'C':coords[:,2]};self.geom=process_coords(c)
 def tensors(self,items):
  batch=[]
  for seq,pos in items:
   x=list(seq);x[pos]='#';batch.append([''.join(x)]+[torch.tensor(y,dtype=torch.float) for y in self.geom])
  return [x.cuda() for x in self.collater(batch)]
 @torch.inference_mode()
 def logits(self,requests,batch):
  u=list(dict.fromkeys(requests));o={}
  for i in range(0,len(u),batch):
   c=u[i:i+batch];x=self.tensors(c);z=self.model(*x,result='logits').float().cpu()
   for r,k in enumerate(c):o[k]=z[r,k[1]].numpy()
  return o
 @torch.inference_mode()
 def pll(self,seqs,batch):
  req=[(r,p) for r,s in enumerate(seqs) for p in range(len(s))];tot=np.zeros(len(seqs))
  for i in range(0,len(req),batch):
   c=req[i:i+batch];items=[(seqs[r],p) for r,p in c];z=torch.log_softmax(self.model(*self.tensors(items),result='logits').float(),-1).cpu()
   for q,(r,p) in enumerate(c):tot[r]+=float(z[q,p,AAID[seqs[r][p]]])
  return tot
def main():
 p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);p.add_argument('--model',choices=['MIF','MIFST'],required=True);p.add_argument('--weights',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--batch-size',type=int,default=16);p.add_argument('--max-doubles',type=int);a=p.parse_args();t=time.perf_counter();a.output.mkdir(parents=True,exist_ok=True)
 if (a.output/'scores.csv').exists()and(a.output/'runtime.json').exists():return
 d=pd.read_csv(INPUT/f'{a.dataset}.csv');d=d[d.num_mutations.eq(2)].copy()
 if a.max_doubles and len(d)>a.max_doubles:d=d.iloc[np.sort(np.random.default_rng(20261008).choice(len(d),a.max_doubles,False))]
 wt=infer_wt(d);b=Backend(a.model,a.dataset,wt,a.weights);recs=[];req=[];keys=[];unique={wt:None}
 for row in d.itertuples():
  ma,mb=parse_double(row.mutant);sa,sb=mutate(wt,ma),mutate(wt,mb);sab=mutate(sa,mb);assert sab==row.mutated_sequence;k=((wt,ma[1]),(sb,ma[1]),(wt,mb[1]),(sa,mb[1]));req+=list(k);keys.append(k);recs.append((row,ma,mb,sa,sb,sab));unique.update({sa:None,sb:None,sab:None})
 z=b.logits(req,a.batch_size);rows=[]
 for rec,k in zip(recs,keys):
  row,ma,mb,sa,sb,sab=rec;delta=lambda q,m:float(z[q][AAID[m[2]]]-z[q][AAID[m[0]]]);da0,da_b,db0,db_a=delta(k[0],ma),delta(k[1],ma),delta(k[2],mb),delta(k[3],mb);eba,eab=da_b-da0,db_a-db0
  rows.append(dict(dataset=a.dataset,model_key=a.model,mutant=row.mutant,mutated_sequence=row.mutated_sequence,pair_name=row.pair_name,experimental_dG=row.dG,experimental_coupling=row.thermodynamic_coupling,epistatic=row.epistatic,ProteinGym_default=d.loc[row.Index,a.model],delta_A_WT=da0,delta_A_B=da_b,delta_B_WT=db0,delta_B_A=db_a,E_B_to_A=eba,E_A_to_B=eab,E_cond=.5*(eba+eab),recomputed_default_MM=da0+db0,S_cond_direct=.5*((da0+db_a)+(db0+da_b))))
 out=pd.DataFrame(rows);atomic_csv(out,a.output/'conditional_scores.csv');cache={};cp=a.output/'sequence_pll_cache.csv'
 if cp.exists():q=pd.read_csv(cp);cache=dict(zip(q.sequence,q.PLL))
 miss=[x for x in unique if x not in cache]
 for i in range(0,len(miss),64):c=miss[i:i+64];cache.update(zip(c,b.pll(c,a.batch_size)));atomic_csv(pd.DataFrame({'sequence':list(cache),'PLL':[cache[x] for x in cache]}),cp)
 out['PLL_WT']=cache[wt];out['PLL_A']=[cache[x[3]] for x in recs];out['PLL_B']=[cache[x[4]] for x in recs];out['PLL_AB']=[cache[x[5]] for x in recs];out['epsilon_PLL']=out.PLL_AB-out.PLL_A-out.PLL_B+out.PLL_WT;atomic_csv(out,a.output/'scores.csv');(a.output/'runtime.json').write_text(json.dumps({'dataset':a.dataset,'model_key':a.model,'n_double_mutants':len(out),'seconds':time.perf_counter()-t},indent=2)+'\n')
if __name__=='__main__':main()
