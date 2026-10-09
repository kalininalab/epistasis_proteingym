#!/usr/bin/env python
"""Five-seed ProteinGym MSA-Transformer conditional MM and PLL."""
from __future__ import annotations
import argparse,json,time,sys
from pathlib import Path
import numpy as np,pandas as pd,torch
ROOT=Path('/data/users/akolchina/epistasis_proteingym');PG=Path('/data/users/akolchina/software/ProteinGym');ASSET=Path('/data/users/akolchina/proteingym_assets');INPUT=ROOT/'results/tables/intermediate/tsuboyama_epistatic'
sys.path[:0]=[str(PG/'proteingym'),str(PG/'proteingym/baselines/esm')];from esm import pretrained
from compute_fitness import process_msa,sample_msa
sys.path.insert(0,str(ROOT/'scripts/reanalysis/multimodel_mm_pll'));from score import AA,atomic_csv,infer_wt,mutate,parse_double

def metadata(ds):
 r=pd.read_csv(ASSET/'v1.3/DMS_substitutions.csv');return r[r.DMS_id.eq(ds)].iloc[0]
class Backend:
 def __init__(self,ds,wt,nseq,seeds):
  self.model,self.alphabet=pretrained.esm_msa1b_t12_100M_UR50S();self.model=self.model.cuda().eval();self.ids={a:self.alphabet.get_idx(a) for a in AA};self.converter=self.alphabet.get_batch_converter();m=metadata(ds)
  msa=ASSET/'msa/files'/m.MSA_filename;weights=ASSET/'msa/weights'/m.weight_file_name;proc=process_msa(str(msa),str(weights),False,'')
  self.homologs=[]
  for seed in seeds:
   sampled=sample_msa(str(msa),nseq,'sequence-reweighting',seed,str(weights),proc)
   if sampled[0][1].replace('.','-').replace('-','')!=wt:raise ValueError(f'MSA focus sequence differs from WT for {ds}')
   self.homologs.append(sampled[1:])
 def one_seed_batch(self,req,homologs):
  alignments=[[('query',s)]+homologs for s,_ in req];_,_,x=self.converter(alignments);x=x.cuda()
  for r,(_,p) in enumerate(req):x[r,0,p+1]=self.alphabet.mask_idx
  with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):z=torch.log_softmax(self.model(x)['logits'].float(),-1)[:,0,1:-1]
  return z.cpu()
 def batch(self,req):return torch.stack([self.one_seed_batch(req,h) for h in self.homologs]).mean(0)
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
 p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--batch-size',type=int,default=1);p.add_argument('--msa-sequences',type=int,default=400);p.add_argument('--seeds',type=int,nargs='+',default=[1,2,3,4,5]);p.add_argument('--max-doubles',type=int);a=p.parse_args();t=time.perf_counter();a.output.mkdir(parents=True,exist_ok=True)
 if (a.output/'scores.csv').exists() and (a.output/'runtime.json').exists():return
 d=pd.read_csv(INPUT/f'{a.dataset}.csv');d=d[d.num_mutations.eq(2)].copy().reset_index(drop=True)
 if a.max_doubles and len(d)>a.max_doubles:d=d.iloc[np.sort(np.random.default_rng(20261009).choice(len(d),a.max_doubles,False))].reset_index(drop=True)
 wt=infer_wt(d);b=Backend(a.dataset,wt,a.msa_sequences,a.seeds);recs=[];req=[];keys=[];unique={wt:None}
 for row in d.itertuples():
  ma,mb=parse_double(row.mutant);sa,sb=mutate(wt,ma),mutate(wt,mb);sab=mutate(sa,mb);assert sab==row.mutated_sequence;k=((wt,ma[1]),(sb,ma[1]),(wt,mb[1]),(sa,mb[1]));req+=list(k);keys.append(k);recs.append((row,ma,mb,sa,sb,sab));unique.update({sa:None,sb:None,sab:None})
 z=b.logits(req,a.batch_size);rows=[]
 for rec,k in zip(recs,keys):
  row,ma,mb,sa,sb,sab=rec;delta=lambda q,m:float(z[q][b.ids[m[2]]]-z[q][b.ids[m[0]]]);da0,da_b,db0,db_a=delta(k[0],ma),delta(k[1],ma),delta(k[2],mb),delta(k[3],mb);eba,eab=da_b-da0,db_a-db0
  rows.append(dict(dataset=a.dataset,model_key='MSA_Transformer_ensemble',mutant=row.mutant,mutated_sequence=row.mutated_sequence,pair_name=row.pair_name,experimental_dG=row.dG,experimental_coupling=row.thermodynamic_coupling,epistatic=row.epistatic,ProteinGym_default=d.loc[row.Index,'MSA_Transformer_ensemble'],delta_A_WT=da0,delta_A_B=da_b,delta_B_WT=db0,delta_B_A=db_a,E_B_to_A=eba,E_A_to_B=eab,E_cond=.5*(eba+eab),recomputed_default_MM=da0+db0,S_cond_direct=.5*((da0+db_a)+(db0+da_b))))
 out=pd.DataFrame(rows);atomic_csv(out,a.output/'conditional_scores.csv');cache={};cp=a.output/'sequence_pll_cache.csv'
 if cp.exists():q=pd.read_csv(cp);cache=dict(zip(q.sequence,q.PLL))
 miss=[x for x in unique if x not in cache]
 for i in range(0,len(miss),8):cur=miss[i:i+8];cache.update(zip(cur,b.pll(cur,a.batch_size)));atomic_csv(pd.DataFrame({'sequence':list(cache),'PLL':[cache[x] for x in cache]}),cp)
 out['PLL_WT']=cache[wt];out['PLL_A']=[cache[x[3]] for x in recs];out['PLL_B']=[cache[x[4]] for x in recs];out['PLL_AB']=[cache[x[5]] for x in recs];out['epsilon_PLL']=out.PLL_AB-out.PLL_A-out.PLL_B+out.PLL_WT;atomic_csv(out,a.output/'scores.csv');(a.output/'runtime.json').write_text(json.dumps({'dataset':a.dataset,'model_key':'MSA_Transformer_ensemble','n_double_mutants':len(out),'msa_sequences':a.msa_sequences,'seeds':a.seeds,'seconds':time.perf_counter()-t},indent=2)+'\n')
if __name__=='__main__':main()
