#!/usr/bin/env python
"""Conditional masked-marginals and full PLL for ProSST-1024 and SaProt-35M-AF2."""
from __future__ import annotations
import argparse,csv,json,sys,time,subprocess,tempfile,os
from pathlib import Path
import numpy as np,pandas as pd,torch
from transformers import AutoModelForMaskedLM,AutoTokenizer
ROOT=Path('/data/users/akolchina/epistasis_proteingym');INPUT=ROOT/'results/tables/intermediate/tsuboyama_epistatic';ASSET=Path('/data/users/akolchina/proteingym_assets')
sys.path.insert(0,str(ROOT/'scripts/reanalysis/multimodel_mm_pll'));from score import AA,atomic_csv,infer_wt,mutate,parse_double
CFG={'ProSST-1024':('AI4Protein/ProSST-1024','ProSST-1024'),'SaProt_35M_AF2':('westlake-repl/SaProt_35M_AF2','SaProt_35M_AF2')};SV='pynwrqhgdlvtmfsaeikc#'
def read_fasta(p):return ''.join(x.strip() for x in open(p) if not x.startswith('>'))
def pdb_for(ds):
 r=pd.read_csv(ASSET/'v1.3/DMS_substitutions.csv');x=r[r.DMS_id.eq(ds)].iloc[0];name=str(x.pdb_file).split('|')[0];return next((ASSET/'v1.3/structures').rglob(name))
def saprot_3di(ds,wt):
 cache=ASSET/'saprot_3di';cache.mkdir(exist_ok=True);out=cache/f'{ds}.txt'
 if out.exists():return out.read_text().strip()
 fold='/data/users/akolchina/software/foldseek/foldseek/bin/foldseek';util='/data/users/akolchina/software/ProteinGym/proteingym/baselines/saprot/foldseek_util.py';sys.path.insert(0,str(Path(util).parent));from foldseek_util import get_struc_seq
 # ProteinGym's helper uses a fixed relative temporary filename.  Run it in a
 # private directory because Condor workers share the same initial directory.
 old=os.getcwd()
 with tempfile.TemporaryDirectory(prefix='saprot_foldseek_') as tmp:
  try:
   os.chdir(tmp)
   seqs=get_struc_seq(fold,str(pdb_for(ds)),['A'],plddt_mask=True,plddt_threshold=70)
  finally:os.chdir(old)
 s=seqs['A'][1].lower();assert len(s)==len(wt),(len(s),len(wt))
 try:out.write_text(s+'\n')
 except OSError:pass
 return s

def prosst_tokens(ds,wt):
 root=ASSET/'prosst/structure_sequence/1024';p=root/f'{ds}.fasta'
 if not p.exists():
  # Some ProteinGym releases renamed the protein prefix while retaining the
  # same Tsuboyama structure (for example PSAE_PICP2 -> PSAE_SYNP2, PDB 1PSE).
  suffix='_Tsuboyama_'+ds.split('_Tsuboyama_',1)[1]
  candidates=list(root.glob(f'*{suffix}.fasta'))
  if len(candidates)!=1:raise FileNotFoundError(f'No unique ProSST token file for {ds}: {candidates}')
  p=candidates[0]
 v=[int(x) for x in read_fasta(p).split(',')]
 if len(v)!=len(wt):raise ValueError(f'ProSST token/WT length mismatch for {ds}: {len(v)} != {len(wt)} ({p})')
 return v
class Backend:
 def __init__(self,key,ds,wt):
  repo,_=CFG[key];self.key=key;self.model=AutoModelForMaskedLM.from_pretrained(repo,trust_remote_code=True).cuda().eval();self.tok=AutoTokenizer.from_pretrained(repo,trust_remote_code=True);self.wt=wt
  if key.startswith('ProSST'):
   v=prosst_tokens(ds,wt);self.ss=torch.tensor([[1]+[x+3 for x in v]+[2]],device='cuda')
  else:self.di=saprot_3di(ds,wt);self.vocab=self.tok.get_vocab();self.aa_ids={a:[self.vocab[a+s] for s in SV] for a in AA}
 def batch(self,req):
  seqs=[s for s,p in req];pos=[p for s,p in req]
  if self.key.startswith('ProSST'):
   x=self.tok(seqs,return_tensors='pt',padding=True).to('cuda');ids=x['input_ids'];
   for i,p in enumerate(pos):ids[i,p+1]=self.tok.mask_token_id
   with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):z=self.model(**x,ss_input_ids=self.ss.expand(len(req),-1)).logits.float()
   return torch.log_softmax(z,-1)[:,1:-1].cpu(),{a:self.tok.get_vocab()[a] for a in AA}
  texts=[]
  for seq,p in req:
   t=[a+b for a,b in zip(seq,self.di)];t[p]='#'+t[p][-1];texts.append(' '.join(t))
  x=self.tok(texts,return_tensors='pt',padding=True).to('cuda')
  with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):lp=torch.log_softmax(self.model(**x).logits.float(),-1)
  # marginalize the fixed model output across all 3Di states exactly as ProteinGym SaProt scoring
  z=torch.stack([torch.logsumexp(lp[:,:,self.aa_ids[a]],-1) for a in AA],-1)[:,1:-1].cpu();return z,{a:i for i,a in enumerate(AA)}
 def logits(self,requests,batch):
  u=list(dict.fromkeys(requests));o={}
  for i in range(0,len(u),batch):
   c=u[i:i+batch];z,ids=self.batch(c)
   for r,k in enumerate(c):o[k]=(z[r,k[1]].numpy(),ids)
  return o
 def pll(self,seqs,batch):
  req=[(r,p) for r,s in enumerate(seqs) for p in range(len(s))];tot=np.zeros(len(seqs))
  for i in range(0,len(req),batch):
   c=req[i:i+batch];z,ids=self.batch([(seqs[r],p) for r,p in c])
   for q,(r,p) in enumerate(c):tot[r]+=float(z[q,p,ids[seqs[r][p]]])
  return tot
def main():
 p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);p.add_argument('--model',choices=CFG,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--batch-size',type=int,default=8);p.add_argument('--max-doubles',type=int);a=p.parse_args();t=time.perf_counter();a.output.mkdir(parents=True,exist_ok=True)
 if (a.output/'scores.csv').exists()and(a.output/'runtime.json').exists():return
 d=pd.read_csv(INPUT/f'{a.dataset}.csv');d=d[d.num_mutations.eq(2)].copy()
 if a.max_doubles and len(d)>a.max_doubles:d=d.iloc[np.sort(np.random.default_rng(20261008).choice(len(d),a.max_doubles,False))]
 wt=infer_wt(d);b=Backend(a.model,a.dataset,wt);recs=[];req=[];keys=[];unique={wt:None}
 for row in d.itertuples():
  ma,mb=parse_double(row.mutant);sa,sb=mutate(wt,ma),mutate(wt,mb);sab=mutate(sa,mb);assert sab==row.mutated_sequence;k=((wt,ma[1]),(sb,ma[1]),(wt,mb[1]),(sa,mb[1]));req+=list(k);keys.append(k);recs.append((row,ma,mb,sa,sb,sab));unique.update({sa:None,sb:None,sab:None})
 z=b.logits(req,a.batch_size);rows=[]
 for rec,k in zip(recs,keys):
  row,ma,mb,sa,sb,sab=rec;delta=lambda q,m:float(z[q][0][z[q][1][m[2]]]-z[q][0][z[q][1][m[0]]]);da0,da_b,db0,db_a=delta(k[0],ma),delta(k[1],ma),delta(k[2],mb),delta(k[3],mb);eba,eab=da_b-da0,db_a-db0
  rows.append(dict(dataset=a.dataset,model_key=a.model,mutant=row.mutant,mutated_sequence=row.mutated_sequence,pair_name=row.pair_name,experimental_dG=row.dG,experimental_coupling=row.thermodynamic_coupling,epistatic=row.epistatic,ProteinGym_default=d.loc[row.Index,a.model],delta_A_WT=da0,delta_A_B=da_b,delta_B_WT=db0,delta_B_A=db_a,E_B_to_A=eba,E_A_to_B=eab,E_cond=.5*(eba+eab),recomputed_default_MM=da0+db0,S_cond_direct=.5*((da0+db_a)+(db0+da_b))))
 out=pd.DataFrame(rows);atomic_csv(out,a.output/'conditional_scores.csv');cache={};cp=a.output/'sequence_pll_cache.csv'
 if cp.exists():q=pd.read_csv(cp);cache=dict(zip(q.sequence,q.PLL))
 miss=[x for x in unique if x not in cache]
 for i in range(0,len(miss),32):c=miss[i:i+32];cache.update(zip(c,b.pll(c,a.batch_size)));atomic_csv(pd.DataFrame({'sequence':list(cache),'PLL':[cache[x] for x in cache]}),cp)
 out['PLL_WT']=cache[wt];out['PLL_A']=[cache[x[3]] for x in recs];out['PLL_B']=[cache[x[4]] for x in recs];out['PLL_AB']=[cache[x[5]] for x in recs];out['epsilon_PLL']=out.PLL_AB-out.PLL_A-out.PLL_B+out.PLL_WT;atomic_csv(out,a.output/'scores.csv');(a.output/'runtime.json').write_text(json.dumps({'dataset':a.dataset,'model_key':a.model,'n_double_mutants':len(out),'seconds':time.perf_counter()-t},indent=2)+'\n')
if __name__=='__main__':main()
