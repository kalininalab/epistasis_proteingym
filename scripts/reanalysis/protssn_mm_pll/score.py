#!/usr/bin/env python
"""Conditional masked marginals and PLL for ProteinGym's ProtSSN k30_h1280."""
from __future__ import annotations
import argparse, copy, json, sys, time, types
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch_geometric.data import Batch
from torch_geometric.inspector import Inspector
from transformers import AutoModel, AutoTokenizer

# Official ProtSSN carries a custom propagate() implementation written for
# PyG <=2.4. PyG 2.5 renamed Inspector.distribute() to collect_param_data()
# without changing the mapping semantics used here.
if not hasattr(Inspector, "distribute"):
    Inspector.distribute = Inspector.collect_param_data

ROOT = Path("/data/users/akolchina/epistasis_proteingym")
PG = Path("/data/users/akolchina/software/ProteinGym/proteingym/baselines/protssn")
ASSET = Path("/data/users/akolchina/proteingym_assets")
INPUT = ROOT / "results/tables/intermediate/tsuboyama_epistatic"
WEIGHTS = Path("/data/users/akolchina/model_weights/protssn/protssn_k30_h1280.pt")
sys.path.insert(0, str(PG))
sys.path.insert(0, str(ROOT / "scripts/reanalysis/multimodel_mm_pll"))
# ProteinGym's ProtSSN dataset utility imports seaborn only for optional plotting.
# The scoring path uses no seaborn API, so avoid making this GPU workflow depend
# on an unrelated visualization package.
try:
    import seaborn  # noqa: F401
except ModuleNotFoundError:
    sys.modules["seaborn"] = types.ModuleType("seaborn")
from score import AA, atomic_csv, infer_wt, mutate, parse_double
from src.dataset.mutant_dataset import MutantDataset
from src.models import GNN_model
from src.utils.dataset_utils import NormalizeProtein


class NormalizeProteinCompat(NormalizeProtein):
    """Bridge the official pre-PyG-2.5 transform API to BaseTransform.forward."""

    def forward(self, data):
        return NormalizeProtein.__call__(self, data)


ORDER = "ARNDCQEGHILKMFPSTWYV"


def metadata(ds):
    m = pd.read_csv(ASSET / "v1.3/DMS_substitutions.csv")
    return m[m.DMS_id.eq(ds)].iloc[0]


def pdb_for(ds):
    name = str(metadata(ds).pdb_file).split("|")[0]
    return next((ASSET / "v1.3/structures").rglob(name))


def graph_for(ds, wt):
    root = ASSET / "protssn_graphs" / ds
    raw = root / "raw" / ds
    raw.mkdir(parents=True, exist_ok=True)
    target = raw / f"{ds}.pdb"
    if not target.exists():
        target.symlink_to(pdb_for(ds))
    norm = PG / "norm/cath_k30_mean_attr.pt"
    dataset = MutantDataset(
        root=str(root), name=f"{ds}_k30", raw_dir=str(root / "raw"),
        c_alpha_max_neighbors=30, pre_transform=NormalizeProteinCompat(str(norm)),
    )
    if len(dataset) != 1:
        raise ValueError(f"Expected one ProtSSN graph for {ds}, got {len(dataset)}")
    graph = dataset[0]
    graph_seq = "".join(ORDER[int(i)] for i in graph.y)
    if graph_seq != wt:
        raise ValueError(f"ProtSSN PDB/WT mismatch for {ds}: {graph_seq} != {wt}")
    return graph


class Backend:
    def __init__(self, ds, wt):
        self.graph = graph_for(ds, wt)
        self.tok = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
        self.plm = AutoModel.from_pretrained("facebook/esm2_t33_650M_UR50D").cuda().eval()
        cfg = yaml.safe_load((PG / "src/config/egnn.yaml").read_text())["egnn"]
        cfg["hidden_channels"] = 1280
        args = Namespace(gnn="egnn", gnn_config=cfg, plm_hidden_size=1280)
        self.gnn = GNN_model(args).cuda().eval()
        state = torch.load(WEIGHTS, map_location="cpu", weights_only=False)
        self.gnn.load_state_dict(state)
        self.ids = {a: ORDER.index(a) for a in AA}

    def batch(self, req):
        texts=[]
        for seq,pos in req:
            texts.append(seq[:pos] + self.tok.mask_token + seq[pos+1:])
        x=self.tok(texts,return_tensors="pt",padding=True).to("cuda")
        with torch.inference_mode(), torch.autocast("cuda",dtype=torch.float16):
            rep=self.plm(**x).last_hidden_state[:,1:1+len(req[0][0])]
            graphs=[]
            for i in range(len(req)):
                g=copy.copy(self.graph);g.esm_rep=rep[i];graphs.append(g)
            logits,_=self.gnn(Batch.from_data_list(graphs).cuda())
            z=torch.log_softmax(logits.float(),-1).reshape(len(req),-1,20)
        return z.cpu()

    def logits(self, requests, batch_size):
        unique=list(dict.fromkeys(requests));out={}
        for i in range(0,len(unique),batch_size):
            cur=unique[i:i+batch_size];z=self.batch(cur)
            for r,key in enumerate(cur):out[key]=z[r,key[1]].numpy()
        return out

    def pll(self, seqs, batch_size):
        todo=[(r,p) for r,s in enumerate(seqs) for p in range(len(s))]
        total=np.zeros(len(seqs))
        for i in range(0,len(todo),batch_size):
            cur=todo[i:i+batch_size];z=self.batch([(seqs[r],p) for r,p in cur])
            for q,(r,p) in enumerate(cur):total[r]+=float(z[q,p,self.ids[seqs[r][p]]])
        return total


def main():
    p=argparse.ArgumentParser();p.add_argument("--dataset",required=True);p.add_argument("--output",type=Path,required=True);p.add_argument("--batch-size",type=int,default=4);p.add_argument("--max-doubles",type=int);a=p.parse_args();start=time.perf_counter();a.output.mkdir(parents=True,exist_ok=True)
    if (a.output/"scores.csv").exists() and (a.output/"runtime.json").exists():return
    d=pd.read_csv(INPUT/f"{a.dataset}.csv");d=d[d.num_mutations.eq(2)].copy().reset_index(drop=True)
    if a.max_doubles and len(d)>a.max_doubles:d=d.iloc[np.sort(np.random.default_rng(20261011).choice(len(d),a.max_doubles,False))].reset_index(drop=True)
    wt=infer_wt(d);b=Backend(a.dataset,wt);records=[];requests=[];keys=[];unique={wt:None}
    for row in d.itertuples():
        ma,mb=parse_double(row.mutant);sa,sb=mutate(wt,ma),mutate(wt,mb);sab=mutate(sa,mb);assert sab==row.mutated_sequence
        k=((wt,ma[1]),(sb,ma[1]),(wt,mb[1]),(sa,mb[1]));requests.extend(k);keys.append(k);records.append((row,ma,mb,sa,sb,sab));unique.update({sa:None,sb:None,sab:None})
    z=b.logits(requests,a.batch_size);rows=[]
    for rec,k in zip(records,keys):
        row,ma,mb,sa,sb,sab=rec
        delta=lambda q,m:float(z[q][b.ids[m[2]]]-z[q][b.ids[m[0]]])
        da0,da_b,db0,db_a=delta(k[0],ma),delta(k[1],ma),delta(k[2],mb),delta(k[3],mb);eba,eab=da_b-da0,db_a-db0
        rows.append(dict(dataset=a.dataset,model_key="ProtSSN_k30_h1280",mutant=row.mutant,mutated_sequence=row.mutated_sequence,pair_name=row.pair_name,experimental_dG=row.dG,experimental_coupling=row.thermodynamic_coupling,epistatic=row.epistatic,ProteinGym_default=d.loc[row.Index,"ProtSSN_k30_h1280"],delta_A_WT=da0,delta_A_B=da_b,delta_B_WT=db0,delta_B_A=db_a,E_B_to_A=eba,E_A_to_B=eab,E_cond=.5*(eba+eab),recomputed_default_MM=da0+db0,S_cond_direct=.5*((da0+db_a)+(db0+da_b))))
    out=pd.DataFrame(rows);atomic_csv(out,a.output/"conditional_scores.csv");cache={};cp=a.output/"sequence_pll_cache.csv"
    if cp.exists():q=pd.read_csv(cp);cache=dict(zip(q.sequence,q.PLL))
    missing=[s for s in unique if s not in cache]
    for i in range(0,len(missing),16):
        cur=missing[i:i+16];cache.update(zip(cur,b.pll(cur,a.batch_size)));atomic_csv(pd.DataFrame({"sequence":list(cache),"PLL":[cache[s] for s in cache]}),cp)
    out["PLL_WT"]=cache[wt];out["PLL_A"]=[cache[x[3]] for x in records];out["PLL_B"]=[cache[x[4]] for x in records];out["PLL_AB"]=[cache[x[5]] for x in records];out["epsilon_PLL"]=out.PLL_AB-out.PLL_A-out.PLL_B+out.PLL_WT
    atomic_csv(out,a.output/"scores.csv");(a.output/"runtime.json").write_text(json.dumps({"dataset":a.dataset,"model_key":"ProtSSN_k30_h1280","n_double_mutants":len(out),"seconds":time.perf_counter()-start},indent=2)+"\n")


if __name__=="__main__":main()
