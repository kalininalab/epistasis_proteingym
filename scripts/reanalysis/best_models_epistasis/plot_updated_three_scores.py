#!/usr/bin/env python
"""Presentation heatmaps for original, conditional-MM, and PLL epistasis."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

ROOT=Path("/data/users/akolchina/epistasis_proteingym")
BASE=ROOT/"results/reanalysis/tsuboyama"
OUT=BASE/"updated_best_models_three_scores"
ORIGINAL=BASE/"best_models_fitted_epistasis/per_assay_model_correlations.csv"
SUMMARY=BASE/"best_models_fitted_epistasis/model_summary.csv"
ROOTS=[
 BASE/"multimodel_mm_pll/fits", BASE/"carp_mm_pll/fits",
 BASE/"modern_esm_mm_pll/full", BASE/"structure_plm_mm_pll/full",
 BASE/"next_sitewise_mm_pll/full", BASE/"vespa_family/fits", BASE/"vespa_family/vespa_full/fits",
]

def short_ds(x): return x.replace("_Tsuboyama_2023_"," ").replace("_Tsuboyama_2023","")

def new_metrics():
 frames=[]
 for root in ROOTS:
  for p in root.rglob("metrics.csv"):
   try:
    d=pd.read_csv(p)
    if {"model_key","rho_conditional_fitted_epistasis"}.issubset(d.columns):
     if "rho_PLL_fitted_epistasis" not in d: d["rho_PLL_fitted_epistasis"]=np.nan
     frames.append(d)
   except Exception: pass
 if frames:
  d=pd.concat(frames,ignore_index=True)
 else: d=pd.DataFrame()
 # Earlier ESM2-650M run has the same values in long form.
 p=BASE/"esm2_650m_three_scoring_comparison/per_assay_correlations.csv"
 if p.exists():
  e=pd.read_csv(p);e=e[e.metric.eq("fitted_epistasis")].copy();e["model_key"]="ESM2_650M"
  e=e.pivot_table(index=["dataset","model_key","subset"],columns="method",values="rho",aggfunc="first").reset_index()
  e=e.rename(columns={"conditional_mm":"rho_conditional_fitted_epistasis","pll":"rho_PLL_fitted_epistasis"})
  d=pd.concat([d,e],ignore_index=True,sort=False)
 d=d.drop_duplicates(["dataset","model_key","subset"],keep="last")
 return d

def model_order(meta):
 order={"joint":0,"sitewise":1,"explicit_pairwise":2,"user_model":3}
 return meta.assign(_o=meta.scoring_type.map(order)).sort_values(["_o","model_family","model"]).drop(columns="_o")

def matrix(d,models,datasets,value,model_col="model"):
 return d.pivot(index=model_col,columns="dataset",values=value).reindex(index=models,columns=datasets)

def draw(subset,original,new,meta):
 models=meta.model.tolist();datasets=sorted(original.dataset.unique())
 mats=[
  matrix(original[original.subset.eq(subset)],models,datasets,"rho"),
  matrix(new[new.subset.eq(subset)],models,datasets,"rho_conditional_fitted_epistasis","model_key"),
  matrix(new[new.subset.eq(subset)],models,datasets,"rho_PLL_fitted_epistasis","model_key"),
 ]
 titles=["Original ProteinGym scoring","Conditional masked-marginal","Pseudo-log-likelihood (PLL)"]
 recomputed=set(new.groupby("model_key").dataset.nunique().loc[lambda x:x==50].index)
 fig,axes=plt.subplots(3,1,figsize=(24,27),sharex=True)
 cmap=plt.get_cmap("RdBu_r").copy();cmap.set_bad("#E5E7EB")
 types=meta.scoring_type.tolist();bounds=[i for i in range(1,len(types)) if types[i]!=types[i-1]]
 for ax,p,title in zip(axes,mats,titles):
  im=ax.imshow(np.ma.masked_invalid(p.to_numpy(float)),aspect="auto",cmap=cmap,vmin=-1,vmax=1)
  ax.set_yticks(np.arange(len(models)),models,fontsize=8.5)
  for tick,name in zip(ax.get_yticklabels(),models):
   if name in recomputed: tick.set_fontweight("bold")
  for b in bounds: ax.axhline(b-.5,color="#172033",lw=1.5)
  ax.set_title(title,loc="left",weight="bold",fontsize=14,pad=8)
  ax.set_ylabel("Best-model representative")
 axes[-1].set_xticks(np.arange(len(datasets)),[short_ds(x) for x in datasets],rotation=65,ha="right",fontsize=7)
 cb=fig.colorbar(im,ax=axes,pad=.008,shrink=.75);cb.set_label("Spearman ρ: predicted vs experimental thermodynamic coupling")
 label="all double mutants" if subset=="all_doubles" else "predefined epistatic subset"
 fig.suptitle(f"Tsuboyama epistasis prediction — {label}",fontsize=19,weight="bold",y=.998)
 fig.text(.5,.004,"Bold model names: at least one new scoring method recomputed for all 50 assays. Gray cells: that method is unavailable or still running.",ha="center",fontsize=10,color="#475569")
 fig.tight_layout(rect=(0,.012,1,.99));OUT.mkdir(parents=True,exist_ok=True)
 for ext in ("png","pdf"): fig.savefig(OUT/f"three_score_epistasis_heatmap_{subset}.{ext}",dpi=220 if ext=="png" else None,bbox_inches="tight")
 plt.close(fig)
 return recomputed

def main():
 original=pd.read_csv(ORIGINAL);summary=pd.read_csv(SUMMARY)
 meta=summary[["model","model_family","scoring_type"]].drop_duplicates();meta=model_order(meta)
 new=new_metrics();new=new[new.model_key.isin(meta.model)]
 recomputed=set()
 for subset in ("all_doubles","epistatic"): recomputed|=draw(subset,original,new,meta)
 coverage=(new.groupby("model_key").agg(assays=("dataset","nunique"),conditional_assays=("rho_conditional_fitted_epistasis","count"),pll_assays=("rho_PLL_fitted_epistasis","count")).reset_index().sort_values("model_key"))
 # Counts above include both subsets, so a complete method has 100 rows.
 coverage["complete_conditional_50"]=coverage.conditional_assays.ge(99);coverage["complete_pll_50"]=coverage.pll_assays.ge(99);coverage.to_csv(OUT/"recomputed_model_coverage.csv",index=False)
 new.to_csv(OUT/"recomputed_per_assay_metrics.csv",index=False)
 (OUT/"README.md").write_text("# Updated best-model heatmaps\n\nThree aligned panels compare original ProteinGym-derived, conditional-MM, and PLL fitted epistasis Spearman correlations. Rows are the 39 best-family representatives; columns are the 50 Tsuboyama assays. Bold rows have complete recomputation for all 50 assays. Gray cells are unavailable or in progress.\n")
 print("complete:",", ".join(sorted(recomputed)));print(coverage.to_string(index=False));print(OUT)
if __name__=="__main__":main()
