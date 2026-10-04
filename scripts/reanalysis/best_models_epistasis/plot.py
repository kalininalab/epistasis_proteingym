#!/usr/bin/env python
"""Plot fitted epistasis correlations for ProteinGym best-model representatives."""
from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import argparse

ROOT = Path("/data/users/akolchina/epistasis_proteingym")
BEST = ROOT / "results/tables/intermediate/models_evaluation/tsuboyama_best_models.csv"
CLASS = ROOT / "results/tables/intermediate/models_evaluation/somermeyer_all_models_scoring_types.csv"
FITS = ROOT / "results/reanalysis/tsuboyama/analysis2/raw_unclipped"
OUT = ROOT / "results/reanalysis/tsuboyama/best_models_fitted_epistasis"
COLORS = {"all_doubles": "#4267AC", "epistatic": "#BE577C"}
TYPE_ORDER = {"sitewise": 0, "joint": 1, "explicit_pairwise": 2, "user_model": 3}

def safe(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)

def short_dataset(value):
    return value.replace("_Tsuboyama_2023_", "\n").replace("_Tsuboyama_2023", "")

def collect():
    best = pd.read_csv(BEST, index_col=0)
    cls = pd.read_csv(CLASS, index_col=0)
    available, excluded, frames = [], [], []
    for model in best.index:
        files = sorted(FITS.glob(f"*/{model}/metrics.csv"))
        if len(files) == 50:
            available.append(model)
            frames.extend(pd.read_csv(f) for f in files)
        else:
            excluded.append({"model": model, "model_family": cls.loc[model, "model_family"], "scoring_type": cls.loc[model, "scoring_type"], "reason": f"existing fitted metrics for {len(files)}/50 assays"})
    d = pd.concat(frames, ignore_index=True)
    d = d[d.model.isin(available)].copy()
    meta = cls.loc[available, ["model_family", "scoring_type"]].rename_axis("model").reset_index()
    d = d.merge(meta, on=["model", "scoring_type"], validate="many_to_one")
    d["subset"] = pd.Categorical(d.subset, ["all_doubles", "epistatic"], ordered=True)
    assert not d.duplicated(["dataset", "model", "subset"]).any()
    assert d.dataset.nunique() == 50 and d.model.nunique() == 39
    return d, pd.DataFrame(excluded), meta

def add_category_spans(ax, ordered_meta):
    start = 0
    types = ordered_meta.scoring_type.tolist()
    for i in range(1, len(types) + 1):
        if i == len(types) or types[i] != types[start]:
            if start:
                ax.axvline(start - .5, color="#CBD5E1", lw=.8)
            ax.text((start + i - 1) / 2, 1.015, types[start].replace("_", " "), transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=8, color="#475569")
            start = i

def dataset_figure(dataset, d, ordered_meta):
    sub = d[d.dataset.eq(dataset)]
    models = ordered_meta.model.tolist(); x = np.arange(len(models)); width = .39
    p = sub.pivot(index="model", columns="subset", values="rho").reindex(models)
    fig, ax = plt.subplots(figsize=(16, 6.3))
    ax.bar(x-width/2, p.all_doubles, width, color=COLORS["all_doubles"], label="All double mutants")
    ax.bar(x+width/2, p.epistatic, width, color=COLORS["epistatic"], label="Epistatic subset")
    ax.axhline(0, color="#64748B", lw=.85); ax.set_ylim(-1.05, 1.05)
    ax.set_xticks(x, models, rotation=63, ha="right", fontsize=8)
    ax.set_ylabel("Spearman ρ: predicted vs experimental epistasis")
    ax.set_title(dataset, loc="left", fontsize=14, weight="bold", pad=26)
    ax.grid(axis="y", alpha=.22); ax.set_axisbelow(True); ax.legend(frameon=False, ncol=2, loc="lower left")
    add_category_spans(ax, ordered_meta); fig.tight_layout(); return fig

def model_figure(model, d, meta_row):
    sub = d[d.model.eq(model)]; datasets = sorted(sub.dataset.unique()); x = np.arange(len(datasets)); width=.39
    p = sub.pivot(index="dataset", columns="subset", values="rho").reindex(datasets)
    fig, ax = plt.subplots(figsize=(19, 6.5))
    ax.bar(x-width/2, p.all_doubles, width, color=COLORS["all_doubles"], label="All double mutants")
    ax.bar(x+width/2, p.epistatic, width, color=COLORS["epistatic"], label="Epistatic subset")
    ax.axhline(0, color="#64748B", lw=.85); ax.set_ylim(-1.05, 1.05)
    ax.set_xticks(x, [short_dataset(v) for v in datasets], rotation=65, ha="right", fontsize=7.5)
    ax.set_ylabel("Spearman ρ: predicted vs experimental epistasis")
    ax.set_title(f"{model}  |  {meta_row.model_family}  |  {meta_row.scoring_type.replace('_',' ')}", loc="left", fontsize=14, weight="bold")
    ax.grid(axis="y", alpha=.22); ax.set_axisbelow(True); ax.legend(frameon=False, ncol=2, loc="lower left")
    fig.tight_layout(); return fig

def heatmap(d, ordered_meta, subset, stem):
    # Use a reviewer-friendly biological ordering rather than the bar-plot
    # ordering: interaction-aware joint models first, then sitewise models,
    # with the explicit-pairwise baseline retained as its own final section.
    heat_order={"joint":0,"sitewise":1,"explicit_pairwise":2,"user_model":3}
    heat_meta=ordered_meta.assign(_heat=ordered_meta.scoring_type.map(heat_order)).sort_values(["_heat","model_family","model"])
    models=heat_meta.model.tolist(); datasets=sorted(d.dataset.unique())
    p=d[d.subset.eq(subset)].pivot(index="model",columns="dataset",values="rho").reindex(index=models,columns=datasets)
    fig,ax=plt.subplots(figsize=(22,12)); im=ax.imshow(p,aspect="auto",cmap="RdBu_r",vmin=-1,vmax=1)
    ax.set_yticks(np.arange(len(models)),models,fontsize=8); ax.set_xticks(np.arange(len(datasets)),[short_dataset(v).replace("\n"," ") for v in datasets],rotation=65,ha="right",fontsize=7)
    types=heat_meta.scoring_type.tolist()
    boundaries=[i for i in range(1,len(types)) if types[i] != types[i-1]]
    for boundary in boundaries:
        ax.axhline(boundary-.5,color="#172033",lw=1.4)
    for tick,kind in zip(ax.get_yticklabels(),types):
        if kind == "joint": tick.set_fontweight("bold")
    ax.set_title(f"Best ProteinGym models: fitted epistasis Spearman ρ ({subset.replace('_',' ')})",loc="left",weight="bold")
    cb=fig.colorbar(im,ax=ax,pad=.01); cb.set_label("Spearman ρ"); fig.tight_layout()
    fig.savefig(OUT/f"{stem}.png",dpi=220,bbox_inches="tight"); fig.savefig(OUT/f"{stem}.pdf",bbox_inches="tight"); plt.close(fig)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--mode",choices=["prepare","dataset","model","finish"],default="prepare"); ap.add_argument("--start",type=int,default=0); ap.add_argument("--end",type=int)
    args=ap.parse_args()
    OUT.mkdir(parents=True,exist_ok=True); (OUT/"by_dataset").mkdir(exist_ok=True); (OUT/"by_model").mkdir(exist_ok=True)
    if args.mode == "finish" and (OUT/"per_assay_model_correlations.csv").exists():
        d=pd.read_csv(OUT/"per_assay_model_correlations.csv")
        excluded=pd.read_csv(OUT/"excluded_best_models.csv")
        meta=d[["model","model_family","scoring_type"]].drop_duplicates()
    else:
        d, excluded, meta = collect(); d.to_csv(OUT/"per_assay_model_correlations.csv",index=False); excluded.to_csv(OUT/"excluded_best_models.csv",index=False)
    ordered_meta=meta.assign(_type=meta.scoring_type.map(TYPE_ORDER)).sort_values(["_type","model_family","model"]).drop(columns="_type")
    if args.mode=="dataset":
        items=sorted(d.dataset.unique())[args.start:args.end]
        for dataset in items:
            fig=dataset_figure(dataset,d,ordered_meta); fig.savefig(OUT/"by_dataset"/f"{safe(dataset)}.png",dpi=180,bbox_inches="tight"); plt.close(fig)
    elif args.mode=="model":
        items=list(ordered_meta.itertuples(index=False))[args.start:args.end]
        for row in items:
            fig=model_figure(row.model,d,row); fig.savefig(OUT/"by_model"/f"{safe(row.model)}.png",dpi=180,bbox_inches="tight"); plt.close(fig)
    elif args.mode=="finish":
        heatmap(d,ordered_meta,"all_doubles","overview_heatmap_all_doubles"); heatmap(d,ordered_meta,"epistatic","overview_heatmap_epistatic")
    summary=d.groupby(["model","model_family","scoring_type","subset"],observed=True).agg(assays=("rho","count"),mean_rho=("rho","mean"),median_rho=("rho","median"),total_variants=("n","sum")).reset_index()
    summary.to_csv(OUT/"model_summary.csv",index=False)
    print(f"Wrote {d.model.nunique()} models x {d.dataset.nunique()} assays; {len(excluded)} exclusions")
    print(excluded.to_string(index=False))

if __name__ == "__main__": main()
