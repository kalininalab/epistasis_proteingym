#!/usr/bin/env python
"""Compare default MM, conditional MM, and PLL across Tsuboyama assays."""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/data/users/akolchina/epistasis_proteingym")
DEFAULT = ROOT / "results/reanalysis/tsuboyama/analysis2/raw_unclipped"
COND = ROOT / "results/reanalysis/tsuboyama/conditional_mm_direct_path_esm2_650m/fits"
PLL = ROOT / "results/reanalysis/tsuboyama/pll_esm2_650m/fits"
TRUTH = ROOT / "results/tables/intermediate/tsuboyama_epistatic"
OUT = ROOT / "results/reanalysis/tsuboyama/esm2_650m_three_scoring_comparison"
METHODS = [("default_mm", "Default MM", "#4267AC"), ("conditional_mm", "Conditional MM", "#BE577C"), ("pll", "PLL", "#D67F32")]

def rho_n(d, x, y):
    v = d[[x, y]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(v) < 3 or v[x].nunique() < 2 or v[y].nunique() < 2:
        return np.nan, len(v)
    return float(spearmanr(v[x], v[y])[0]), len(v)

def short_name(s):
    return s.replace("_Tsuboyama_2023_", "\n").replace("_Tsuboyama_2023", "")

def collect():
    rows = []
    datasets = sorted(p.parent.name for p in COND.glob("*/couplings.csv"))
    if len(datasets) != 50:
        raise RuntimeError(f"Expected 50 assays, found {len(datasets)}")
    for ds in datasets:
        base = pd.read_csv(DEFAULT / ds / "ESM2_650M/couplings.csv")[["mutant", "predicted_dG", "predicted_coupling"]]
        con = pd.read_csv(COND / ds / "couplings.csv")[["mutant", "S_cond_direct", "predicted_conditional_direct_epistasis"]]
        pll = pd.read_csv(PLL / ds / "couplings.csv")[["mutant", "PLL_AB", "predicted_PLL_epistasis"]]
        exp = pd.read_csv(TRUTH / f"{ds}.csv", low_memory=False)
        exp = exp.loc[exp.num_mutations.eq(2), ["mutant", "dG", "thermodynamic_coupling", "epistatic"]]
        d = exp.merge(base, on="mutant", validate="one_to_one").merge(con, on="mutant", validate="one_to_one").merge(pll, on="mutant", validate="one_to_one")
        d["epistatic"] = d.epistatic.astype(str).str.lower().eq("true")
        columns = {"default_mm": ("predicted_dG", "predicted_coupling"), "conditional_mm": ("S_cond_direct", "predicted_conditional_direct_epistasis"), "pll": ("PLL_AB", "predicted_PLL_epistasis")}
        for subset in ["all_doubles", "epistatic"]:
            sub = d if subset == "all_doubles" else d.loc[d.epistatic]
            for method, (raw_col, epi_col) in columns.items():
                raw_rho, raw_n = rho_n(sub, raw_col, "dG")
                epi_rho, epi_n = rho_n(sub, epi_col, "thermodynamic_coupling")
                rows += [dict(dataset=ds, subset=subset, method=method, metric="raw_fitness", n=raw_n, rho=raw_rho), dict(dataset=ds, subset=subset, method=method, metric="fitted_epistasis", n=epi_n, rho=epi_rho)]
    return pd.DataFrame(rows)

def draw(data, metric, title, ylabel, stem):
    datasets = sorted(data.dataset.unique()); x = np.arange(len(datasets)); width = .25
    fig, axes = plt.subplots(2, 1, figsize=(20, 11.5), sharex=True, sharey=True)
    for ax, subset in zip(axes, ["all_doubles", "epistatic"]):
        sub = data[(data.metric == metric) & (data.subset == subset)]
        for offset, (method, label, color) in zip([-1, 0, 1], METHODS):
            values = sub[sub.method == method].set_index("dataset").reindex(datasets).rho
            ax.bar(x + offset * width, values, width, color=color, label=label)
        ax.axhline(0, color="#64748B", lw=.85); ax.grid(axis="y", alpha=.22); ax.set_axisbelow(True)
        ax.set_ylabel(ylabel); ax.set_title("All double mutants" if subset == "all_doubles" else "Predefined epistatic subset", loc="left", weight="bold")
    axes[0].legend(frameon=False, ncol=3, loc="upper right")
    axes[1].set_xticks(x, [short_name(v) for v in datasets], rotation=65, ha="right", fontsize=8)
    fig.suptitle(title, fontsize=16, weight="bold", y=.995); fig.tight_layout()
    fig.savefig(OUT / f"{stem}.png", dpi=240, bbox_inches="tight"); fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight"); plt.close(fig)

def main():
    OUT.mkdir(parents=True, exist_ok=True); d = collect(); d.to_csv(OUT / "per_assay_correlations.csv", index=False)
    draw(d, "raw_fitness", "Raw ESM2-650M sequence scores vs experimental double-mutant stability", "Spearman ρ", "raw_fitness_three_scores")
    draw(d, "fitted_epistasis", "Fitted ESM2-650M residuals vs experimental thermodynamic coupling", "Spearman ρ", "fitted_epistasis_three_scores")
    summary = d.groupby(["metric", "subset", "method"]).agg(assays=("rho", "count"), mean_rho=("rho", "mean"), median_rho=("rho", "median"), variants=("n", "sum")).reset_index()
    summary.to_csv(OUT / "summary.csv", index=False); print(summary.to_string(index=False))

if __name__ == "__main__": main()
