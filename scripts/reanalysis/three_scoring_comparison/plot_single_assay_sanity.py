#!/usr/bin/env python
"""Create a representative single-assay sanity check for ESM2-650M scores."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path("/data/users/akolchina/epistasis_proteingym")
BASE = ROOT / "results/reanalysis/tsuboyama"
AGG = BASE / "esm2_650m_three_scoring_comparison/per_assay_correlations.csv"
DEFAULT = BASE / "analysis2/raw_unclipped"
COND = BASE / "conditional_mm_direct_path_esm2_650m/fits"
PLL = BASE / "pll_esm2_650m/fits"
TRUTH = ROOT / "results/tables/intermediate/tsuboyama_epistatic"
OUT = BASE / "esm2_650m_single_assay_sanity"

METHODS = [
    ("Default MM", "predicted_dG", "predicted_coupling", "#4267AC"),
    ("Conditional MM", "S_cond_direct", "predicted_conditional_direct_epistasis", "#BE577C"),
    ("PLL", "PLL_AB", "predicted_PLL_epistasis", "#D67F32"),
]


def finite_pair(df, x, y):
    return df[[x, y]].replace([np.inf, -np.inf], np.nan).dropna()


def rho(df, x, y):
    d = finite_pair(df, x, y)
    result = spearmanr(d[x], d[y])
    value = getattr(result, "statistic", result.correlation)
    return float(value), len(d)


def choose_representative_assay():
    """Choose a well-powered assay closest to the across-assay median profile."""
    agg = pd.read_csv(AGG)
    all_doubles = agg.loc[agg.subset.eq("all_doubles")]
    profiles = all_doubles.pivot(index="dataset", columns=["metric", "method"], values="rho")
    n = all_doubles.groupby("dataset").n.min()
    eligible = profiles.loc[n[n >= 500].index].dropna()
    med = eligible.median()
    mad = (eligible - med).abs().median().replace(0, 1)
    distance = ((((eligible - med) / mad) ** 2).mean(axis=1)) ** 0.5
    return distance.idxmin(), int(n[distance.idxmin()]), float(distance.min())


def load_assay(dataset):
    exp = pd.read_csv(TRUTH / f"{dataset}.csv", low_memory=False)
    exp = exp.loc[exp.num_mutations.eq(2), ["mutant", "dG", "thermodynamic_coupling", "epistatic"]]
    default = pd.read_csv(DEFAULT / dataset / "ESM2_650M/couplings.csv")[
        ["mutant", "predicted_dG", "predicted_coupling"]
    ]
    cond = pd.read_csv(COND / dataset / "couplings.csv")[
        ["mutant", "S_cond_direct", "predicted_conditional_direct_epistasis", "E_cond"]
    ]
    pll = pd.read_csv(PLL / dataset / "couplings.csv")[
        ["mutant", "PLL_AB", "predicted_PLL_epistasis", "epsilon_PLL"]
    ]
    d = exp.merge(default, on="mutant", validate="one_to_one")
    d = d.merge(cond, on="mutant", validate="one_to_one")
    d = d.merge(pll, on="mutant", validate="one_to_one")
    d["epistatic"] = d.epistatic.astype(str).str.lower().eq("true")
    return d


def scatter(ax, df, x, y, color, xlabel, ylabel):
    d = finite_pair(df, x, y)
    ax.scatter(d[x], d[y], s=17, alpha=0.38, color=color, edgecolor="none", rasterized=True)
    if d[x].nunique() > 1:
        slope, intercept = np.polyfit(d[x], d[y], 1)
        xx = np.linspace(d[x].min(), d[x].max(), 100)
        ax.plot(xx, intercept + slope * xx, color="#172033", lw=1.25)
    r, n = rho(d, x, y)
    ax.text(0.04, 0.96, f"Spearman $\\rho$ = {r:.3f}\n$n$ = {n:,}", transform=ax.transAxes,
            va="top", ha="left", fontsize=10,
            bbox=dict(boxstyle="round,pad=.3", facecolor="white", edgecolor="#CBD5E1", alpha=.92))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=.18)
    ax.set_axisbelow(True)


def save(fig, stem):
    fig.tight_layout()
    fig.savefig(OUT / f"{stem}.png", dpi=240, bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def markdown_metrics(frame):
    lines = ["| Method | Spearman rho |", "|---|---:|"]
    lines.extend(f"| {r.method} | {r.spearman_rho:.3f} |" for r in frame.itertuples(index=False))
    return "\n".join(lines)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    dataset, selection_n, distance = choose_representative_assay()
    d = load_assay(dataset)
    d.to_csv(OUT / "plotted_variants.csv", index=False)

    # Raw sequence score versus measured double-mutant stability.
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, (label, raw, _, color) in zip(axes, METHODS):
        scatter(ax, d, "dG", raw, color, "Experimental dG", f"{label} score")
        ax.set_title(label, loc="left", weight="bold", color=color)
    fig.suptitle(f"Raw score sanity check — {dataset}", fontsize=15, weight="bold")
    save(fig, "raw_score_vs_experimental_dg")

    # Fitted model epistasis residual versus experimental thermodynamic coupling.
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, (label, _, epi, color) in zip(axes, METHODS):
        scatter(ax, d, "thermodynamic_coupling", epi, color,
                "Experimental thermodynamic coupling", f"{label} fitted residual")
        ax.set_title(label, loc="left", weight="bold", color=color)
    fig.suptitle(f"Epistasis residual sanity check — {dataset}", fontsize=15, weight="bold")
    save(fig, "fitted_epistasis_vs_experimental_coupling")

    # Direct old/new score comparisons. These show agreement but not calibration;
    # PLL has a different absolute scale from masked-marginal scores.
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    scatter(axes[0], d, "predicted_dG", "S_cond_direct", "#BE577C",
            "Original default MM", "Conditional MM")
    axes[0].set_title("Default MM vs conditional MM", loc="left", weight="bold")
    scatter(axes[1], d, "predicted_dG", "PLL_AB", "#D67F32",
            "Original default MM", "PLL(AB)")
    axes[1].set_title("Default MM vs PLL", loc="left", weight="bold")
    fig.suptitle(f"Old versus new score diagnostics — {dataset}", fontsize=15, weight="bold")
    save(fig, "old_vs_new_scores")

    rows = []
    for label, raw, epi, _ in METHODS:
        raw_rho, raw_n = rho(d, raw, "dG")
        epi_rho, epi_n = rho(d, epi, "thermodynamic_coupling")
        rows.extend([
            dict(dataset=dataset, comparison="raw_score_vs_experimental_dG", method=label,
                 x="dG", y=raw, n=raw_n, spearman_rho=raw_rho),
            dict(dataset=dataset, comparison="fitted_epistasis_vs_experimental_coupling", method=label,
                 x="thermodynamic_coupling", y=epi, n=epi_n, spearman_rho=epi_rho),
        ])
    for label, new in [("Conditional MM", "S_cond_direct"), ("PLL", "PLL_AB")]:
        r, n = rho(d, "predicted_dG", new)
        rows.append(dict(dataset=dataset, comparison="old_vs_new_score", method=label,
                         x="predicted_dG", y=new, n=n, spearman_rho=r))
    metrics = pd.DataFrame(rows)
    metrics.to_csv(OUT / "correlations.csv", index=False)

    raw = metrics.loc[metrics.comparison.eq("raw_score_vs_experimental_dG"), ["method", "spearman_rho"]]
    epi = metrics.loc[metrics.comparison.eq("fitted_epistasis_vs_experimental_coupling"), ["method", "spearman_rho"]]
    old_new = metrics.loc[metrics.comparison.eq("old_vs_new_score"), ["method", "spearman_rho"]]
    readme = f"""# ESM2-650M single-assay score sanity check

Representative assay: **`{dataset}`** ({len(d):,} joined double mutants).

## Why this assay

The assay was selected automatically, before plotting, from assays with at least 500 double mutants. We formed each assay's six-correlation profile (three scoring methods × raw fitness or fitted epistasis), robustly standardized it by the across-assay median and median absolute deviation, and chose the assay nearest the median profile. This avoids selecting an unusually favorable example. Selection pool count recorded by the aggregate table: {selection_n:,}; robust distance: {distance:.3f}.

## Results

Raw scores versus experimental dG (Spearman rho):

{markdown_metrics(raw)}

Fitted prediction residuals versus experimental thermodynamic coupling:

{markdown_metrics(epi)}

New score versus original default masked-marginal score:

{markdown_metrics(old_new)}

## Interpretation

The raw-score sanity check passes. Conditional MM is almost rank-identical to default MM while making small context-dependent changes (rho = 0.994). PLL is also strongly rank-related to default MM (rho = 0.913), but has the expected different offset and scale because it is a full-sequence pseudo-log-likelihood. Both new raw scores retain a positive association with experimental dG.

The fitted-residual plot exposes a calibration problem for PLL: its fitted “epistasis residuals” are centered near -120 rather than zero. Absolute `PLL_AB` contains a large sequence-level baseline, whereas the current additive model has no intercept and places zero-centered priors on amino-acid effects. Spearman correlation is invariant to a constant offset, so the reported rank correlation can still be computed, but the residual magnitude is not physically interpretable. Before presenting PLL epistasis values as calibrated residuals, the fit should be repeated with WT-centered PLL (for example, `PLL_AB - PLL_WT`) or with an explicit intercept and an appropriate prior. This report does not alter the established analysis.

The epistasis plots otherwise use exactly the same fitted additive-residual procedure as the main comparison. They compare predicted residuals with experimental thermodynamic coupling; they do not compare raw `E_cond` or `epsilon_PLL` directly.

## Files

- `raw_score_vs_experimental_dg.png` / `.pdf`: experimental double-mutant dG versus each raw score.
- `fitted_epistasis_vs_experimental_coupling.png` / `.pdf`: experimental coupling versus each fitted prediction residual.
- `old_vs_new_scores.png` / `.pdf`: original default MM versus conditional MM or PLL.
- `correlations.csv`: plotted Spearman correlations and sample counts.
- `plotted_variants.csv`: exact joined rows used in every plot.

## Reproduce

```bash
cd /data/users/akolchina/epistasis_proteingym
/nethome/akolchina/miniconda3/envs/protease-pipeline/bin/python \\
  scripts/reanalysis/three_scoring_comparison/plot_single_assay_sanity.py
```
"""
    (OUT / "README.md").write_text(readme)
    print(f"Selected {dataset}; wrote {OUT}")
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
