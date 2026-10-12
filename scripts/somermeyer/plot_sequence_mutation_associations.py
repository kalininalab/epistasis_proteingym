#!/usr/bin/env python
"""Create a sequence-level cgreGFP association map and an executive report."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd

AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")


def reconstruct_wild_type(data: pd.DataFrame) -> str:
    sequence = list(data.iloc[0].mutated_sequence)
    for mutation in data.iloc[0].mutant.split(":"):
        sequence[int(mutation[1:-1]) - 1] = mutation[0]
    wild_type = "".join(sequence)
    for mutant, mutated_sequence in data[["mutant", "mutated_sequence"]].itertuples(index=False):
        if len(mutated_sequence) != len(wild_type):
            raise ValueError("Sequences do not all have the same length")
        for mutation in mutant.split(":"):
            position = int(mutation[1:-1])
            if wild_type[position - 1] != mutation[0] or mutated_sequence[position - 1] != mutation[-1]:
                raise ValueError(f"Mutation indexing failed for {mutation}")
    return wild_type


def create_position_table(associations: pd.DataFrame, wild_type: str, min_frequency: int) -> pd.DataFrame:
    table = associations.copy()
    table["position"] = table.mutation.str[1:-1].astype(int)
    table["mutant_amino_acid"] = table.mutation.str[-1]
    tested = table[table.frequency >= min_frequency].copy()
    rows = []
    for position in range(1, len(wild_type) + 1):
        group = tested[tested.position.eq(position)]
        if group.empty:
            rows.append({"position": position, "wild_type_amino_acid": wild_type[position - 1],
                         "tested_substitutions": 0, "strongest_mutation": np.nan,
                         "strongest_spearman_rho": np.nan, "strongest_spearman_q": np.nan,
                         "minimum_spearman_rho": np.nan, "maximum_spearman_rho": np.nan})
            continue
        strongest = group.loc[group.spearman_rho.abs().idxmax()]
        rows.append({"position": position, "wild_type_amino_acid": wild_type[position - 1],
                     "tested_substitutions": len(group), "strongest_mutation": strongest.mutation,
                     "strongest_spearman_rho": strongest.spearman_rho,
                     "strongest_spearman_q": strongest.spearman_q,
                     "minimum_spearman_rho": group.spearman_rho.min(),
                     "maximum_spearman_rho": group.spearman_rho.max()})
    return pd.DataFrame(rows)


def draw_heatmap(associations: pd.DataFrame, positions: pd.DataFrame, wild_type: str,
                 min_frequency: int, output: Path) -> None:
    associations = associations[associations.frequency >= min_frequency].copy()
    associations["position"] = associations.mutation.str[1:-1].astype(int)
    associations["mutant_amino_acid"] = associations.mutation.str[-1]
    matrix = associations.pivot_table(index="mutant_amino_acid", columns="position",
                                      values="spearman_rho", aggfunc="first").reindex(AMINO_ACIDS)
    matrix = matrix.reindex(columns=range(1, len(wild_type) + 1))
    blocks = [(1, 80), (81, 160), (161, len(wild_type))]
    fig = plt.figure(figsize=(24, 16))
    gs = fig.add_gridspec(6, 1, height_ratios=[4, 1, 4, 1, 4, 1], hspace=.30)
    cmap = plt.get_cmap("RdBu_r").copy(); cmap.set_bad("#E5E7EB")
    norm = TwoSlopeNorm(vmin=-.125, vcenter=0, vmax=.125)
    image = None
    for block_index, (start, end) in enumerate(blocks):
        ax = fig.add_subplot(gs[block_index * 2])
        values = matrix.loc[:, start:end].to_numpy(float)
        image = ax.imshow(np.ma.masked_invalid(values), aspect="auto", interpolation="nearest",
                          cmap=cmap, norm=norm)
        ax.set_yticks(range(len(AMINO_ACIDS)), AMINO_ACIDS, fontsize=9)
        ticks = np.arange(start, end + 1, 5)
        ax.set_xticks(ticks - start, [f"{p}\n{wild_type[p-1]}" for p in ticks], fontsize=8)
        ax.set_ylabel("Substituted amino acid")
        ax.set_title(f"Positions {start}–{end}", loc="left", weight="bold")
        track = fig.add_subplot(gs[block_index * 2 + 1], sharex=ax)
        block = positions.set_index("position").loc[start:end]
        x = np.arange(end - start + 1)
        values = block.strongest_spearman_rho.to_numpy(float)
        colors = np.where(values >= 0, "#D55E00", "#3B6FB6")
        track.bar(x, np.nan_to_num(values), color=colors, width=.9)
        track.axhline(0, color="#475569", lw=.8)
        track.set_ylim(-.13, .05); track.set_ylabel("Strongest\nSpearman ρ", fontsize=9)
        track.set_xticks(ticks - start, [f"{p} {wild_type[p-1]}" for p in ticks], fontsize=8)
        for p, row in block.iterrows():
            if pd.notna(row.strongest_spearman_rho) and abs(row.strongest_spearman_rho) >= .07:
                track.text(p - start, row.strongest_spearman_rho - .006,
                           str(row.strongest_mutation), rotation=90, ha="center", va="top", fontsize=7)
    cbar = fig.colorbar(image, ax=fig.axes, location="right", fraction=.018, pad=.012)
    cbar.set_label("Mutation–fluorescence Spearman ρ")
    fig.suptitle("cgreGFP mutation–fluorescence association map", fontsize=22, weight="bold", y=.998)
    fig.text(.5, .956,
             f"Each cell is one substitution observed in at least {min_frequency} variants; gray = not sufficiently observed. "
             "Bottom tracks show the strongest association at each sequence position.",
             ha="center", fontsize=11, color="#475569")
    fig.text(.5, .006, "Blue: lower fluorescence · Orange/red: higher fluorescence · Position numbering is 1-based and validated against the 235-aa cgreGFP sequence.",
             ha="center", fontsize=10, color="#475569")
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_report(out: Path, summary: dict, associations: pd.DataFrame, positions: pd.DataFrame,
                 strong_pairs: pd.DataFrame, min_frequency: int) -> None:
    frequent = associations[associations.frequency >= min_frequency]
    negative = frequent.nsmallest(6, "spearman_rho")
    positive = frequent.nlargest(6, "spearman_rho")
    def mutation_list(frame):
        return ", ".join(f"**{r.mutation}** (ρ={r.spearman_rho:.3f})" for r in frame.itertuples())
    perf = summary["performance"]
    report = f"""# Executive report: mutation associations with cgreGFP fluorescence

## Executive summary

We analyzed **{summary['n_variants']:,} cgreGFP variants** containing **{summary['n_mutations']:,} distinct amino-acid substitutions**. Mutations were tested individually for association with fluorescence, mutation pairs were tested for co-occurrence, and Elastic Net models measured how well the complete mutation pattern predicts fluorescence.

The strongest signals are predominantly **loss-of-fluorescence substitutions**. Mutations around positions **65–69, 97, 107, 168, 186, and 231** repeatedly show negative associations. Positive associations exist but have smaller univariate effect sizes. The models predict held-out variants well, showing that the combined mutation pattern contains substantial fluorescence information.

![Sequence-level mutation association map](cgre_sequence_mutation_heatmap.png)

## What we did

1. Converted each variant into a binary vector indicating which substitutions are present.
2. Log10-transformed fluorescence and fitted a two-component Gaussian mixture. The intersection between the two fitted modes, **log10 {summary['threshold_log10']:.3f}** or **{summary['threshold_raw']:,.0f} raw units**, defines active versus inactive fluorescence. This agrees with **{100*summary['agreement_with_supplied_bin']:.1f}%** of the existing ProteinGym binary labels.
3. For each mutation, calculated an active/inactive odds ratio and Fisher exact test, plus point-biserial and Spearman correlations with continuous fluorescence. Each test family was FDR-corrected with Benjamini–Hochberg.
4. For mutation pairs observed often enough for stable analysis, calculated phi correlation, Fisher exact p-values, and global FDR q-values.
5. Evaluated Elastic Net logistic and linear models with nested **5-fold outer / 3-fold inner cross-validation**. All reported performance is based on out-of-fold predictions.

## Formulas and computation

### 1. Mutation indicator matrix

For every variant \(v\) and specific substitution \(m\), we constructed:

```text
X[v,m] = 1  if mutation m is present in variant v
X[v,m] = 0  otherwise
```

Thus, `A42V` is a different feature from `A42T`. The matrix contains one row per variant and one column per observed substitution. Mutation labels and sequence positions were checked against the reconstructed 235-aa wild-type cgreGFP sequence using **1-based indexing**.

The continuous response was:

```text
y[v] = log10(raw fluorescence[v])
```

Spearman correlation is unchanged if raw fluorescence is used because log10 is strictly monotonic and therefore preserves ranks.

### 2. Active/inactive threshold

We fitted two Gaussian components to log10 fluorescence. One component represents the low-fluorescence mode and the other the high-fluorescence mode. The automatic threshold is the value \(t\) where their posterior probabilities are equal:

```text
P(low component | y=t) = P(high component | y=t)
active[v] = 1 if y[v] >= t, otherwise 0
```

For this assay, \(t={summary['threshold_log10']:.3f}\), corresponding to {summary['threshold_raw']:,.0f} raw fluorescence units. This threshold follows the observed bimodal distribution rather than copying the supplied ProteinGym binary label.

### 3. Mutation association with binary activity

For each mutation, variants were arranged in a 2×2 contingency table:

| | Active | Inactive |
|---|---:|---:|
| Mutation present | \(a\) | \(c\) |
| Mutation absent | \(b\) | \(d\) |

The odds ratio was:

```text
OR = (a × d) / (b × c)
```

An odds ratio above 1 means the mutation is enriched among active variants; below 1 means it is enriched among inactive variants. Fisher's exact test supplied an exact p-value for the same table. Infinite or zero odds ratios can occur when one cell is zero.

### 4. Mutation association with continuous fluorescence

The point-biserial correlation compares mean log10 fluorescence with and without a mutation:

```text
r_pb = (mean(y | X=1) - mean(y | X=0)) / sd(y)
       × sqrt(n1 × n0 / n²)
```

This is Pearson correlation between a binary mutation indicator and continuous log10 fluorescence.

Spearman correlation was computed independently for every mutation:

```text
rho_m = Pearson correlation(X[:,m], rank(y))
```

Equivalently, it asks whether variants containing mutation \(m\) tend to occupy higher or lower fluorescence ranks than variants without it. Positive rho means higher fluorescence ranks and negative rho means lower ranks. This is an **unadjusted association across combinatorial variants**: other mutations remain present and can confound the result.

Point-biserial and Spearman p-values were calculated from their correlation test statistics. Fisher, point-biserial, and Spearman p-values were each corrected as separate test families using Benjamini–Hochberg false-discovery-rate control:

```text
ordered q(i) = min over j>=i of [p(j) × M / j]
```

where \(M\) is the number of mutations tested. A q-value estimates significance after accounting for the large number of simultaneous mutation tests.

### 5. Mutation–mutation association

For two mutations A and B, the phi coefficient is Pearson correlation between their two binary indicator columns. Using counts \(n11\) (both present), \(n10\) (only A), \(n01\) (only B), and \(n00\) (neither):

```text
phi = (n11 × n00 - n10 × n01)
      / sqrt[(n11+n10)(n01+n00)(n11+n01)(n10+n00)]
```

Phi ranges from −1 to +1. Positive values indicate co-occurrence, values near zero indicate little association, and negative values indicate mutual avoidance. Pairwise Fisher exact tests and Benjamini–Hochberg q-values were also computed. Only mutations observed in at least {summary['pairwise_min_frequency']} variants entered this analysis.

Phi measures **co-occurrence, not fluorescence and not epistasis**. Strong phi can result from library construction, shared ancestry, or mutations introduced as a block.

### 6. Multivariate Elastic Net models

Mutations observed in at least {summary['regression_min_frequency']} variants were included as predictors. Features were standardized within the modeling pipeline. Elastic Net combines L1 sparsity with L2 shrinkage:

```text
penalty(beta) = lambda × [alpha × sum(|beta|)
                         + (1-alpha)/2 × sum(beta²)]
```

The classification model minimized logistic loss plus this penalty and predicted active/inactive fluorescence. The regression model minimized squared error plus the same type of penalty and predicted log10 fluorescence.

Evaluation used nested cross-validation:

1. Split the data into five outer folds.
2. Within each outer training set, use three-fold cross-validation to choose regularization strength and L1/L2 mixing.
3. Fit the selected model on that outer training set.
4. Predict the untouched outer test fold.
5. Pool the five out-of-fold predictions and calculate ROC-AUC, R², and Spearman correlation.

This prevents the test fold from influencing hyperparameter selection. Final coefficient rankings come from a separately tuned descriptive model fitted to the full dataset; predictive performance comes only from out-of-fold predictions.

## Main findings

### Individual mutations

Among the **{len(frequent):,} substitutions observed in at least {min_frequency} variants**, **{sum(frequent.spearman_q < .05):,}** have a Spearman FDR q-value below 0.05 and **{sum(frequent.fisher_q < .05):,}** have a Fisher FDR q-value below 0.05. Statistical significance is common because the dataset is large; correlation magnitude and recurrence across substitutions at a position are more useful for prioritization.

Strongest negative associations: {mutation_list(negative)}.

Strongest positive associations: {mutation_list(positive)}.

The strongest positive correlation is only **ρ={positive.iloc[0].spearman_rho:.3f}**, compared with **ρ={negative.iloc[0].spearman_rho:.3f}** for the strongest negative association. This library therefore provides clearer evidence for fluorescence-disrupting mutations than for individually beneficial mutations.

### Mutation–mutation associations

We tested **{summary['pairwise_tests']:,} mutation pairs** among {summary['pairwise_mutations']:,} substitutions observed at least {summary['pairwise_min_frequency']} times. Only **{summary['strong_pairs']} pairs** meet the prespecified strong-pair definition, |phi| ≥ 0.5 and Fisher q < 0.05. They connect **T233R, Y234T, and R235A** (phi 0.83–0.96), indicating a tightly linked C-terminal mutation block. This likely reflects library design or jointly introduced variants and should not automatically be interpreted as biochemical interaction.

![Strongest mutation-pair correlations](mutation_mutation_phi_heatmap_strongest_pairs.png)

### Multivariate prediction

- Active/inactive Elastic Net ROC-AUC: **{perf['classification_oof_roc_auc']:.3f}**
- Continuous log10-fluorescence Elastic Net R²: **{perf['regression_oof_r2']:.3f}**
- Continuous prediction Spearman ρ: **{perf['regression_oof_spearman']:.3f}**

![Elastic Net out-of-fold predictions and coefficients](elastic_net_regression.png)

These values show strong predictive signal, but they do not prove that every selected coefficient is causal. Linked mutations, unequal library sampling, and epistasis can redistribute coefficients between correlated substitutions.

## Recommended interpretation

- Use the sequence map to prioritize positions with repeated negative signals across several substitutions.
- Treat the positive mutations as candidates for experimental validation rather than confirmed brightness improvements.
- Validate candidate substitutions individually or in controlled backgrounds before making mechanistic claims.
- For publication, describe these results as **associations in a combinatorial library**, not single-mutation causal effects.

## Files

- `cgre_sequence_mutation_heatmap.png`: sequence-level substitution map.
- `position_association_summary.csv`: strongest association and tested substitutions at every sequence position.
- `mutation_fluorescence_associations.csv`: complete univariate mutation results.
- `mutation_mutation_associations.csv.gz`: complete pairwise results.
- `elastic_net_mutation_coefficients.csv` and `elastic_net_oof_predictions.csv`: multivariate results.
- `summary.json`: machine-readable settings and performance.

The reproducible analysis commands are documented in `README.md`.
"""
    (out / "BOSS_REPORT.md").write_text(report)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--min-frequency", type=int, default=10)
    args = parser.parse_args()
    data = pd.read_csv(args.data, usecols=["mutant", "mutated_sequence"])
    associations = pd.read_csv(args.results / "mutation_fluorescence_associations.csv")
    strong_pairs = pd.read_csv(args.results / "strong_mutation_pairs.csv")
    summary = json.loads((args.results / "summary.json").read_text())
    wild_type = reconstruct_wild_type(data)
    positions = create_position_table(associations, wild_type, args.min_frequency)
    positions.to_csv(args.results / "position_association_summary.csv", index=False)
    (args.results / "cgre_wild_type.fasta").write_text(f">D7PM05_CLYGR_reconstructed_and_validated\n{wild_type}\n")
    draw_heatmap(associations, positions, wild_type, args.min_frequency,
                 args.results / "cgre_sequence_mutation_heatmap.png")
    write_report(args.results, summary, associations, positions, strong_pairs, args.min_frequency)


if __name__ == "__main__":
    main()
