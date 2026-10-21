# ESM2-650M single-assay score sanity check

Representative assay: **`CBPA2_HUMAN_Tsuboyama_2023_1O6X`** (711 joined double mutants).

## Why this assay

The assay was selected automatically, before plotting, from assays with at least 500 double mutants. We formed each assay's six-correlation profile (three scoring methods × raw fitness or fitted epistasis), robustly standardized it by the across-assay median and median absolute deviation, and chose the assay nearest the median profile. This avoids selecting an unusually favorable example. Selection pool count recorded by the aggregate table: 711; robust distance: 0.574.

## Results

Raw scores versus experimental dG (Spearman rho):

| Method | Spearman rho |
|---|---:|
| Default MM | 0.333 |
| Conditional MM | 0.325 |
| PLL | 0.269 |

Fitted prediction residuals versus experimental thermodynamic coupling:

| Method | Spearman rho |
|---|---:|
| Default MM | 0.003 |
| Conditional MM | 0.068 |
| PLL | 0.017 |

New score versus original default masked-marginal score:

| Method | Spearman rho |
|---|---:|
| Conditional MM | 0.994 |
| PLL | 0.913 |

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
/nethome/akolchina/miniconda3/envs/protease-pipeline/bin/python \
  scripts/reanalysis/three_scoring_comparison/plot_single_assay_sanity.py
```
