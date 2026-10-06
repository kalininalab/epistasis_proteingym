#!/usr/bin/env python
"""Combine per-assay default-MM, conditional-MM, and PLL correlations."""
from pathlib import Path
import pandas as pd

ROOT = Path('/data/users/akolchina/epistasis_proteingym')
BASE = ROOT / 'results/reanalysis/tsuboyama'
OUT = BASE / 'multimodel_mm_pll/summary'

rows = []
for path in sorted((BASE / 'multimodel_mm_pll/fits').glob('*/*/metrics.csv')):
    d = pd.read_csv(path)
    for r in d.itertuples(index=False):
        for method, raw, epi in [
            ('default_mm', r.rho_default_fitness, None),
            ('conditional_mm', r.rho_conditional_fitness, r.rho_conditional_fitted_epistasis),
            ('pll', r.rho_PLL_fitness, r.rho_PLL_fitted_epistasis),
        ]:
            rows.append(dict(dataset=r.dataset, model=r.model_key, subset=r.subset,
                             method=method, metric='raw_fitness', n=r.n, rho=raw))
            if epi is not None:
                rows.append(dict(dataset=r.dataset, model=r.model_key, subset=r.subset,
                                 method=method, metric='fitted_epistasis', n=r.n, rho=epi))

# Default fitted epistasis was computed earlier with the identical additive fit.
default = pd.read_csv(BASE / 'best_models_fitted_epistasis/per_assay_model_correlations.csv')
default = default[default.model.isin(['ESM1b', 'ESM1v_ensemble', 'ESM2_150M'])]
for r in default.itertuples(index=False):
    rows.append(dict(dataset=r.dataset, model=r.model, subset=r.subset,
                     method='default_mm', metric='fitted_epistasis', n=r.n, rho=r.rho))

# ESM2-650M was completed in the preceding analysis.
esm650 = pd.read_csv(BASE / 'esm2_650m_three_scoring_comparison/per_assay_correlations.csv')
esm650.insert(1, 'model', 'ESM2_650M')
rows.extend(esm650[['dataset','model','subset','method','metric','n','rho']].to_dict('records'))

per_assay = pd.DataFrame(rows).sort_values(['metric','subset','model','dataset','method'])
expected = 4 * 2 * 3 * 2 * 50
if len(per_assay) != expected:
    raise RuntimeError(f'expected {expected} rows, found {len(per_assay)}')
if per_assay.duplicated(['dataset','model','subset','method','metric']).any():
    raise RuntimeError('duplicate assay/model/subset/method/metric rows')

summary = (per_assay.groupby(['metric','subset','model','method'], as_index=False)
           .agg(assays=('rho','count'), mean_rho=('rho','mean'),
                median_rho=('rho','median'), total_variants=('n','sum')))
OUT.mkdir(parents=True, exist_ok=True)
per_assay.to_csv(OUT / 'per_assay_correlations.csv', index=False)
summary.to_csv(OUT / 'summary.csv', index=False)
print(summary.to_string(index=False))
