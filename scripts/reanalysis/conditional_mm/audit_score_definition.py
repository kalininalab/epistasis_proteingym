#!/usr/bin/env python
"""Audit direct path-average S_cond against the formerly used stored-default construction."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT=Path('/data/users/akolchina/epistasis_proteingym')
SOURCE=ROOT/'results/reanalysis/tsuboyama/conditional_mm_esm2_650m/scores'
OUT=ROOT/'results/reanalysis/tsuboyama/conditional_mm_score_definition_audit'

def main():
    OUT.mkdir(parents=True,exist_ok=True); (OUT/'tables').mkdir(exist_ok=True)
    rows=[]; detail=[]
    for path in sorted(SOURCE.glob('*.csv')):
        d=pd.read_csv(path)
        direct=.5*((d.delta_A_WT+d.delta_B_A)+(d.delta_B_WT+d.delta_A_B))
        current=d.default_MM_fitness+d.E_cond
        algebraic=(d.delta_A_WT+d.delta_B_WT)+d.E_cond
        delta=direct-current
        detail.append(pd.DataFrame({'dataset':path.stem,'mutant':d.mutant,'S_cond_direct':direct,
          'S_cond_current_stored_default_plus_E_cond':current,'difference_direct_minus_current':delta}))
        rows.append(dict(dataset=path.stem,n=len(d),max_absolute_difference=float(delta.abs().max()),
          mean_absolute_difference=float(delta.abs().mean()),
          direct_vs_same_components_identity_max_error=float((direct-algebraic).abs().max())))
    detail=pd.concat(detail,ignore_index=True); per=pd.DataFrame(rows)
    detail.to_csv(OUT/'tables/per_variant_score_comparison.csv',index=False)
    per.to_csv(OUT/'tables/per_assay_score_comparison.csv',index=False)
    summary={'n_assays':len(per),'n_double_mutants':len(detail),
      'max_absolute_difference_direct_vs_current':float(detail.difference_direct_minus_current.abs().max()),
      'mean_absolute_difference_direct_vs_current':float(detail.difference_direct_minus_current.abs().mean()),
      'rmse_direct_vs_current':float(np.sqrt(np.mean(detail.difference_direct_minus_current**2))),
      'max_identity_error_direct_vs_same_components_default_plus_Econd':float(per.direct_vs_same_components_identity_max_error.max())}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    report=f'''# Conditional score-definition audit

No ESM inference was run. This audit uses the four saved conditional masked-marginal components for all {summary['n_double_mutants']:,} double mutants in {summary['n_assays']} assays.

The ProteinGym ESM implementation masks each WT position independently, then `label_row` sums `log p(mut) - log p(WT)` over the substitutions. Its exact double-mutant definition is therefore:

`S_default_AB = ΔA_WT + ΔB_WT`.

The conditional score is now defined directly as:

`S_cond_direct = 0.5 * [(ΔA_WT + ΔB_A) + (ΔB_WT + ΔA_B)]`.

Using components from the same inference, this equals `(ΔA_WT + ΔB_WT) + E_cond` to floating-point precision: maximum absolute identity error **{summary['max_identity_error_direct_vs_same_components_default_plus_Econd']:.3g}**.

The formerly fitted construction mixed the stored ProteinGym score with newly computed Hugging Face FP16 components: `S_default_stored + E_cond`. Compared with `S_cond_direct`, its maximum absolute difference is **{summary['max_absolute_difference_direct_vs_current']:.9f}** log-score units, mean absolute difference **{summary['mean_absolute_difference_direct_vs_current']:.9f}**, and RMSE **{summary['rmse_direct_vs_current']:.9f}**.

The formulas are compatible, but the implementations are not bit-identical. Future fits use `S_cond_direct`; existing fitted outputs remain untouched and are explicitly treated as the former construction.
'''
    (OUT/'REPORT.md').write_text(report)
    print(json.dumps(summary,indent=2)); print(OUT/'REPORT.md')
if __name__=='__main__': main()
