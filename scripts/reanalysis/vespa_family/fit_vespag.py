#!/usr/bin/env python
"""Fit the established additive model to VespaG conditional scores."""
import argparse, importlib.util, json, time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path("/data/users/akolchina/epistasis_proteingym")
MODEL_FILE = ROOT / "scripts/reanalysis/conditional_mm/thermodynamic_model.py"

def rho(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y); x, y = x[ok], y[ok]
    return float(spearmanr(x, y)[0]) if len(x) >= 3 and len(set(x)) > 1 and len(set(y)) > 1 else np.nan

def main():
    p = argparse.ArgumentParser(); p.add_argument("--scores", type=Path, required=True); p.add_argument("--output", type=Path, required=True); a = p.parse_args()
    started = time.perf_counter(); a.output.mkdir(parents=True, exist_ok=True)
    if (a.output / "metrics.csv").exists() and (a.output / "couplings.csv").exists(): return
    spec = importlib.util.spec_from_file_location("thermo", MODEL_FILE); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    data = pd.read_csv(a.scores); fit = data.copy(); fit["num_mutations"] = 2; fit["dG"] = fit.VespaG_S_cond
    fit = mod.fill_recon_columns_full(fit, clip_predictions=False)
    results = data.copy(); results["VespaG_fitted_additive_conditional"] = fit.recon_dg.to_numpy(); results["VespaG_predicted_conditional_epistasis"] = fit.thermodynamic_coupling.to_numpy()
    pd.DataFrame(fit.attrs.get("fit_diagnostics", [])).to_csv(a.output / "fit_diagnostics.csv", index=False); results.to_csv(a.output / "couplings.csv", index=False)
    rows = []
    for name, sub in (("all_doubles", results), ("epistatic", results[results.epistatic.astype(str).str.lower().eq("true")])):
        rows.append(dict(dataset=str(results.dataset.iloc[0]), model_key="VespaG", subset=name, n=len(sub),
                         rho_default_fitness=rho(sub.VespaG_default, sub.experimental_dG), rho_conditional_fitness=rho(sub.VespaG_S_cond, sub.experimental_dG),
                         rho_conditional_fitted_epistasis=rho(sub.VespaG_predicted_conditional_epistasis, sub.experimental_coupling),
                         rho_direct_E_cond=rho(sub.VespaG_E_cond, sub.experimental_coupling),
                         max_abs_default_reconstruction_error=float(np.nanmax(np.abs(sub.VespaG_default - sub.VespaG_recomputed_default)))))
    pd.DataFrame(rows).to_csv(a.output / "metrics.csv", index=False)
    (a.output / "runtime.json").write_text(json.dumps({"fit_seconds": time.perf_counter()-started, "n_double_mutants": len(results), "n_position_pairs": int(results.pair_name.nunique())}, indent=2)+"\n")

if __name__ == "__main__": main()
