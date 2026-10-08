#!/usr/bin/env python
"""Fit the established additive model to VESPA-family conditional scores."""
import argparse
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path("/data/users/akolchina/epistasis_proteingym")
MODEL_FILE = ROOT / "scripts/reanalysis/conditional_mm/thermodynamic_model.py"


def rho(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    return float(spearmanr(x, y)[0]) if len(x) >= 3 and len(set(x)) > 1 and len(set(y)) > 1 else np.nan


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "metrics.csv").exists() and (args.output / "couplings.csv").exists():
        print(f"complete: {args.output}", flush=True)
        return
    spec = importlib.util.spec_from_file_location("thermo", MODEL_FILE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    data = pd.read_csv(args.scores)
    results = data.copy()
    diagnostics = []
    for model in ("VESPA", "VESPAl"):
        fit = data.copy()
        fit["num_mutations"] = 2
        fit["dG"] = fit[f"{model}_S_cond"]
        fit = mod.fill_recon_columns_full(fit, clip_predictions=False)
        results[f"{model}_fitted_additive_conditional"] = fit.recon_dg.to_numpy()
        results[f"{model}_predicted_conditional_epistasis"] = fit.thermodynamic_coupling.to_numpy()
        diag = pd.DataFrame(fit.attrs.get("fit_diagnostics", []))
        diag["model"] = model
        diagnostics.append(diag)
    pd.concat(diagnostics, ignore_index=True).to_csv(args.output / "fit_diagnostics.csv", index=False)
    results.to_csv(args.output / "couplings.csv", index=False)
    rows = []
    for subset_name, subset in (("all_doubles", results),
                                ("epistatic", results[results.epistatic.astype(str).str.lower().eq("true")])):
        for model in ("VESPA", "VESPAl"):
            rows.append(dict(
                dataset=str(results.dataset.iloc[0]), model_key=model, subset=subset_name, n=len(subset),
                rho_default_fitness=rho(subset[f"{model}_default"], subset.experimental_dG),
                rho_conditional_fitness=rho(subset[f"{model}_S_cond"], subset.experimental_dG),
                rho_conditional_fitted_epistasis=rho(subset[f"{model}_predicted_conditional_epistasis"], subset.experimental_coupling),
                rho_direct_E_cond=rho(subset[f"{model}_E_cond"], subset.experimental_coupling),
                max_abs_default_reconstruction_error=float(np.nanmax(
                    np.abs(subset[f"{model}_default"] - subset[f"{model}_recomputed_default"])
                )),
            ))
    pd.DataFrame(rows).to_csv(args.output / "metrics.csv", index=False)
    runtime = {"fit_seconds": time.perf_counter() - started, "n_double_mutants": len(results),
               "n_position_pairs": int(results.pair_name.nunique())}
    (args.output / "runtime.json").write_text(json.dumps(runtime, indent=2) + "\n")
    print(json.dumps(runtime, indent=2), flush=True)


if __name__ == "__main__":
    main()
