import argparse
from pathlib import Path
import pandas as pd
import numpy as np

# --- heavy deps live in the external env ---
from jax import random
import jax.numpy
from numpyro.infer import MCMC, NUTS
from numpyro.infer import initialization
import numpyro
import numpyro.distributions as dist

aas = 'QENHDRKTSAGMCLVIWYFP'  # amino acids

def additive_model(df_dg, aa_len):
    first = numpyro.sample("first_aa_effect", dist.Normal(np.resize(0.1, (aa_len)), 3))
    second = numpyro.sample("second_aa_effect", dist.Normal(np.resize(0.1, (aa_len)), 3))

    first_idx = list(df_dg['aa1'])
    second_idx = list(df_dg['aa2'])
    obs = jax.numpy.array(df_dg['dG'])

    pred = jax.numpy.array([first[i] for i in first_idx]) + \
           jax.numpy.array([second[j] for j in second_idx])
    pred = jax.numpy.clip(pred, -1, 5)

    sigma = numpyro.sample("sigma", dist.Exponential(1))
    numpyro.sample(
        "obs_dg",
        dist.TransformedDistribution(
            dist.Normal(0, 1),
            dist.transforms.AffineTransform(pred, sigma)
        ),
        obs=obs
    )

def fill_recon_columns_full(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a copy of df with two new columns:
      - 'recon_dg'
      - 'thermodynamic_coupling' = dG - recon_dg
    Only double mutants (num_mutations==2) are modeled; others are left NaN in these columns.
    """
    out = df.copy()

    # add empty columns so all rows are kept
    out['recon_dg'] = np.nan
    out['thermodynamic_coupling'] = np.nan

    # work only on doubles
    mask = (out['num_mutations'] == 2)
    if not mask.any():
        return out  # nothing to compute, but keep full table with NaNs

    doubles = out.loc[mask].copy()

    # ensure required columns exist
    required = {'mutant', 'dG', 'pair_name'}
    missing = [c for c in required if c not in doubles.columns]
    if missing:
        raise ValueError(f"Missing required columns for reconstruction: {missing}")

    # derive aa indices
    doubles['aa1'] = [aas.index(m.split(':')[0][-1]) for m in doubles['mutant']]
    doubles['aa2'] = [aas.index(m.split(':')[1][-1]) for m in doubles['mutant']]

    # run model per pair_name (independent fits)
    recon_vals = pd.Series(index=doubles.index, dtype=float)

    for pname, sub in doubles.groupby('pair_name'):
        if len(sub) == 0:
            continue

        key = random.PRNGKey(1)
        _, key2 = random.split(key)
        kernel = NUTS(additive_model, init_strategy=initialization.init_to_feasible())
        mcmc = MCMC(kernel, num_warmup=100, num_samples=50, num_chains=1)
        mcmc.run(key2, df_dg=sub, aa_len=20)
        samples = mcmc.get_samples()

        first = np.median(samples['first_aa_effect'], axis=0)
        second = np.median(samples['second_aa_effect'], axis=0)

        recon = [first[a] + second[b] for a, b in zip(sub['aa1'], sub['aa2'])]
        recon_vals.loc[sub.index] = recon

    # write results back into full table
    out.loc[recon_vals.index, 'recon_dg'] = recon_vals
    out.loc[recon_vals.index, 'thermodynamic_coupling'] = out.loc[recon_vals.index, 'dG'] - recon_vals

    # optional: drop helper cols if you don't want them in output
    # out = out.drop(columns=['aa1','aa2'], errors='ignore')

    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in",  dest="inp",  required=True, help="Folder with input CSVs")
    ap.add_argument("--out", dest="outp", required=True, help="Folder to write processed CSVs")
    args = ap.parse_args()

    in_dir  = Path(args.inp)
    out_dir = Path(args.outp)
    out_dir.mkdir(parents=True, exist_ok=True)

    for csv in sorted(in_dir.glob("*.csv")):
        df = pd.read_csv(csv, low_memory=False)
        df_full = fill_recon_columns_full(df)
        df_full.to_csv(out_dir / csv.name, index=False)
        print(f"[OK] wrote {out_dir / csv.name} (rows kept: {len(df_full)})")

if __name__ == "__main__":
    main()
