#!/usr/bin/env python
"""Mutation association analysis for combinatorial fluorescence landscapes."""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import sparse, stats
from sklearn.linear_model import ElasticNet, SGDClassifier
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

MUTATION_RE = re.compile(r"^[A-Z*][0-9]+[A-Z*]$")


def bh_fdr(pvalues: np.ndarray) -> np.ndarray:
    p = np.asarray(pvalues, dtype=float)
    q = np.full(p.shape, np.nan)
    ok = np.isfinite(p)
    vals = p[ok]
    order = np.argsort(vals)
    ranked = vals[order]
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1].clip(0, 1)
    restored = np.empty_like(adjusted)
    restored[order] = adjusted
    q[ok] = restored
    return q


def parse_mutations(series: pd.Series) -> tuple[list[list[str]], list[str]]:
    rows, invalid = [], set()
    for value in series.fillna("").astype(str):
        muts = [] if value in ("", "WT", "wildtype") else value.split(":")
        for mutation in muts:
            if not MUTATION_RE.fullmatch(mutation):
                invalid.add(mutation)
        rows.append(muts)
    if invalid:
        raise ValueError(f"Invalid mutation labels: {sorted(invalid)[:20]}")
    names = sorted({m for row in rows for m in row}, key=lambda x: (int(x[1:-1]), x[0], x[-1]))
    return rows, names


def indicator_matrix(rows: list[list[str]], names: list[str]) -> sparse.csr_matrix:
    lookup = {name: i for i, name in enumerate(names)}
    rr, cc = [], []
    for i, muts in enumerate(rows):
        rr.extend([i] * len(muts)); cc.extend(lookup[m] for m in muts)
    return sparse.csr_matrix((np.ones(len(rr), dtype=np.int8), (rr, cc)), shape=(len(rows), len(names)))


def mixture_threshold(y_log10: np.ndarray, seed: int) -> tuple[float, dict]:
    gm = GaussianMixture(2, covariance_type="full", n_init=10, max_iter=500, random_state=seed)
    gm.fit(y_log10[:, None])
    grid = np.linspace(y_log10.min(), y_log10.max(), 100_000)
    posterior = gm.predict_proba(grid[:, None])
    order = np.argsort(gm.means_.ravel())
    difference = posterior[:, order[0]] - posterior[:, order[1]]
    crossings = np.where(np.diff(np.sign(difference)) != 0)[0]
    if not len(crossings):
        raise RuntimeError("The fitted mixture components have no posterior intersection")
    midpoint = gm.means_.ravel().mean()
    index = crossings[np.argmin(np.abs(grid[crossings] - midpoint))]
    threshold = float(grid[index])
    details = {
        "component_means_log10": gm.means_.ravel().tolist(),
        "component_weights": gm.weights_.tolist(),
        "component_variances": gm.covariances_.reshape(-1).tolist(),
        "threshold_log10": threshold,
        "threshold_raw": 10 ** threshold,
    }
    return threshold, details


def binary_continuous_stats(X: sparse.csr_matrix, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = X.shape[0]
    freq = np.asarray(X.sum(axis=0)).ravel().astype(float)
    sum1 = np.asarray(X.T @ y).ravel()
    mean1 = sum1 / freq
    mean0 = (y.sum() - sum1) / (n - freq)
    r = (mean1 - mean0) * np.sqrt((freq / n) * ((n - freq) / n)) / y.std(ddof=0)
    df = n - 2
    t = r * np.sqrt(df / np.maximum(1e-300, 1 - r * r))
    p = 2 * stats.t.sf(np.abs(t), df)
    return r, p


def mutation_associations(X, names, y_log, active):
    n = len(active); freq = np.asarray(X.sum(axis=0)).ravel().astype(int)
    active_n = int(active.sum())
    a = np.asarray(X[active].sum(axis=0)).ravel().astype(int)
    c = freq - a; b = active_n - a; d = n - active_n - c
    odds, fisher_p = [], []
    for table in zip(a, b, c, d):
        odds_ratio, pvalue = stats.fisher_exact([[table[0], table[1]], [table[2], table[3]]])
        odds.append(odds_ratio); fisher_p.append(pvalue)
    pb_r, pb_p = binary_continuous_stats(X, y_log)
    ranks = stats.rankdata(y_log)
    sp_r, sp_p = binary_continuous_stats(X, ranks)
    out = pd.DataFrame({
        "mutation": names, "frequency": freq, "frequency_fraction": freq / n,
        "active_with_mutation": a, "inactive_with_mutation": c,
        "odds_ratio": odds, "fisher_p": fisher_p,
        "point_biserial_r_log10": pb_r, "point_biserial_p": pb_p,
        "spearman_rho": sp_r, "spearman_p": sp_p,
        "mean_log10_with": np.asarray(X.T @ y_log).ravel() / freq,
    })
    out["mean_log10_without"] = (y_log.sum() - np.asarray(X.T @ y_log).ravel()) / (n - freq)
    out["mean_log10_difference"] = out.mean_log10_with - out.mean_log10_without
    out["fisher_q"] = bh_fdr(out.fisher_p)
    out["point_biserial_q"] = bh_fdr(out.point_biserial_p)
    out["spearman_q"] = bh_fdr(out.spearman_p)
    return out.sort_values(["fisher_q", "frequency"], ascending=[True, False])


def pair_associations(X, names, min_frequency):
    freq_all = np.asarray(X.sum(axis=0)).ravel().astype(int)
    keep = np.flatnonzero(freq_all >= min_frequency)
    Xm = X[:, keep].astype(np.int32); kept_names = np.asarray(names)[keep]; freq = freq_all[keep]
    co = (Xm.T @ Xm).toarray().astype(int); n = X.shape[0]
    i, j = np.triu_indices(len(keep), 1)
    n11 = co[i, j]; n10 = freq[i] - n11; n01 = freq[j] - n11; n00 = n - n11 - n10 - n01
    denom = np.sqrt(freq[i] * (n - freq[i]) * freq[j] * (n - freq[j]))
    phi = (n * n11 - freq[i] * freq[j]) / denom
    cache, p = {}, np.empty(len(i), dtype=float)
    for k, table in enumerate(zip(n11, n10, n01, n00)):
        key = tuple(map(int, table))
        if key not in cache:
            cache[key] = stats.fisher_exact([[key[0], key[1]], [key[2], key[3]]])[1]
        p[k] = cache[key]
    out = pd.DataFrame({"mutation_1": kept_names[i], "mutation_2": kept_names[j],
                        "frequency_1": freq[i], "frequency_2": freq[j], "cooccurrence": n11,
                        "phi": phi, "fisher_p": p})
    out["fisher_q"] = bh_fdr(p)
    out["abs_phi"] = np.abs(out.phi)
    return out.sort_values(["fisher_q", "abs_phi"], ascending=[True, False]), Xm, kept_names, freq


def nested_elastic_net(X, y_class, y_reg, names, seed, outer_folds, inner_folds):
    class_cv = StratifiedKFold(outer_folds, shuffle=True, random_state=seed)
    reg_cv = KFold(outer_folds, shuffle=True, random_state=seed)
    class_oof = np.full(len(y_class), np.nan); reg_oof = np.full(len(y_reg), np.nan)
    selected_class, selected_reg = [], []
    for fold, (tr, te) in enumerate(class_cv.split(X, y_class), 1):
        pipe = Pipeline([("scale", StandardScaler(with_mean=False)),
                         ("model", SGDClassifier(loss="log_loss", penalty="elasticnet", max_iter=3000, tol=1e-3,
                                                 early_stopping=True, validation_fraction=.1, n_iter_no_change=10,
                                                 class_weight="balanced", random_state=seed + fold))])
        search = GridSearchCV(pipe, {"model__alpha": [1e-5, 1e-4, 1e-3], "model__l1_ratio": [0.1, 0.5, 0.9]},
                              scoring="roc_auc", cv=StratifiedKFold(inner_folds, shuffle=True, random_state=seed + fold), n_jobs=-1)
        search.fit(X[tr], y_class[tr]); class_oof[te] = search.predict_proba(X[te])[:, 1]
        selected_class.append(search.best_params_)
    for fold, (tr, te) in enumerate(reg_cv.split(X), 1):
        pipe = Pipeline([("scale", StandardScaler(with_mean=False)), ("model", ElasticNet(max_iter=10000, random_state=seed + fold))])
        search = GridSearchCV(pipe, {"model__alpha": [0.001, 0.003, 0.01, 0.03, 0.1], "model__l1_ratio": [0.1, 0.5, 0.9]},
                              scoring="r2", cv=KFold(inner_folds, shuffle=True, random_state=seed + fold), n_jobs=-1)
        search.fit(X[tr], y_reg[tr]); reg_oof[te] = search.predict(X[te]); selected_reg.append(search.best_params_)
    # Tune final descriptive models on all data; OOF predictions above remain the unbiased evaluation.
    cp = Pipeline([("scale", StandardScaler(with_mean=False)),
                   ("model", SGDClassifier(loss="log_loss", penalty="elasticnet", max_iter=3000, tol=1e-3,
                                           early_stopping=True, validation_fraction=.1, n_iter_no_change=10,
                                           class_weight="balanced", random_state=seed))])
    cs = GridSearchCV(cp, {"model__alpha": [1e-5, 1e-4, 1e-3], "model__l1_ratio": [0.1, 0.5, 0.9]}, scoring="roc_auc", cv=inner_folds, n_jobs=-1).fit(X, y_class)
    rp = Pipeline([("scale", StandardScaler(with_mean=False)), ("model", ElasticNet(max_iter=10000, random_state=seed))])
    rs = GridSearchCV(rp, {"model__alpha": [0.001, 0.003, 0.01, 0.03, 0.1], "model__l1_ratio": [0.1, 0.5, 0.9]}, scoring="r2", cv=inner_folds, n_jobs=-1).fit(X, y_reg)
    coef = pd.DataFrame({"mutation": names, "logistic_coefficient_standardized": cs.best_estimator_["model"].coef_[0],
                         "linear_coefficient_standardized": rs.best_estimator_["model"].coef_})
    coef["abs_logistic_coefficient"] = coef.logistic_coefficient_standardized.abs()
    coef["abs_linear_coefficient"] = coef.linear_coefficient_standardized.abs()
    metrics = {"classification_oof_roc_auc": roc_auc_score(y_class, class_oof),
               "regression_oof_r2": r2_score(y_reg, reg_oof),
               "regression_oof_spearman": stats.spearmanr(y_reg, reg_oof)[0],
               "outer_folds": outer_folds, "inner_folds": inner_folds,
               "final_classification_params": cs.best_params_, "final_regression_params": rs.best_params_,
               "outer_classification_params": selected_class, "outer_regression_params": selected_reg}
    pred = pd.DataFrame({"observed_active": y_class, "predicted_active_probability_oof": class_oof,
                         "observed_log10_fluorescence": y_reg, "predicted_log10_fluorescence_oof": reg_oof})
    return metrics, coef, pred


def plots(df, threshold, assoc, pairs, Xm, pair_names, pair_freq, coef, predictions, out):
    sns.set_theme(style="whitegrid", context="talk")
    fig, ax = plt.subplots(figsize=(8, 5)); sns.histplot(df.fluorescence_log10, bins=80, ax=ax, color="#4C72B0")
    ax.axvline(threshold, color="#D55E00", lw=2, label=f"threshold = {threshold:.3f}")
    ax.set(xlabel="log10 fluorescence", ylabel="Variants"); ax.legend(); fig.tight_layout(); fig.savefig(out/"fluorescence_threshold.png", dpi=220); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 6)); q=assoc.fisher_q.clip(lower=1e-300)
    sc=ax.scatter(assoc.spearman_rho, -np.log10(q), c=np.log10(assoc.frequency), cmap="viridis", s=18, alpha=.75)
    ax.axvline(0,color="#777",lw=1); ax.set(xlabel="Mutation–fluorescence Spearman ρ",ylabel="−log10 Fisher FDR q")
    fig.colorbar(sc,ax=ax,label="log10 mutation frequency");fig.tight_layout();fig.savefig(out/"mutation_association_volcano.png",dpi=220);plt.close(fig)
    strongest=pairs.nlargest(50,"abs_phi")
    selected=list(dict.fromkeys(strongest.mutation_1.tolist()+strongest.mutation_2.tolist()))[:60]
    lookup={name:i for i,name in enumerate(pair_names)};top=np.array([lookup[name] for name in selected])
    dense=Xm[:,top].toarray();corr=np.corrcoef(dense,rowvar=False)
    fig,ax=plt.subplots(figsize=(14,12));sns.heatmap(corr,xticklabels=pair_names[top],yticklabels=pair_names[top],cmap="RdBu_r",center=0,vmin=-1,vmax=1,ax=ax,cbar_kws={"label":"Phi coefficient"})
    ax.tick_params(labelsize=7);ax.set_title("Mutations represented in the 50 strongest pairwise associations")
    fig.tight_layout();fig.savefig(out/"mutation_mutation_phi_heatmap_strongest_pairs.png",dpi=220);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(13,5));
    sns.scatterplot(x="observed_log10_fluorescence",y="predicted_log10_fluorescence_oof",data=predictions,s=8,alpha=.25,ax=axes[0],color="#4C72B0")
    axes[0].set(xlabel="Observed log10 fluorescence",ylabel="OOF predicted log10 fluorescence")
    topc=coef.nlargest(20,"abs_linear_coefficient").sort_values("linear_coefficient_standardized")
    axes[1].barh(topc.mutation,topc.linear_coefficient_standardized,color=np.where(topc.linear_coefficient_standardized>=0,"#D55E00","#4C72B0"));axes[1].set_xlabel("Elastic Net coefficient (standardized)")
    fig.tight_layout();fig.savefig(out/"elastic_net_regression.png",dpi=220);plt.close(fig)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--input",type=Path,required=True);ap.add_argument("--output",type=Path,required=True)
    ap.add_argument("--threshold-log10",type=float,default=None)
    ap.add_argument("--min-pair-frequency",type=int,default=10);ap.add_argument("--min-regression-frequency",type=int,default=5)
    ap.add_argument("--strong-phi",type=float,default=.5);ap.add_argument("--outer-folds",type=int,default=5);ap.add_argument("--inner-folds",type=int,default=3);ap.add_argument("--seed",type=int,default=42)
    ap.add_argument("--resume",action="store_true",help="Reuse completed association tables in the output directory")
    args=ap.parse_args(); start=time.time();args.output.mkdir(parents=True,exist_ok=True)
    raw=pd.read_csv(args.input); rows,names=parse_mutations(raw.mutant);X=indicator_matrix(rows,names)
    y_log=np.log10(raw.DMS_score.to_numpy(float)); auto,mixture=mixture_threshold(y_log,args.seed);threshold=auto if args.threshold_log10 is None else args.threshold_log10
    active=y_log>=threshold;df=pd.DataFrame({"mutant":raw.mutant,"DMS_score":raw.DMS_score,"fluorescence_log10":y_log,"active":active.astype(int)})
    assoc_path=args.output/"mutation_fluorescence_associations.csv"
    assoc=pd.read_csv(assoc_path) if args.resume and assoc_path.exists() else mutation_associations(X,names,y_log,active)
    assoc.to_csv(assoc_path,index=False)
    freq_all=np.asarray(X.sum(axis=0)).ravel().astype(int);pair_keep=np.flatnonzero(freq_all>=args.min_pair_frequency)
    Xm=X[:,pair_keep].astype(np.int32);pair_names=np.asarray(names)[pair_keep];pair_freq=freq_all[pair_keep]
    pair_path=args.output/"mutation_mutation_associations.csv.gz"
    if args.resume and pair_path.exists(): pairs=pd.read_csv(pair_path)
    else: pairs,_,_,_=pair_associations(X,names,args.min_pair_frequency)
    pairs.to_csv(pair_path,index=False,compression="gzip")
    strong=pairs[(pairs.abs_phi>=args.strong_phi)&(pairs.fisher_q<.05)];strong.to_csv(args.output/"strong_mutation_pairs.csv",index=False)
    freq=np.asarray(X.sum(axis=0)).ravel();reg_keep=np.flatnonzero(freq>=args.min_regression_frequency);Xreg=X[:,reg_keep];reg_names=np.asarray(names)[reg_keep]
    metrics,coef,pred=nested_elastic_net(Xreg,active.astype(int),y_log,reg_names,args.seed,args.outer_folds,args.inner_folds)
    coef=coef.merge(assoc[["mutation","frequency","odds_ratio","fisher_q","point_biserial_r_log10","spearman_rho","spearman_q"]],on="mutation",how="left")
    coef.sort_values("abs_linear_coefficient",ascending=False).to_csv(args.output/"elastic_net_mutation_coefficients.csv",index=False);pred.to_csv(args.output/"elastic_net_oof_predictions.csv",index=False)
    df.to_csv(args.output/"variant_activity_labels.csv",index=False);plots(df,threshold,assoc,pairs,Xm,pair_names,pair_freq,coef,pred,args.output)
    supplied={}
    if "DMS_score_bin" in raw:
        supplied={"supplied_active_fraction":float(raw.DMS_score_bin.mean()),"agreement_with_supplied_bin":float(np.mean(active==raw.DMS_score_bin.to_numpy())),
                  "supplied_max_inactive":float(raw.loc[raw.DMS_score_bin.eq(0),"DMS_score"].max()),"supplied_min_active":float(raw.loc[raw.DMS_score_bin.eq(1),"DMS_score"].min())}
    summary={"input":str(args.input),"n_variants":len(raw),"n_mutations":len(names),"threshold_source":"two_component_Gaussian_mixture_intersection" if args.threshold_log10 is None else "user_override",
             "threshold_log10":threshold,"threshold_raw":10**threshold,"active_n":int(active.sum()),"inactive_n":int((~active).sum()),"mixture":mixture,**supplied,
             "pairwise_min_frequency":args.min_pair_frequency,"pairwise_mutations":len(pair_names),"pairwise_tests":len(pairs),"strong_pair_definition":f"abs(phi)>={args.strong_phi} and Fisher q<0.05","strong_pairs":len(strong),
             "regression_min_frequency":args.min_regression_frequency,"regression_mutations":len(reg_names),"performance":metrics,"runtime_seconds":time.time()-start}
    (args.output/"summary.json").write_text(json.dumps(summary,indent=2))
    report=f"""# Mutation association analysis: D7PM05 CLYGR

Analyzed **{len(raw):,} variants** and **{len(names):,} distinct substitutions**. Continuous analyses use log10 fluorescence. Activity was defined by the intersection of a two-component Gaussian mixture fitted to log10 fluorescence: **{threshold:.3f}** (**{10**threshold:,.1f} raw units**), producing {active.sum():,} active and {(~active).sum():,} inactive variants. The existing ProteinGym label instead changes near 12,500 raw units; agreement with the mixture label is {supplied.get('agreement_with_supplied_bin',float('nan')):.3f}.

## Predictive evaluation

Performance uses nested {args.outer_folds}-fold outer / {args.inner_folds}-fold inner cross-validation. Hyperparameters are tuned only inside each training fold.

- Elastic Net logistic ROC-AUC: **{metrics['classification_oof_roc_auc']:.3f}**
- Elastic Net linear R²: **{metrics['regression_oof_r2']:.3f}**
- Elastic Net linear Spearman ρ: **{metrics['regression_oof_spearman']:.3f}**

## Outputs

- `mutation_fluorescence_associations.csv`: frequency, odds ratio, Fisher p/q, point-biserial r/p/q, and Spearman rho/p/q for every mutation.
- `mutation_mutation_associations.csv.gz`: pairwise phi, Fisher p/q, frequencies, and co-occurrences for mutations observed at least {args.min_pair_frequency} times.
- `strong_mutation_pairs.csv`: pairs with |phi| ≥ {args.strong_phi} and Fisher q < 0.05.
- `elastic_net_mutation_coefficients.csv`: fitted standardized coefficients and univariate statistics.
- `elastic_net_oof_predictions.csv`: out-of-fold predictions used for performance estimates.
- `variant_activity_labels.csv`: reproducible per-variant log10 fluorescence and activity label.
- PNG figures show the threshold, mutation associations, mutation-pair heatmap, and regression predictions/coefficients.

Association does not by itself establish a causal mutation effect: library design, linked mutations, and epistasis can produce correlations. Elastic Net coefficients are conditional associations within this dataset.

## Reproduce

```bash
/nethome/akolchina/miniconda3/envs/protease-pipeline/bin/python \
  scripts/somermeyer/mutation_association.py \
  --input data/somermeyer/D7PM05_CLYGR_Somermeyer_2022.csv \
  --output results/reanalysis/somermeyer/D7PM05_CLYGR_mutation_associations
```

Use `--threshold-log10 VALUE` to override the mixture-derived activity threshold. `--resume` reuses completed association tables when only the multivariate analysis or plots need to be regenerated.
"""
    (args.output/"README.md").write_text(report)
    print(json.dumps(summary,indent=2))

if __name__=="__main__": main()
