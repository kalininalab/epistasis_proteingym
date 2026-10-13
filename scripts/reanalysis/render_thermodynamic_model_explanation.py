#!/usr/bin/env python
"""Render the Tsuboyama thermodynamic-model explanation as Markdown and PDF."""
from pathlib import Path
import re
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

OUT = Path("results/reanalysis/tsuboyama/thermodynamic_model_explanation")

MARKDOWN = r"""# How the Tsuboyama “thermodynamic model” works

## Overview

Based strictly on the implementation in `external/tsuboyama/additive_model.py`, the so-called thermodynamic model is a Bayesian two-way additive model fitted independently for every pair of mutated positions.

For a double mutant containing amino acid a at the first position and amino acid b at the second position, it assumes:

$$dG_{ab} = f_a + s_b + \varepsilon_{ab}$$

- `f_a` is the fitted effect of amino acid a at the first position.
- `s_b` is the fitted effect of amino acid b at the second position.
- `epsilon_ab` is the remaining non-additive residual.

The residual is saved as `thermodynamic_coupling` and used as experimental epistasis.

## 1. Only double mutants are modeled

```python
mask = out['num_mutations'] == 2
doubles = out.loc[mask].copy()
```

Single mutants, wild type, and variants containing more than two mutations are not used. Consequently, the additive effects are not estimated from measured single-mutant effects.

## 2. Fits are independent for every position pair

```python
for pname, sub in doubles.groupby('pair_name'):
```

A `pair_name` identifies two mutated sequence positions. All observed amino-acid combinations at those two positions are fitted together. This is often described as a 20 by 20 matrix, but the code does not require all 400 combinations to be observed; it fits the available cells.

## 3. Mutant amino acids become integer indices

```python
aas = 'QENHDRKTSAGMCLVIWYFP'
doubles['aa1'] = [aas.index(m.split(':')[0][-1]) for m in doubles['mutant']]
doubles['aa2'] = [aas.index(m.split(':')[1][-1]) for m in doubles['mutant']]
```

For a mutant such as `A10V:G20L`, the model extracts `V` and `L`. The wild-type amino acids and position numbers do not enter the statistical equation directly; `pair_name` determines which two positions are being fitted.

## 4. Forty additive amino-acid effects are sampled

```python
first = numpyro.sample('first_aa_effect', dist.Normal(np.resize(0.1, (20)), 3))
second = numpyro.sample('second_aa_effect', dist.Normal(np.resize(0.1, (20)), 3))
```

For every position pair, the model contains 20 effects for the first position and 20 effects for the second position:

$$f_a \sim \mathcal{N}(0.1, 3^2), \qquad s_b \sim \mathcal{N}(0.1, 3^2)$$

The priors are broad. There is no separate intercept; the expected score is represented by the sum of the two amino-acid effects.

## 5. The additive prediction is constructed

```python
pred = [first[i] for i in first_idx] + [second[j] for j in second_idx]
```

For combination ab:

$$\mu_{ab} = f_a + s_b$$

The contribution of amino acid a at the first position is assumed to be the same regardless of the amino acid at the second position, and vice versa. Combination-specific behavior cannot be represented by this additive term.

## 6. Experimental predictions are clipped in the likelihood

```python
if clip_predictions:
    pred = jax.numpy.clip(pred, -1, 5)
```

With the default experimental fit:

$$\widetilde{\mu}_{ab} = \operatorname{clip}(f_a+s_b,-1,5)$$

Sums below -1 are treated as -1 and sums above 5 as 5. In our PLM-score fits, including conditional MM and PLL, we use `clip_predictions=False` because model scores do not use the experimental dG scale.

## 7. Gaussian observation noise is assumed

```python
sigma = numpyro.sample('sigma', dist.Exponential(1))
numpyro.sample('obs_dg', Normal(pred, sigma), obs=obs)
```

The actual code expresses the normal distribution as an affine transformation of a standard normal, but it is equivalent to:

$$\sigma \sim \operatorname{Exponential}(1)$$
$$dG_{ab} \sim \mathcal{N}(\widetilde{\mu}_{ab}, \sigma^2)$$

All double mutants in one position-pair fit share the same noise parameter.

## 8. NUTS samples the posterior

```python
kernel = NUTS(additive_model, init_strategy=initialization.init_to_feasible())
mcmc = MCMC(kernel, num_warmup=100, num_samples=50, num_chains=1)
```

Each position pair is fitted with one MCMC chain, 100 warm-up iterations, and 50 retained posterior samples. NUTS jointly samples 20 first-position effects, 20 second-position effects, and one noise parameter. The code records divergences, mean acceptance probability, and maximum NUTS steps as diagnostics.

## 9. Posterior samples become a single additive reconstruction

```python
first = np.median(samples['first_aa_effect'], axis=0)
second = np.median(samples['second_aa_effect'], axis=0)
recon = [first[a] + second[b] for a, b in zip(sub['aa1'], sub['aa2'])]
```

The code takes the marginal posterior median for each amino-acid effect and then adds them:

$$\widehat{dG}^{add}_{ab} = \operatorname{median}(f_a) + \operatorname{median}(s_b)$$

This value is saved as `recon_dg`.

## 10. Epistasis is the residual

```python
thermodynamic_coupling = dG - recon_dg
```

Therefore:

$$\varepsilon_{ab} = dG_{ab} - \widehat{dG}^{add}_{ab}$$

- A residual near zero means the score is explained by the additive reconstruction.
- A positive residual means the observed score is above the additive reconstruction.
- A negative residual means it is below the additive reconstruction.
- A large absolute residual means a strong departure from additivity.

The precise biochemical meaning of the sign depends on the direction of the score. The safest interpretation is “above or below the fitted additive expectation on this score scale.”

## Concrete example

Suppose the fit estimates:

$$\widehat f_V = 0.8, \qquad \widehat s_L = 1.1$$

The additive reconstruction for combination VL is 1.9. If the measured score is 2.5:

$$\varepsilon_{VL}=2.5-1.9=0.6$$

The combination lies 0.6 units above its fitted additive expectation.

## Important implementation subtleties

### The saved reconstruction is not clipped

Clipping is applied to `pred` inside the likelihood during fitting. After fitting, however, `recon_dg` is calculated as `first[a] + second[b]` without clipping. Thus, when `clip_predictions=True`, posterior fitting uses a clipped mean but the saved reconstruction and residual use the un-clipped sum.

### It sums marginal posterior medians

The implementation calculates:

$$\operatorname{median}(f_a)+\operatorname{median}(s_b)$$

It does not calculate:

$$\operatorname{median}(f_a+s_b)$$

These are often similar but are not mathematically guaranteed to be identical.

### The individual effects are not uniquely identifiable

For any constant c:

$$(f_a+c)+(s_b-c)=f_a+s_b$$

Therefore, individual first- and second-position coefficients should not be interpreted as uniquely determined biochemical effects. The priors partially anchor the decomposition. The fitted sums and residuals are the relevant outputs.

### The random seed is reset for every pair

The code initializes `PRNGKey(1)` inside the loop. Every position-pair fit therefore uses the same deterministic random-number stream. This makes runs reproducible but does not give independent random seeds across pairs.

### Posterior sampling is short

Only 50 posterior samples from one chain are retained after 100 warm-up iterations. This reproduces the established workflow, but individual pair-level posterior estimates should be interpreted with the MCMC diagnostics in mind.

## Why it is called “thermodynamic”

The model does not simulate folding equilibria, residue contacts, molecular energies, or a mechanistic thermodynamic cycle. It is called thermodynamic because it is applied to experimental stability-related dG values.

Statistically, it is a Bayesian two-factor additive model with Gaussian noise:

```text
observed double-mutant score
    = fitted additive component
    + residual coupling
```

For a protein-language-model score, we repeat the same decomposition:

$$S_{ab} = \widehat S^{add}_{ab} + \varepsilon^{pred}_{ab}$$

Predicted epistasis is then compared with experimental thermodynamic coupling using Spearman correlation.
"""


def clean_inline(text: str) -> str:
    text = text.replace("**", "").replace("`", "")
    return text


def render_pdf(markdown: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    page_w, page_h = 8.27, 11.69
    left, right, top, bottom = 0.72, 0.58, 0.72, 0.62
    content_w = page_w - left - right
    pdf = PdfPages(destination)
    fig = None
    y = 0
    page = 0

    def new_page():
        nonlocal fig, y, page
        if fig is not None:
            fig.text(0.5, 0.025, str(page), ha="center", fontsize=8, color="#64748B")
            pdf.savefig(fig, bbox_inches=None)
            plt.close(fig)
        page += 1
        fig = plt.figure(figsize=(page_w, page_h), facecolor="white")
        y = page_h - top

    def require(height):
        nonlocal y
        if y - height < bottom:
            new_page()

    def add_lines(text, fontsize=10, weight="normal", color="#172033", family="DejaVu Sans",
                  width=98, spacing=0.19, before=0.0, after=0.08):
        nonlocal y
        lines = []
        for paragraph in text.split("\n"):
            lines.extend(textwrap.wrap(clean_inline(paragraph), width=width, break_long_words=False,
                                       break_on_hyphens=False) or [""])
        height = before + len(lines) * spacing + after
        require(height)
        y -= before
        for line in lines:
            fig.text(left / page_w, y / page_h, line, ha="left", va="top", fontsize=fontsize,
                     fontweight=weight, color=color, family=family)
            y -= spacing
        y -= after

    def add_equation(equation):
        nonlocal y
        require(0.48)
        y -= 0.10
        try:
            fig.text(0.5, y / page_h, f"${equation}$", ha="center", va="top", fontsize=12,
                     color="#0F172A")
        except Exception:
            fig.text(left / page_w, y / page_h, equation, ha="left", va="top", fontsize=10,
                     family="DejaVu Sans Mono")
        y -= 0.38

    new_page()
    in_code = False
    code = []
    for raw in markdown.splitlines():
        line = raw.rstrip()
        if line.startswith("```"):
            if in_code:
                add_lines("\n".join(code), fontsize=8.2, family="DejaVu Sans Mono", color="#334155",
                          width=100, spacing=0.155, before=.06, after=.13)
                code = []; in_code = False
            else:
                in_code = True
            continue
        if in_code:
            code.append(line); continue
        if not line:
            y -= .05
            continue
        if line.startswith("# "):
            add_lines(line[2:], fontsize=20, weight="bold", color="#172033", width=38,
                      spacing=.34, before=.12, after=.22)
        elif line.startswith("## "):
            add_lines(line[3:], fontsize=14, weight="bold", color="#234E78", width=72,
                      spacing=.26, before=.22, after=.10)
        elif line.startswith("### "):
            add_lines(line[4:], fontsize=11.5, weight="bold", color="#7C3E16", width=82,
                      spacing=.22, before=.16, after=.08)
        elif line.startswith("$$") and line.endswith("$$"):
            add_equation(line[2:-2])
        elif line.startswith("- "):
            add_lines("•  " + line[2:], fontsize=9.6, width=94, spacing=.18, before=.01, after=.02)
        else:
            add_lines(line, fontsize=9.6, width=98, spacing=.18, before=.01, after=.07)
    if fig is not None:
        fig.text(0.5, 0.025, str(page), ha="center", fontsize=8, color="#64748B")
        pdf.savefig(fig, bbox_inches=None); plt.close(fig)
    info = pdf.infodict()
    info["Title"] = "How the Tsuboyama thermodynamic model works"
    info["Subject"] = "Code-based explanation of additive thermodynamic coupling"
    info["Author"] = "epistasis_proteingym analysis"
    pdf.close()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    md = OUT / "thermodynamic_model_explained.md"
    pdf = OUT / "thermodynamic_model_explained.pdf"
    md.write_text(MARKDOWN)
    render_pdf(MARKDOWN, pdf)
    print(md)
    print(pdf)


if __name__ == "__main__":
    main()
