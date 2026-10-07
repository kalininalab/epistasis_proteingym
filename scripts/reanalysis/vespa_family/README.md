# ProtT5 and VESPA-family extension

ProtT5 uses the official VESPA reconstruction convention and supports default
MM, conditional MM, and PLL. The checkpoint is
[`Rostlab/prot_t5_xl_uniref50` at revision `973be27c`](https://huggingface.co/Rostlab/prot_t5_xl_uniref50/tree/973be27c52ee6474de9c945952a8008aeb2a1a73).

VESPA, VESPAl, and VespaG will be evaluated separately with conditional
variant-score interactions because their outputs are effect-predictor scores,
not log probabilities; PLL is therefore undefined for these predictors.
Official pinned sources are VESPA
[`34fd1ab`](https://github.com/Rostlab/VESPA/tree/34fd1ab1b7fdc2b4e1a241fdae1db5c2f09d8944)
and VespaG [`3d47582`](https://github.com/JSchlensok/VespaG/tree/3d4758252dbd423249e694d6f7d195c707f72a92).
