# ESM2-650M PLL pilot

This is an implementation and runtime pilot. It is not the final biological
analysis.

## Pilot workload

- Assay: `PIN1_HUMAN_Tsuboyama_2023_1I6C`
- Deterministic subset: 32 of 116 double mutants, seed 20261002
- Unique sequences after caching/deduplication: 67
- One-position-masked examples: 2,613
- Protein length: 39 residues
- GPU batch size: 32
- Model: `facebook/esm2_t33_650M_UR50D`, FP16 autocast

## Measured runtime

| Stage | Time |
|---|---:|
| Model loading | 40.34 s |
| PLL inference | 55.91 s |
| Total PLL job | 96.46 s |
| Additive-model pilot fit | 22.15 s |

Inference throughput was 46.74 masked examples/s, or 1.20 complete PIN1
sequences/s. Re-running the same output can reuse all 67 cached sequence PLLs.

The end-to-end pilot saved finite `PLL_WT`, `PLL_A`, `PLL_B`, `PLL_AB`, and
`epsilon_PLL` values for all 32 doubles. The inclusion-exclusion recomputation
error was at most 7.33e-15. The additive fit covered one position-pair matrix
and had zero NUTS divergences.

The pilot correlations are not estimates of final performance: only five of
the selected variants are in the predefined epistatic subset.

## Projected full workload

Across the 50 assays there are 51,924 doubles, 56,871 unique assay-specific
WT/single/double sequences after caching, and 3,351,842 masked examples. At the
PIN1 pilot throughput this is 19.9 serial GPU-hours, approximately 2.0 ideal
wall-clock hours on 10 GPUs plus queue time and CPU additive fits. Longer
proteins require more attention computation, so 2.0 hours is an optimistic
estimate; a practical allocation should allow roughly 2–4 hours with 10 GPUs.

No full-assay PLL jobs have been launched.
