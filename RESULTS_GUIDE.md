# Results Guide

Start with:

1. `PROJECT_SUMMARY.pdf` (and `.tex` source)
   High-level claim, setup, main result, limitations, and next steps.

2. `figures/diagnostic_timeseries_weak_layer8_scale0.05_K512_zoom_agg.png`
   Main diagnostic figure. Shows that paired-delta training rapidly learns target-aligned and target-selective latents, while mixed reconstruction SAEs plateau at low decoder alignment and near-zero selectivity.

3. `results/main_comparison/AGGREGATE_RESULTS.csv`
   Main recovery comparison across direction groups and training conditions.

4. `figures/weak_diagnostic_final_bars_weak_layer8_scale0.05_K512.png`
   Final diagnostic metrics for the weak target directions.

5. `results/public_sae_comparison/ALL_ALLOCATION_SUMMARY.csv`
   Public SAE decoder-allocation comparison.

## Core interpretation

The paired-delta condition is a positive control showing that target directions are learnable when isolated. The main result is that mixed reconstruction SAEs can fail to recover the same directions as selective, functionally usable features. This supports the claim that decoder alignment is not sufficient evidence of SAE feature recovery.
