# SAE Feature Recovery Artifact

## One-sentence summary

This project studies whether decoder alignment in sparse autoencoders is sufficient evidence of functional feature recovery.

## Main finding

In controlled GPT-2-small residual-stream experiments, standard mixed-reconstruction SAE training shows weak functional recovery of target directions, even when decoder allocation or gradient signal is nonzero. In contrast, paired-delta training recovers the same directions almost perfectly, while random-pair controls fail.

This supports the claim that **decoder alignment is not sufficient evidence of feature recovery**.

The paired-delta condition is intentionally easier than mixed reconstruction: it isolates the target direction. It is used as a positive control showing that the target directions are learnable in principle, not as evidence that paired-delta training is a complete replacement for reconstruction SAEs.

## Artifact status

This is a compact anonymized work-in-progress research artifact. It contains cleaned scripts, processed result tables, diagnostic outputs, and selected figures. It is not the full raw experiment workspace.

## Repository contents

```text
.
├── README.md
├── PROJECT_SUMMARY.tex
├── PROJECT_SUMMARY.pdf
├── RESULTS_GUIDE.md
├── requirements.txt
├── figures/
├── results/
│   ├── main_comparison/
│   ├── diagnostics/
│   └── public_sae_comparison/
└── scripts/
    ├── run_comparison.py
    ├── find_weak_directions.py
    ├── compare_public_sae.py
    ├── plot_diagnostics.py
    ├── make_dataset.py
    └── extra_logging_utils.py
```

## Recommended reading order

1. `PROJECT_SUMMARY.pdf` (and `.tex` source) for the high-level claim and result.
2. `figures/diagnostic_timeseries_weak_layer8_scale0.05_K512_zoom_agg.png` for the main diagnostic figure.
3. `results/main_comparison/AGGREGATE_RESULTS.csv` for the main recovery comparison.
4. `results/public_sae_comparison/ALL_ALLOCATION_SUMMARY.csv` for the public SAE decoder-allocation comparison.

`RESULTS_GUIDE.md` provides a short tour of the same files in narrative form.

## Main components

### Main comparison

Evaluates mixed-reconstruction training, paired-delta training, and random-pair delta controls.

Key files:

```text
results/main_comparison/AGGREGATE_RESULTS.csv
results/main_comparison/ALL_SUMMARY.csv
results/main_comparison/ALL_PER_CONCEPT.csv
```

### Diagnostics

Tracks decoder allocation, gradient signal, selectivity, and recovery for weak, strong, and random target directions.

Key folders:

```text
results/diagnostics/weak/
results/diagnostics/strong/
results/diagnostics/random/
```

### Public SAE comparison

Tests whether pretrained/public SAE decoders show geometric allocation toward the same direction groups.

Key files:

```text
results/public_sae_comparison/ALL_ALLOCATION_SUMMARY.csv
results/public_sae_comparison/ALL_PUBLIC_SAE_ALLOCATION.csv
```

## Notes

Large tensors, model caches, raw training logs, and machine-specific configuration files are intentionally omitted. The included files are sufficient to inspect the core result and reproduce the main analysis logic.
