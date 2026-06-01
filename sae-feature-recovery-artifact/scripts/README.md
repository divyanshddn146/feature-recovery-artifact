# Scripts

This folder contains cleaned scripts for the SAE feature-recovery artifact.

## Files

- `run_comparison.py`
  Runs the main comparison between mixed reconstruction, TopK mixed reconstruction, paired-delta training, and random-pair controls.

- `extra_logging_utils.py`
  Diagnostic logging utilities for decoder allocation, decoder-gradient signal, selectivity, and recovery.

- `make_dataset.py`
  Builds controlled datasets of background activations, weak/strong/random directions, mixed activations, paired deltas, and random-pair deltas.

- `compare_public_sae.py`
  Compares internal target directions with decoder directions from public GPT-2-small SAEs.

- `plot_diagnostics.py`
  Generates selected diagnostic figures from processed result files.

- `find_weak_directions.py`
  Scans candidate concept pairs to identify weak or low-SNR target directions.

## Scope

These scripts are included to document the experimental structure and support reproducibility of the main analysis logic. Large model caches, tensors, raw logs, and machine-specific configuration files are intentionally omitted.

The scripts are intended to document and support the artifact, but running the full pipeline may require downloading models and regenerating omitted activation caches.
