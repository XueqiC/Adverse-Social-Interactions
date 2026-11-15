# Adverse-Social-Interactions

This is the official repository for the WSDM 2027 paper "Adverse Online Social Interactions: A Multi-Level Evolutionary Analysis of Local Patterns, Diffusion, and Community Disruption"

This repository implements a multi-level framework that combines structural disconnection signals, such as blocking or unfollowing, with content-level toxicity. The code supports three analytical levels:

- `micro`: snapshot-level signed clustering and node toxicity summaries
- `meso`: local exposure, matching, cascade, and outcome analyses
- `macro`: tie-change analyses with random and recommendation-based null models

## Repository Layout

```text
Adverse-Social-Interactions/
├── common/   Shared utilities for micro and macro analyses
├── micro/    Community-level and node-level analysis scripts
├── meso/     Pair construction, cascade extraction, and outcome analysis
└── macro/    Long-range tie-change and null-model comparison scripts
```

## Requirements

Core dependencies:

- `numpy`
- `scipy`
- `scikit-learn`
- `signet`

Optional:

- `matplotlib` for plot output
- `torch` for extended deterministic seeding support

## Data Assumptions

The repository expects preprocessed `.npy` snapshot files rather than raw platform exports.

Default locations used by the current scripts:

- Twitter snapshots: `final_twitter_dataset/largest_cluster/<week>weeks_largest_cluster.npy`
- Bluesky snapshots: `bluesky/filter/d<snapshot>.npy`

Raw data are not distributed with this repository.

## Quick Start

Run from the repository root.

### Micro

```bash
python -m micro.run_analysis --dataset twitter --snapshot 0
python -m micro.run_analysis --dataset bluesky --snapshot 3 --negative-edge-weight 1
bash micro/run_batch.sh
```

### Meso

The `meso/` directory contains study-specific scripts for:

- interaction triple extraction
- matched exposed/unexposed pair construction
- cascade tree construction
- downstream outcome analysis

Representative entry points:

- `meso/extract_triple.py`
- `meso/pair_construction/`
- `meso/cascade/`
- `meso/outcome_analysis/`

### Macro

```bash
python -m macro.run_analysis --dataset twitter --old-snapshot 0 --new-snapshot 12
python -m macro.run_analysis --dataset bluesky --old-snapshot 1 --new-snapshot 4
bash macro/run_twitter_batch.sh
bash macro/run_bluesky_batch.sh
```

## Method Summary

At a high level, the pipeline:

1. loads a signed interaction snapshot
2. filters to valid toxicity-bearing edges
3. assigns or reassigns signed edge labels
4. performs signed spectral clustering or matched local analysis
5. measures user-level, local, or network-level outcomes
6. compares observed dynamics against null baselines when applicable

Macro analysis reports both weight-based and toxicity-based separation statistics over repeated null trials.

## Notes

- The code is intended for analysis on preprocessed research datasets.
- Scripts may generate derived CSV files, logs, and plot data.
