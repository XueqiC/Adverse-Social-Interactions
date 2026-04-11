# Macro Analysis

`macro/run_analysis.py` runs the tie-change comparison against random and matrix-factorization nulls.

Examples:

```bash
python -m macro.run_analysis --dataset twitter --old-snapshot 0 --new-snapshot 12
python -m macro.run_analysis --dataset bluesky --old-snapshot 1 --new-snapshot 4
```

Batch runs:

```bash
bash macro/run_twitter_batch.sh
bash macro/run_bluesky_batch.sh
```

Expected data locations by default:

- Twitter: `final_twitter_dataset/largest_cluster/<week>weeks_largest_cluster.npy`
- Bluesky: `bluesky/filter/d<snapshot>.npy`

Required Python packages: `numpy`, `scipy`, `scikit-learn`, and `signet`.
