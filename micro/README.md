# Micro Analysis

`micro/run_analysis.py` performs snapshot-level signed clustering and writes node-level toxicity summaries.

Examples:

```bash
python -m micro.run_analysis --dataset twitter --snapshot 0
python -m micro.run_analysis --dataset bluesky --snapshot 3 --negative-edge-weight 1
```

Batch run:

```bash
bash micro/run_batch.sh
DATASET=bs bash micro/run_batch.sh
```

Expected data locations by default:

- Twitter: `final_twitter_dataset/largest_cluster/<week>weeks_largest_cluster.npy`
- Bluesky: `bluesky/filter/d<snapshot>.npy`

Required Python packages: `numpy`, `scipy`, `scikit-learn`, and `signet`.
