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

If you stay in `/largedisk/xueqic/Dropbox/polar`, you can run:

```bash
python code_submission/Adverse-Social-Interactions/micro/run_analysis.py --dataset twitter --snapshot 0
bash code_submission/Adverse-Social-Interactions/micro/run_batch.sh
```

Expected data locations by default:

- Twitter: `final_twitter_dataset/largest_cluster/<week>weeks_largest_cluster.npy`
- Bluesky: `bluesky/filter/d<snapshot>.npy`

Required Python packages: `numpy`, `scipy`, `scikit-learn`, and `signet`.
