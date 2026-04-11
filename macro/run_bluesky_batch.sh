#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

OUT_DIR="${OUT_DIR:-macro/rec_bluesky}"
LOG_DIR="${LOG_DIR:-macro/logs_bluesky}"
SNAPSHOT_OLD="${SNAPSHOT_OLD:-1}"
SNAPSHOT_NEWS=(${SNAPSHOT_NEWS:-2 3 4})
OPTIONS=(${OPTIONS:-1 2 3 4 5})
RUNS="${RUNS:-20}"
SEED="${SEED:-1028}"
GTHR="${GTHR:-0.1}"
MF_TOPK="${MF_TOPK:-20}"
OPT_K="${OPT_K:-0}"
SAVE_PLOTS="${SAVE_PLOTS:-0}"
BASE_DIR="${BASE_DIR:-}"

mkdir -p "${OUT_DIR}" "${LOG_DIR}"

echo "Running Bluesky macro batch into ${OUT_DIR} ..."
for snapshot_new in "${SNAPSHOT_NEWS[@]}"; do
    for option in "${OPTIONS[@]}"; do
        stamp="$(date +%Y%m%d_%H%M%S)"
        log_file="${LOG_DIR}/rec_bluesky_d${SNAPSHOT_OLD}_to_d${snapshot_new}_opt${option}_${stamp}.log"

        cmd=(
            python -m macro.run_analysis
            --dataset bluesky
            --old-snapshot "${SNAPSHOT_OLD}"
            --new-snapshot "${snapshot_new}"
            --edge-label-option "${option}"
            --global-toxicity-threshold "${GTHR}"
            --run "${RUNS}"
            --seed "${SEED}"
            --mf-topk "${MF_TOPK}"
            --opt-k "${OPT_K}"
            --output-dir "${OUT_DIR}"
        )

        if [[ -n "${BASE_DIR}" ]]; then
            cmd+=(--base-dir "${BASE_DIR}")
        fi
        if [[ "${SAVE_PLOTS}" == "1" || "${SAVE_PLOTS}" == "true" ]]; then
            cmd+=(--save-plots)
        fi

        echo "== snapshot_old=d${SNAPSHOT_OLD}, snapshot_new=d${snapshot_new}, option=${option} =="
        "${cmd[@]}" 2>&1 | tee "${log_file}"
    done
done

echo "Done. Results in ${OUT_DIR}, logs in ${LOG_DIR}"
