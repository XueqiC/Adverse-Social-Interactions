#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

OUT_DIR="${OUT_DIR:-macro/rec_twitter}"
LOG_DIR="${LOG_DIR:-macro/logs}"
WEEK_OLD="${WEEK_OLD:-0}"
WEEK_NEWS=(${WEEK_NEWS:-4 8 12})
OPTIONS=(${OPTIONS:-1 2 3 4 5})
RUNS="${RUNS:-20}"
SEED="${SEED:-1028}"
GTHR="${GTHR:-0.1}"
MF_TOPK="${MF_TOPK:-20}"
OPT_K="${OPT_K:-0}"
SAVE_PLOTS="${SAVE_PLOTS:-0}"
BASE_DIR="${BASE_DIR:-}"

mkdir -p "${OUT_DIR}" "${LOG_DIR}"

echo "Running Twitter macro batch into ${OUT_DIR} ..."
for week_new in "${WEEK_NEWS[@]}"; do
    for option in "${OPTIONS[@]}"; do
        stamp="$(date +%Y%m%d_%H%M%S)"
        log_file="${LOG_DIR}/rec_w${WEEK_OLD}_to_w${week_new}_opt${option}_${stamp}.log"

        cmd=(
            python -m macro.run_analysis
            --dataset twitter
            --old-snapshot "${WEEK_OLD}"
            --new-snapshot "${week_new}"
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

        echo "== week_old=${WEEK_OLD}, week_new=${week_new}, option=${option} =="
        "${cmd[@]}" 2>&1 | tee "${log_file}"
    done
done

echo "Done. Results in ${OUT_DIR}, logs in ${LOG_DIR}"
