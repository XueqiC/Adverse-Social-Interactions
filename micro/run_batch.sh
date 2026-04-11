#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

NEGATIVE_WEIGHTS=(${NEGATIVE_WEIGHTS:-1 10})
OPTIONS=(${OPTIONS:-1 2 3 4 5})
DATASET="${DATASET:-twitter}"
SAVE_PLOTS="${SAVE_PLOTS:-0}"
RECLUSTER_LARGEST="${RECLUSTER_LARGEST:-0}"
SPECTRAL_NORMALISATION="${SPECTRAL_NORMALISATION:-rw_sep}"

dataset_lc="$(printf '%s' "${DATASET}" | tr '[:upper:]' '[:lower:]')"
if [[ "${dataset_lc}" == "bs" || "${dataset_lc}" == "bluesky" ]]; then
    SNAPSHOTS=(${SNAPSHOTS:-1 2 3 4})
else
    SNAPSHOTS=(${SNAPSHOTS:-0})
fi

TIMESTAMP="$(date +"%Y%m%d_%H%M%S")"
RESULT_ROOT="micro/batch_results_${TIMESTAMP}"
mkdir -p "${RESULT_ROOT}"

TOTAL_RUNS=$(( ${#NEGATIVE_WEIGHTS[@]} * ${#SNAPSHOTS[@]} * ${#OPTIONS[@]} ))
RUN_IDX=0

{
    echo "timestamp=${TIMESTAMP}"
    echo "dataset=${DATASET}"
    echo "snapshots=${SNAPSHOTS[*]}"
    echo "negative_weights=${NEGATIVE_WEIGHTS[*]}"
    echo "options=${OPTIONS[*]}"
    echo "spectral_normalisation=${SPECTRAL_NORMALISATION}"
    echo "recluster_largest=${RECLUSTER_LARGEST}"
    echo "save_plots=${SAVE_PLOTS}"
    echo "total_runs=${TOTAL_RUNS}"
} > "${RESULT_ROOT}/run_config.txt"

echo "Saving all run outputs to: ${RESULT_ROOT}"

for neg_weight in "${NEGATIVE_WEIGHTS[@]}"; do
    for snapshot in "${SNAPSHOTS[@]}"; do
        for option in "${OPTIONS[@]}"; do
            RUN_IDX=$((RUN_IDX + 1))

            if [[ "${dataset_lc}" == "bs" || "${dataset_lc}" == "bluesky" ]]; then
                SNAPSHOT_TAG="d${snapshot}"
            else
                SNAPSHOT_TAG="${snapshot}weeks"
            fi

            RUN_TAG="neg${neg_weight}_${SNAPSHOT_TAG}_opt${option}"
            RUN_DIR="${RESULT_ROOT}/${RUN_TAG}"
            LOG_FILE="${RUN_DIR}/run.log"
            mkdir -p "${RUN_DIR}"

            echo "[${RUN_IDX}/${TOTAL_RUNS}] Running ${RUN_TAG}"

            cmd=(
                python -m micro.run_analysis
                --dataset "${DATASET}"
                --snapshot "${snapshot}"
                --edge-label-option "${option}"
                --negative-edge-weight "${neg_weight}"
                --spectral-normalisation "${SPECTRAL_NORMALISATION}"
                --output-dir "${RUN_DIR}"
            )

            if [[ "${RECLUSTER_LARGEST}" == "1" || "${RECLUSTER_LARGEST}" == "true" ]]; then
                cmd+=(--recluster-largest)
            fi
            if [[ "${SAVE_PLOTS}" == "1" || "${SAVE_PLOTS}" == "true" ]]; then
                cmd+=(--save-plots)
            fi

            "${cmd[@]}" 2>&1 | tee "${LOG_FILE}"
        done
    done
done

echo "Completed ${TOTAL_RUNS} runs."
echo "Results directory: ${RESULT_ROOT}"
