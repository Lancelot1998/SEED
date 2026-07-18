#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
export PYTHONUNBUFFERED=1
MODES=(${CMASD_ABLATION_MODES:-${DEFAULT_CMASD_MODES}})
GPU_IDS=(${CMASD_ABLATION_GPU_IDS:-${DEFAULT_CMASD_GPU_IDS}})
MAX_JOBS=${#GPU_IDS[@]}
PIDS=()
NAMES=()

wait_one_if_full() {
  while [ "${#PIDS[@]}" -ge "${MAX_JOBS}" ]; do
    wait "${PIDS[0]}"
    PIDS=("${PIDS[@]:1}")
    NAMES=("${NAMES[@]:1}")
  done
}

for IDX in "${!MODES[@]}"; do
  wait_one_if_full
  MODE="${MODES[$IDX]}"
  GPU="${GPU_IDS[$((IDX % MAX_JOBS))]}"
  echo "[cmasd-ablation:parallel4] launch mode=${MODE} gpu=${GPU}"
  (CUDA_VISIBLE_DEVICES="${GPU}" CMASD_ABLATION_MODE="${MODE}" python "${ABLATION_ENTRY}" --mode "${MODE}" "$@") &
  PIDS+=("$!")
  NAMES+=("${MODE}")
done
for PID in "${PIDS[@]}"; do
  wait "${PID}"
done
echo "[cmasd-ablation:parallel4] all modes done"
