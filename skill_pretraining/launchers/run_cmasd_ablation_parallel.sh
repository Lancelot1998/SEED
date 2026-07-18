#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
export PYTHONUNBUFFERED=1
MODES=(${CMASD_ABLATION_MODES:-${DEFAULT_CMASD_MODES}})
PIDS=()
for MODE in "${MODES[@]}"; do
  echo "[cmasd-ablation:parallel] launch mode=${MODE}"
  (CMASD_ABLATION_MODE="${MODE}" python "${ABLATION_ENTRY}" --mode "${MODE}" "$@") &
  PIDS+=("$!")
done
for PID in "${PIDS[@]}"; do
  wait "${PID}"
done
echo "[cmasd-ablation:parallel] all modes done"
