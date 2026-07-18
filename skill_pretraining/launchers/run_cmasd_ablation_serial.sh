#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
export PYTHONUNBUFFERED=1
MODES=(${CMASD_ABLATION_MODES:-${DEFAULT_CMASD_MODES}})
for MODE in "${MODES[@]}"; do
  echo "[cmasd-ablation:serial] start mode=${MODE}"
  CMASD_ABLATION_MODE="${MODE}" python "${ABLATION_ENTRY}" --mode "${MODE}" "$@"
  echo "[cmasd-ablation:serial] done mode=${MODE}"
done
