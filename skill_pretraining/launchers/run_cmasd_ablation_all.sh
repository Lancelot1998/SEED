#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODE="${1:-serial}"
shift || true
case "${MODE}" in
  serial)
    bash "${SCRIPT_DIR}/run_cmasd_ablation_serial.sh" "$@"
    ;;
  parallel)
    bash "${SCRIPT_DIR}/run_cmasd_ablation_parallel.sh" "$@"
    ;;
  parallel4|4gpu)
    bash "${SCRIPT_DIR}/run_cmasd_ablation_parallel_4gpu.sh" "$@"
    ;;
  single|one)
    bash "${SCRIPT_DIR}/run_cmasd_ablation_single.sh" "$@"
    ;;
  *)
    echo "Usage: $0 {serial|parallel|parallel4|single} [extra train args]" >&2
    exit 2
    ;;
esac
