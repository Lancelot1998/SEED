#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-${CMASD_ABLATION_MODE:-single_modal_numeric}}"
shift || true
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
export PYTHONUNBUFFERED=1
export CMASD_ABLATION_MODE="${MODE}"
python "${ABLATION_ENTRY}" --mode "${MODE}" "$@"
