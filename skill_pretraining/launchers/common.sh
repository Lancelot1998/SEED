#!/usr/bin/env bash

# Shared launcher locations and defaults. Paths are resolved from this file so the
# launchers work from any current working directory.
LAUNCHER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${LAUNCHER_DIR}/.." && pwd)"
ABLATION_ENTRY="${PROJECT_ROOT}/train_knowledge_dynamic_tools_ablation.py"

DEFAULT_CMASD_MODES="single_modal_numeric no_fusion no_topo no_wire action_out_only"
DEFAULT_CMASD_GPU_IDS="0 1 2 3"

