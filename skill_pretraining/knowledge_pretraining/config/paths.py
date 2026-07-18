"""Centralized input and output paths.

Edit this file to relocate datasets, models, tool libraries, or generated files.
The values below intentionally preserve every path from the original project.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

# Input: generated task-graph dataset directory.
DATA_DIR = Path("/models/Qwen/lyq_jsac_2026_03_17/new_code/new_llm/newcode0411/deepseek_intent_dag_dataset_500")
DATASET_FILE_PATTERNS = ("dataset_*.json", "sample_*.json")
DATASET_METADATA_FILES = ("pool_metadata.json", "dataset_pool_metadata.json")

# Input: local Qwen model directories, checked in this order.
QWEN_MODEL_PATH = "/models/Qwen"
QWEN_MODEL_PATH_FALLBACK = "/model/Qwen"

# Input: optional pretrained knowledge weights and memory.
PRETRAINED_KNOWLEDGE_MODEL_PATH = Path(
    "/models/Qwen/lyq_jsac_2026_03_17/new_code/new_llm/newcode0411/new_0419/"
    "tmpjyz_100/knowledge_bootstrap_runs/knowledge_heuristic_bootstrap_20260420_230745/"
    "knowledge_aspect_adapter.pt"
)
OPTIONAL_PRETRAINED_KNOWLEDGE_MEMORY_PATH: Optional[Path] = None

# Inputs: dynamic tool libraries. Keep the order unchanged for deterministic merging.
TOOL_LIBRARY_PATHS = [
    Path("/models/Qwen/lyq_jsac_2026_03_17/new_code/new_llm/newcode0411/new_0419/new_0425_api/tool/api_tool_0_500/subset_api_schema_retry_20260427_141944/tool_library.json"),
    Path("/models/Qwen/lyq_jsac_2026_03_17/new_code/new_llm/newcode0411/new_0419/new_0425_api/tool/api_tool_500_999/subset_api_schema_retry_20260427_141947/tool_library.json"),
    Path("/models/Qwen/lyq_jsac_2026_03_17/new_code/new_llm/newcode0411/new_0419/new_0425_api/tool_new/api_tool_1000_1499/subset_api_schema_retry_20260427_164323/tool_library.json"),
    Path("/models/Qwen/lyq_jsac_2026_03_17/new_code/new_llm/newcode0411/new_0419/new_0425_api/tool/api_tool_1500_1999/subset_api_schema_retry_20260427_141953/tool_library.json"),
]

# Input fallback: relative to the project root, matching the original location.
LOCAL_TOOL_LIBRARY_PATH = Path("./tool_library.json")

# Output: run root and file names. Relative paths remain relative to the launch directory.
OUTPUT_ROOT = Path("./knowledge_dynamic_tool_runs_c")
BASE_RUN_NAME = "knowledge_dynamic_tools_heuristic"
ABLATION_RUN_NAME_BASE = "knowledge_dynamic_tools_heuristic_ablation"
LOSS_CSV_FILE = "loss.csv"
BEST_MODEL_FILE = "best.pt"
MEMORY_FILE = "memory.json"


def get_ablation_output_root() -> Path:
    """Return the original output root or its existing environment override."""

    return Path(os.environ.get("CMASD_ABLATION_OUTPUT_ROOT", str(OUTPUT_ROOT)))

