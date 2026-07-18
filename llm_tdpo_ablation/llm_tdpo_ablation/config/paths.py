"""Single source of truth for all default input and output paths.

Every value preserves the original code. Edit this file when relocating the
dataset, Qwen model, knowledge checkpoints, memories, tool libraries, or
experiment outputs. Environment-variable overrides retain their original names.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Final, List, Optional, Tuple


# Shared model and dataset inputs.
QWEN_MODEL_PATH: Final[str] = "/models/Qwen"
QWEN_MODEL_PATH_FALLBACK: Final[str] = "/model/Qwen"
DATA_DIR: Final[Path] = Path(
    "/models/Qwen/lyq_jsac_2026_03_17/new_code/new_llm/newcode0411/"
    "deepseek_intent_dag_dataset_500"
)
DATASET_FILE_PATTERNS: Final[Tuple[str, ...]] = ("dataset_*.json", "sample_*.json")
DATASET_METADATA_FILES: Final[Tuple[str, ...]] = (
    "pool_metadata.json",
    "dataset_pool_metadata.json",
)

# Knowledge inputs used by the no-skill and blank-direct variants.
BASE_KNOWLEDGE_MODEL_PATH: Final[Path] = Path(
    "/models/Qwen/lyq_jsac_2026_03_17/new_code/new_llm/newcode0411/new_0419/"
    "new_0503_yes/knowledge_dynamic_tool_runs_c/"
    "knowledge_dynamic_tools_heuristic_20260504_131050/best.pt"
)
BASE_KNOWLEDGE_MEMORY_PATH: Final[Optional[Path]] = (  # type: ignore[assignment]
    "/models/Qwen/lyq_jsac_2026_03_17/new_code/new_llm/newcode0411/new_0419/"
    "new_0503_yes/knowledge_dynamic_tool_runs_c/"
    "knowledge_dynamic_tools_heuristic_20260504_131050/memory.json"
)

# Knowledge inputs used by the blank-LLM with skill-memory variant.
SKILL_KNOWLEDGE_MODEL_PATH: Final[Path] = Path(
    "/models/Qwen/lyq_jsac_2026_03_17/new_code/new_llm/newcode0411/new_0419/"
    "new_0509_skill/knowledge_dynamic_tool_runs_c/"
    "knowledge_dynamic_tools_heuristic_20260509_171515/best.pt"
)
SKILL_KNOWLEDGE_MEMORY_PATH: Final[Optional[Path]] = (  # type: ignore[assignment]
    "/models/Qwen/lyq_jsac_2026_03_17/new_code/new_llm/newcode0411/new_0419/"
    "new_0509_skill/knowledge_dynamic_tool_runs_c/"
    "knowledge_dynamic_tools_heuristic_20260509_171515/memory.json"
)

# Optional external memory shared by the original runners.
EXTERNAL_MEMORY_LIBRARY_PATH: Final[Optional[Path]] = (  # type: ignore[assignment]
    "/models/Qwen/lyq_jsac_2026_03_17/new_code/new_llm/newcode0411/new_0419/"
    "new_0503_yes/knowledge_dynamic_tool_runs_c/"
    "knowledge_dynamic_tools_heuristic_20260504_131050/memory.json"
)

# Dynamic tool-library inputs.
TOOL_LIBRARY_PATHS: Final[List[Path]] = [
    Path(
        "/models/Qwen/lyq_jsac_2026_03_17/new_code/new_llm/newcode0411/new_0419/"
        "new_0425_api/tool/api_tool_0_500/"
        "subset_api_schema_retry_20260427_141944/tool_library.json"
    ),
    Path(
        "/models/Qwen/lyq_jsac_2026_03_17/new_code/new_llm/newcode0411/new_0419/"
        "new_0425_api/tool/api_tool_500_999/"
        "subset_api_schema_retry_20260427_141947/tool_library.json"
    ),
    Path(
        "/models/Qwen/lyq_jsac_2026_03_17/new_code/new_llm/newcode0411/new_0419/"
        "new_0425_api/tool_new/api_tool_1000_1499/"
        "subset_api_schema_retry_20260427_164323/tool_library.json"
    ),
    Path(
        "/models/Qwen/lyq_jsac_2026_03_17/new_code/new_llm/newcode0411/new_0419/"
        "new_0425_api/tool/api_tool_1500_1999/"
        "subset_api_schema_retry_20260427_141953/tool_library.json"
    ),
]
# The runner resolves this relative path against the project root, matching the
# original THIS_DIR / "tool_library.json" behavior.
LOCAL_TOOL_LIBRARY_PATH: Final[Path] = Path("./tool_library.json")

# Policy offload outputs.
POLICY_OFFLOAD_DIR: Final[Path] = Path("./tdpo_offload")
RUN_OFFLOAD_DIR_NAME: Final[str] = "tdpo_offload"

# Variant-specific output roots and run names. The two blank-LLM variants keep
# the original TDPO_OUTPUT_ROOT and TDPO_RUN_NAME environment overrides.
NO_SKILL_OUTPUT_ROOT: Final[Path] = Path("./tdpo_decision_runs_batch_fast_no")
NO_SKILL_RUN_NAME: Final[str] = "qwen7b_tdpo_no_skill"
BLANK_DIRECT_OUTPUT_ROOT_DEFAULT: Final[str] = "./llm_ablation_runs/direct"
BLANK_DIRECT_RUN_NAME_DEFAULT: Final[str] = "ablation_blank_llm_direct_generation"
SKILL_MEMORY_OUTPUT_ROOT_DEFAULT: Final[str] = "./llm_ablation_runs/skill_memory"
SKILL_MEMORY_RUN_NAME_DEFAULT: Final[str] = (
    "ablation_blank_llm_skill_memory_direct_generation"
)


def blank_direct_output_root() -> Path:
    """Resolve the blank-direct output root with the original override."""
    return Path(os.environ.get("TDPO_OUTPUT_ROOT", BLANK_DIRECT_OUTPUT_ROOT_DEFAULT))


def blank_direct_run_name() -> str:
    """Resolve the blank-direct run name with the original override."""
    return os.environ.get("TDPO_RUN_NAME", BLANK_DIRECT_RUN_NAME_DEFAULT)


def skill_memory_output_root() -> Path:
    """Resolve the skill-memory output root with the original override."""
    return Path(os.environ.get("TDPO_OUTPUT_ROOT", SKILL_MEMORY_OUTPUT_ROOT_DEFAULT))


def skill_memory_run_name() -> str:
    """Resolve the skill-memory run name with the original override."""
    return os.environ.get("TDPO_RUN_NAME", SKILL_MEMORY_RUN_NAME_DEFAULT)


# Run-local directories and output filenames.
CHECKPOINT_DIR_NAME: Final[str] = "ckpts"
RECORDS_DIR_NAME: Final[str] = "records"
PAIRS_DIR_NAME: Final[str] = "pairs"
TRACES_DIR_NAME: Final[str] = "traces"
KNOWLEDGE_DIR_NAME: Final[str] = "knowledge"
DATASET_SCAN_FILE: Final[str] = "dataset_scan.json"
RUN_CONFIG_FILE: Final[str] = "run_config.json"
KNOWLEDGE_METRICS_FILE: Final[str] = "knowledge_metrics.csv"
EVAL_METRICS_FILE: Final[str] = "eval_metrics.csv"
ENERGY_FILE: Final[str] = "energy.csv"
DELAY_FILE: Final[str] = "delay.csv"
EPOCH_METRICS_FILE: Final[str] = "epoch_metrics.csv"
LATEST_SUMMARY_FILE: Final[str] = "latest_summary.json"
FINAL_RESULT_FILE: Final[str] = "final_result.json"
TRACE_FILE: Final[str] = "traces.jsonl"
PAIR_FILE: Final[str] = "pairs.jsonl"
BEST_CHECKPOINT_FILE: Final[str] = "best.pt"
KNOWLEDGE_MEMORY_FILE: Final[str] = "memory.json"
PRETRAINED_INPUT_COPY_FILE: Final[str] = "pretrained_input_copy.pt"
LOADED_KNOWLEDGE_INFO_FILE: Final[str] = "loaded_knowledge_info.json"
MEMORY_BOOTSTRAP_SUMMARY_FILE: Final[str] = "memory_bootstrap_summary.json"
MEMORY_BOOTSTRAP_TASK_RECORDS_FILE: Final[str] = "memory_bootstrap_task_records.json"
MEMORY_BOOTSTRAP_STATS_FILE: Final[str] = "memory_bootstrap_stats.json"
RELEASE_BEFORE_TDPO_FILE: Final[str] = "release_before_tdpo.json"


def checkpoint_epoch_file(epoch: int) -> str:
    return f"tdpo_policy_epoch_{int(epoch):03d}.pt"


def task_records_epoch_file(epoch: int) -> str:
    return f"task_records_epoch_{int(epoch):03d}.json"


def knowledge_update_epoch_file(epoch: int) -> str:
    return f"update_stats_epoch_{int(epoch):03d}.json"


def prompt_samples_epoch_file(epoch: int) -> str:
    return f"prompt_samples_epoch_{int(epoch):03d}.json"


# Prompt-invalid diagnostic outputs and compatibility entry names.
PROMPT_INVALID_OUTPUT_ROOT_DEFAULT: Final[str] = (
    "./llm_ablation_runs/skill_memory_prompt_invalid_test"
)
PROMPT_INVALID_TEST_DIR_PREFIX: Final[str] = "prompt_invalid_two_epoch"
PROMPT_INVALID_TEST_CONFIG_FILE: Final[str] = "test_config.json"
PROMPT_INVALID_ALL_EPOCHS_FILE: Final[str] = "invalid_summary_all_epochs.json"


def prompt_invalid_output_root() -> Path:
    """Resolve the diagnostic root with its original environment override."""
    return Path(
        os.environ.get(
            "PROMPT_INVALID_TEST_OUTPUT_ROOT",
            PROMPT_INVALID_OUTPUT_ROOT_DEFAULT,
        )
    ).expanduser()


def invalid_actions_epoch_file(epoch: int) -> str:
    return f"invalid_actions_epoch_{int(epoch):03d}.jsonl"


def all_actions_epoch_file(epoch: int) -> str:
    return f"all_actions_epoch_{int(epoch):03d}.jsonl"


def truncated_prompt_samples_epoch_file(epoch: int) -> str:
    return f"prompt_after_max_length_samples_epoch_{int(epoch):03d}.json"


def invalid_summary_epoch_file(epoch: int) -> str:
    return f"invalid_summary_epoch_{int(epoch):03d}.json"
