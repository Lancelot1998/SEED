"""Single source of truth for every default input and output path.

The values intentionally preserve the paths used by the original code. Edit
this file when moving datasets, models, tool libraries, memories, or outputs.
Command-line arguments can still override the defaults exposed by test tools.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final, List, Optional, Tuple


# Model inputs. Keep the original primary and fallback locations unchanged.
QWEN_MODEL_PATH: Final[str] = "/models/Qwen"
QWEN_MODEL_PATH_FALLBACK: Final[str] = "/model/Qwen"

# Dataset inputs. Patterns and metadata names are centralized because they
# control which files are read from DATA_DIR.
DATA_DIR: Final[Path] = Path(
    "/models/Qwen/lyq_jsac_2026_03_17/new_code/new_llm/newcode0411/"
    "deepseek_intent_dag_dataset_500"
)
DATASET_FILE_PATTERNS: Final[Tuple[str, ...]] = ("dataset_*.json", "sample_*.json")
DATASET_METADATA_FILES: Final[Tuple[str, ...]] = (
    "pool_metadata.json",
    "dataset_pool_metadata.json",
)

# Pretrained knowledge and memory inputs.
PRETRAINED_KNOWLEDGE_MODEL_PATH: Final[Path] = Path(
    "/models/Qwen/lyq_jsac_2026_03_17/new_code/new_llm/newcode0411/new_0419/"
    "new_0509_skill/knowledge_dynamic_tool_runs_c/"
    "knowledge_dynamic_tools_heuristic_20260509_171515/best.pt"
)
OPTIONAL_PRETRAINED_KNOWLEDGE_MEMORY_PATH: Final[Optional[Path]] = (  # type: ignore[assignment]
    "/models/Qwen/lyq_jsac_2026_03_17/new_code/new_llm/newcode0411/new_0419/"
    "new_0509_skill/knowledge_dynamic_tool_runs_c/"
    "knowledge_dynamic_tools_heuristic_20260509_171515/memory.json"
)
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
# This relative fallback preserves the original "./tool_library.json" behavior.
LOCAL_TOOL_LIBRARY_PATH: Final[Path] = Path("./tool_library.json")

# Training outputs. Relative locations intentionally retain the original paths.
OUTPUT_ROOT: Final[Path] = Path("./tdpo_decision_runs_batch_fast_skill")
RUN_NAME: Final[str] = "qwen7b_tdpo_with_skill_memory"
POLICY_OFFLOAD_DIR: Final[Path] = Path("./tdpo_offload")
RUN_OFFLOAD_DIR_NAME: Final[str] = "tdpo_offload"

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
    """Return the unchanged per-epoch checkpoint filename."""
    return f"tdpo_policy_epoch_{int(epoch):03d}.pt"


def task_records_epoch_file(epoch: int) -> str:
    """Return the unchanged per-epoch task-record filename."""
    return f"task_records_epoch_{int(epoch):03d}.json"


def knowledge_update_epoch_file(epoch: int) -> str:
    """Return the unchanged per-epoch knowledge-update filename."""
    return f"update_stats_epoch_{int(epoch):03d}.json"


def prompt_samples_epoch_file(epoch: int) -> str:
    """Return the unchanged prompt-sample filename."""
    return f"prompt_samples_epoch_{int(epoch):03d}.json"


# Offline rematching inputs and output.
REMATCH_TRACE_PATH: Final[Path] = Path(
    "/models/Qwen/lyq_jsac_2026_03_17/new_code/new_llm/newcode0411/new_0419/"
    "new_0509_all/llm_tdpo_skill_fast1/tdpo_decision_runs_batch_fast_skill/"
    "qwen7b_tdpo_with_skill_memory_20260511_140148/traces/traces.jsonl"
)
REMATCH_NEW_MEMORY_PATH: Final[Path] = Path(
    "/models/Qwen/lyq_jsac_2026_03_17/new_code/new_llm/newcode0411/new_0419/"
    "new_0509_all/llm_tdpo_skill_fast1/tdpo_decision_runs_batch_fast_skill/"
    "qwen7b_tdpo_with_skill_memory_20260511_140148/knowledge/memory.json"
)
REMATCH_OLD_MEMORY_PATH: Final[Path] = Path(
    "/models/Qwen/lyq_jsac_2026_03_17/new_code/new_llm/newcode0411/new_0419/"
    "new_0509_skill/knowledge_dynamic_tool_runs_c/"
    "knowledge_dynamic_tools_heuristic_20260509_171515/memory.json"
)
REMATCH_OUTPUT_PATH: Final[Path] = Path("./epoch75_task1090_node_skill_matches.json")

# Package implementation files used by dynamically loaded diagnostics.
# These paths replace removed root compatibility modules without changing behavior.
TRAIN_ENTRY_FILE: Final[str] = "./llm_tdpo/training/runner.py"
CURRENT_POLICY_ENTRY_FILE: Final[str] = "./llm_tdpo/policy/tdpo.py"
LEGACY_POLICY_ENTRY_FILE: Final[str] = "./llm_tdpo/policy/legacy_tdpo.py"
TASK_ENV_ENTRY_FILE: Final[str] = "./llm_tdpo/environment/task_env.py"
KNOWLEDGE_ENTRY_FILE: Final[str] = "./llm_tdpo/knowledge/adapter.py"

# Diagnostic output defaults.
POLICY_LEGALITY_OUTPUT_DIR: Final[Path] = Path("./policy_legality_test_runs")
POLICY_LEGALITY_REPORT_FILE: Final[str] = "policy_legality_report.json"
POLICY_DECISIONS_FILE: Final[str] = "policy_decisions.jsonl"
POLICY_TASK_RECORDS_FILE: Final[str] = "policy_task_records.json"
PROMPT_COMPARISON_REPORT_FILE: Final[str] = "prompt_old_new_examples.json"
PROMPT_LENGTH_REPORT_FILE: Final[str] = "prompt_token_length_report.json"
