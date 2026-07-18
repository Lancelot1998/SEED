"""Centralized input and output paths for every experiment.

All values preserve the original source paths. Edit this file only when moving
datasets, models, checkpoints, memories, tool libraries, or output directories.
"""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LLM_TOKENIZER_PATH = None
EVAL_OVERRIDE_QWEN_MODEL_PATH = ""
EVAL_OVERRIDE_QWEN_TOKENIZER_PATH = ""
SAME_SKILL_KNOWLEDGE_BEST_PATH = ""
SAME_SKILL_KNOWLEDGE_MEMORY_PATH = ""
KNOWLEDGE_CASE_DIR_TEMPLATE = "case_{case_idx:03d}_arrival_{arrival_tag}_deadline_{deadline_tag}"

# Paths used by experiment_suite/knowledge_training/bind_sep_runner.py.
KNOWLEDGE_BIND_DATA_DIR = Path("/models/Qwen/lyq_jsac_2026_03_17/new_code/new_llm/newcode0411/deepseek_intent_dag_dataset_500")
KNOWLEDGE_BIND_DATASET_FILE_PATTERNS = ("dataset_*.json", "sample_*.json")
KNOWLEDGE_BIND_DATASET_METADATA_FILES = ("pool_metadata.json", "dataset_pool_metadata.json")
KNOWLEDGE_BIND_QWEN_MODEL_PATH = "/models/Qwen"
KNOWLEDGE_BIND_QWEN_MODEL_PATH_FALLBACK = "/model/Qwen"
KNOWLEDGE_BIND_PRETRAINED_KNOWLEDGE_MODEL_PATH = Path(
    "/models/Qwen/lyq_jsac_2026_03_17/new_code/new_llm/newcode0411/new_0419/"
    "tmpjyz_100/knowledge_bootstrap_runs/knowledge_heuristic_bootstrap_20260420_230745/"
    "knowledge_aspect_adapter.pt"
)
KNOWLEDGE_BIND_OPTIONAL_PRETRAINED_KNOWLEDGE_MEMORY_PATH = None
KNOWLEDGE_BIND_OUTPUT_ROOT = Path("./knowledge_env_0")
KNOWLEDGE_BIND_TOOL_LIBRARY_PATHS = [
    Path("/models/Qwen/lyq_jsac_2026_03_17/new_code/new_llm/newcode0411/new_0419/new_0425_api/tool/api_tool_0_500/subset_api_schema_retry_20260427_141944/tool_library.json"),
    Path("/models/Qwen/lyq_jsac_2026_03_17/new_code/new_llm/newcode0411/new_0419/new_0425_api/tool/api_tool_500_999/subset_api_schema_retry_20260427_141947/tool_library.json"),
    Path("/models/Qwen/lyq_jsac_2026_03_17/new_code/new_llm/newcode0411/new_0419/new_0425_api/tool_new/api_tool_1000_1499/subset_api_schema_retry_20260427_164323/tool_library.json"),
    Path("/models/Qwen/lyq_jsac_2026_03_17/new_code/new_llm/newcode0411/new_0419/new_0425_api/tool/api_tool_1500_1999/subset_api_schema_retry_20260427_141953/tool_library.json"),
]

# Paths used by experiment_suite/knowledge_training/deadline_sweep.py.
KNOWLEDGE_DEADLINE_OUTPUT_ROOT = Path("./knowledge_env_dead")

# Paths used by experiment_suite/knowledge_training/time_sweep.py.
KNOWLEDGE_TIME_OUTPUT_ROOT = Path("./knowledge_env_time")

# Paths used by experiment_suite/tdpo_skill/trainer.py.
TDPO_SKILL_TRAIN_DATA_DIR = Path("/models/Qwen/lyq_jsac_2026_03_17/new_code/new_llm/newcode0411/deepseek_intent_dag_dataset_500")
TDPO_SKILL_TRAIN_DATASET_FILE_PATTERNS = ("dataset_*.json", "sample_*.json")
TDPO_SKILL_TRAIN_DATASET_METADATA_FILES = ("pool_metadata.json", "dataset_pool_metadata.json")
TDPO_SKILL_TRAIN_QWEN_MODEL_PATH = "/models/Qwen"
TDPO_SKILL_TRAIN_QWEN_MODEL_PATH_FALLBACK = "/model/Qwen"
TDPO_SKILL_TRAIN_PRETRAINED_KNOWLEDGE_MODEL_PATH = Path( "/models/Qwen/lyq_jsac_2026_03_17/new_code/new_llm/newcode0411/new_0419/new_0509_skill/knowledge_dynamic_tool_runs_c/knowledge_dynamic_tools_heuristic_20260509_171515/best.pt"
)
TDPO_SKILL_TRAIN_OPTIONAL_PRETRAINED_KNOWLEDGE_MEMORY_PATH = "/models/Qwen/lyq_jsac_2026_03_17/new_code/new_llm/newcode0411/new_0419/new_0509_skill/knowledge_dynamic_tool_runs_c/knowledge_dynamic_tools_heuristic_20260509_171515/memory.json"
TDPO_SKILL_TRAIN_OUTPUT_ROOT = Path("./tdpo_decision_runs_batch_fast_skill")
TDPO_SKILL_TRAIN_TOOL_LIBRARY_PATHS = [
    Path("/models/Qwen/lyq_jsac_2026_03_17/new_code/new_llm/newcode0411/new_0419/new_0425_api/tool/api_tool_0_500/subset_api_schema_retry_20260427_141944/tool_library.json"),
    Path("/models/Qwen/lyq_jsac_2026_03_17/new_code/new_llm/newcode0411/new_0419/new_0425_api/tool/api_tool_500_999/subset_api_schema_retry_20260427_141947/tool_library.json"),
    Path("/models/Qwen/lyq_jsac_2026_03_17/new_code/new_llm/newcode0411/new_0419/new_0425_api/tool_new/api_tool_1000_1499/subset_api_schema_retry_20260427_164323/tool_library.json"),
    Path("/models/Qwen/lyq_jsac_2026_03_17/new_code/new_llm/newcode0411/new_0419/new_0425_api/tool/api_tool_1500_1999/subset_api_schema_retry_20260427_141953/tool_library.json"),
]
TDPO_SKILL_TRAIN_EXTERNAL_MEMORY_LIBRARY_PATH = "/models/Qwen/lyq_jsac_2026_03_17/new_code/new_llm/newcode0411/new_0419/new_0503_yes/knowledge_dynamic_tool_runs_c/knowledge_dynamic_tools_heuristic_20260504_131050/memory.json"

# Paths used by experiment_suite/tdpo_no_skill/trainer.py.
TDPO_NO_SKILL_TRAIN_DATA_DIR = Path("/models/Qwen/lyq_jsac_2026_03_17/new_code/new_llm/newcode0411/deepseek_intent_dag_dataset_500")
TDPO_NO_SKILL_TRAIN_DATASET_FILE_PATTERNS = ("dataset_*.json", "sample_*.json")
TDPO_NO_SKILL_TRAIN_DATASET_METADATA_FILES = ("pool_metadata.json", "dataset_pool_metadata.json")
TDPO_NO_SKILL_TRAIN_QWEN_MODEL_PATH = "/models/Qwen"
TDPO_NO_SKILL_TRAIN_QWEN_MODEL_PATH_FALLBACK = "/model/Qwen"
TDPO_NO_SKILL_TRAIN_PRETRAINED_KNOWLEDGE_MODEL_PATH = Path( "/models/Qwen/lyq_jsac_2026_03_17/new_code/new_llm/newcode0411/new_0419/new_0503_yes/knowledge_dynamic_tool_runs_c/knowledge_dynamic_tools_heuristic_20260504_131050/best.pt"
)
TDPO_NO_SKILL_TRAIN_OPTIONAL_PRETRAINED_KNOWLEDGE_MEMORY_PATH = "/models/Qwen/lyq_jsac_2026_03_17/new_code/new_llm/newcode0411/new_0419/new_0503_yes/knowledge_dynamic_tool_runs_c/knowledge_dynamic_tools_heuristic_20260504_131050/memory.json"
TDPO_NO_SKILL_TRAIN_OUTPUT_ROOT = Path("./tdpo_decision_runs_batch_fast_no")
TDPO_NO_SKILL_TRAIN_TOOL_LIBRARY_PATHS = [
    Path("/models/Qwen/lyq_jsac_2026_03_17/new_code/new_llm/newcode0411/new_0419/new_0425_api/tool/api_tool_0_500/subset_api_schema_retry_20260427_141944/tool_library.json"),
    Path("/models/Qwen/lyq_jsac_2026_03_17/new_code/new_llm/newcode0411/new_0419/new_0425_api/tool/api_tool_500_999/subset_api_schema_retry_20260427_141947/tool_library.json"),
    Path("/models/Qwen/lyq_jsac_2026_03_17/new_code/new_llm/newcode0411/new_0419/new_0425_api/tool_new/api_tool_1000_1499/subset_api_schema_retry_20260427_164323/tool_library.json"),
    Path("/models/Qwen/lyq_jsac_2026_03_17/new_code/new_llm/newcode0411/new_0419/new_0425_api/tool/api_tool_1500_1999/subset_api_schema_retry_20260427_141953/tool_library.json"),
]
TDPO_NO_SKILL_TRAIN_EXTERNAL_MEMORY_LIBRARY_PATH = "/models/Qwen/lyq_jsac_2026_03_17/new_code/new_llm/newcode0411/new_0419/new_0503_yes/knowledge_dynamic_tool_runs_c/knowledge_dynamic_tools_heuristic_20260504_131050/memory.json"

# Paths used by experiment_suite/tdpo_skill/evaluation/deadline_sweep.py.
TDPO_SKILL_EVAL_DEADLINE_OUTPUT_ROOT = Path("./tdpo_eval_knowledge_env_sweep_dead")
TDPO_SKILL_EVAL_DEADLINE_DATA_DIR = Path("/models/Qwen/lyq_jsac_2026_03_17/new_code/new_llm/newcode0411/deepseek_intent_dag_dataset_500")
TDPO_SKILL_EVAL_DEADLINE_TDPO_CHECKPOINTS = [
    {"epoch": 100, "path": r"/models/Qwen/lyq_jsac_2026_03_17/new_code/new_llm/newcode0411/new_0419/new_0509_all/llm_tdpo_skill_fast1/tdpo_decision_runs_batch_fast_skill/qwen7b_tdpo_with_skill_memory_20260511_140148/ckpts/best.pt"},
    # {"epoch": 80, "path": r"/path/to/tdpo_run/ckpts/tdpo_policy_epoch_080.pt"},
]
TDPO_SKILL_EVAL_DEADLINE_TDPO_CKPT_DIR = Path("")
TDPO_SKILL_EVAL_DEADLINE_TDPO_CKPT_TEMPLATE = "tdpo_policy_epoch_{epoch:03d}.pt"
TDPO_SKILL_EVAL_DEADLINE_TDPO_BEST_CKPT_NAME = "best.pt"
TDPO_SKILL_EVAL_DEADLINE_KNOWLEDGE_CASES = [
    # {
    #     "case_name": "arrival_1_deadline_0p25",
    #     "knowledge_best_path": r"/models/Qwen/lyq_jsac_2026_03_17/new_code/new_llm/newcode0411/new_0419/new_0515_skill/knowledge_env_dead/knowledge_dynamic_tools_heuristic_sweep_20260515_010543/case_001_arrival_1_deadline_0p25/best.pt",
    #     "knowledge_memory_path": r"/models/Qwen/lyq_jsac_2026_03_17/new_code/new_llm/newcode0411/new_0419/new_0515_skill/knowledge_env_dead/knowledge_dynamic_tools_heuristic_sweep_20260515_010543/case_001_arrival_1_deadline_0p25/memory.json",
    #     "knowledge_loss_path": r"/models/Qwen/lyq_jsac_2026_03_17/new_code/new_llm/newcode0411/new_0419/new_0515_skill/knowledge_env_dead/knowledge_dynamic_tools_heuristic_sweep_20260515_010543/case_001_arrival_1_deadline_0p25/loss.csv",
    #     "arrival_gap": 1.0,
    #     "task_deadline_multiplier": 0.25,
    # },
    # {
    #     "case_name": "arrival_1_deadline_0p75",
    #     "knowledge_best_path": r"/models/Qwen/lyq_jsac_2026_03_17/new_code/new_llm/newcode0411/new_0419/new_0515_skill/knowledge_env_dead/knowledge_dynamic_tools_heuristic_sweep_20260515_010543/case_002_arrival_1_deadline_0p75/best.pt",
    #     "knowledge_memory_path": r"/models/Qwen/lyq_jsac_2026_03_17/new_code/new_llm/newcode0411/new_0419/new_0515_skill/knowledge_env_dead/knowledge_dynamic_tools_heuristic_sweep_20260515_010543/case_002_arrival_1_deadline_0p75/memory.json",
    #     "knowledge_loss_path": r"/models/Qwen/lyq_jsac_2026_03_17/new_code/new_llm/newcode0411/new_0419/new_0515_skill/knowledge_env_dead/knowledge_dynamic_tools_heuristic_sweep_20260515_010543/case_002_arrival_1_deadline_0p75/loss.csv",
    #     "arrival_gap": 1.0,
    #     "task_deadline_multiplier": 0.75,
    # },
    # {
    #     "case_name": "arrival_1_deadline_1p25",
    #     "knowledge_best_path": r"/models/Qwen/lyq_jsac_2026_03_17/new_code/new_llm/newcode0411/new_0419/new_0515_skill/knowledge_env_dead/knowledge_dynamic_tools_heuristic_sweep_20260515_010543/case_003_arrival_1_deadline_1p25/best.pt",
    #     "knowledge_memory_path": r"/models/Qwen/lyq_jsac_2026_03_17/new_code/new_llm/newcode0411/new_0419/new_0515_skill/knowledge_env_dead/knowledge_dynamic_tools_heuristic_sweep_20260515_010543/case_003_arrival_1_deadline_1p25/memory.json",
    #     "knowledge_loss_path": r"/models/Qwen/lyq_jsac_2026_03_17/new_code/new_llm/newcode0411/new_0419/new_0515_skill/knowledge_env_dead/knowledge_dynamic_tools_heuristic_sweep_20260515_010543/case_003_arrival_1_deadline_1p25/loss.csv",
    #     "arrival_gap": 1.0,
    #     "task_deadline_multiplier": 1.25,
    # },
    # {
    #     "case_name": "arrival_1_deadline_1p5",
    #     "knowledge_best_path": r"/models/Qwen/lyq_jsac_2026_03_17/new_code/new_llm/newcode0411/new_0419/new_0515_skill/knowledge_env_dead/knowledge_dynamic_tools_heuristic_sweep_20260515_010543/case_004_arrival_1_deadline_1p5/best.pt",
    #     "knowledge_memory_path": r"/models/Qwen/lyq_jsac_2026_03_17/new_code/new_llm/newcode0411/new_0419/new_0515_skill/knowledge_env_dead/knowledge_dynamic_tools_heuristic_sweep_20260515_010543/case_004_arrival_1_deadline_1p5/memory.json",
    #     "knowledge_loss_path": r"/models/Qwen/lyq_jsac_2026_03_17/new_code/new_llm/newcode0411/new_0419/new_0515_skill/knowledge_env_dead/knowledge_dynamic_tools_heuristic_sweep_20260515_010543/case_004_arrival_1_deadline_1p5/loss.csv",
    #     "arrival_gap": 1.0,
    #     "task_deadline_multiplier": 1.5,
    # },
    # {
    #     "case_name": "arrival_1_deadline_2",
    #     "knowledge_best_path": r"/models/Qwen/lyq_jsac_2026_03_17/new_code/new_llm/newcode0411/new_0419/new_0515_skill/knowledge_env_dead/knowledge_dynamic_tools_heuristic_sweep_20260515_010543/case_005_arrival_1_deadline_2/best.pt",
    #     "knowledge_memory_path": r"/models/Qwen/lyq_jsac_2026_03_17/new_code/new_llm/newcode0411/new_0419/new_0515_skill/knowledge_env_dead/knowledge_dynamic_tools_heuristic_sweep_20260515_010543/case_005_arrival_1_deadline_2/memory.json",
    #     "knowledge_loss_path": r"/models/Qwen/lyq_jsac_2026_03_17/new_code/new_llm/newcode0411/new_0419/new_0515_skill/knowledge_env_dead/knowledge_dynamic_tools_heuristic_sweep_20260515_010543/case_005_arrival_1_deadline_2/loss.csv",
    #     "arrival_gap": 1.0,
    #     "task_deadline_multiplier": 2.0,
    # },
    {
        "case_name": "arrival_1_deadline_4",
        "knowledge_best_path": r"/models/Qwen/lyq_jsac_2026_03_17/new_code/new_llm/newcode0411/new_0419/new_0515_skill/knowledge_env_dead/knowledge_dynamic_tools_heuristic_sweep_20260515_173459/case_001_arrival_1_deadline_4/best.pt",
        "knowledge_memory_path": r"/models/Qwen/lyq_jsac_2026_03_17/new_code/new_llm/newcode0411/new_0419/new_0515_skill/knowledge_env_dead/knowledge_dynamic_tools_heuristic_sweep_20260515_173459/case_001_arrival_1_deadline_4/memory.json",
        "knowledge_loss_path": r"/models/Qwen/lyq_jsac_2026_03_17/new_code/new_llm/newcode0411/new_0419/new_0515_skill/knowledge_env_dead/knowledge_dynamic_tools_heuristic_sweep_20260515_173459/case_001_arrival_1_deadline_4/loss.csv",
        "arrival_gap": 1.0,
        "task_deadline_multiplier": 4.0,
    },
]
TDPO_SKILL_EVAL_DEADLINE_KNOWLEDGE_SWEEP_ROOT = Path("")
TDPO_SKILL_EVAL_DEADLINE_KNOWLEDGE_BEST_FILENAME = "best.pt"
TDPO_SKILL_EVAL_DEADLINE_KNOWLEDGE_MEMORY_FILENAME = "memory.json"

# Paths used by experiment_suite/tdpo_skill/evaluation/deadline_same_environment.py.
TDPO_SKILL_EVAL_DEADLINE_SAME_OUTPUT_ROOT = Path("./tdpo_eval_knowledge_env_sweep_dead_same")
TDPO_SKILL_EVAL_DEADLINE_SAME_DATA_DIR = Path("/models/Qwen/lyq_jsac_2026_03_17/new_code/new_llm/newcode0411/deepseek_intent_dag_dataset_500")
TDPO_SKILL_EVAL_DEADLINE_SAME_TDPO_CHECKPOINTS = [
    {"epoch": 100, "path": r"/models/Qwen/lyq_jsac_2026_03_17/new_code/new_llm/newcode0411/new_0419/new_0509_all/llm_tdpo_skill_fast1/tdpo_decision_runs_batch_fast_skill/qwen7b_tdpo_with_skill_memory_20260511_140148/ckpts/best.pt"},
    # {"epoch": 80, "path": r"/path/to/tdpo_run/ckpts/tdpo_policy_epoch_080.pt"},
]
TDPO_SKILL_EVAL_DEADLINE_SAME_TDPO_CKPT_DIR = Path("")
TDPO_SKILL_EVAL_DEADLINE_SAME_TDPO_CKPT_TEMPLATE = "tdpo_policy_epoch_{epoch:03d}.pt"
TDPO_SKILL_EVAL_DEADLINE_SAME_TDPO_BEST_CKPT_NAME = "best.pt"
TDPO_SKILL_EVAL_DEADLINE_SAME_KNOWLEDGE_CASES = [
    # {
    #     "case_name": "arrival_1_deadline_0p25",
    #     "knowledge_best_path": r"/models/Qwen/lyq_jsac_2026_03_17/new_code/new_llm/newcode0411/new_0419/new_0515_skill/knowledge_env_dead/knowledge_dynamic_tools_heuristic_sweep_20260515_010543/case_001_arrival_1_deadline_0p25/best.pt",
    #     "knowledge_memory_path": r"/models/Qwen/lyq_jsac_2026_03_17/new_code/new_llm/newcode0411/new_0419/new_0515_skill/knowledge_env_dead/knowledge_dynamic_tools_heuristic_sweep_20260515_010543/case_001_arrival_1_deadline_0p25/memory.json",
    #     "knowledge_loss_path": r"/models/Qwen/lyq_jsac_2026_03_17/new_code/new_llm/newcode0411/new_0419/new_0515_skill/knowledge_env_dead/knowledge_dynamic_tools_heuristic_sweep_20260515_010543/case_001_arrival_1_deadline_0p25/loss.csv",
    #     "arrival_gap": 1.0,
    #     "task_deadline_multiplier": 0.25,
    # },
    # {
    #     "case_name": "arrival_1_deadline_0p75",
    #     "knowledge_best_path": r"/models/Qwen/lyq_jsac_2026_03_17/new_code/new_llm/newcode0411/new_0419/new_0515_skill/knowledge_env_dead/knowledge_dynamic_tools_heuristic_sweep_20260515_010543/case_002_arrival_1_deadline_0p75/best.pt",
    #     "knowledge_memory_path": r"/models/Qwen/lyq_jsac_2026_03_17/new_code/new_llm/newcode0411/new_0419/new_0515_skill/knowledge_env_dead/knowledge_dynamic_tools_heuristic_sweep_20260515_010543/case_002_arrival_1_deadline_0p75/memory.json",
    #     "knowledge_loss_path": r"/models/Qwen/lyq_jsac_2026_03_17/new_code/new_llm/newcode0411/new_0419/new_0515_skill/knowledge_env_dead/knowledge_dynamic_tools_heuristic_sweep_20260515_010543/case_002_arrival_1_deadline_0p75/loss.csv",
    #     "arrival_gap": 1.0,
    #     "task_deadline_multiplier": 0.75,
    # },
    # {
    #     "case_name": "arrival_1_deadline_1p25",
    #     "knowledge_best_path": r"/models/Qwen/lyq_jsac_2026_03_17/new_code/new_llm/newcode0411/new_0419/new_0515_skill/knowledge_env_dead/knowledge_dynamic_tools_heuristic_sweep_20260515_010543/case_003_arrival_1_deadline_1p25/best.pt",
    #     "knowledge_memory_path": r"/models/Qwen/lyq_jsac_2026_03_17/new_code/new_llm/newcode0411/new_0419/new_0515_skill/knowledge_env_dead/knowledge_dynamic_tools_heuristic_sweep_20260515_010543/case_003_arrival_1_deadline_1p25/memory.json",
    #     "knowledge_loss_path": r"/models/Qwen/lyq_jsac_2026_03_17/new_code/new_llm/newcode0411/new_0419/new_0515_skill/knowledge_env_dead/knowledge_dynamic_tools_heuristic_sweep_20260515_010543/case_003_arrival_1_deadline_1p25/loss.csv",
    #     "arrival_gap": 1.0,
    #     "task_deadline_multiplier": 1.25,
    # },
    # {
    #     "case_name": "arrival_1_deadline_1p5",
    #     "knowledge_best_path": r"/models/Qwen/lyq_jsac_2026_03_17/new_code/new_llm/newcode0411/new_0419/new_0515_skill/knowledge_env_dead/knowledge_dynamic_tools_heuristic_sweep_20260515_010543/case_004_arrival_1_deadline_1p5/best.pt",
    #     "knowledge_memory_path": r"/models/Qwen/lyq_jsac_2026_03_17/new_code/new_llm/newcode0411/new_0419/new_0515_skill/knowledge_env_dead/knowledge_dynamic_tools_heuristic_sweep_20260515_010543/case_004_arrival_1_deadline_1p5/memory.json",
    #     "knowledge_loss_path": r"/models/Qwen/lyq_jsac_2026_03_17/new_code/new_llm/newcode0411/new_0419/new_0515_skill/knowledge_env_dead/knowledge_dynamic_tools_heuristic_sweep_20260515_010543/case_004_arrival_1_deadline_1p5/loss.csv",
    #     "arrival_gap": 1.0,
    #     "task_deadline_multiplier": 1.5,
    # },
    # {
    #     "case_name": "arrival_1_deadline_2",
    #     "knowledge_best_path": r"/models/Qwen/lyq_jsac_2026_03_17/new_code/new_llm/newcode0411/new_0419/new_0515_skill/knowledge_env_dead/knowledge_dynamic_tools_heuristic_sweep_20260515_010543/case_005_arrival_1_deadline_2/best.pt",
    #     "knowledge_memory_path": r"/models/Qwen/lyq_jsac_2026_03_17/new_code/new_llm/newcode0411/new_0419/new_0515_skill/knowledge_env_dead/knowledge_dynamic_tools_heuristic_sweep_20260515_010543/case_005_arrival_1_deadline_2/memory.json",
    #     "knowledge_loss_path": r"/models/Qwen/lyq_jsac_2026_03_17/new_code/new_llm/newcode0411/new_0419/new_0515_skill/knowledge_env_dead/knowledge_dynamic_tools_heuristic_sweep_20260515_010543/case_005_arrival_1_deadline_2/loss.csv",
    #     "arrival_gap": 1.0,
    #     "task_deadline_multiplier": 2.0,
    # },
    {
        "case_name": "arrival_1_deadline_4",
        "knowledge_best_path": r"/models/Qwen/lyq_jsac_2026_03_17/new_code/new_llm/newcode0411/new_0419/new_0515_skill/knowledge_env_dead/knowledge_dynamic_tools_heuristic_sweep_20260515_173459/case_001_arrival_1_deadline_4/best.pt",
        "knowledge_memory_path": r"/models/Qwen/lyq_jsac_2026_03_17/new_code/new_llm/newcode0411/new_0419/new_0515_skill/knowledge_env_dead/knowledge_dynamic_tools_heuristic_sweep_20260515_173459/case_001_arrival_1_deadline_4/memory.json",
        "knowledge_loss_path": r"/models/Qwen/lyq_jsac_2026_03_17/new_code/new_llm/newcode0411/new_0419/new_0515_skill/knowledge_env_dead/knowledge_dynamic_tools_heuristic_sweep_20260515_173459/case_001_arrival_1_deadline_4/loss.csv",
        "arrival_gap": 1.0,
        "task_deadline_multiplier": 4.0,
    },
]
TDPO_SKILL_EVAL_DEADLINE_SAME_KNOWLEDGE_SWEEP_ROOT = Path("")
TDPO_SKILL_EVAL_DEADLINE_SAME_KNOWLEDGE_BEST_FILENAME = "best.pt"
TDPO_SKILL_EVAL_DEADLINE_SAME_KNOWLEDGE_MEMORY_FILENAME = "memory.json"

# Paths used by experiment_suite/tdpo_skill/evaluation/time_sweep.py.
TDPO_SKILL_EVAL_TIME_OUTPUT_ROOT = Path("./tdpo_eval_knowledge_env_sweep")
TDPO_SKILL_EVAL_TIME_DATA_DIR = Path("/models/Qwen/lyq_jsac_2026_03_17/new_code/new_llm/newcode0411/deepseek_intent_dag_dataset_500")
TDPO_SKILL_EVAL_TIME_TDPO_CHECKPOINTS = [
    {"epoch": 100, "path": r"/models/Qwen/lyq_jsac_2026_03_17/new_code/new_llm/newcode0411/new_0419/new_0509_all/llm_tdpo_skill_fast1/tdpo_decision_runs_batch_fast_skill/qwen7b_tdpo_with_skill_memory_20260511_140148/ckpts/best.pt"},
    # {"epoch": 80, "path": r"/path/to/tdpo_run/ckpts/tdpo_policy_epoch_080.pt"},
]
TDPO_SKILL_EVAL_TIME_TDPO_CKPT_DIR = Path("")
TDPO_SKILL_EVAL_TIME_TDPO_CKPT_TEMPLATE = "tdpo_policy_epoch_{epoch:03d}.pt"
TDPO_SKILL_EVAL_TIME_TDPO_BEST_CKPT_NAME = "best.pt"
TDPO_SKILL_EVAL_TIME_KNOWLEDGE_CASES = [
    {
        "case_name": "arrival_0p25_deadline_2p6",
        "knowledge_best_path": r"/models/Qwen/lyq_jsac_2026_03_17/new_code/new_llm/newcode0411/new_0419/new_0515_skill/knowledge_env_time/knowledge_dynamic_tools_heuristic_sweep_20260515_005650/case_001_arrival_0p25_deadline_2p6/best.pt",
        "knowledge_memory_path": r"/models/Qwen/lyq_jsac_2026_03_17/new_code/new_llm/newcode0411/new_0419/new_0515_skill/knowledge_env_time/knowledge_dynamic_tools_heuristic_sweep_20260515_005650/case_001_arrival_0p25_deadline_2p6/memory.json",
        "arrival_gap": 0.25,
        "task_deadline_multiplier": 2.6,
    },
    {
        "case_name": "arrival_0p5_deadline_2p6",
        "knowledge_best_path": r"/models/Qwen/lyq_jsac_2026_03_17/new_code/new_llm/newcode0411/new_0419/new_0515_skill/knowledge_env_time/knowledge_dynamic_tools_heuristic_sweep_20260515_005650/case_002_arrival_0p5_deadline_2p6/best.pt",
        "knowledge_memory_path": r"/models/Qwen/lyq_jsac_2026_03_17/new_code/new_llm/newcode0411/new_0419/new_0515_skill/knowledge_env_time/knowledge_dynamic_tools_heuristic_sweep_20260515_005650/case_002_arrival_0p5_deadline_2p6/memory.json",
        "arrival_gap": 0.5,
        "task_deadline_multiplier": 2.6,
    },
    {
        "case_name": "arrival_0p75_deadline_2p6",
        "knowledge_best_path": r"/models/Qwen/lyq_jsac_2026_03_17/new_code/new_llm/newcode0411/new_0419/new_0515_skill/knowledge_env_time/knowledge_dynamic_tools_heuristic_sweep_20260515_005650/case_003_arrival_0p75_deadline_2p6/best.pt",
        "knowledge_memory_path": r"/models/Qwen/lyq_jsac_2026_03_17/new_code/new_llm/newcode0411/new_0419/new_0515_skill/knowledge_env_time/knowledge_dynamic_tools_heuristic_sweep_20260515_005650/case_003_arrival_0p75_deadline_2p6/memory.json",
        "arrival_gap": 0.75,
        "task_deadline_multiplier": 2.6,
    },
    {
        "case_name": "arrival_1_deadline_2p6",
        "knowledge_best_path": r"/models/Qwen/lyq_jsac_2026_03_17/new_code/new_llm/newcode0411/new_0419/new_0515_skill/knowledge_env_time/knowledge_dynamic_tools_heuristic_sweep_20260515_005650/case_004_arrival_1_deadline_2p6/best.pt",
        "knowledge_memory_path": r"/models/Qwen/lyq_jsac_2026_03_17/new_code/new_llm/newcode0411/new_0419/new_0515_skill/knowledge_env_time/knowledge_dynamic_tools_heuristic_sweep_20260515_005650/case_004_arrival_1_deadline_2p6/memory.json",
        "knowledge_loss_path": r"/models/Qwen/lyq_jsac_2026_03_17/new_code/new_llm/newcode0411/new_0419/new_0515_skill/knowledge_env_time/knowledge_dynamic_tools_heuristic_sweep_20260515_005650/case_004_arrival_1_deadline_2p6/loss.csv",
        "arrival_gap": 1.0,
        "task_deadline_multiplier": 2.6,
    },
    {
        "case_name": "arrival_2_deadline_2p6",
        "knowledge_best_path": r"/models/Qwen/lyq_jsac_2026_03_17/new_code/new_llm/newcode0411/new_0419/new_0515_skill/knowledge_env_time/knowledge_dynamic_tools_heuristic_sweep_20260515_005650/case_005_arrival_2_deadline_2p6/best.pt",
        "knowledge_memory_path": r"/models/Qwen/lyq_jsac_2026_03_17/new_code/new_llm/newcode0411/new_0419/new_0515_skill/knowledge_env_time/knowledge_dynamic_tools_heuristic_sweep_20260515_005650/case_005_arrival_2_deadline_2p6/memory.json",
        "knowledge_loss_path": r"/models/Qwen/lyq_jsac_2026_03_17/new_code/new_llm/newcode0411/new_0419/new_0515_skill/knowledge_env_time/knowledge_dynamic_tools_heuristic_sweep_20260515_005650/case_005_arrival_2_deadline_2p6/loss.csv",
        "arrival_gap": 2.0,
        "task_deadline_multiplier": 2.6,
    },
    {
        "case_name": "arrival_4_deadline_2p6",
        "knowledge_best_path": r"/models/Qwen/lyq_jsac_2026_03_17/new_code/new_llm/newcode0411/new_0419/new_0515_skill/knowledge_env_time/knowledge_dynamic_tools_heuristic_sweep_20260515_005650/case_006_arrival_4_deadline_2p6/best.pt",
        "knowledge_memory_path": r"/models/Qwen/lyq_jsac_2026_03_17/new_code/new_llm/newcode0411/new_0419/new_0515_skill/knowledge_env_time/knowledge_dynamic_tools_heuristic_sweep_20260515_005650/case_006_arrival_4_deadline_2p6/memory.json",
        "knowledge_loss_path": r"/models/Qwen/lyq_jsac_2026_03_17/new_code/new_llm/newcode0411/new_0419/new_0515_skill/knowledge_env_time/knowledge_dynamic_tools_heuristic_sweep_20260515_005650/case_006_arrival_4_deadline_2p6/loss.csv",
        "arrival_gap": 4.0,
        "task_deadline_multiplier": 2.6,
    },
]
TDPO_SKILL_EVAL_TIME_KNOWLEDGE_SWEEP_ROOT = Path("")
TDPO_SKILL_EVAL_TIME_KNOWLEDGE_BEST_FILENAME = "best.pt"
TDPO_SKILL_EVAL_TIME_KNOWLEDGE_MEMORY_FILENAME = "memory.json"

# Paths used by experiment_suite/tdpo_skill/evaluation/time_same_environment.py.
TDPO_SKILL_EVAL_TIME_SAME_OUTPUT_ROOT = Path("./tdpo_eval_knowledge_env_sweep_time_same")
TDPO_SKILL_EVAL_TIME_SAME_DATA_DIR = Path("/models/Qwen/lyq_jsac_2026_03_17/new_code/new_llm/newcode0411/deepseek_intent_dag_dataset_500")
TDPO_SKILL_EVAL_TIME_SAME_TDPO_CHECKPOINTS = [
    {"epoch": 100, "path": r"/models/Qwen/lyq_jsac_2026_03_17/new_code/new_llm/newcode0411/new_0419/new_0509_all/llm_tdpo_skill_fast1/tdpo_decision_runs_batch_fast_skill/qwen7b_tdpo_with_skill_memory_20260511_140148/ckpts/best.pt"},
    # {"epoch": 80, "path": r"/path/to/tdpo_run/ckpts/tdpo_policy_epoch_080.pt"},
]
TDPO_SKILL_EVAL_TIME_SAME_TDPO_CKPT_DIR = Path("")
TDPO_SKILL_EVAL_TIME_SAME_TDPO_CKPT_TEMPLATE = "tdpo_policy_epoch_{epoch:03d}.pt"
TDPO_SKILL_EVAL_TIME_SAME_TDPO_BEST_CKPT_NAME = "best.pt"
TDPO_SKILL_EVAL_TIME_SAME_KNOWLEDGE_CASES = [
    {
        "case_name": "arrival_0p25_deadline_2p6",
        "knowledge_best_path": r"/models/Qwen/lyq_jsac_2026_03_17/new_code/new_llm/newcode0411/new_0419/new_0515_skill/knowledge_env_time/knowledge_dynamic_tools_heuristic_sweep_20260515_005650/case_001_arrival_0p25_deadline_2p6/best.pt",
        "knowledge_memory_path": r"/models/Qwen/lyq_jsac_2026_03_17/new_code/new_llm/newcode0411/new_0419/new_0515_skill/knowledge_env_time/knowledge_dynamic_tools_heuristic_sweep_20260515_005650/case_001_arrival_0p25_deadline_2p6/memory.json",
        "arrival_gap": 0.25,
        "task_deadline_multiplier": 2.6,
    },
    {
        "case_name": "arrival_0p5_deadline_2p6",
        "knowledge_best_path": r"/models/Qwen/lyq_jsac_2026_03_17/new_code/new_llm/newcode0411/new_0419/new_0515_skill/knowledge_env_time/knowledge_dynamic_tools_heuristic_sweep_20260515_005650/case_002_arrival_0p5_deadline_2p6/best.pt",
        "knowledge_memory_path": r"/models/Qwen/lyq_jsac_2026_03_17/new_code/new_llm/newcode0411/new_0419/new_0515_skill/knowledge_env_time/knowledge_dynamic_tools_heuristic_sweep_20260515_005650/case_002_arrival_0p5_deadline_2p6/memory.json",
        "arrival_gap": 0.5,
        "task_deadline_multiplier": 2.6,
    },
    {
        "case_name": "arrival_0p75_deadline_2p6",
        "knowledge_best_path": r"/models/Qwen/lyq_jsac_2026_03_17/new_code/new_llm/newcode0411/new_0419/new_0515_skill/knowledge_env_time/knowledge_dynamic_tools_heuristic_sweep_20260515_005650/case_003_arrival_0p75_deadline_2p6/best.pt",
        "knowledge_memory_path": r"/models/Qwen/lyq_jsac_2026_03_17/new_code/new_llm/newcode0411/new_0419/new_0515_skill/knowledge_env_time/knowledge_dynamic_tools_heuristic_sweep_20260515_005650/case_003_arrival_0p75_deadline_2p6/memory.json",
        "arrival_gap": 0.75,
        "task_deadline_multiplier": 2.6,
    },
    {
        "case_name": "arrival_1_deadline_2p6",
        "knowledge_best_path": r"/models/Qwen/lyq_jsac_2026_03_17/new_code/new_llm/newcode0411/new_0419/new_0515_skill/knowledge_env_time/knowledge_dynamic_tools_heuristic_sweep_20260515_005650/case_004_arrival_1_deadline_2p6/best.pt",
        "knowledge_memory_path": r"/models/Qwen/lyq_jsac_2026_03_17/new_code/new_llm/newcode0411/new_0419/new_0515_skill/knowledge_env_time/knowledge_dynamic_tools_heuristic_sweep_20260515_005650/case_004_arrival_1_deadline_2p6/memory.json",
        "knowledge_loss_path": r"/models/Qwen/lyq_jsac_2026_03_17/new_code/new_llm/newcode0411/new_0419/new_0515_skill/knowledge_env_time/knowledge_dynamic_tools_heuristic_sweep_20260515_005650/case_004_arrival_1_deadline_2p6/loss.csv",
        "arrival_gap": 1.0,
        "task_deadline_multiplier": 2.6,
    },
    {
        "case_name": "arrival_2_deadline_2p6",
        "knowledge_best_path": r"/models/Qwen/lyq_jsac_2026_03_17/new_code/new_llm/newcode0411/new_0419/new_0515_skill/knowledge_env_time/knowledge_dynamic_tools_heuristic_sweep_20260515_005650/case_005_arrival_2_deadline_2p6/best.pt",
        "knowledge_memory_path": r"/models/Qwen/lyq_jsac_2026_03_17/new_code/new_llm/newcode0411/new_0419/new_0515_skill/knowledge_env_time/knowledge_dynamic_tools_heuristic_sweep_20260515_005650/case_005_arrival_2_deadline_2p6/memory.json",
        "knowledge_loss_path": r"/models/Qwen/lyq_jsac_2026_03_17/new_code/new_llm/newcode0411/new_0419/new_0515_skill/knowledge_env_time/knowledge_dynamic_tools_heuristic_sweep_20260515_005650/case_005_arrival_2_deadline_2p6/loss.csv",
        "arrival_gap": 2.0,
        "task_deadline_multiplier": 2.6,
    },
    {
        "case_name": "arrival_4_deadline_2p6",
        "knowledge_best_path": r"/models/Qwen/lyq_jsac_2026_03_17/new_code/new_llm/newcode0411/new_0419/new_0515_skill/knowledge_env_time/knowledge_dynamic_tools_heuristic_sweep_20260515_005650/case_006_arrival_4_deadline_2p6/best.pt",
        "knowledge_memory_path": r"/models/Qwen/lyq_jsac_2026_03_17/new_code/new_llm/newcode0411/new_0419/new_0515_skill/knowledge_env_time/knowledge_dynamic_tools_heuristic_sweep_20260515_005650/case_006_arrival_4_deadline_2p6/memory.json",
        "knowledge_loss_path": r"/models/Qwen/lyq_jsac_2026_03_17/new_code/new_llm/newcode0411/new_0419/new_0515_skill/knowledge_env_time/knowledge_dynamic_tools_heuristic_sweep_20260515_005650/case_006_arrival_4_deadline_2p6/loss.csv",
        "arrival_gap": 4.0,
        "task_deadline_multiplier": 2.6,
    },
]
TDPO_SKILL_EVAL_TIME_SAME_KNOWLEDGE_SWEEP_ROOT = Path("")
TDPO_SKILL_EVAL_TIME_SAME_KNOWLEDGE_BEST_FILENAME = "best.pt"
TDPO_SKILL_EVAL_TIME_SAME_KNOWLEDGE_MEMORY_FILENAME = "memory.json"

# Paths used by experiment_suite/tdpo_no_skill/evaluation/deadline_sweep.py.
TDPO_NO_SKILL_EVAL_DEADLINE_OUTPUT_ROOT = Path("./tdpo_no_skill_eval_dead")
TDPO_NO_SKILL_EVAL_DEADLINE_DATA_DIR = Path("/models/Qwen/lyq_jsac_2026_03_17/new_code/new_llm/newcode0411/deepseek_intent_dag_dataset_500")
TDPO_NO_SKILL_EVAL_DEADLINE_TDPO_CHECKPOINTS = [
    {"epoch": 100, "path": r"/models/Qwen/lyq_jsac_2026_03_17/new_code/new_llm/newcode0411/new_0419/new_0509_all/llm_tdpo_no/tdpo_decision_runs_batch_fast_no/qwen7b_tdpo_no_skill_20260509_235950/ckpts/best.pt"},
    # {"epoch": 80, "path": r"/path/to/tdpo_run/ckpts/tdpo_policy_epoch_080.pt"},
]
TDPO_NO_SKILL_EVAL_DEADLINE_TDPO_CKPT_DIR = Path("")
TDPO_NO_SKILL_EVAL_DEADLINE_TDPO_CKPT_TEMPLATE = "tdpo_policy_epoch_{epoch:03d}.pt"
TDPO_NO_SKILL_EVAL_DEADLINE_TDPO_BEST_CKPT_NAME = "best.pt"
TDPO_NO_SKILL_EVAL_DEADLINE_KNOWLEDGE_CASES = [
    {
        "case_name": "arrival_1_deadline_0p25",
        "knowledge_best_path": r"/models/Qwen/lyq_jsac_2026_03_17/new_code/new_llm/newcode0411/new_0419/new_0515_skill/knowledge_env_dead/knowledge_dynamic_tools_heuristic_sweep_20260515_010543/case_001_arrival_1_deadline_0p25/best.pt",
        "knowledge_memory_path": r"/models/Qwen/lyq_jsac_2026_03_17/new_code/new_llm/newcode0411/new_0419/new_0515_skill/knowledge_env_dead/knowledge_dynamic_tools_heuristic_sweep_20260515_010543/case_001_arrival_1_deadline_0p25/memory.json",
        "knowledge_loss_path": r"/models/Qwen/lyq_jsac_2026_03_17/new_code/new_llm/newcode0411/new_0419/new_0515_skill/knowledge_env_dead/knowledge_dynamic_tools_heuristic_sweep_20260515_010543/case_001_arrival_1_deadline_0p25/loss.csv",
        "arrival_gap": 1.0,
        "task_deadline_multiplier": 0.25,
    },
    {
        "case_name": "arrival_1_deadline_0p75",
        "knowledge_best_path": r"/models/Qwen/lyq_jsac_2026_03_17/new_code/new_llm/newcode0411/new_0419/new_0515_skill/knowledge_env_dead/knowledge_dynamic_tools_heuristic_sweep_20260515_010543/case_002_arrival_1_deadline_0p75/best.pt",
        "knowledge_memory_path": r"/models/Qwen/lyq_jsac_2026_03_17/new_code/new_llm/newcode0411/new_0419/new_0515_skill/knowledge_env_dead/knowledge_dynamic_tools_heuristic_sweep_20260515_010543/case_002_arrival_1_deadline_0p75/memory.json",
        "knowledge_loss_path": r"/models/Qwen/lyq_jsac_2026_03_17/new_code/new_llm/newcode0411/new_0419/new_0515_skill/knowledge_env_dead/knowledge_dynamic_tools_heuristic_sweep_20260515_010543/case_002_arrival_1_deadline_0p75/loss.csv",
        "arrival_gap": 1.0,
        "task_deadline_multiplier": 0.75,
    },
    {
        "case_name": "arrival_1_deadline_1p25",
        "knowledge_best_path": r"/models/Qwen/lyq_jsac_2026_03_17/new_code/new_llm/newcode0411/new_0419/new_0515_skill/knowledge_env_dead/knowledge_dynamic_tools_heuristic_sweep_20260515_010543/case_003_arrival_1_deadline_1p25/best.pt",
        "knowledge_memory_path": r"/models/Qwen/lyq_jsac_2026_03_17/new_code/new_llm/newcode0411/new_0419/new_0515_skill/knowledge_env_dead/knowledge_dynamic_tools_heuristic_sweep_20260515_010543/case_003_arrival_1_deadline_1p25/memory.json",
        "knowledge_loss_path": r"/models/Qwen/lyq_jsac_2026_03_17/new_code/new_llm/newcode0411/new_0419/new_0515_skill/knowledge_env_dead/knowledge_dynamic_tools_heuristic_sweep_20260515_010543/case_003_arrival_1_deadline_1p25/loss.csv",
        "arrival_gap": 1.0,
        "task_deadline_multiplier": 1.25,
    },
    {
        "case_name": "arrival_1_deadline_4",
        "knowledge_best_path": r"/models/Qwen/lyq_jsac_2026_03_17/new_code/new_llm/newcode0411/new_0419/new_0515_skill/knowledge_env_dead/knowledge_dynamic_tools_heuristic_sweep_20260515_173459/case_001_arrival_1_deadline_4/best.pt",
        "knowledge_memory_path": r"/models/Qwen/lyq_jsac_2026_03_17/new_code/new_llm/newcode0411/new_0419/new_0515_skill/knowledge_env_dead/knowledge_dynamic_tools_heuristic_sweep_20260515_173459/case_001_arrival_1_deadline_4/memory.json",
        "knowledge_loss_path": r"/models/Qwen/lyq_jsac_2026_03_17/new_code/new_llm/newcode0411/new_0419/new_0515_skill/knowledge_env_dead/knowledge_dynamic_tools_heuristic_sweep_20260515_173459/case_001_arrival_1_deadline_4/loss.csv",
        "arrival_gap": 1.0,
        "task_deadline_multiplier": 4.0,
    },
    {
        "case_name": "arrival_1_deadline_2",
        "knowledge_best_path": r"/models/Qwen/lyq_jsac_2026_03_17/new_code/new_llm/newcode0411/new_0419/new_0515_skill/knowledge_env_dead/knowledge_dynamic_tools_heuristic_sweep_20260515_010543/case_005_arrival_1_deadline_2/best.pt",
        "knowledge_memory_path": r"/models/Qwen/lyq_jsac_2026_03_17/new_code/new_llm/newcode0411/new_0419/new_0515_skill/knowledge_env_dead/knowledge_dynamic_tools_heuristic_sweep_20260515_010543/case_005_arrival_1_deadline_2/memory.json",
        "knowledge_loss_path": r"/models/Qwen/lyq_jsac_2026_03_17/new_code/new_llm/newcode0411/new_0419/new_0515_skill/knowledge_env_dead/knowledge_dynamic_tools_heuristic_sweep_20260515_010543/case_005_arrival_1_deadline_2/loss.csv",
        "arrival_gap": 1.0,
        "task_deadline_multiplier": 2.0,
    },
    {
        "case_name": "arrival_1_deadline_1p5",
        "knowledge_best_path": r"/models/Qwen/lyq_jsac_2026_03_17/new_code/new_llm/newcode0411/new_0419/new_0515_skill/knowledge_env_dead/knowledge_dynamic_tools_heuristic_sweep_20260515_010543/case_004_arrival_1_deadline_1p5/best.pt",
        "knowledge_memory_path": r"/models/Qwen/lyq_jsac_2026_03_17/new_code/new_llm/newcode0411/new_0419/new_0515_skill/knowledge_env_dead/knowledge_dynamic_tools_heuristic_sweep_20260515_010543/case_004_arrival_1_deadline_1p5/memory.json",
        "knowledge_loss_path": r"/models/Qwen/lyq_jsac_2026_03_17/new_code/new_llm/newcode0411/new_0419/new_0515_skill/knowledge_env_dead/knowledge_dynamic_tools_heuristic_sweep_20260515_010543/case_004_arrival_1_deadline_1p5/loss.csv",
        "arrival_gap": 1.0,
        "task_deadline_multiplier": 1.5,
    },
]
TDPO_NO_SKILL_EVAL_DEADLINE_KNOWLEDGE_SWEEP_ROOT = Path("")
TDPO_NO_SKILL_EVAL_DEADLINE_KNOWLEDGE_BEST_FILENAME = "best.pt"
TDPO_NO_SKILL_EVAL_DEADLINE_KNOWLEDGE_MEMORY_FILENAME = "memory.json"

# Paths used by experiment_suite/tdpo_no_skill/evaluation/time_sweep.py.
TDPO_NO_SKILL_EVAL_TIME_OUTPUT_ROOT = Path("./tdpo_no_skill_eval_time")
TDPO_NO_SKILL_EVAL_TIME_DATA_DIR = Path("/models/Qwen/lyq_jsac_2026_03_17/new_code/new_llm/newcode0411/deepseek_intent_dag_dataset_500")
TDPO_NO_SKILL_EVAL_TIME_TDPO_CHECKPOINTS = [
    {"epoch": 100, "path": r"/models/Qwen/lyq_jsac_2026_03_17/new_code/new_llm/newcode0411/new_0419/new_0509_all/llm_tdpo_no/tdpo_decision_runs_batch_fast_no/qwen7b_tdpo_no_skill_20260509_235950/ckpts/best.pt"},
    # {"epoch": 80, "path": r"/path/to/tdpo_run/ckpts/tdpo_policy_epoch_080.pt"},
]
TDPO_NO_SKILL_EVAL_TIME_TDPO_CKPT_DIR = Path("")
TDPO_NO_SKILL_EVAL_TIME_TDPO_CKPT_TEMPLATE = "tdpo_policy_epoch_{epoch:03d}.pt"
TDPO_NO_SKILL_EVAL_TIME_TDPO_BEST_CKPT_NAME = "best.pt"
TDPO_NO_SKILL_EVAL_TIME_KNOWLEDGE_CASES = [
    {
        "case_name": "arrival_0p25_deadline_2p6",
        "knowledge_best_path": r"/models/Qwen/lyq_jsac_2026_03_17/new_code/new_llm/newcode0411/new_0419/new_0515_skill/knowledge_env_time/knowledge_dynamic_tools_heuristic_sweep_20260515_005650/case_001_arrival_0p25_deadline_2p6/best.pt",
        "knowledge_memory_path": r"/models/Qwen/lyq_jsac_2026_03_17/new_code/new_llm/newcode0411/new_0419/new_0515_skill/knowledge_env_time/knowledge_dynamic_tools_heuristic_sweep_20260515_005650/case_001_arrival_0p25_deadline_2p6/memory.json",
        "arrival_gap": 0.25,
        "task_deadline_multiplier": 2.6,
    },
    {
        "case_name": "arrival_0p5_deadline_2p6",
        "knowledge_best_path": r"/models/Qwen/lyq_jsac_2026_03_17/new_code/new_llm/newcode0411/new_0419/new_0515_skill/knowledge_env_time/knowledge_dynamic_tools_heuristic_sweep_20260515_005650/case_002_arrival_0p5_deadline_2p6/best.pt",
        "knowledge_memory_path": r"/models/Qwen/lyq_jsac_2026_03_17/new_code/new_llm/newcode0411/new_0419/new_0515_skill/knowledge_env_time/knowledge_dynamic_tools_heuristic_sweep_20260515_005650/case_002_arrival_0p5_deadline_2p6/memory.json",
        "arrival_gap": 0.5,
        "task_deadline_multiplier": 2.6,
    },
    {
        "case_name": "arrival_0p75_deadline_2p6",
        "knowledge_best_path": r"/models/Qwen/lyq_jsac_2026_03_17/new_code/new_llm/newcode0411/new_0419/new_0515_skill/knowledge_env_time/knowledge_dynamic_tools_heuristic_sweep_20260515_005650/case_003_arrival_0p75_deadline_2p6/best.pt",
        "knowledge_memory_path": r"/models/Qwen/lyq_jsac_2026_03_17/new_code/new_llm/newcode0411/new_0419/new_0515_skill/knowledge_env_time/knowledge_dynamic_tools_heuristic_sweep_20260515_005650/case_003_arrival_0p75_deadline_2p6/memory.json",
        "arrival_gap": 0.75,
        "task_deadline_multiplier": 2.6,
    },
    {
        "case_name": "arrival_1_deadline_2p6",
        "knowledge_best_path": r"/models/Qwen/lyq_jsac_2026_03_17/new_code/new_llm/newcode0411/new_0419/new_0515_skill/knowledge_env_time/knowledge_dynamic_tools_heuristic_sweep_20260515_005650/case_004_arrival_1_deadline_2p6/best.pt",
        "knowledge_memory_path": r"/models/Qwen/lyq_jsac_2026_03_17/new_code/new_llm/newcode0411/new_0419/new_0515_skill/knowledge_env_time/knowledge_dynamic_tools_heuristic_sweep_20260515_005650/case_004_arrival_1_deadline_2p6/memory.json",
        "knowledge_loss_path": r"/models/Qwen/lyq_jsac_2026_03_17/new_code/new_llm/newcode0411/new_0419/new_0515_skill/knowledge_env_time/knowledge_dynamic_tools_heuristic_sweep_20260515_005650/case_004_arrival_1_deadline_2p6/loss.csv",
        "arrival_gap": 1.0,
        "task_deadline_multiplier": 2.6,
    },
    {
        "case_name": "arrival_2_deadline_2p6",
        "knowledge_best_path": r"/models/Qwen/lyq_jsac_2026_03_17/new_code/new_llm/newcode0411/new_0419/new_0515_skill/knowledge_env_time/knowledge_dynamic_tools_heuristic_sweep_20260515_005650/case_005_arrival_2_deadline_2p6/best.pt",
        "knowledge_memory_path": r"/models/Qwen/lyq_jsac_2026_03_17/new_code/new_llm/newcode0411/new_0419/new_0515_skill/knowledge_env_time/knowledge_dynamic_tools_heuristic_sweep_20260515_005650/case_005_arrival_2_deadline_2p6/memory.json",
        "knowledge_loss_path": r"/models/Qwen/lyq_jsac_2026_03_17/new_code/new_llm/newcode0411/new_0419/new_0515_skill/knowledge_env_time/knowledge_dynamic_tools_heuristic_sweep_20260515_005650/case_005_arrival_2_deadline_2p6/loss.csv",
        "arrival_gap": 2.0,
        "task_deadline_multiplier": 2.6,
    },
    {
        "case_name": "arrival_4_deadline_2p6",
        "knowledge_best_path": r"/models/Qwen/lyq_jsac_2026_03_17/new_code/new_llm/newcode0411/new_0419/new_0515_skill/knowledge_env_time/knowledge_dynamic_tools_heuristic_sweep_20260515_005650/case_006_arrival_4_deadline_2p6/best.pt",
        "knowledge_memory_path": r"/models/Qwen/lyq_jsac_2026_03_17/new_code/new_llm/newcode0411/new_0419/new_0515_skill/knowledge_env_time/knowledge_dynamic_tools_heuristic_sweep_20260515_005650/case_006_arrival_4_deadline_2p6/memory.json",
        "knowledge_loss_path": r"/models/Qwen/lyq_jsac_2026_03_17/new_code/new_llm/newcode0411/new_0419/new_0515_skill/knowledge_env_time/knowledge_dynamic_tools_heuristic_sweep_20260515_005650/case_006_arrival_4_deadline_2p6/loss.csv",
        "arrival_gap": 4.0,
        "task_deadline_multiplier": 2.6,
    },
]
TDPO_NO_SKILL_EVAL_TIME_KNOWLEDGE_SWEEP_ROOT = Path("")
TDPO_NO_SKILL_EVAL_TIME_KNOWLEDGE_BEST_FILENAME = "best.pt"
TDPO_NO_SKILL_EVAL_TIME_KNOWLEDGE_MEMORY_FILENAME = "memory.json"

# Local fallback tool-library paths retain their original project locations.
KNOWLEDGE_LOCAL_TOOL_LIBRARY_PATH = PROJECT_ROOT / "tool_library.json"
TDPO_SKILL_LOCAL_TOOL_LIBRARY_PATH = PROJECT_ROOT / "tdpo" / "tool_library.json"
TDPO_NO_SKILL_LOCAL_TOOL_LIBRARY_PATH = PROJECT_ROOT / "tdpo_no" / "tool_library.json"

# Generated file names shared by the training and evaluation workflows.
LOSS_FILE = "loss.csv"
BEST_MODEL_FILE = "best.pt"
MEMORY_FILE = "memory.json"
EPOCH_METRICS_FILE = "epoch_metrics.csv"
EPOCH_ALIAS_FILE = "epoch.csv"
ENERGY_FILE = "energy.csv"
DELAY_FILE = "delay.csv"
CASE_CONFIG_FILE = "case_config.json"
EVAL_CONFIG_FILE = "eval_config.json"
COMBINED_SUMMARY_FILE = "combined_summary.csv"
CHECKPOINT_DIR = "ckpts"
RECORDS_DIR = "records"
PAIRS_DIR = "pairs"
TRACES_DIR = "traces"
KNOWLEDGE_DIR = "knowledge"
TDPO_OFFLOAD_DIR = "tdpo_offload"
DATASET_SCAN_FILE = "dataset_scan.json"
RUN_CONFIG_FILE = "run_config.json"
PRETRAINED_INPUT_COPY_FILE = "pretrained_input_copy.pt"
LOADED_KNOWLEDGE_INFO_FILE = "loaded_knowledge_info.json"
MEMORY_BOOTSTRAP_SUMMARY_FILE = "memory_bootstrap_summary.json"
MEMORY_BOOTSTRAP_RECORDS_FILE = "memory_bootstrap_task_records.json"
MEMORY_BOOTSTRAP_STATS_FILE = "memory_bootstrap_stats.json"
KNOWLEDGE_METRICS_FILE = "knowledge_metrics.csv"
RELEASE_BEFORE_TDPO_FILE = "release_before_tdpo.json"
EVAL_METRICS_FILE = "eval_metrics.csv"
TRACES_FILE = "traces.jsonl"
PAIRS_FILE = "pairs.jsonl"
PROMPT_SAMPLES_FILE = "prompt_samples_epoch_001.json"
LATEST_SUMMARY_FILE = "latest_summary.json"
FINAL_RESULT_FILE = "final_result.json"


def knowledge_update_filename(epoch: int) -> str:
    """Return the original per-epoch knowledge update file name."""

    return f"update_stats_epoch_{epoch:03d}.json"


def tdpo_checkpoint_filename(epoch: int) -> str:
    """Return the original per-epoch TDPO checkpoint file name."""

    return f"tdpo_policy_epoch_{epoch:03d}.pt"


def task_records_filename(epoch: int) -> str:
    """Return the original per-epoch task-record file name."""

    return f"task_records_epoch_{epoch:03d}.json"
