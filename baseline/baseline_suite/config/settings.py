"""Behavioral settings for all baseline algorithms.

Filesystem locations are intentionally kept in config/paths.py. Edit this module
for sampling, environment, optimization, GPU, knowledge, and logging behavior.
"""

import os

from baseline_suite.config.paths import DEFAULT_POLICY_OFFLOAD_PATH, LOCAL_TOOL_LIBRARY_PATH

# ============================================================
# User-editable globals: reproducibility and data scale
# ============================================================

SEED = 0
SAMPLE_NUM_GRAPHS = 20
SAMPLE_IN_ORDER = False
SAMPLE_RANDOM_POOL_LIMIT = 1500
MAX_TRAIN_GRAPHS = SAMPLE_NUM_GRAPHS
ROLLOUT_CHUNK_GRAPHS = 20
SHUFFLE_DATASET_ON_LOAD = False

# ============================================================
# User-editable globals: environment.  Kept aligned with the uploaded TDPO run.
# ============================================================

ENV_DT = 0.10
ENV_Q_MAX = 12
ENV_ARRIVAL_GAP_RANGE = (1.05, 1.05)
ENV_METRICS_RECORD_PERIOD_STEPS = 1
ENV_NODE_DEADLINE_MULTIPLIER = 3.3
ENV_TASK_DEADLINE_MULTIPLIER = 2.7
ENV_MIN_NODE_DEADLINE_S = 2.0
ENV_MIN_TASK_DEADLINE_S = 6.0
ENV_LOCAL_MAX_CONCURRENCY = 5
ENV_LOCAL_QUEUE_CAPACITY = 80
ENV_RETRY_RESTART_PENALTY_S = 0.10
ENV_INVALID_ACTION_MODE = "fail_task"
ENV_INVALID_ACTION_PENALTY_S = 0.1
ENV_FAIL_ON_QUEUE_REJECTION = True
ENV_MAX_INVALID_ATTEMPTS_PER_NODE = 2
ENV_TARGET_TOOL_COMPUTE_S = 1.0
ENV_TARGET_TOOL_UPLINK_S = 0.25
ENV_TARGET_TOOL_DOWNLINK_S = 0.25

ENV_RELAXED_RESOURCE_MODE = True
ENV_TOOL_INSTANCE_MULTIPLIER = 1.25
ENV_TOOL_INSTANCE_ADDITIVE = 0
ENV_MIN_TOOL_INSTANCES = 1
FORCE_MAX_ENV_STEPS = 0

USE_DYNAMIC_TOOL_LIBRARY = True
USE_MULTIPLE_TOOL_LIBRARIES = True
TOOL_LIBRARY_PATH = LOCAL_TOOL_LIBRARY_PATH
ALLOW_TOOL_LIBRARY_FALLBACK = False
# ============================================================
# User-editable globals: training and evaluation
# ============================================================

NUM_EPOCHS = 100
BATCH_SIZE = 64
INNER_EPOCHS = 1
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 1e-4
TRAIN_WITH_SAMPLING = True
TRAIN_SAMPLE_PROBABILITY = 0.35
ACTION_TEMPERATURE = 0.50
RUN_GREEDY_EVAL_EACH_EPOCH = False
EVAL_USE_FIXED_SEED = True
EVAL_SEED_OFFSET = 100000
EVAL_PRINT_PROGRESS = True
SAVE_EVERY_EPOCH = False

# DPO
DPO_BETA = 0.10
DPO_MAX_PAIR_BUFFER = 512
DPO_BUFFER_SAMPLE_SIZE = 64
DPO_MIN_PAIRS_TO_TRAIN = 16
DPO_MAX_NEW_PAIRS_PER_EPOCH = 64
DPO_RANDOM_COUNTERFACTUAL = True
DPO_MIN_UTILITY_GAP = 0.25

# SFT
SFT_MAX_EXAMPLE_BUFFER = 512
SFT_BUFFER_SAMPLE_SIZE = 64
SFT_MIN_EXAMPLES_TO_TRAIN = 16
SFT_MAX_NEW_EXAMPLES_PER_EPOCH = 96

# PPO, implemented as a contextual-bandit PPO over node-level decisions.
PPO_BATCH_SIZE = 64
PPO_INNER_EPOCHS = 1
PPO_CLIP_RANGE = 0.20
PPO_VALUE_COEF = 0.50
PPO_ENTROPY_COEF = 0.01
PPO_MAX_GRAD_NORM = 1.0
PPO_REWARD_NORMALIZE = True

# Reward / utility used by PPO and simple DPO pair labeling.
UTILITY_W_MISSION = 100.0
UTILITY_W_NODE = 20.0
UTILITY_W_VIOL = 70.0
UTILITY_W_LATENCY = 6.0
UTILITY_LATENCY_CAP = 2.0
PAUSE_DURATION_S = 0.20
PAUSE_UTILITY_PENALTY = 2.0

# ============================================================
# User-editable globals: Qwen policy
# ============================================================

LLM_USE_LORA = False
LLM_LORA_NUM_LAYERS = 0
LLM_LORA_LAYER_IDS = tuple(range(-int(LLM_LORA_NUM_LAYERS), 0)) if int(LLM_LORA_NUM_LAYERS) > 0 else ()
LLM_LORA_TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
LLM_LORA_R = 4
LLM_LORA_ALPHA = 8.0
LLM_LORA_DROPOUT = 0.05
LLM_TUNE_LAST_N_BLOCKS = 0
LLM_TRAIN_FINAL_NORM = False
LLM_DTYPE = "bfloat16"
LLM_MAX_LENGTH = 256
LLM_GRADIENT_CHECKPOINTING = False
LLM_GENERATE_MAX_NEW_TOKENS = 16
LLM_COMPLETION_SCORE_BATCH_SIZE = 32
LLM_ACTION_OUTPUT_PREFIX = "ACTION_INDEX="

# Fast baseline mode for expensive single-A100 runs.  This keeps the same env,
# knowledge retrieval, and Qwen ACTION_INDEX policy, but shortens prompt/context
# and avoids an extra value-head forward during rollout.  PPO still trains the
# value head during the PPO update step.
FAST_COMPACT_PROMPT = True
KNOWLEDGE_PROMPT_MAX_CHARS = 240
ROLLOUT_COMPUTE_VALUE_HEAD = False
# Ultra-light baseline mode: Qwen is frozen and used only as a no-gradient
# state encoder. PPO/SFT/DPO train only small action/value heads, avoiding
# 7B backpropagation and per-action completion scoring. This keeps env and
# knowledge interfaces unchanged while making 20-graph single-env epochs practical.
LIGHTWEIGHT_HEAD_ONLY_POLICY = True
LIGHTWEIGHT_FEATURE_CACHE_MAX = 20000

GPU_PROFILE = "1xA100"  # "1xA100", "2x4090", or "4x4090"
if GPU_PROFILE.lower() in {"1xa100", "a100", "single_a100"}:
    CUDA_VISIBLE_DEVICES_DEFAULT = "0"
    LLM_ENABLE_MULTI_GPU = False
    LLM_DEVICE_MAP = "none"
    LLM_MAX_MEMORY_PER_GPU = "76GiB"
elif GPU_PROFILE.lower() in {"2x4090", "2xrtx4090", "dual_4090"}:
    CUDA_VISIBLE_DEVICES_DEFAULT = "0,1"
    LLM_ENABLE_MULTI_GPU = True
    LLM_DEVICE_MAP = "balanced"
    LLM_MAX_MEMORY_PER_GPU = "20GiB"
else:
    CUDA_VISIBLE_DEVICES_DEFAULT = "0,1,2,3"
    LLM_ENABLE_MULTI_GPU = True
    LLM_DEVICE_MAP = "balanced"
    LLM_MAX_MEMORY_PER_GPU = "18GiB"
LLM_LOW_CPU_MEM_USAGE = True
LLM_OFFLOAD_FOLDER_NAME = DEFAULT_POLICY_OFFLOAD_PATH.name

# ============================================================
# User-editable globals: knowledge/memory
# ============================================================

USE_KNOWLEDGE_MEMORY = True
KNOWLEDGE_USE_EXISTING_MEMORY_JSON = True
KNOWLEDGE_CLEAR_MEMORY_AFTER_LOAD = False
KNOWLEDGE_UPDATE_MEMORY_DURING_TRAINING = False
KNOWLEDGE_UPDATE_EVERY_EPOCHS = 10
KNOWLEDGE_UPDATE_WEIGHTS_DURING_TRAINING = False
RELEASE_KNOWLEDGE_BACKBONE_BEFORE_POLICY = True
KNOWLEDGE_FIT_EPOCHS_PER_UPDATE = 1
KNOWLEDGE_BATCH_SIZE = 16
KNOWLEDGE_VERBOSE = True
KNOWLEDGE_TRAIN_LOG_MODE = "both"
KNOWLEDGE_PROGRESS_EVERY_BATCHES = 2
KNOWLEDGE_USE_LLM_VERBALIZER = False
KNOWLEDGE_FAST_UPDATE_TEXT_ENCODER = True
KNOWLEDGE_FREEZE_LLM_ADAPTERS_DURING_UPDATE = True
KNOWLEDGE_FIT_MAX_SEGMENTS_PER_UPDATE = 32
KNOWLEDGE_FIT_MAX_NEW_ATOMS_PER_UPDATE = 8
RUN_MEMORY_MAX_ATOMS = 20000
RUN_MEMORY_MAX_PROTOTYPES = 4096
CONSERVATIVE_MAX_NEW_ATOMS_PER_UPDATE = 8
CONSERVATIVE_MAX_TASK_RECORDS_PER_UPDATE = 4
CONSERVATIVE_COMPLETED_ONLY = True
CONSERVATIVE_MIN_COMPLETION_RATE_TO_UPDATE = 0.45
KNOWLEDGE_UPDATE_STRATEGY = 3  # 1 conservative memory-only; 2 normal; 3 off; 4 delayed conservative
KNOWLEDGE_DELAYED_START_EPOCHS = 10
USE_EXTERNAL_MEMORY_LIBRARY = False
EXTERNAL_MEMORY_USE_WHEN_RUN_PROTOTYPES_LT = 16
EXTERNAL_MEMORY_BLEND_WEIGHT = 0.10
COPY_PRETRAINED_KNOWLEDGE_TO_RUN_DIR = False

# ============================================================
# Logging / save switches
# ============================================================

PRINT_EVERY_ENV_STEPS = 200
ROLLOUT_PRINT_EVERY_SECONDS = 10.0
ROLLOUT_PROGRESS_BAR_WIDTH = 28
ROLLOUT_DECISION_HEARTBEAT_EVERY = 50
BATCH_DECISION_ENABLED = True
BATCH_DECISION_MAX_SIZE = 16
ROLLOUT_SLOW_SELECT_SECONDS = 10.0
SAVE_AUXILIARY_JSON = True
SAVE_TASK_RECORDS_EVERY_EPOCH = False
SAVE_TRACE_JSONL = True
SAVE_FINAL_RESULT_JSON = True
SAVE_LATEST_SUMMARY_JSON = True
WRITE_PROMPT_SAMPLES = False
MAX_PROMPT_SAMPLES = 8

TX_POWER_W = 0.10
RX_POWER_W = 0.06
TOOL_EXEC_POWER_W = 1.50
LOCAL_POWER_W = 0.90
PAUSE_POWER_W = 0.05

os.environ.setdefault("CUDA_VISIBLE_DEVICES", CUDA_VISIBLE_DEVICES_DEFAULT)
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)
