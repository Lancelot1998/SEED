# -*- coding: utf-8 -*-
"""
train_TDPO.py

Training entry for decision_TDPO.py.

This script uses:
  1) task_env.py as the execution environment,
  2) knowledge_aspect_adapter.py as the pretrained knowledge/memory module,
  3) decision_TDPO.py as the LLM + TDPO policy module.

All experiment-level hyperparameters are placed in this file header.  The
pretrained knowledge checkpoint is loaded read-only by default.  Newly generated
memory and all TDPO fine-tuned LLM parameters are saved under OUTPUT_ROOT and do
not overwrite the original Qwen or knowledge files.
"""

from __future__ import annotations

from experiment_suite.config.paths import (
    TDPO_NO_SKILL_TRAIN_DATA_DIR as DATA_DIR,
    TDPO_NO_SKILL_TRAIN_DATASET_FILE_PATTERNS as DATASET_FILE_PATTERNS,
    TDPO_NO_SKILL_TRAIN_DATASET_METADATA_FILES as DATASET_METADATA_FILES,
    TDPO_NO_SKILL_TRAIN_QWEN_MODEL_PATH as QWEN_MODEL_PATH,
    TDPO_NO_SKILL_TRAIN_QWEN_MODEL_PATH_FALLBACK as QWEN_MODEL_PATH_FALLBACK,
    TDPO_NO_SKILL_TRAIN_PRETRAINED_KNOWLEDGE_MODEL_PATH as PRETRAINED_KNOWLEDGE_MODEL_PATH,
    TDPO_NO_SKILL_TRAIN_OPTIONAL_PRETRAINED_KNOWLEDGE_MEMORY_PATH as OPTIONAL_PRETRAINED_KNOWLEDGE_MEMORY_PATH,
    TDPO_NO_SKILL_TRAIN_OUTPUT_ROOT as OUTPUT_ROOT,
    TDPO_NO_SKILL_TRAIN_TOOL_LIBRARY_PATHS as TOOL_LIBRARY_PATHS,
    TDPO_NO_SKILL_TRAIN_EXTERNAL_MEMORY_LIBRARY_PATH as EXTERNAL_MEMORY_LIBRARY_PATH,
    TDPO_NO_SKILL_LOCAL_TOOL_LIBRARY_PATH as TOOL_LIBRARY_PATH,
    PROJECT_ROOT,
)

from experiment_suite.config.paths import (
    BEST_MODEL_FILE,
    CASE_CONFIG_FILE,
    CHECKPOINT_DIR,
    COMBINED_SUMMARY_FILE,
    DATASET_SCAN_FILE,
    DELAY_FILE,
    ENERGY_FILE,
    EPOCH_ALIAS_FILE,
    EPOCH_METRICS_FILE,
    EVAL_CONFIG_FILE,
    EVAL_METRICS_FILE,
    FINAL_RESULT_FILE,
    KNOWLEDGE_DIR,
    KNOWLEDGE_METRICS_FILE,
    LATEST_SUMMARY_FILE,
    LOADED_KNOWLEDGE_INFO_FILE,
    LOSS_FILE,
    MEMORY_BOOTSTRAP_RECORDS_FILE,
    MEMORY_BOOTSTRAP_STATS_FILE,
    MEMORY_BOOTSTRAP_SUMMARY_FILE,
    MEMORY_FILE,
    PAIRS_DIR,
    PAIRS_FILE,
    PRETRAINED_INPUT_COPY_FILE,
    PROMPT_SAMPLES_FILE,
    RECORDS_DIR,
    RELEASE_BEFORE_TDPO_FILE,
    RUN_CONFIG_FILE,
    TDPO_OFFLOAD_DIR,
    TRACES_DIR,
    TRACES_FILE,
    knowledge_update_filename,
    task_records_filename,
    tdpo_checkpoint_filename,
)

import csv
import json
import math
import os
import random
import shutil
import sys
import time
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

# ============================================================
# User-editable globals: paths
# ============================================================




RUN_NAME = "qwen7b_tdpo_no_skill"

# ============================================================
# User-editable globals: reproducibility and data scale
# ============================================================

SEED = 20260425
# Dataset sampling switch.  Dataset files are first sorted by filename.
# SAMPLE_RANDOM_POOL_LIMIT restricts the selectable range to the first N sorted files.
# SAMPLE_IN_ORDER=True means use the first SAMPLE_NUM_GRAPHS files from that pool.
# SAMPLE_IN_ORDER=False means randomly sample SAMPLE_NUM_GRAPHS files from that pool.
SAMPLE_NUM_GRAPHS = 20
SAMPLE_IN_ORDER = False
SAMPLE_RANDOM_POOL_LIMIT = 1500
MAX_TRAIN_GRAPHS = SAMPLE_NUM_GRAPHS          # Backward-compatible alias. 0 means use all loaded graphs.
# Split one epoch into several independent rollout environments.
# Example: 100 graphs with chunk=20 runs 5 env.reset() episodes, then aggregates
# all records/traces and performs one TDPO update. 0 disables chunking.
TDPO_ROLLOUT_CHUNK_GRAPHS = 20
MAX_MEMORY_BOOTSTRAP_GRAPHS = 0
SHUFFLE_DATASET_ON_LOAD = False              # Deprecated by SAMPLE_IN_ORDER; kept for compatibility.

# ============================================================
# User-editable globals: environment
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
# Prompt-only action mode: the policy no longer masks invalid actions, so the
# environment converts invalid/rejected decisions into explicit negative events.
# Options: "fail_task" or "keep_ready".
ENV_INVALID_ACTION_MODE = "fail_task"
ENV_INVALID_ACTION_PENALTY_S = 0.1
ENV_FAIL_ON_QUEUE_REJECTION = True
ENV_MAX_INVALID_ATTEMPTS_PER_NODE = 2
ENV_TARGET_TOOL_COMPUTE_S = 1.0
ENV_TARGET_TOOL_UPLINK_S = 0.25
ENV_TARGET_TOOL_DOWNLINK_S = 0.25

# Relaxed environment resource scaling.
# These switches only affect resource capacity, not the action space or TDPO route.
# Dynamic tool-library timing attributes are still used; instances/queue_capacity
# are taken from the scaled base slot profiles below.
ENV_RELAXED_RESOURCE_MODE = True
ENV_TOOL_INSTANCE_MULTIPLIER = 1.25
ENV_TOOL_INSTANCE_ADDITIVE = 0
ENV_MIN_TOOL_INSTANCES = 1
FORCE_MAX_ENV_STEPS = 0      # 0 means dynamic limit.

# Dynamic tool library.  Keep the TDPO action space and masks unchanged;
# task_env.py will read per-(graph_id, node_id, slot) tool attributes from this JSON.
USE_DYNAMIC_TOOL_LIBRARY = True
USE_MULTIPLE_TOOL_LIBRARIES = True
# If False, every allowed (graph_id, node_id, slot) in the selected dataset must exist in the tool library.
ALLOW_TOOL_LIBRARY_FALLBACK = False
_DYNAMIC_TOOL_LIBRARY_CACHE: Optional[Dict[str, object]] = None
_DYNAMIC_TOOL_LIBRARY_CACHE_KEY: Optional[Tuple[str, ...]] = None

# ============================================================
# User-editable globals: policy and TDPO training
# ============================================================
NUM_EPOCHS = 100
TDPO_BATCH_SIZE = 16
TDPO_INNER_EPOCHS = 1
TDPO_LEARNING_RATE = 1e-5
TDPO_WEIGHT_DECAY = 1e-4
TDPO_MAX_PREFERENCE_BUFFER = 8192
# Optional speed-up: keep a small preference buffer and train only on a bounded subset.
# With TDPO_MAX_PREFERENCE_BUFFER=512 and TDPO_BATCH_SIZE=64, each TDPO update has at most 8 optimizer steps.
TDPO_SAMPLE_FROM_BUFFER = True
TDPO_BUFFER_SAMPLE_SIZE = 512
TDPO_MAX_NEW_TIER_B_PAIRS_PER_EPOCH = 128
TDPO_MIN_PAIRS_TO_TRAIN = 32
ROLLING_REF_UPDATE_STEPS = 0
TDPO_REF_PRINT_EVERY_BATCHES = 2

# Quality-aware preference buffering.  The total buffer can remain moderate;
# pair admission and sampling are now completion/quality driven rather than FIFO-only.
PAIR_QUALITY_FILTER_ENABLED = True
PAIR_MAX_NEW_PAIRS_PER_EPOCH = 192
PAIR_MIN_CONFIDENCE = 0.60
PAIR_MIN_UTILITY_GAP = 3.0
PAIR_FAILED_ONLY_MIN_GAP = 10.0
PAIR_DROP_SKILL_CONFLICT = False
PAIR_SKILL_TOP_K = 2
PAIR_SUCCESS_BUFFER_RATIO = 0.80
PAIR_TRAIN_SUCCESS_RATIO = 0.85
PAIR_TIER_WARMUP_EPOCHS = 10
PAIR_TIER_A_RATIO_WARMUP = 0.60
PAIR_TIER_A_RATIO_AFTER_WARMUP = 0.20
PAIR_QUALITY_RECENCY_WEIGHT = 0.05
# Anti-collapse controls: keep Tier-B alive, avoid confidence saturation, and prevent buffers from becoming top-only.
PAIR_TIER_B_MIN_CONFIDENCE = 0.35
PAIR_TIER_B_MIN_UTILITY_GAP = 1.0
PAIR_TIER_B_FAILED_ONLY_MIN_GAP = 6.0
PAIR_CONFIDENCE_MAX = 0.85
PAIR_CONFIDENCE_BOOST_COMPLETED = 0.25
PAIR_CONFIDENCE_BOOST_NODE = 0.08
PAIR_CONFIDENCE_BOOST_SKILL = 0.05
PAIR_BUFFER_TOP_KEEP_RATIO = 0.70
PAIR_BUFFER_RECENT_KEEP_RATIO = 0.30
PAIR_TRAIN_TOP_SAMPLE_RATIO = 0.60
PAIR_TRAIN_RANDOM_SAMPLE_RATIO = 0.40
PAIR_TIER_B_RESERVOIR_PER_EPOCH = 32

# Greedy-gap guard: if rollout sampling looks good but greedy eval remains poor,
# reduce knowledge action-bias strength and increase sampling entropy for future rollouts.
# This keeps the technical route unchanged while preventing pair-learning from optimizing
# only the sampling policy and leaving greedy top-1 decisions weak.
GREEDY_GAP_GUARD_ENABLED = True
GREEDY_GAP_THRESHOLD = 0.20
GREEDY_GAP_LOW_EVAL = 0.25
GREEDY_GAP_PATIENCE = 2
GREEDY_GAP_BIAS_DECAY = 0.75
GREEDY_GAP_MIN_MEMORY_ACTION_BIAS_SCALE = 0.01
GREEDY_GAP_TEMPERATURE_STEP = 0.05
GREEDY_GAP_MAX_TEMPERATURE = 1.05

# Prefer the checkpoint that performs best under the same greedy policy used for reporting.
BEST_SCORE_USE_GREEDY_EVAL = True



# NUM_EPOCHS = 100
# TDPO_BATCH_SIZE = 16
# TDPO_INNER_EPOCHS = 1
# TDPO_LEARNING_RATE = 2e-5
# TDPO_WEIGHT_DECAY = 1e-4
# TDPO_MAX_PREFERENCE_BUFFER = 4096
# TDPO_MAX_NEW_TIER_B_PAIRS_PER_EPOCH = 512
# TDPO_MIN_PAIRS_TO_TRAIN = 4
# ROLLING_REF_UPDATE_STEPS = 0

USE_KNOWLEDGE_MEMORY = False
TRAIN_WITH_SAMPLING = True
# Probability of stochastic sampling during training rollout when TRAIN_WITH_SAMPLING=True.
# This is distinct from ACTION_TEMPERATURE: 0.55 means 55% of decision points use sampling,
# while the remaining 45% use greedy argmax.
TRAIN_SAMPLE_PROBABILITY = 0.55
SAVE_EVERY_EPOCH = False

# Greedy evaluation switch.  Training rollout now uses greedy action selection when TRAIN_WITH_SAMPLING=False.
# If enabled, after each epoch update finishes, a no-gradient greedy rollout is run
# and written to eval_metrics.csv.  It does not build pairs or update parameters.
RUN_GREEDY_EVAL_EACH_EPOCH = False
EVAL_WITH_GREEDY = RUN_GREEDY_EVAL_EACH_EPOCH  # backward-compatible alias
EVAL_USE_FIXED_SEED = True
EVAL_SEED_OFFSET = 100000
EVAL_PRINT_PROGRESS = True

# Qwen LoRA fine-tuning.  The base Qwen weights remain frozen; only LoRA
# adapters in the selected two blocks and the lightweight decision heads are trained.
LLM_USE_LORA = True
LLM_LORA_NUM_LAYERS = 4                 # Number of last transformer blocks to receive LoRA. Set to 1 for fastest TDPO.
LLM_LORA_LAYER_IDS = tuple(range(-int(LLM_LORA_NUM_LAYERS), 0)) if int(LLM_LORA_NUM_LAYERS) > 0 else ()
LLM_LORA_TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
LLM_LORA_R = 8
LLM_LORA_ALPHA = 16.0
LLM_LORA_DROPOUT = 0.05

# Fallback only when LLM_USE_LORA=False.  Keep 0 to avoid full-block tuning.
LLM_TUNE_LAST_N_BLOCKS = 0
LLM_TRAIN_FINAL_NORM = False
LLM_DTYPE = "bfloat16"
LLM_MAX_LENGTH = 512
LLM_GRADIENT_CHECKPOINTING = True
LLM_DEVICE = "auto"

# GPU profile switch.  Choose one of: "4x4090" or "1xA100".
# - "4x4090": shard Qwen across four visible RTX 4090 cards with device_map="balanced".
# - "1xA100": place the whole Qwen policy on one visible A100; no model-parallel device_map.
GPU_PROFILE = "1xA100"#"4x4090"

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
LLM_OFFLOAD_FOLDER_NAME = "tdpo_offload"

# ============================================================
# User-editable globals: knowledge/memory generation
# ============================================================

# The original checkpoint is never overwritten.  These switches only control the
# run-local copy and the newly generated memory JSON under OUTPUT_ROOT.
GENERATE_NEW_MEMORY_BEFORE_TDPO = False
MEMORY_BOOTSTRAP_POLICY = "completion_first_feasible"    # completion_first_feasible, fastest_feasible, tool_preferred, local_preferred, random_feasible
KNOWLEDGE_USE_EXISTING_MEMORY_JSON = False      # If True and OPTIONAL_PRETRAINED_KNOWLEDGE_MEMORY_PATH exists, load it first.
KNOWLEDGE_CLEAR_MEMORY_AFTER_LOAD = False        # Keep loaded memory; TDPO-time updates are success-driven and memory-only.
KNOWLEDGE_UPDATE_MEMORY_DURING_TDPO = False      # Success-driven memory-only updates during TDPO.
KNOWLEDGE_UPDATE_EVERY_EPOCHS = 5
KNOWLEDGE_UPDATE_WEIGHTS_DURING_TDPO = False   # Do not move the knowledge encoder during TDPO; update memory/prototypes only.
RELEASE_KNOWLEDGE_BACKBONE_BEFORE_TDPO = True   # In fast-safe mode, only the heavy Qwen backbone is released; the small distiller remains trainable.
KNOWLEDGE_FIT_EPOCHS_PER_UPDATE = 1
KNOWLEDGE_BATCH_SIZE = 16
KNOWLEDGE_VERBOSE = True
KNOWLEDGE_TRAIN_LOG_MODE = "both"       # epoch, step, both. "both" shows update progress and epoch summary.
KNOWLEDGE_PROGRESS_EVERY_BATCHES = 2
KNOWLEDGE_USE_LLM_VERBALIZER = False

# Fast-safe knowledge update mode.
# This keeps periodic knowledge weight updates, but trains only the lightweight
# distiller with deterministic segment embeddings instead of four Qwen text
# encoding passes per batch. The Qwen aspect adapters are frozen during TDPO-time
# updates to avoid unstable overwrites and to make the update much faster.
KNOWLEDGE_FAST_UPDATE_TEXT_ENCODER = True
KNOWLEDGE_FREEZE_LLM_ADAPTERS_DURING_TDPO_UPDATE = True
KNOWLEDGE_FIT_MAX_SEGMENTS_PER_UPDATE = 64
KNOWLEDGE_FIT_MAX_NEW_ATOMS_PER_UPDATE = 16

# Run-local memory capacity. These limits apply only to newly generated / run-local memory.
# External memory loaded below is kept separate and does not count toward these limits.
RUN_MEMORY_MAX_ATOMS = 20000
RUN_MEMORY_MAX_PROTOTYPES = 4096
CONSERVATIVE_MAX_NEW_ATOMS_PER_UPDATE = 16
CONSERVATIVE_MAX_TASK_RECORDS_PER_UPDATE = 8
CONSERVATIVE_COMPLETED_ONLY = True
CONSERVATIVE_MIN_COMPLETION_RATE_TO_UPDATE = 0.45

# Knowledge update strategy during TDPO.
# 1: conservative memory-only update; 2: normal update; 3: no update;
# 4: skip the first KNOWLEDGE_DELAYED_START_EPOCHS epochs, then conservative update.
KNOWLEDGE_UPDATE_STRATEGY = 3
KNOWLEDGE_DELAYED_START_EPOCHS = 10

# Optional external old-memory library. It is used only for retrieval augmentation and is not saved
# into run-local memory.json, so it does not consume RUN_MEMORY_MAX_* capacity.
USE_EXTERNAL_MEMORY_LIBRARY = False
# Historical threshold was 16.  It is kept only for config compatibility;
# external-vs-run memory is now selected by highest retrieval similarity.
EXTERNAL_MEMORY_USE_WHEN_RUN_PROTOTYPES_LT = 16
EXTERNAL_MEMORY_BLEND_WEIGHT = 0.10
EXTERNAL_MEMORY_APPEND_PROMPT = False

# Keep an explicit copy of the input pretrained knowledge checkpoint inside the run directory.
# The original PRETRAINED_KNOWLEDGE_MODEL_PATH is never overwritten.
COPY_PRETRAINED_KNOWLEDGE_TO_RUN_DIR = False

# ============================================================
# User-editable globals: utility, logging, and resource controls
# ============================================================

PRINT_EVERY_ENV_STEPS = 200
ROLLOUT_PRINT_EVERY_SECONDS = 20.0
ROLLOUT_PROGRESS_BAR_WIDTH = 28
ROLLOUT_DECISION_HEARTBEAT_EVERY = 50
ROLLOUT_SLOW_SELECT_SECONDS = 8.0
# Batch all frontier decision points in one Qwen forward when possible.
# 0 means no chunk limit for one env step; set a small positive value if OOM occurs.
BATCH_DECISION_ENABLED = True
BATCH_DECISION_MAX_SIZE = 0
BATCH_DECISION_FALLBACK_ON_ERROR = True
TDPO_UPDATE_PRINT_EVERY = 50
TDPO_UPDATE_PRINT_EVERY_SECONDS = 20.0
SAVE_TASK_RECORDS_EVERY_EPOCH = True
SAVE_TRACE_JSONL = True
WRITE_PROMPT_SAMPLES = True
MAX_PROMPT_SAMPLES = 20
SAVE_MEMORY_BOOTSTRAP_TASK_RECORDS = False
SAVE_LATEST_SUMMARY_JSON = False
SAVE_FINAL_RESULT_JSON = True
SAVE_AUXILIARY_JSON = False

TX_POWER_W = 0.10
RX_POWER_W = 0.06
TOOL_EXEC_POWER_W = 1.50
LOCAL_POWER_W = 0.90
PAUSE_POWER_W = 0.05

# CUDA memory behavior.  Set these before importing torch/transformers.
# Newer PyTorch prefers PYTORCH_ALLOC_CONF; remove the legacy variable unless
# the user explicitly disables the cleanup.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", CUDA_VISIBLE_DEVICES_DEFAULT)
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
if os.environ.get("TDPO_KEEP_LEGACY_CUDA_ALLOC_CONF", "0") != "1":
    os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)


# ============================================================
# Imports from local codebase
# ============================================================

import torch

def safe_cuda_empty_cache() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def print_cuda_layout(prefix: str = "[cuda]") -> None:
    if not torch.cuda.is_available():
        print(f"{prefix} cuda is not available", flush=True)
        return
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    print(f"{prefix} visible={visible} | device_count={torch.cuda.device_count()}", flush=True)
    for i in range(torch.cuda.device_count()):
        try:
            prop = torch.cuda.get_device_properties(i)
            total_gib = prop.total_memory / (1024 ** 3)
            print(f"{prefix} cuda:{i} {prop.name} total={total_gib:.2f}GiB", flush=True)
        except Exception:
            pass


from experiment_suite.shared import knowledge_adapter as knowledge_mod
from experiment_suite.tdpo_no_skill import decision as decision_mod
from experiment_suite.tdpo_no_skill import environment as task_env_mod


# ============================================================
# Basic IO helpers
# ============================================================


def now_tag() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _resolve_library_path(path: Path) -> Path:
    path = Path(path).expanduser()
    if not path.is_absolute():
        path = THIS_DIR / path
    return path


def _selected_tool_library_paths() -> List[Path]:
    if bool(USE_MULTIPLE_TOOL_LIBRARIES):
        return [_resolve_library_path(Path(p)) for p in TOOL_LIBRARY_PATHS]
    return [_resolve_library_path(Path(TOOL_LIBRARY_PATH))]


def load_dynamic_tool_library() -> Optional[Dict[str, object]]:
    global _DYNAMIC_TOOL_LIBRARY_CACHE, _DYNAMIC_TOOL_LIBRARY_CACHE_KEY
    if not bool(USE_DYNAMIC_TOOL_LIBRARY):
        if not bool(ALLOW_TOOL_LIBRARY_FALLBACK):
            raise RuntimeError("USE_DYNAMIC_TOOL_LIBRARY=False while ALLOW_TOOL_LIBRARY_FALLBACK=False. Enable the tool library or allow fallback.")
        return None

    paths = _selected_tool_library_paths()
    cache_key = tuple(str(p.resolve()) if p.exists() else str(p) for p in paths)
    if _DYNAMIC_TOOL_LIBRARY_CACHE is not None and _DYNAMIC_TOOL_LIBRARY_CACHE_KEY == cache_key:
        return _DYNAMIC_TOOL_LIBRARY_CACHE

    merged_entries: List[Dict[str, object]] = []
    seen_entry_keys = set()
    loaded_paths = []
    for path in paths:
        if not path.exists():
            msg = f"dynamic tool library file not found: {path}"
            if bool(ALLOW_TOOL_LIBRARY_FALLBACK):
                print(f"[train-decision] WARNING: {msg}; continuing with fallback allowed", flush=True)
                continue
            raise FileNotFoundError(msg)
        data = load_json(path)
        if not isinstance(data, dict) or not isinstance(data.get("entries", None), list):
            raise ValueError(f"Invalid tool library format. Expected a dict with an entries list: {path}")
        for entry in data.get("entries", []):
            if not isinstance(entry, dict):
                continue
            key = (str(entry.get("graph_id", "")), int(entry.get("node_id", -1)))
            if key in seen_entry_keys:
                continue
            seen_entry_keys.add(key)
            merged_entries.append(entry)
        loaded_paths.append(str(path))

    if not merged_entries and not bool(ALLOW_TOOL_LIBRARY_FALLBACK):
        raise RuntimeError("No dynamic tool library entries loaded and fallback is disabled.")

    merged = {
        "schema_version": "merged_api_tool_library_v2",
        "created_by": "train_TDPO.py",
        "source_paths": loaded_paths,
        "num_entries": len(merged_entries),
        "entries": merged_entries,
    }
    _DYNAMIC_TOOL_LIBRARY_CACHE = merged
    _DYNAMIC_TOOL_LIBRARY_CACHE_KEY = cache_key
    print(
        f"[train-decision] dynamic tool libraries loaded: files={len(loaded_paths)} | entries={len(merged_entries)} | fallback_allowed={bool(ALLOW_TOOL_LIBRARY_FALLBACK)}",
        flush=True,
    )
    return merged


def _tool_library_key_set(tool_library: Optional[Dict[str, object]]) -> set:
    keys = set()
    if not isinstance(tool_library, dict):
        return keys
    for entry in tool_library.get("entries", []) or []:
        if not isinstance(entry, dict):
            continue
        graph_id = str(entry.get("graph_id", ""))
        try:
            node_id = int(entry.get("node_id"))
        except Exception:
            continue
        for tool in entry.get("tools", []) or []:
            if not isinstance(tool, dict):
                continue
            try:
                slot = int(tool.get("slot", tool.get("tool_type_id")))
            except Exception:
                continue
            keys.add((graph_id, node_id, slot))
    return keys


def validate_dynamic_tool_library_coverage(dataset: Dict[str, object], tool_library: Optional[Dict[str, object]]) -> Dict[str, object]:
    keys = _tool_library_key_set(tool_library)
    missing: List[Dict[str, object]] = []
    required = 0
    for graph in dataset.get("graphs", []) or []:
        graph_id = str(graph.get("graph_id"))
        for node in graph.get("nodes", []) or []:
            node_id = int(node.get("node_id", 0))
            for slot, allowed in enumerate(list(node.get("allowed_tools_mask", []))):
                if int(allowed) != 1:
                    continue
                required += 1
                if (graph_id, node_id, int(slot)) not in keys:
                    if len(missing) < 20:
                        missing.append({"graph_id": graph_id, "node_id": node_id, "slot": int(slot)})
    covered = required - len(missing)
    # If there are more than 20 missing keys, count them exactly without storing all examples.
    missing_count = 0
    if required:
        for graph in dataset.get("graphs", []) or []:
            graph_id = str(graph.get("graph_id"))
            for node in graph.get("nodes", []) or []:
                node_id = int(node.get("node_id", 0))
                for slot, allowed in enumerate(list(node.get("allowed_tools_mask", []))):
                    if int(allowed) == 1 and (graph_id, node_id, int(slot)) not in keys:
                        missing_count += 1
    info = {
        "required_allowed_tool_slots": int(required),
        "covered_allowed_tool_slots": int(required - missing_count),
        "missing_allowed_tool_slots": int(missing_count),
        "coverage_rate": float((required - missing_count) / max(1, required)),
        "missing_examples": missing,
    }
    print(
        f"[tool-library:coverage] required={info['required_allowed_tool_slots']} covered={info['covered_allowed_tool_slots']} "
        f"missing={info['missing_allowed_tool_slots']} coverage={info['coverage_rate']:.4f}",
        flush=True,
    )
    if missing_count > 0 and not bool(ALLOW_TOOL_LIBRARY_FALLBACK):
        raise RuntimeError(
            "Dynamic tool library coverage check failed while ALLOW_TOOL_LIBRARY_FALLBACK=False. "
            f"missing={missing_count}, examples={missing}"
        )
    return info


def save_json(path: Path, data: object) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def append_jsonl(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def safe_mean(values: Sequence[float], default: float = 0.0) -> float:
    vals = [float(v) for v in values if v is not None]
    return float(sum(vals) / len(vals)) if vals else float(default)


def fmt_seconds(x: float) -> str:
    x = max(0.0, float(x))
    if x < 60.0:
        return f"{x:.1f}s"
    m, sec = divmod(int(x), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h:d}h{m:02d}m{sec:02d}s"
    return f"{m:d}m{sec:02d}s"


def text_progress_bar(done: int, total: int, width: int = 28) -> str:
    total = max(1, int(total))
    done = max(0, min(int(done), total))
    filled = int(round(float(done) / float(total) * int(width)))
    return "[" + "#" * filled + "." * (int(width) - filled) + "]"


def env_runtime_counts(env) -> Dict[str, int]:
    counts = defaultdict(int)
    try:
        for task in getattr(env, "tasks", {}).values():
            counts[f"task_{getattr(task, 'status', 'unknown')}"] += 1
            for node in getattr(task, "nodes", {}).values():
                counts[f"node_{getattr(node, 'status', 'unknown')}"] += 1
    except Exception as exc:
        counts["count_error"] = 1
    return dict(counts)


def compact_env_counts(counts: Dict[str, int]) -> str:
    keys = [
        "task_waiting", "task_active", "task_completed", "task_failed",
        "node_ready", "node_queued", "node_queued_local", "node_running_tool",
        "node_running_local", "node_paused", "node_done",
    ]
    parts = [f"{k}={int(counts.get(k, 0))}" for k in keys if int(counts.get(k, 0)) > 0]
    return " ".join(parts) if parts else "empty"


# ============================================================
# Dataset loading
# ============================================================


def list_dataset_files(data_dir: Path) -> List[Path]:
    files: List[Path] = []
    seen = set()
    for pattern in DATASET_FILE_PATTERNS:
        for path in sorted(data_dir.glob(pattern)):
            if path.name in DATASET_METADATA_FILES:
                continue
            rp = str(path.resolve())
            if rp not in seen:
                files.append(path)
                seen.add(rp)
    return files


def load_dataset_from_dir(data_dir: Path, seed: int = SEED) -> Tuple[Dict[str, object], Dict[str, object]]:
    if not data_dir.exists():
        raise FileNotFoundError(f"DATA_DIR does not exist: {data_dir}")

    all_dataset_files = list_dataset_files(data_dir)
    if not all_dataset_files:
        raise RuntimeError(f"No dataset files found under {data_dir} with patterns {DATASET_FILE_PATTERNS}")

    pool_limit = int(SAMPLE_RANDOM_POOL_LIMIT) if int(SAMPLE_RANDOM_POOL_LIMIT) > 0 else len(all_dataset_files)
    candidate_files = list(all_dataset_files[: min(pool_limit, len(all_dataset_files))])

    rng = random.Random(seed)
    sample_n = int(SAMPLE_NUM_GRAPHS if SAMPLE_NUM_GRAPHS is not None else MAX_TRAIN_GRAPHS)
    if sample_n and sample_n > 0:
        if bool(SAMPLE_IN_ORDER):
            dataset_files = candidate_files[: sample_n]
        else:
            dataset_files = rng.sample(candidate_files, k=min(sample_n, len(candidate_files)))
    else:
        dataset_files = list(candidate_files)
        if SHUFFLE_DATASET_ON_LOAD:
            rng.shuffle(dataset_files)

    metadata = {}
    metadata_path = None
    for name in DATASET_METADATA_FILES:
        p = data_dir / name
        if p.exists():
            metadata_path = p
            metadata = load_json(p)
            break

    raw_items = [load_json(path) for path in dataset_files]
    dataset = {
        "schema_version": metadata.get("schema_version", "2.0.0"),
        "dataset_name": f"tdpo_dataset_from_{data_dir.name}",
        "config": {
            "source": "disk_pool",
            "data_dir": str(data_dir),
            "sample_count": len(raw_items),
            "seed": int(seed),
        },
        "task_type_specs": metadata.get("task_type_specs", []),
        "tool_catalog": metadata.get("tool_catalog", []),
        "graphs": raw_items,
    }
    dataset = task_env_mod._normalize_dataset_structure(dataset, fallback_dataset_name=dataset["dataset_name"])

    scan_info = {
        "data_dir": str(data_dir),
        "metadata_path": str(metadata_path) if metadata_path else None,
        "num_all_files": len(all_dataset_files),
        "sample_random_pool_limit": int(SAMPLE_RANDOM_POOL_LIMIT),
        "num_candidate_files": len(candidate_files),
        "num_input_files": len(dataset_files),
        "num_graphs_after_normalization": len(dataset.get("graphs", [])),
        "num_tool_types": len(dataset.get("tool_catalog", [])),
        "file_names_head": [p.name for p in dataset_files[:10]],
    }
    return dataset, scan_info


def subset_dataset(dataset: Dict[str, object], max_graphs: int) -> Dict[str, object]:
    if not max_graphs or int(max_graphs) <= 0:
        return dataset
    out = dict(dataset)
    out["graphs"] = list(dataset.get("graphs", []))[: int(max_graphs)]
    out["config"] = dict(dataset.get("config", {}))
    out["config"]["subsampled_graphs"] = int(max_graphs)
    return task_env_mod._normalize_dataset_structure(out, fallback_dataset_name=str(dataset.get("dataset_name", "tdpo_subset")))


def build_graph_meta_map(dataset: Dict[str, object]) -> Dict[str, Dict[str, object]]:
    meta = {}
    for g in dataset.get("graphs", []):
        gid = str(g.get("graph_id"))
        meta[gid] = g
    return meta


# ============================================================
# Environment helpers
# ============================================================


def build_relaxed_tool_profiles(dataset: Dict[str, object]) -> Optional[List[object]]:
    """Build base tool profiles with relaxed instance counts and queue lengths.

    task_env.py still resolves per-(graph,node,slot) timing attributes from the
    dynamic tool library.  This helper only scales slot-level resource capacity:
    concurrent instances and queue capacity.  Returning None restores the
    original task_env default profiles.
    """
    if not bool(ENV_RELAXED_RESOURCE_MODE):
        return None
    num_tools = len(dataset.get("tool_catalog", []))
    profiles: List[object] = []
    for i in range(int(num_tools)):
        base_instances = 1 + (i % 3)
        scaled_instances = int(math.ceil(float(base_instances) * float(ENV_TOOL_INSTANCE_MULTIPLIER))) + int(ENV_TOOL_INSTANCE_ADDITIVE)
        scaled_instances = max(int(ENV_MIN_TOOL_INSTANCES), scaled_instances)
        profiles.append(
            task_env_mod.ToolTypeProfile(
                tool_type_id=i,
                name=f"tool_type_{i}",
                instances=int(scaled_instances),
                compute_frequency_hz=2.5e9 + 0.4e9 * i,
                uplink_bandwidth_hz=1.2e6 + 0.15e6 * i,
                downlink_bandwidth_hz=1.0e6 + 0.12e6 * i,
                queue_capacity=int(ENV_Q_MAX),
                coverage_distance_m=60.0 + 8.0 * i,
                server_reliability=max(0.85, min(0.995, 0.90 + 0.01 * i)),
                info_richness=max(0.4, min(0.98, 0.55 + 0.04 * i)),
                validity_horizon_s=4.0 + 1.0 * i,
            )
        )
    return profiles


def build_env(dataset: Dict[str, object], seed: int):
    env_cfg = task_env_mod.EnvironmentConfig(
        dt=float(ENV_DT),
        seed=int(seed),
        q_max=int(ENV_Q_MAX),
        task_arrival_default_gap_range=tuple(ENV_ARRIVAL_GAP_RANGE),
        metrics_record_period_steps=int(ENV_METRICS_RECORD_PERIOD_STEPS),
        node_deadline_multiplier=float(ENV_NODE_DEADLINE_MULTIPLIER),
        task_deadline_multiplier=float(ENV_TASK_DEADLINE_MULTIPLIER),
        min_node_deadline_s=float(ENV_MIN_NODE_DEADLINE_S),
        min_task_deadline_s=float(ENV_MIN_TASK_DEADLINE_S),
        local_max_concurrency=int(ENV_LOCAL_MAX_CONCURRENCY),
        local_queue_capacity=int(ENV_LOCAL_QUEUE_CAPACITY),
        retry_restart_penalty_s=float(ENV_RETRY_RESTART_PENALTY_S),
        invalid_action_mode=str(ENV_INVALID_ACTION_MODE),
        invalid_action_penalty_s=float(ENV_INVALID_ACTION_PENALTY_S),
        fail_on_queue_rejection=bool(ENV_FAIL_ON_QUEUE_REJECTION),
        max_invalid_attempts_per_node=int(ENV_MAX_INVALID_ATTEMPTS_PER_NODE),
        target_tool_compute_s=float(ENV_TARGET_TOOL_COMPUTE_S),
        target_tool_uplink_s=float(ENV_TARGET_TOOL_UPLINK_S),
        target_tool_downlink_s=float(ENV_TARGET_TOOL_DOWNLINK_S),
    )
    tool_library = load_dynamic_tool_library()
    tool_profiles = build_relaxed_tool_profiles(dataset)
    env = task_env_mod.MissionEnvironment(
        dataset=dataset,
        env_cfg=env_cfg,
        tool_profiles=tool_profiles,
        tool_library=tool_library,
    )
    return env


def compute_env_step_limit(dataset: Dict[str, object]) -> int:
    if FORCE_MAX_ENV_STEPS and int(FORCE_MAX_ENV_STEPS) > 0:
        return int(FORCE_MAX_ENV_STEPS)
    num_tasks = len(dataset.get("graphs", []))
    num_nodes = sum(len(g.get("nodes", [])) for g in dataset.get("graphs", []))
    avg_nodes = num_nodes / max(1, num_tasks)
    arrival_span_s = max(1.0, float(ENV_ARRIVAL_GAP_RANGE[1]) * max(1, num_tasks + 2))
    max_deadline = max([float(g.get("task_deadline", 0.0) or 0.0) for g in dataset.get("graphs", [])] + [float(ENV_MIN_TASK_DEADLINE_S)])
    horizon_s = arrival_span_s + 3.0 * max_deadline + 2.0 * avg_nodes * max(1.0, ENV_TARGET_TOOL_COMPUTE_S)
    return int(max(1000, math.ceil(horizon_s / max(1e-6, float(ENV_DT))) + 100))


def build_complete_task_records(env) -> List[Dict[str, object]]:
    finalized = {str(rec["task_id"]): rec for rec in env.completed_task_records}
    for task_id, task in env.tasks.items():
        if str(task_id) in finalized:
            continue
        finalized[str(task_id)] = {
            "task_id": task.graph_id,
            "status": task.status,
            "arrival_time": task.arrival_time,
            "end_time": task.end_time,
            "latency_s": None if task.end_time is None else round(task.end_time - task.arrival_time, 4),
            "task_deadline_s": float(getattr(task, "task_deadline", 0.0)),
            "restart_count": task.restart_count,
            "failure_reason": task.failure_reason,
            "log": list(task.log),
        }
    return [finalized[k] for k in sorted(finalized.keys())]


# ============================================================
# Heuristic policy for memory bootstrap and fallback
# ============================================================


def best_tool_option(dp: Dict[str, object]) -> Optional[Dict[str, object]]:
    feasible = [opt for opt in dp.get("tool_options", []) if not bool(opt.get("queue_blocked", False))]
    if not feasible:
        return None
    feasible.sort(key=lambda x: (
        float(x.get("predicted_total_s", 1e9)),
        float(x.get("risk_estimate", 1e9)),
        int(x.get("tool_type_id", 10**9)),
    ))
    return feasible[0]


def choose_heuristic_action(policy_name: str, dp: Dict[str, object], rng: random.Random):
    local_opt = dp.get("local_option", {}) or {}
    local_feasible = not bool(local_opt.get("queue_blocked", False))
    best_tool = best_tool_option(dp)

    if policy_name == "completion_first_feasible":
        # Memory bootstrap preference only: rank feasible actions with the same
        # completion-first utility used by TDPO preference construction.  This
        # avoids seeding memory with a pure fastest-feasible bias while leaving
        # the environment dynamics, action space, and TDPO update logic unchanged.
        deadline = max(1.0e-6, float(dp.get("node_remaining_deadline_s", 1.0) or 1.0))
        candidates = []

        for opt in dp.get("tool_options", []):
            if bool(opt.get("queue_blocked", False)):
                continue
            latency = float(opt.get("predicted_total_s", 1e9) or 1e9)
            risk = max(0.0, min(1.0, float(opt.get("risk_estimate", 0.0) or 0.0)))
            loss = max(0.0, min(1.0, float(opt.get("packet_loss_rate", 0.0) or 0.0)))
            success_prob = max(0.0, min(1.0, 1.0 - 0.65 * risk - 0.35 * loss))
            violation = 1.0 if latency > deadline else (1.0 - success_prob)
            utility = decision_mod.completion_first_utility(
                node_success=success_prob,
                mission_success=success_prob,
                violation=violation,
                latency_s=latency,
                deadline_s=deadline,
            )
            action = task_env_mod.DecisionAction(action_type="tool", tool_type_id=int(opt["tool_type_id"]))
            candidates.append((utility, -latency, -risk, action))

        if local_feasible:
            latency = float(local_opt.get("predicted_total_s", local_opt.get("compute_s", 1e9)) or 1e9)
            fail_prob = max(0.0, min(1.0, float(local_opt.get("fail_probability", 0.0) or 0.0)))
            success_prob = max(0.0, min(1.0, 1.0 - fail_prob))
            violation = 1.0 if latency > deadline else fail_prob
            utility = decision_mod.completion_first_utility(
                node_success=success_prob,
                mission_success=success_prob,
                violation=violation,
                latency_s=latency,
                deadline_s=deadline,
            )
            action = task_env_mod.DecisionAction(action_type="local")
            candidates.append((utility, -latency, -fail_prob, action))

        if candidates:
            return max(candidates, key=lambda x: (x[0], x[1], x[2]))[3]
        return task_env_mod.DecisionAction(action_type="pause", pause_duration_s=0.20)

    if policy_name == "tool_preferred":
        if best_tool is not None:
            return task_env_mod.DecisionAction(action_type="tool", tool_type_id=int(best_tool["tool_type_id"]))
        if local_feasible:
            return task_env_mod.DecisionAction(action_type="local")
        return task_env_mod.DecisionAction(action_type="pause", pause_duration_s=0.20)

    if policy_name == "local_preferred":
        if local_feasible:
            return task_env_mod.DecisionAction(action_type="local")
        if best_tool is not None:
            return task_env_mod.DecisionAction(action_type="tool", tool_type_id=int(best_tool["tool_type_id"]))
        return task_env_mod.DecisionAction(action_type="pause", pause_duration_s=0.20)

    if policy_name == "random_feasible":
        candidates = []
        for opt in dp.get("tool_options", []):
            if not bool(opt.get("queue_blocked", False)):
                candidates.append(task_env_mod.DecisionAction(action_type="tool", tool_type_id=int(opt["tool_type_id"])))
        if local_feasible:
            candidates.append(task_env_mod.DecisionAction(action_type="local"))
        if candidates:
            return rng.choice(candidates)
        return task_env_mod.DecisionAction(action_type="pause", pause_duration_s=0.20)

    # fastest_feasible
    best_score = float("inf")
    best_action = None
    if local_feasible:
        local_score = float(local_opt.get("predicted_total_s", local_opt.get("compute_s", 1e9)))
        best_score = local_score
        best_action = task_env_mod.DecisionAction(action_type="local")
    if best_tool is not None:
        tool_score = float(best_tool.get("predicted_total_s", 1e9))
        if tool_score < best_score:
            best_action = task_env_mod.DecisionAction(action_type="tool", tool_type_id=int(best_tool["tool_type_id"]))
    return best_action or task_env_mod.DecisionAction(action_type="pause", pause_duration_s=0.20)


# ============================================================
# Metrics
# ============================================================


def _event_energy_proxy(ev: Dict[str, object]) -> float:
    et = str(ev.get("event_type", ""))
    payload = ev.get("payload", {}) or {}
    if et == "decision_tool":
        return float(
            TX_POWER_W * float(payload.get("uplink_s", 0.0) or 0.0)
            + RX_POWER_W * float(payload.get("downlink_s", 0.0) or 0.0)
            + TOOL_EXEC_POWER_W * float(payload.get("exec_s", 0.0) or 0.0)
        )
    if et == "decision_local":
        return float(LOCAL_POWER_W * float(payload.get("compute_s", 0.0) or 0.0))
    if et == "decision_pause":
        return float(PAUSE_POWER_W * float(payload.get("pause_duration_s", 0.0) or 0.0))
    return 0.0


def _record_energy_proxy(record: Dict[str, object]) -> float:
    return float(sum(_event_energy_proxy(ev) for ev in list(record.get("log", []) or [])))


def estimate_energy_proxy(task_records: Sequence[Dict[str, object]]) -> float:
    return float(sum(_record_energy_proxy(rec) for rec in task_records))


def _record_latency_for_summary(record: Dict[str, object]) -> float:
    """Latency metric used for reporting.

    Completed tasks use their observed latency. Failed/unfinished tasks are
    charged by the task deadline so average latency is not biased toward only
    successful tasks.
    """
    status = str(record.get("status"))
    latency = record.get("latency_s")
    if status == "completed" and latency is not None:
        return float(latency)
    deadline = record.get("task_deadline_s", record.get("task_deadline", record.get("deadline_s", None)))
    if deadline is not None:
        return max(0.0, float(deadline))
    if latency is not None:
        return max(0.0, float(latency))
    arrival = record.get("arrival_time")
    end = record.get("end_time")
    if arrival is not None and end is not None:
        return max(0.0, float(end) - float(arrival))
    return 0.0


def _completed_task_records(task_records: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    return [r for r in task_records if str(r.get("status")) == "completed"]


def _completed_task_latency(record: Dict[str, object]) -> float:
    latency = record.get("latency_s")
    if latency is not None:
        return max(0.0, float(latency))
    arrival = record.get("arrival_time")
    end = record.get("end_time")
    if arrival is not None and end is not None:
        return max(0.0, float(end) - float(arrival))
    return 0.0


def _node_delay_from_events(events: Sequence[Dict[str, object]]) -> float:
    evs = list(events)
    success_events = [ev for ev in evs if str(ev.get("event_type")) in {"tool_success", "local_success"}]
    if not success_events:
        return 0.0
    last_success = success_events[-1]
    finish_time = float(last_success.get("time", 0.0) or 0.0)
    start_candidates = [
        float(ev.get("time", 0.0) or 0.0)
        for ev in evs
        if str(ev.get("event_type")) in {
            "decision_tool",
            "decision_local",
            "decision_pause",
            "tool_execution_started",
            "local_execution_started",
        }
    ]
    if start_candidates:
        return max(0.0, finish_time - min(start_candidates))

    payload = last_success.get("payload", {}) or {}
    if str(last_success.get("event_type")) == "tool_success":
        return float(
            float(payload.get("queue_delay_s", 0.0) or 0.0)
            + float(payload.get("uplink_s", 0.0) or 0.0)
            + float(payload.get("exec_s", 0.0) or 0.0)
            + float(payload.get("downlink_s", 0.0) or 0.0)
        )
    return float(
        float(payload.get("queue_delay_s", 0.0) or 0.0)
        + float(payload.get("compute_s", payload.get("predicted_compute_s", 0.0)) or 0.0)
    )


def _completed_node_metric_rows(record: Dict[str, object]) -> List[Dict[str, object]]:
    """Return per-node energy and delay rows for a completed graph only.

    Failed graphs are intentionally excluded, even if some of their nodes reached
    a done state before the graph failed.  For a completed graph, each returned
    node row aggregates all logged work for that node inside the graph record,
    including queueing, pause, retry, local, and tool decisions.
    """
    if str(record.get("status")) != "completed":
        return []
    by_node: Dict[int, List[Dict[str, object]]] = defaultdict(list)
    for ev in list(record.get("log", []) or []):
        node_id = ev.get("node_id")
        if node_id is None:
            continue
        try:
            by_node[int(node_id)].append(ev)
        except Exception:
            continue

    rows: List[Dict[str, object]] = []
    for node_id in sorted(by_node):
        events = by_node[node_id]
        has_success = any(str(ev.get("event_type")) in {"tool_success", "local_success"} for ev in events)
        if not has_success:
            continue
        rows.append(
            {
                "task_id": str(record.get("task_id", "")),
                "node_id": int(node_id),
                "energy_proxy": float(sum(_event_energy_proxy(ev) for ev in events)),
                "delay_s": float(_node_delay_from_events(events)),
            }
        )
    return rows


def completed_task_metric_summary(task_records: Sequence[Dict[str, object]]) -> Dict[str, object]:
    completed = _completed_task_records(task_records)
    node_rows: List[Dict[str, object]] = []
    for rec in completed:
        node_rows.extend(_completed_node_metric_rows(rec))
    total_energy = float(sum(_record_energy_proxy(rec) for rec in completed))
    total_delay = float(sum(_completed_task_latency(rec) for rec in completed))
    return {
        "completed_total_energy_proxy": total_energy,
        "avg_completed_energy_proxy_per_graph": float(total_energy / max(1, len(completed))),
        "completed_total_delay_s": total_delay,
        "avg_completed_delay_s_per_graph": float(total_delay / max(1, len(completed))),
        "num_completed_nodes_logged": int(len(node_rows)),
    }


def build_energy_delay_detail_rows(
    epoch: int,
    stage: str,
    task_records: Sequence[Dict[str, object]],
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    completed = _completed_task_records(task_records)
    node_rows: List[Dict[str, object]] = []
    for rec in completed:
        node_rows.extend(_completed_node_metric_rows(rec))

    total_energy = float(sum(_record_energy_proxy(rec) for rec in completed))
    total_delay = float(sum(_completed_task_latency(rec) for rec in completed))
    base = {
        "stage": str(stage),
        "epoch": int(epoch),
        "num_completed_tasks": int(len(completed)),
        "num_completed_nodes": int(len(node_rows)),
    }

    energy_rows: List[Dict[str, object]] = [
        {
            **base,
            "record_type": "epoch_total",
            "task_id": "",
            "node_id": "",
            "status": "completed_only",
            "energy_proxy": total_energy,
            "epoch_completed_total_energy_proxy": total_energy,
        }
    ]
    delay_rows: List[Dict[str, object]] = [
        {
            **base,
            "record_type": "epoch_total",
            "task_id": "",
            "node_id": "",
            "status": "completed_only",
            "delay_s": total_delay,
            "epoch_completed_total_delay_s": total_delay,
        }
    ]

    for rec in completed:
        task_id = str(rec.get("task_id", ""))
        task_energy = float(_record_energy_proxy(rec))
        task_delay = float(_completed_task_latency(rec))
        energy_rows.append(
            {
                **base,
                "record_type": "task",
                "task_id": task_id,
                "node_id": "",
                "status": "completed",
                "energy_proxy": task_energy,
                "epoch_completed_total_energy_proxy": total_energy,
            }
        )
        delay_rows.append(
            {
                **base,
                "record_type": "task",
                "task_id": task_id,
                "node_id": "",
                "status": "completed",
                "delay_s": task_delay,
                "epoch_completed_total_delay_s": total_delay,
            }
        )

    for item in node_rows:
        energy_rows.append(
            {
                **base,
                "record_type": "node",
                "task_id": item.get("task_id", ""),
                "node_id": item.get("node_id", ""),
                "status": "completed",
                "energy_proxy": float(item.get("energy_proxy", 0.0)),
                "epoch_completed_total_energy_proxy": total_energy,
            }
        )
        delay_rows.append(
            {
                **base,
                "record_type": "node",
                "task_id": item.get("task_id", ""),
                "node_id": item.get("node_id", ""),
                "status": "completed",
                "delay_s": float(item.get("delay_s", 0.0)),
                "epoch_completed_total_delay_s": total_delay,
            }
        )
    return energy_rows, delay_rows


def write_energy_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    ensure_dir(path.parent)
    fieldnames = [
        "stage",
        "epoch",
        "record_type",
        "task_id",
        "node_id",
        "status",
        "energy_proxy",
        "epoch_completed_total_energy_proxy",
        "num_completed_tasks",
        "num_completed_nodes",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})


def write_delay_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    ensure_dir(path.parent)
    fieldnames = [
        "stage",
        "epoch",
        "record_type",
        "task_id",
        "node_id",
        "status",
        "delay_s",
        "epoch_completed_total_delay_s",
        "num_completed_tasks",
        "num_completed_nodes",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})

def summarize_task_records(task_records: Sequence[Dict[str, object]], policy_name: str, steps_taken: int, hit_step_limit: bool) -> Dict[str, object]:
    num_tasks = len(task_records)
    completed = [r for r in task_records if str(r.get("status")) == "completed"]
    failed = [r for r in task_records if str(r.get("status")) == "failed"]
    unfinished = [r for r in task_records if str(r.get("status")) not in {"completed", "failed"}]
    latencies = [_record_latency_for_summary(r) for r in task_records]
    energy = estimate_energy_proxy(task_records)
    return {
        "policy": policy_name,
        "num_tasks": int(num_tasks),
        "num_completed": int(len(completed)),
        "num_failed": int(len(failed)),
        "num_unfinished": int(len(unfinished)),
        "completion_rate": float(len(completed) / max(1, num_tasks)),
        "avg_latency_s": safe_mean(latencies, 0.0),
        "energy_proxy": float(energy),
        "avg_energy_proxy_per_task": float(energy / max(1, num_tasks)),
        **completed_task_metric_summary(task_records),
        "steps_taken": int(steps_taken),
        "hit_step_limit": bool(hit_step_limit),
    }


def write_epoch_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    ensure_dir(path.parent)
    fieldnames = [
        "epoch",
        "completion_rate",
        "avg_latency_s",
        "energy_proxy",
        "completed_total_energy_proxy",
        "avg_completed_energy_proxy_per_graph",
        "completed_total_delay_s",
        "avg_completed_delay_s_per_graph",
        "num_completed_nodes_logged",
        "num_completed",
        "num_failed",
        "num_unfinished",
        "steps_taken",
        "hit_step_limit",
        "num_traces",
        "num_pairs_added",
        "num_pairs_buffer",
        "num_success_buffer",
        "num_explore_buffer",
        "num_pairs_train",
        "num_tier_a_pairs",
        "num_tier_b_pairs",
        "num_tier_a_raw",
        "num_tier_b_raw",
        "tdpo_loss",
        "tdpo_pref_acc",
        "tdpo_margin",
        "tdpo_confidence",
        "tdpo_grad_norm",
        "tdpo_grad_norm_clipped",
        "tdpo_completion_pair_ratio",
        "tdpo_completion_advantage_ratio",
        "tdpo_node_success_advantage_ratio",
        "tdpo_pair_gap_mean",
        "rollout_seconds",
        "pair_seconds",
        "new_ref_seconds",
        "tdpo_seconds",
        "tdpo_ref_seconds",
        "tdpo_update_seconds",
        "knowledge_update_seconds",
        "epoch_seconds",
        "knowledge_num_atoms",
        "knowledge_num_prototypes",
        "is_best",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})


def write_eval_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    ensure_dir(path.parent)
    fieldnames = [
        "epoch",
        "completion_rate",
        "avg_latency_s",
        "energy_proxy",
        "completed_total_energy_proxy",
        "avg_completed_energy_proxy_per_graph",
        "completed_total_delay_s",
        "avg_completed_delay_s_per_graph",
        "num_completed_nodes_logged",
        "num_completed",
        "num_failed",
        "num_unfinished",
        "steps_taken",
        "hit_step_limit",
        "num_traces",
        "rollout_seconds",
        "eval_seconds",
        "eval_seed",
        "use_memory",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})


def write_knowledge_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    ensure_dir(path.parent)
    fieldnames = [
        "stage",
        "epoch",
        "mode",
        "loss_total",
        "loss_topo",
        "loss_wire",
        "loss_tool",
        "loss_out",
        "loss_need",
        "loss_atom",
        "num_segments",
        "num_new_atoms",
        "num_new_prototypes",
        "num_atoms",
        "num_prototypes",
        "prototype_risk_mean",
        "prototype_risk_drift",
        "fit_seconds",
        "total_updates",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})


def build_knowledge_row(stage: str, epoch: int, stats: Dict[str, object], digest: Dict[str, object]) -> Dict[str, object]:
    stats = stats or {}
    digest = digest or {}
    return {
        "stage": str(stage),
        "epoch": int(epoch),
        "mode": str(stats.get("mode", "fit" if stats.get("total_updates", 0) else "memory_only")),
        "loss_total": float(stats.get("loss_total", 0.0)),
        "loss_topo": float(stats.get("loss_topo", 0.0)),
        "loss_wire": float(stats.get("loss_wire", 0.0)),
        "loss_tool": float(stats.get("loss_tool", 0.0)),
        "loss_out": float(stats.get("loss_out", 0.0)),
        "loss_need": float(stats.get("loss_need", 0.0)),
        "loss_atom": float(stats.get("loss_atom", 0.0)),
        "num_segments": int(stats.get("num_segments", 0)),
        "num_new_atoms": int(stats.get("num_new_atoms", 0)),
        "num_new_prototypes": int(stats.get("num_new_prototypes", 0)),
        "num_atoms": int(digest.get("num_atoms", 0)),
        "num_prototypes": int(digest.get("num_prototypes", 0)),
        "prototype_risk_mean": float(stats.get("prototype_risk_mean", 0.0)),
        "prototype_risk_drift": float(stats.get("prototype_risk_drift", 0.0)),
        "fit_seconds": float(stats.get("fit_seconds", 0.0)),
        "total_updates": int(stats.get("total_updates", 0)),
    }


# ============================================================
# Knowledge loading and run-local memory generation
# ============================================================


def enforce_run_memory_limits(knowledge_trainer) -> Dict[str, int]:
    """Trim only the run-local memory owned by knowledge_trainer.

    External memory, when enabled, is stored outside knowledge_trainer.memory and is never counted here.
    """
    max_atoms = int(max(0, RUN_MEMORY_MAX_ATOMS))
    max_prototypes = int(max(0, RUN_MEMORY_MAX_PROTOTYPES))
    before_atoms = len(getattr(knowledge_trainer.memory, "buffer", []))
    before_prototypes = len(getattr(knowledge_trainer.memory, "prototypes", []))

    if max_atoms > 0 and before_atoms > max_atoms:
        knowledge_trainer.memory.buffer = list(knowledge_trainer.memory.buffer)[-max_atoms:]
    if max_prototypes > 0 and before_prototypes > max_prototypes:
        protos = list(knowledge_trainer.memory.prototypes)
        protos.sort(key=lambda p: (int(getattr(p, "support", 0)), float(getattr(p, "confidence_mean", 0.0))), reverse=True)
        knowledge_trainer.memory.prototypes = protos[:max_prototypes]

    after_atoms = len(getattr(knowledge_trainer.memory, "buffer", []))
    after_prototypes = len(getattr(knowledge_trainer.memory, "prototypes", []))
    return {
        "atoms_before": int(before_atoms),
        "atoms_after": int(after_atoms),
        "prototypes_before": int(before_prototypes),
        "prototypes_after": int(after_prototypes),
    }


def _memory_result_from_retrieved(memory_obj, retrieved, action_dim: int, cfg, state_query_embedding: List[float]) -> Dict[str, object]:
    if not retrieved:
        return {
            "memory_vector": [0.0] * (int(action_dim) + 4),
            "action_bias": [0.0] * int(action_dim),
            "selected": [],
            "knowledge_prompt": "",
        }
    total_weight = sum(max(1e-6, score) for _, score in retrieved)
    risk = 0.0
    conf = 0.0
    support = 0.0
    action_bias = [0.0] * int(action_dim)
    selected = []
    prompt_lines = []
    for proto, score in retrieved:
        w = max(1e-6, score) / max(1e-12, total_weight)
        risk += w * float(getattr(proto, "no_tool_risk_mean", 0.0))
        conf += w * float(getattr(proto, "confidence_mean", 0.0))
        support += w * float(getattr(proto, "support", 0))
        gains = list(getattr(proto, "action_gain_means", []) or [])
        for i in range(min(int(action_dim), len(gains))):
            action_bias[i] += w * float(gains[i])
        selected.append({
            "prototype_id": str(getattr(proto, "prototype_id", "")),
            "score": float(score),
            "support": int(getattr(proto, "support", 0)),
            "node_type_id": int(getattr(proto, "node_type_id", 0)),
            "queue_class": int(getattr(proto, "queue_class", 0)),
            "slack_class": int(getattr(proto, "slack_class", 0)),
            "source": "external_memory",
        })
        prompt_text = str(getattr(proto, "prompt_text", "") or "")
        if prompt_text:
            prompt_lines.append(prompt_text)
    k_retrieve = int(getattr(cfg, "k_retrieve", 3))
    memory_vector = [risk, conf, support / 10.0, float(len(retrieved)) / max(1.0, float(k_retrieve))] + list(action_bias)
    return {
        "memory_vector": memory_vector,
        "action_bias": action_bias,
        "selected": selected,
        "knowledge_prompt": "\n".join(prompt_lines[:k_retrieve]),
        "query_embedding_dim": len(state_query_embedding),
        "proto_embedding_dim": memory_obj.embedding_dim() if hasattr(memory_obj, "embedding_dim") else 0,
    }


def load_external_memory(action_dim: int, cfg) -> Optional[object]:
    if not bool(USE_EXTERNAL_MEMORY_LIBRARY):
        return None
    if EXTERNAL_MEMORY_LIBRARY_PATH is None:
        print("[external-memory] disabled because EXTERNAL_MEMORY_LIBRARY_PATH is None", flush=True)
        return None
    path = Path(EXTERNAL_MEMORY_LIBRARY_PATH).expanduser()
    if not path.is_absolute():
        path = THIS_DIR / path
    if not path.exists():
        raise FileNotFoundError(f"EXTERNAL_MEMORY_LIBRARY_PATH does not exist: {path}")
    data = load_json(path)
    memory = knowledge_mod.PrototypeMemory(action_dim=int(action_dim), cfg=cfg)
    memory.load_json(data)
    print(
        f"[external-memory] loaded: {path} | atoms={len(memory.buffer)} | prototypes={len(memory.prototypes)} | not_counted_in_run_memory=True",
        flush=True,
    )
    return memory


def attach_external_memory_query(knowledge_trainer, external_memory) -> None:
    if external_memory is None:
        return
    original_query_memory = knowledge_trainer.query_memory
    action_dim = int(knowledge_trainer.action_dim)
    cfg = knowledge_trainer.cfg

    def _best_score(out: Dict[str, object]) -> float:
        vals = []
        for item in list(out.get("selected", []) or []):
            if isinstance(item, dict):
                try:
                    vals.append(float(item.get("score", float("-inf"))))
                except Exception:
                    pass
        return max(vals) if vals else float("-inf")

    def _combined_query_memory(state_query_embedding: List[float], node_type_id: int, queue_class: int, slack_class: int) -> Dict[str, object]:
        run_out = original_query_memory(state_query_embedding, node_type_id, queue_class, slack_class)
        retrieved = external_memory.query(
            query_embedding=state_query_embedding,
            node_type_id=int(node_type_id),
            queue_class=int(queue_class),
            slack_class=int(slack_class),
            k=int(cfg.k_retrieve),
        )
        ext_out = _memory_result_from_retrieved(external_memory, retrieved, action_dim, cfg, state_query_embedding)

        run_score = _best_score(run_out)
        ext_score = _best_score(ext_out)
        if ext_score > run_score:
            ext_out = dict(ext_out)
            ext_out["external_memory_used"] = True
            ext_out["selected_memory_source"] = "external_memory"
            ext_out["selected_memory_score"] = float(ext_score)
            ext_out["run_memory_best_score"] = float(run_score) if math.isfinite(run_score) else None
            return ext_out

        run_out = dict(run_out)
        run_out["external_memory_used"] = False
        run_out["selected_memory_source"] = "run_memory" if run_out.get("selected") else "none"
        run_out["selected_memory_score"] = float(run_score) if math.isfinite(run_score) else None
        run_out["external_memory_best_score"] = float(ext_score) if math.isfinite(ext_score) else None
        return run_out

    knowledge_trainer.query_memory = _combined_query_memory
    knowledge_trainer.external_memory = external_memory


def copy_pretrained_knowledge_to_run_dir(run_dir: Path) -> Optional[str]:
    if not bool(COPY_PRETRAINED_KNOWLEDGE_TO_RUN_DIR):
        return None
    if not PRETRAINED_KNOWLEDGE_MODEL_PATH.exists():
        return None
    ensure_dir(run_dir / KNOWLEDGE_DIR)
    dst = run_dir / KNOWLEDGE_DIR / PRETRAINED_INPUT_COPY_FILE
    if not dst.exists():
        shutil.copy2(PRETRAINED_KNOWLEDGE_MODEL_PATH, dst)
    return str(dst.resolve())


def load_knowledge_trainer(dataset: Dict[str, object], action_dim: int, run_dir: Path):
    if not PRETRAINED_KNOWLEDGE_MODEL_PATH.exists():
        raise FileNotFoundError(f"PRETRAINED_KNOWLEDGE_MODEL_PATH does not exist: {PRETRAINED_KNOWLEDGE_MODEL_PATH}")

    copied_path = copy_pretrained_knowledge_to_run_dir(run_dir)
    if copied_path:
        print(f"[knowledge-load] copied pretrained input checkpoint to run dir: {copied_path}", flush=True)

    memory_path = None
    if KNOWLEDGE_USE_EXISTING_MEMORY_JSON and OPTIONAL_PRETRAINED_KNOWLEDGE_MEMORY_PATH is not None:
        if Path(OPTIONAL_PRETRAINED_KNOWLEDGE_MEMORY_PATH).exists():
            memory_path = str(OPTIONAL_PRETRAINED_KNOWLEDGE_MEMORY_PATH)

    trainer = knowledge_mod.KnowledgeTrainer.load(
        dataset=dataset,
        model_path=str(PRETRAINED_KNOWLEDGE_MODEL_PATH),
        memory_path=memory_path,
        action_dim=action_dim,
    )

    # Run-time overrides. These do not edit the checkpoint on disk.
    trainer.cfg.batch_size = int(KNOWLEDGE_BATCH_SIZE)
    trainer.cfg.epochs_per_fit = int(KNOWLEDGE_FIT_EPOCHS_PER_UPDATE)
    trainer.cfg.verbose = bool(KNOWLEDGE_VERBOSE)
    trainer.cfg.use_llm_verbalizer = bool(KNOWLEDGE_USE_LLM_VERBALIZER)
    trainer.cfg.verbalize_on_update = bool(KNOWLEDGE_USE_LLM_VERBALIZER)
    trainer.cfg.train_log_mode = str(KNOWLEDGE_TRAIN_LOG_MODE)
    trainer.cfg.log_every_fit_step = max(1, int(KNOWLEDGE_PROGRESS_EVERY_BATCHES))
    trainer.cfg.fast_update_text_encoder = bool(KNOWLEDGE_FAST_UPDATE_TEXT_ENCODER)
    trainer.cfg.freeze_llm_adapters_during_fit = bool(KNOWLEDGE_FREEZE_LLM_ADAPTERS_DURING_TDPO_UPDATE)
    trainer.cfg.fit_max_segments = int(KNOWLEDGE_FIT_MAX_SEGMENTS_PER_UPDATE)
    trainer.cfg.fit_max_new_atoms = int(KNOWLEDGE_FIT_MAX_NEW_ATOMS_PER_UPDATE)
    trainer.cfg.max_buffer_atoms = int(RUN_MEMORY_MAX_ATOMS)
    trainer.cfg.max_prototypes = int(RUN_MEMORY_MAX_PROTOTYPES)

    if KNOWLEDGE_CLEAR_MEMORY_AFTER_LOAD:
        trainer.memory.buffer = []
        trainer.memory.prototypes = []
        trainer.last_snapshot = {"prototype_risk_mean": 0.0, "prototype_count": 0.0}

    enforce_run_memory_limits(trainer)
    external_memory = load_external_memory(action_dim=action_dim, cfg=trainer.cfg)
    attach_external_memory_query(trainer, external_memory)

    if SAVE_AUXILIARY_JSON:
        save_json(run_dir / KNOWLEDGE_DIR / LOADED_KNOWLEDGE_INFO_FILE, {
            "pretrained_model_path": str(PRETRAINED_KNOWLEDGE_MODEL_PATH),
            "pretrained_input_copy_path": copied_path,
            "loaded_memory_path": memory_path,
            "cleared_memory_after_load": bool(KNOWLEDGE_CLEAR_MEMORY_AFTER_LOAD),
            "action_dim": int(action_dim),
            "run_memory_max_atoms": int(RUN_MEMORY_MAX_ATOMS),
            "run_memory_max_prototypes": int(RUN_MEMORY_MAX_PROTOTYPES),
            "use_external_memory_library": bool(USE_EXTERNAL_MEMORY_LIBRARY),
            "external_memory_library_path": str(EXTERNAL_MEMORY_LIBRARY_PATH) if EXTERNAL_MEMORY_LIBRARY_PATH is not None else None,
        })
    return trainer


def should_load_knowledge_trainer_for_run() -> bool:
    """Return True only when this run actually consumes or updates knowledge/skill.

    The no-skill baseline keeps the environment, dynamic tool library, action mask,
    policy architecture, and TDPO hyperparameters unchanged, but skips loading the
    knowledge trainer so no memory/skill can enter the decision path.
    """
    if bool(USE_KNOWLEDGE_MEMORY):
        return True
    if bool(GENERATE_NEW_MEMORY_BEFORE_TDPO):
        return True
    if int(KNOWLEDGE_UPDATE_STRATEGY) in {1, 2, 4}:
        return True
    return False


def knowledge_digest_or_empty(knowledge_trainer) -> Dict[str, object]:
    if knowledge_trainer is None:
        return {"num_atoms": 0, "num_prototypes": 0, "disabled": True}
    return knowledge_trainer.knowledge_digest()


def save_knowledge_artifacts_if_enabled(knowledge_trainer, run_dir: Path, tag: str = "best") -> Dict[str, str]:
    if knowledge_trainer is None:
        return {"disabled": "true"}
    return save_knowledge_artifacts(knowledge_trainer, run_dir, tag=tag)


def _task_records_completion_rate(task_records: Sequence[Dict[str, object]]) -> float:
    records = list(task_records)
    if not records:
        return 0.0
    completed = [r for r in records if str(r.get("status")) == "completed"]
    return float(len(completed) / max(1, len(records)))


def _filter_conservative_task_records(task_records: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    records = list(task_records)
    if bool(CONSERVATIVE_COMPLETED_ONLY):
        completed = [r for r in records if str(r.get("status")) == "completed"]
        completion_rate = len(completed) / max(1, len(records))
        # Safety gate: do not write run-local memory until the current policy has
        # reached a usable completion rate.  This prevents early low-quality
        # decisions from polluting the memory used by later prompts.
        if completion_rate <= float(CONSERVATIVE_MIN_COMPLETION_RATE_TO_UPDATE) or not completed:
            print(
                f"[knowledge-memory] skip conservative memory update | "
                f"completion_rate={completion_rate:.4f} <= threshold={float(CONSERVATIVE_MIN_COMPLETION_RATE_TO_UPDATE):.4f} "
                f"records={len(records)} completed={len(completed)}",
                flush=True,
            )
            return []
        records = completed
    max_records = int(CONSERVATIVE_MAX_TASK_RECORDS_PER_UPDATE)
    if max_records > 0 and len(records) > max_records:
        records = records[:max_records]
    return records


def add_task_records_to_memory_without_weight_update(
    knowledge_trainer,
    task_records: Sequence[Dict[str, object]],
    max_new_atoms: Optional[int] = None,
    mode: str = "memory_only",
) -> Dict[str, object]:
    t0 = time.time()
    print(f"[knowledge-memory] mode={mode} extract_start records={len(task_records)}", flush=True)
    segments = knowledge_trainer.extractor.extract(list(task_records))
    print(f"[knowledge-memory] mode={mode} extracted segments={len(segments)} time={fmt_seconds(time.time() - t0)}", flush=True)
    if not segments:
        limits = enforce_run_memory_limits(knowledge_trainer)
        return {"num_segments": 0, "num_new_atoms": 0, "num_new_prototypes": 0, "mode": mode, "update_seconds": float(time.time() - t0), **limits}
    raw_segment_count = len(segments)
    if max_new_atoms is not None and int(max_new_atoms) > 0 and len(segments) > int(max_new_atoms):
        segments = segments[: int(max_new_atoms)]
        print(
            f"[knowledge-memory] mode={mode} segment_cap {raw_segment_count}->{len(segments)} before atom_build",
            flush=True,
        )
    before = len(knowledge_trainer.memory.prototypes)
    atoms = knowledge_trainer._segments_to_atoms(
        segments,
        progress_prefix=f"[knowledge-memory:{mode}]",
        progress_every_batches=max(1, int(KNOWLEDGE_PROGRESS_EVERY_BATCHES)),
    )
    add_t0 = time.time()
    total_atoms = len(atoms)
    atom_print_every = max(1, min(128, max(1, int(CONSERVATIVE_MAX_NEW_ATOMS_PER_UPDATE)) // 4))
    for atom_idx, atom in enumerate(atoms, start=1):
        knowledge_trainer.memory.add_atom(atom)
        if atom_idx == total_atoms or atom_idx % atom_print_every == 0:
            print(f"[knowledge-memory] add_atoms {atom_idx}/{total_atoms}", flush=True)
    limits = enforce_run_memory_limits(knowledge_trainer)
    after = len(knowledge_trainer.memory.prototypes)
    total_seconds = time.time() - t0
    print(
        f"[knowledge-memory] mode={mode} done segments={len(segments)} atoms={len(atoms)} "
        f"new_prototypes={max(0, after - before)} add_time={fmt_seconds(time.time() - add_t0)} total={fmt_seconds(total_seconds)}",
        flush=True,
    )
    return {
        "num_segments": int(len(segments)),
        "num_raw_segments": int(raw_segment_count),
        "num_new_atoms": int(len(atoms)),
        "num_new_prototypes": int(max(0, after - before)),
        "mode": mode,
        "update_seconds": float(total_seconds),
        **limits,
    }


def update_knowledge_from_records(knowledge_trainer, task_records: Sequence[Dict[str, object]], epoch: Optional[int] = None) -> Dict[str, object]:
    strategy = int(KNOWLEDGE_UPDATE_STRATEGY)
    epoch_i = int(epoch or 0)

    if strategy == 3:
        return {"mode": "no_update", "num_segments": 0, "num_new_atoms": 0, "num_new_prototypes": 0, **enforce_run_memory_limits(knowledge_trainer)}
    if strategy == 4 and epoch_i > 0 and epoch_i <= int(KNOWLEDGE_DELAYED_START_EPOCHS):
        return {"mode": "delayed_no_update", "num_segments": 0, "num_new_atoms": 0, "num_new_prototypes": 0, **enforce_run_memory_limits(knowledge_trainer)}

    if strategy in {1, 4}:
        record_completion_rate = _task_records_completion_rate(task_records)
        records = _filter_conservative_task_records(task_records)
        if not records:
            return {
                "mode": "memory_gate_no_update",
                "num_segments": 0,
                "num_new_atoms": 0,
                "num_new_prototypes": 0,
                "num_records_used": 0,
                "record_completion_rate": float(record_completion_rate),
                "memory_update_threshold": float(CONSERVATIVE_MIN_COMPLETION_RATE_TO_UPDATE),
                **enforce_run_memory_limits(knowledge_trainer),
            }
        out = add_task_records_to_memory_without_weight_update(
            knowledge_trainer,
            records,
            max_new_atoms=int(CONSERVATIVE_MAX_NEW_ATOMS_PER_UPDATE),
            mode="conservative_memory_only",
        )
        out["num_records_used"] = int(len(records))
        out["record_completion_rate"] = float(record_completion_rate)
        out["memory_update_threshold"] = float(CONSERVATIVE_MIN_COMPLETION_RATE_TO_UPDATE)
        return out

    if strategy == 2:
        record_completion_rate = _task_records_completion_rate(task_records)
        records = _filter_conservative_task_records(task_records)
        if not records:
            return {
                "mode": "memory_gate_no_update",
                "num_segments": 0,
                "num_new_atoms": 0,
                "num_new_prototypes": 0,
                "num_records_used": 0,
                "record_completion_rate": float(record_completion_rate),
                "memory_update_threshold": float(CONSERVATIVE_MIN_COMPLETION_RATE_TO_UPDATE),
                **enforce_run_memory_limits(knowledge_trainer),
            }
        if KNOWLEDGE_UPDATE_WEIGHTS_DURING_TDPO:
            print(
                f"[knowledge-update] conservative_fit records={len(records)} "
                f"fast_text={bool(KNOWLEDGE_FAST_UPDATE_TEXT_ENCODER)} "
                f"fit_max_segments={int(KNOWLEDGE_FIT_MAX_SEGMENTS_PER_UPDATE)} "
                f"max_new_atoms={int(KNOWLEDGE_FIT_MAX_NEW_ATOMS_PER_UPDATE)}",
                flush=True,
            )
            knowledge_trainer.cfg.fast_update_text_encoder = bool(KNOWLEDGE_FAST_UPDATE_TEXT_ENCODER)
            knowledge_trainer.cfg.freeze_llm_adapters_during_fit = bool(KNOWLEDGE_FREEZE_LLM_ADAPTERS_DURING_TDPO_UPDATE)
            knowledge_trainer.cfg.fit_max_segments = int(KNOWLEDGE_FIT_MAX_SEGMENTS_PER_UPDATE)
            knowledge_trainer.cfg.fit_max_new_atoms = int(KNOWLEDGE_FIT_MAX_NEW_ATOMS_PER_UPDATE)
            out = knowledge_trainer.fit_on_task_records(
                list(records),
                batch_size=int(KNOWLEDGE_BATCH_SIZE),
                epochs=int(KNOWLEDGE_FIT_EPOCHS_PER_UPDATE),
            )
            limits = enforce_run_memory_limits(knowledge_trainer)
            out.update(limits)
            out["mode"] = "conservative_fit"
            out["num_records_used"] = int(len(records))
            return out
        out = add_task_records_to_memory_without_weight_update(
            knowledge_trainer,
            records,
            max_new_atoms=int(CONSERVATIVE_MAX_NEW_ATOMS_PER_UPDATE),
            mode="conservative_memory_only",
        )
        out["num_records_used"] = int(len(records))
        out["record_completion_rate"] = float(record_completion_rate)
        out["memory_update_threshold"] = float(CONSERVATIVE_MIN_COMPLETION_RATE_TO_UPDATE)
        return out

    raise ValueError("KNOWLEDGE_UPDATE_STRATEGY must be one of 1, 2, 3, 4")


def save_knowledge_artifacts(knowledge_trainer, run_dir: Path, tag: str = "best") -> Dict[str, str]:
    ensure_dir(run_dir / KNOWLEDGE_DIR)
    enforce_run_memory_limits(knowledge_trainer)
    model_path = run_dir / KNOWLEDGE_DIR / BEST_MODEL_FILE
    memory_path = run_dir / KNOWLEDGE_DIR / MEMORY_FILE
    out = knowledge_trainer.save(str(model_path), str(memory_path))
    out["tag"] = str(tag)
    out["model_path"] = str(model_path)
    out["memory_path"] = str(memory_path)
    out["run_memory_max_atoms"] = int(RUN_MEMORY_MAX_ATOMS)
    out["run_memory_max_prototypes"] = int(RUN_MEMORY_MAX_PROTOTYPES)
    out["external_memory_saved_into_run_memory"] = False
    return out


def release_knowledge_backbone_for_tdpo(knowledge_trainer) -> Dict[str, object]:
    """Move the pretrained knowledge Qwen/distiller off CUDA after memory generation.

    The TDPO policy only needs knowledge_trainer.query_memory(), which uses the
    prototype memory and does not call the Qwen backbone.  Releasing these modules
    prevents a second full Qwen-7B copy from occupying GPU memory while TDPO loads
    and trains its own sharded Qwen policy.
    """
    info = {"released": False, "reason": "", "device_before": ""}
    if knowledge_trainer is None:
        info["reason"] = "knowledge_trainer is None"
        return info
    try:
        if hasattr(knowledge_trainer, "device"):
            info["device_before"] = str(knowledge_trainer.device)
        fast_weight_update = (
            int(KNOWLEDGE_UPDATE_STRATEGY) == 2
            and bool(KNOWLEDGE_UPDATE_WEIGHTS_DURING_TDPO)
            and bool(KNOWLEDGE_FAST_UPDATE_TEXT_ENCODER)
        )
        if hasattr(knowledge_trainer, "llm"):
            llm = knowledge_trainer.llm
            if hasattr(llm, "model"):
                llm.model.to("cpu")
            if hasattr(llm, "aspect_adapters"):
                llm.aspect_adapters.to("cpu")
            if hasattr(llm, "device"):
                llm.device = torch.device("cpu")
        if not fast_weight_update:
            if hasattr(knowledge_trainer, "distiller"):
                knowledge_trainer.distiller.to("cpu")
            if hasattr(knowledge_trainer, "device"):
                knowledge_trainer.device = torch.device("cpu")
        else:
            info["reason"] = "released heavy Qwen backbone only; kept lightweight distiller on CUDA for fast conservative updates"
        safe_cuda_empty_cache()
        info["released"] = True
        return info
    except Exception as exc:
        info["reason"] = repr(exc)
        return info


def run_heuristic_for_records(dataset: Dict[str, object], seed: int, policy_name: str) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    env = build_env(dataset, seed=seed)
    env.reset()
    rng = random.Random(seed + 1009)
    step_limit = compute_env_step_limit(dataset)
    steps_taken = 0
    while not env.done() and steps_taken < step_limit:
        dps = env.collect_decision_points()
        for dp in dps:
            action = choose_heuristic_action(policy_name, dp, rng)
            env.apply_decision(str(dp["task_id"]), int(dp["node_id"]), action)
        env.step()
        steps_taken += 1
    records = build_complete_task_records(env)
    summary = summarize_task_records(records, policy_name=policy_name, steps_taken=steps_taken, hit_step_limit=(not env.done()))
    return summary, records


# ============================================================
# Trace post-processing
# ============================================================


def index_records_by_task(task_records: Sequence[Dict[str, object]]) -> Dict[str, Dict[str, object]]:
    return {str(r.get("task_id")): r for r in task_records}


def find_trace_outcome(trace: decision_mod.DecisionTrace, task_record: Dict[str, object]) -> Dict[str, object]:
    events = list(task_record.get("log", []))
    node_events = [
        ev for ev in events
        if ev.get("node_id") is not None and int(ev.get("node_id")) == int(trace.node_id)
    ]
    mission_success = 1 if str(task_record.get("status")) == "completed" else 0
    start_t = float(trace.env_time)

    invalid_or_rejected_names = {
        "decision_tool_invalid_missing_id",
        "decision_tool_invalid_unknown_tool",
        "decision_tool_invalid_not_allowed",
        "decision_tool_rejected_queue_full",
        "decision_local_rejected_queue_full",
    }
    terminal_names = {
        "tool_success",
        "local_success",
        "local_failed",
        "node_timeout",
        "task_restarted",
        "task_failed",
        *invalid_or_rejected_names,
    }
    candidates = []
    for ev in node_events:
        t = float(ev.get("time", 0.0) or 0.0)
        if t + 1e-12 < start_t:
            continue
        if str(ev.get("event_type")) in terminal_names:
            candidates.append(ev)
    if not candidates:
        pred_u = trace.sandbox_utilities.get(int(trace.exec_action), 0.0)
        return {
            "realized_utility": float(pred_u),
            "node_success": 0,
            "mission_success": int(mission_success),
            "violation": 1 if str(task_record.get("status")) == "failed" else 0,
            "actual_latency_s": max(0.0, decision_mod.PAUSE_SANDBOX_DURATION_S),
            "fallback": True,
        }

    ev = sorted(candidates, key=lambda x: float(x.get("time", 0.0) or 0.0))[0]
    et = str(ev.get("event_type"))
    node_success = 1 if et in {"tool_success", "local_success"} else 0
    violation = 1 if et in {"local_failed", "node_timeout", "task_restarted", "task_failed"} or et in invalid_or_rejected_names else 0
    actual_latency = max(0.0, float(ev.get("time", start_t) or start_t) - start_t)
    utility = decision_mod.realized_utility(
        node_success=node_success,
        mission_success=mission_success,
        violation=violation,
        actual_latency_s=actual_latency,
        node_remaining_deadline_s=float(trace.node_remaining_deadline_s),
    )
    return {
        "realized_utility": float(utility),
        "node_success": int(node_success),
        "mission_success": int(mission_success),
        "violation": int(violation),
        "actual_latency_s": float(actual_latency),
        "event_type": et,
        "fallback": False,
    }


def attach_realized_utilities(traces: Sequence[decision_mod.DecisionTrace], task_records: Sequence[Dict[str, object]]) -> None:
    by_task = index_records_by_task(task_records)
    for tr in traces:
        rec = by_task.get(str(tr.task_id))
        if rec is None:
            continue
        out = find_trace_outcome(tr, rec)
        tr.realized_utility = float(out["realized_utility"])
        tr.node_success = int(out["node_success"])
        tr.mission_success = int(out["mission_success"])
        tr.violation = int(out["violation"])
        tr.actual_latency_s = float(out["actual_latency_s"])


def trace_to_json(trace: decision_mod.DecisionTrace) -> Dict[str, object]:
    d = dict(trace.__dict__)
    d["sandbox_utilities"] = {str(k): float(v) for k, v in trace.sandbox_utilities.items()}
    d["wireless_regime"] = list(trace.wireless_regime)
    return d


def pair_to_json(pair: decision_mod.PreferencePair) -> Dict[str, object]:
    d = dict(pair.__dict__)
    return d


# ============================================================
# TDPO rollout and training
# ============================================================



def select_rollout_actions_with_sampling_mix(
    policy: decision_mod.TDPOPolicy,
    chunk: Sequence[Dict[str, object]],
    graph_metas: Sequence[Dict[str, object]],
    num_tools: int,
    use_memory: bool,
    knowledge_trainer,
    sample: bool,
    rng: random.Random,
) -> List[Tuple[object, decision_mod.TDPOActionInfo]]:
    """Select rollout actions with optional per-decision stochastic/greedy mixing.

    ACTION_TEMPERATURE controls the sharpness of the sampling distribution.
    TRAIN_SAMPLE_PROBABILITY controls how often sampling is used during training;
    the remaining decisions use greedy argmax.  Evaluation calls pass sample=False,
    so they remain fully greedy.
    """
    chunk = list(chunk)
    graph_metas = list(graph_metas)
    if not chunk:
        return []
    if not bool(sample):
        return policy.select_actions(
            dps=chunk,
            num_tools=num_tools,
            use_memory=use_memory,
            knowledge_trainer=knowledge_trainer,
            sample=False,
            temperature=decision_mod.ACTION_TEMPERATURE,
            graph_metas=graph_metas,
        )

    p_sample = max(0.0, min(1.0, float(TRAIN_SAMPLE_PROBABILITY)))
    sample_flags = [bool(rng.random() < p_sample) for _ in chunk]
    results: List[Optional[Tuple[object, decision_mod.TDPOActionInfo]]] = [None] * len(chunk)

    for flag in (False, True):
        idxs = [i for i, v in enumerate(sample_flags) if bool(v) == bool(flag)]
        if not idxs:
            continue
        sub_chunk = [chunk[i] for i in idxs]
        sub_metas = [graph_metas[i] for i in idxs]
        try:
            sub_results = policy.select_actions(
                dps=sub_chunk,
                num_tools=num_tools,
                use_memory=use_memory,
                knowledge_trainer=knowledge_trainer,
                sample=bool(flag),
                temperature=decision_mod.ACTION_TEMPERATURE,
                graph_metas=sub_metas,
            )
        except Exception:
            if not bool(BATCH_DECISION_FALLBACK_ON_ERROR) or len(sub_chunk) <= 1:
                raise
            sub_results = []
            for one_dp, one_meta in zip(sub_chunk, sub_metas):
                sub_results.append(policy.select_action(
                    dp=one_dp,
                    num_tools=num_tools,
                    use_memory=use_memory,
                    knowledge_trainer=knowledge_trainer,
                    sample=bool(flag),
                    temperature=decision_mod.ACTION_TEMPERATURE,
                    graph_meta=one_meta,
                ))
        for i, item in zip(idxs, sub_results):
            results[i] = item

    return [r for r in results if r is not None]

def run_tdpo_rollout(
    dataset: Dict[str, object],
    graph_meta: Dict[str, Dict[str, object]],
    policy: decision_mod.TDPOPolicy,
    knowledge_trainer,
    epoch: int,
    seed: int,
    use_memory: bool,
    sample: bool,
) -> Tuple[Dict[str, object], List[Dict[str, object]], List[decision_mod.DecisionTrace]]:
    env = build_env(dataset, seed=seed)
    env.reset()
    rng = random.Random(seed + epoch * 17)
    num_tools = len(dataset.get("tool_catalog", []))
    step_limit = compute_env_step_limit(dataset)
    traces: List[decision_mod.DecisionTrace] = []
    steps_taken = 0

    rollout_t0 = time.time()
    last_print_t = rollout_t0
    last_decision_print_t = rollout_t0
    collect_seconds = 0.0
    select_seconds = 0.0
    apply_seconds = 0.0
    env_step_seconds = 0.0
    max_select_seconds = 0.0
    decision_count = 0
    slow_select_count = 0

    while not env.done() and steps_taken < step_limit:
        step_t0 = time.time()
        t0 = time.time()
        dps = env.collect_decision_points()
        collect_seconds += time.time() - t0

        if dps:
            if bool(BATCH_DECISION_ENABLED):
                chunk_size = int(BATCH_DECISION_MAX_SIZE) if int(BATCH_DECISION_MAX_SIZE) > 0 else len(dps)
            else:
                chunk_size = 1
            chunk_size = max(1, min(int(chunk_size), len(dps)))

            for chunk_start in range(0, len(dps), chunk_size):
                chunk = list(dps[chunk_start:chunk_start + chunk_size])
                graph_metas = [graph_meta.get(str(dp["task_id"]), {}) for dp in chunk]
                t_select = time.time()
                action_info_list = select_rollout_actions_with_sampling_mix(
                    policy=policy,
                    chunk=chunk,
                    graph_metas=graph_metas,
                    num_tools=num_tools,
                    use_memory=use_memory,
                    knowledge_trainer=knowledge_trainer,
                    sample=sample,
                    rng=rng,
                )
                select_dt = time.time() - t_select
                batch_n = max(1, len(chunk))
                select_seconds += select_dt
                decision_count += batch_n
                max_select_seconds = max(max_select_seconds, select_dt / batch_n)
                if select_dt >= float(ROLLOUT_SLOW_SELECT_SECONDS):
                    slow_select_count += 1
                    counts = compact_env_counts(env_runtime_counts(env))
                    print(
                        f"[tdpo-rollout:slow-select] epoch={epoch} step={steps_taken}/{step_limit} "
                        f"batch={chunk_start + 1}-{chunk_start + len(chunk)}/{len(dps)} "
                        f"select_batch={fmt_seconds(select_dt)} avg_in_batch={select_dt / batch_n:.3f}s "
                        f"avg_select={select_seconds / max(1, decision_count):.3f}s max_select={max_select_seconds:.3f}s "
                        f"counts=({counts})",
                        flush=True,
                    )

                now = time.time()
                if (
                    ROLLOUT_DECISION_HEARTBEAT_EVERY
                    and decision_count % int(ROLLOUT_DECISION_HEARTBEAT_EVERY) < batch_n
                ) or (now - last_decision_print_t >= float(ROLLOUT_PRINT_EVERY_SECONDS) and len(dps) > 1):
                    last_decision_print_t = now
                    elapsed = now - rollout_t0
                    counts = compact_env_counts(env_runtime_counts(env))
                    print(
                        f"[tdpo-rollout:decision] epoch={epoch} step={steps_taken}/{step_limit} "
                        f"batch={chunk_start + 1}-{chunk_start + len(chunk)}/{len(dps)} decisions={decision_count} "
                        f"elapsed={fmt_seconds(elapsed)} avg_select={select_seconds / max(1, decision_count):.3f}s "
                        f"collect={fmt_seconds(collect_seconds)} select={fmt_seconds(select_seconds)} "
                        f"apply={fmt_seconds(apply_seconds)} env_step={fmt_seconds(env_step_seconds)} "
                        f"counts=({counts})",
                        flush=True,
                    )

                for local_idx, (dp, (action, info)) in enumerate(zip(chunk, action_info_list), start=1):
                    dp_idx = chunk_start + local_idx
                    task_id = str(dp["task_id"])
                    node_id = int(dp["node_id"])
                    cf_idx = decision_mod.choose_counterfactual_action(dp, info.action_index, num_tools, rng)
                    su = {
                        int(info.action_index): decision_mod.sandbox_utility_for_action(dp, int(info.action_index), num_tools),
                        int(cf_idx): decision_mod.sandbox_utility_for_action(dp, int(cf_idx), num_tools),
                    }
                    trace = decision_mod.DecisionTrace(
                        trace_id=str(uuid.uuid4()),
                        epoch=int(epoch),
                        step_count=int(env.step_count),
                        env_time=float(env.time),
                        task_id=task_id,
                        node_id=node_id,
                        node_type_id=int(dp.get("node_type_id", 0)),
                        prompt=info.prompt,
                        memory_vector=list(info.memory_vector),
                        memory_action_bias=list(info.memory_action_bias),
                        knowledge_prompt=info.knowledge_prompt,
                        action_mask=list(info.action_mask),
                        prompt_allowed_tool_mask=list(dp.get("allowed_tools_mask", []) or []),
                        exec_action=int(info.action_index),
                        cf_action=int(cf_idx),
                        valid_actions=list(info.valid_action_indices),
                        node_remaining_deadline_s=float(dp.get("node_remaining_deadline_s", 0.0) or 0.0),
                        task_remaining_deadline_s=float(dp.get("task_remaining_deadline_s", 0.0) or 0.0),
                        sandbox_utilities=dict(su),
                        wireless_regime=decision_mod.infer_wireless_regime(dp),
                    )
                    traces.append(trace)

                    t_apply = time.time()
                    env.apply_decision(task_id, node_id, action)
                    apply_seconds += time.time() - t_apply

        t_env = time.time()
        env.step()
        env_step_seconds += time.time() - t_env
        steps_taken += 1

        now = time.time()
        if (
            PRINT_EVERY_ENV_STEPS and steps_taken % int(PRINT_EVERY_ENV_STEPS) == 0
        ) or (now - last_print_t >= float(ROLLOUT_PRINT_EVERY_SECONDS)):
            last_print_t = now
            elapsed = now - rollout_t0
            pct = 100.0 * float(steps_taken) / float(max(1, step_limit))
            bar = text_progress_bar(steps_taken, step_limit, int(ROLLOUT_PROGRESS_BAR_WIDTH))
            counts = compact_env_counts(env_runtime_counts(env))
            avg_step = elapsed / max(1, steps_taken)
            eta = avg_step * max(0, step_limit - steps_taken)
            print(
                f"[tdpo-rollout] epoch={epoch} {bar} {steps_taken}/{step_limit} ({pct:.1f}%) "
                f"env_time={env.time:.2f} dps={len(dps)} traces={len(traces)} decisions={decision_count} "
                f"elapsed={fmt_seconds(elapsed)} eta~{fmt_seconds(eta)} "
                f"last_step={fmt_seconds(now - step_t0)} avg_step={avg_step:.3f}s "
                f"avg_select={select_seconds / max(1, decision_count):.3f}s max_select={max_select_seconds:.3f}s "
                f"slow_select={slow_select_count} counts=({counts})",
                flush=True,
            )

    task_records = build_complete_task_records(env)
    attach_realized_utilities(traces, task_records)
    summary = summarize_task_records(task_records, policy_name="TDPO", steps_taken=steps_taken, hit_step_limit=(not env.done()))
    summary.update({
        "rollout_seconds": float(time.time() - rollout_t0),
        "collect_seconds": float(collect_seconds),
        "select_seconds": float(select_seconds),
        "apply_seconds": float(apply_seconds),
        "env_step_seconds": float(env_step_seconds),
        "decision_count": int(decision_count),
        "max_select_seconds": float(max_select_seconds),
        "slow_select_count": int(slow_select_count),
    })
    return summary, task_records, traces


def chunk_dataset_for_rollout(dataset: Dict[str, object], chunk_size: int) -> List[Dict[str, object]]:
    graphs = list(dataset.get("graphs", []) or [])
    chunk_size = int(chunk_size)
    if chunk_size <= 0 or len(graphs) <= chunk_size:
        return [dataset]
    chunks: List[Dict[str, object]] = []
    for start in range(0, len(graphs), chunk_size):
        end = min(start + chunk_size, len(graphs))
        out = dict(dataset)
        out["graphs"] = list(graphs[start:end])
        out["config"] = dict(dataset.get("config", {}))
        out["config"]["rollout_chunk_start"] = int(start)
        out["config"]["rollout_chunk_end"] = int(end)
        out["config"]["rollout_chunk_size"] = int(chunk_size)
        chunks.append(task_env_mod._normalize_dataset_structure(out, fallback_dataset_name=str(dataset.get("dataset_name", "tdpo_chunk"))))
    return chunks


def run_tdpo_rollout_chunked(
    dataset: Dict[str, object],
    graph_meta: Dict[str, Dict[str, object]],
    policy: decision_mod.TDPOPolicy,
    knowledge_trainer,
    epoch: int,
    seed: int,
    use_memory: bool,
    sample: bool,
) -> Tuple[Dict[str, object], List[Dict[str, object]], List[decision_mod.DecisionTrace]]:
    chunk_size = int(TDPO_ROLLOUT_CHUNK_GRAPHS) if int(TDPO_ROLLOUT_CHUNK_GRAPHS) > 0 else 0
    chunks = chunk_dataset_for_rollout(dataset, chunk_size)
    if len(chunks) <= 1:
        return run_tdpo_rollout(
            dataset=dataset,
            graph_meta=graph_meta,
            policy=policy,
            knowledge_trainer=knowledge_trainer,
            epoch=epoch,
            seed=seed,
            use_memory=use_memory,
            sample=sample,
        )

    rollout_t0 = time.time()
    all_task_records: List[Dict[str, object]] = []
    all_traces: List[decision_mod.DecisionTrace] = []
    total_steps = 0
    hit_step_limit = False
    total_rollout_seconds = 0.0
    total_collect_seconds = 0.0
    total_select_seconds = 0.0
    total_apply_seconds = 0.0
    total_env_step_seconds = 0.0
    total_decision_count = 0
    max_select_seconds = 0.0
    total_slow_select_count = 0

    for chunk_idx, chunk_dataset in enumerate(chunks, start=1):
        print(
            f"[tdpo-rollout:chunk] epoch={epoch} chunk={chunk_idx}/{len(chunks)} "
            f"graphs={len(chunk_dataset.get('graphs', []))} seed={seed + chunk_idx - 1}",
            flush=True,
        )
        summary, task_records, traces = run_tdpo_rollout(
            dataset=chunk_dataset,
            graph_meta=graph_meta,
            policy=policy,
            knowledge_trainer=knowledge_trainer,
            epoch=epoch,
            seed=seed + chunk_idx - 1,
            use_memory=use_memory,
            sample=sample,
        )
        all_task_records.extend(task_records)
        all_traces.extend(traces)
        total_steps += int(summary.get("steps_taken", 0))
        hit_step_limit = bool(hit_step_limit or summary.get("hit_step_limit", False))
        total_rollout_seconds += float(summary.get("rollout_seconds", 0.0) or 0.0)
        total_collect_seconds += float(summary.get("collect_seconds", 0.0) or 0.0)
        total_select_seconds += float(summary.get("select_seconds", 0.0) or 0.0)
        total_apply_seconds += float(summary.get("apply_seconds", 0.0) or 0.0)
        total_env_step_seconds += float(summary.get("env_step_seconds", 0.0) or 0.0)
        total_decision_count += int(summary.get("decision_count", 0) or 0)
        max_select_seconds = max(max_select_seconds, float(summary.get("max_select_seconds", 0.0) or 0.0))
        total_slow_select_count += int(summary.get("slow_select_count", 0) or 0)

    combined_summary = summarize_task_records(
        all_task_records,
        policy_name="TDPO",
        steps_taken=total_steps,
        hit_step_limit=hit_step_limit,
    )
    combined_summary.update({
        "rollout_seconds": float(total_rollout_seconds if total_rollout_seconds > 0.0 else time.time() - rollout_t0),
        "collect_seconds": float(total_collect_seconds),
        "select_seconds": float(total_select_seconds),
        "apply_seconds": float(total_apply_seconds),
        "env_step_seconds": float(total_env_step_seconds),
        "decision_count": int(total_decision_count),
        "max_select_seconds": float(max_select_seconds),
        "slow_select_count": int(total_slow_select_count),
        "num_rollout_chunks": int(len(chunks)),
        "rollout_chunk_graphs": int(chunk_size),
    })
    return combined_summary, all_task_records, all_traces


def _pair_gap(pair: decision_mod.PreferencePair) -> float:
    meta = getattr(pair, "meta", {}) or {}
    if "gap" in meta:
        return abs(float(meta.get("gap", 0.0) or 0.0))
    u_exec = meta.get("u_exec")
    u_cf = meta.get("u_cf")
    if u_exec is not None and u_cf is not None:
        return abs(float(u_exec) - float(u_cf))
    return 0.0


def _pair_skill_top_actions(pair: decision_mod.PreferencePair, top_k: int = None) -> List[int]:
    k = int(PAIR_SKILL_TOP_K if top_k is None else top_k)
    bias = list(getattr(pair, "memory_action_bias", []) or [])
    if not bias:
        return []
    return [int(i) for i in sorted(range(len(bias)), key=lambda i: float(bias[i]), reverse=True)[:max(1, k)]]


def _score_pair_quality(
    pair: decision_mod.PreferencePair,
    epoch: int,
    success_action_counts: Dict[Tuple[int, int], int],
) -> Tuple[bool, float, str]:
    """Completion-oriented pair gate used before writing to the preference buffers."""
    meta = getattr(pair, "meta", {}) or {}
    conf = float(getattr(pair, "confidence", 0.0) or 0.0)
    gap = float(_pair_gap(pair))
    pos_completed = int(meta.get("positive_task_completed", 0) or 0)
    pos_node_success = int(meta.get("positive_node_success", 0) or 0)
    neg_completed = int(meta.get("negative_task_completed", 0) or 0)
    neg_node_success = int(meta.get("negative_node_success", 0) or 0)
    positive_realized = bool(meta.get("positive_realized", False))
    node_type = int(meta.get("node_type_id", 0) or 0)
    pos_action = int(pair.positive_action)
    neg_action = int(pair.negative_action)
    allowed_mask = list(meta.get("prompt_allowed_tool_mask", []) or [])
    if allowed_mask and not decision_mod.is_action_legal_by_indicator(pos_action, allowed_mask, len(allowed_mask)):
        meta["quality_reason"] = "illegal_positive_action"
        pair.meta = meta
        return False, 0.0, "illegal_positive_action"

    tier = str(pair.tier).upper()
    min_conf = float(PAIR_TIER_B_MIN_CONFIDENCE) if tier == "B" else float(PAIR_MIN_CONFIDENCE)
    min_gap = float(PAIR_TIER_B_MIN_UTILITY_GAP) if tier == "B" else float(PAIR_MIN_UTILITY_GAP)
    failed_gap = float(PAIR_TIER_B_FAILED_ONLY_MIN_GAP) if tier == "B" else float(PAIR_FAILED_ONLY_MIN_GAP)

    if conf < min_conf:
        return False, 0.0, "low_confidence"
    if gap < min_gap:
        return False, 0.0, "small_gap"
    if pos_completed == 0 and pos_node_success == 0 and neg_completed == 0 and neg_node_success == 0 and gap < failed_gap:
        return False, 0.0, "failed_only_small_gap"

    top_actions = _pair_skill_top_actions(pair, int(PAIR_SKILL_TOP_K))
    skill_agree = int(pos_action in top_actions) if top_actions else 0
    skill_conflict = int((neg_action in top_actions) and (pos_action not in top_actions)) if top_actions else 0
    # Do not hard-drop Tier-B skill conflicts. Tier-B is the only real cross-episode
    # comparison signal; conflicts are handled by lowering quality/confidence instead.
    if bool(PAIR_DROP_SKILL_CONFLICT) and tier != "B" and skill_conflict and pos_completed == 0:
        return False, 0.0, "skill_conflict_without_completion"

    repeated_success = int(success_action_counts.get((node_type, pos_action), 0))
    quality = 0.0
    quality += 3.0 * float(pos_completed)
    quality += 0.60 * float(pos_node_success)
    quality += 0.50 * conf
    quality += 0.05 * min(gap, 30.0)
    quality += 0.30 * float(skill_agree)
    quality += 0.15 * min(5.0, math.log1p(float(repeated_success)))
    quality += float(PAIR_QUALITY_RECENCY_WEIGHT) * float(epoch)
    quality -= 1.0 * float(skill_conflict)
    if str(pair.tier).upper() == "A" and not positive_realized:
        quality -= 0.50  # sandbox-only positive should not dominate completed real traces
    if neg_completed:
        quality -= 1.00

    # Convert quality into a DPO confidence weight; this keeps PDF-style confidence-weighted DPO.
    mult = 1.0
    if pos_completed:
        mult += float(PAIR_CONFIDENCE_BOOST_COMPLETED)
    elif pos_node_success:
        mult += float(PAIR_CONFIDENCE_BOOST_NODE)
    if skill_agree:
        mult += float(PAIR_CONFIDENCE_BOOST_SKILL)
    if skill_conflict:
        mult *= 0.65 if tier == "B" else 0.50
    # Keep confidence-weighted DPO from saturating to 1.0 for nearly every pair.
    pair.confidence = float(max(0.01, min(float(PAIR_CONFIDENCE_MAX), conf * mult)))
    meta["quality_score"] = float(quality)
    meta["utility_gap"] = float(gap)
    meta["skill_top_actions"] = list(top_actions)
    meta["skill_agreement"] = int(skill_agree)
    meta["skill_conflict"] = int(skill_conflict)
    meta["repeated_success_count"] = int(repeated_success)
    meta["buffer_type"] = "success" if pos_completed else "explore"
    meta["quality_reason"] = "accepted"
    pair.meta = meta
    return True, float(quality), "accepted"


def _select_tier_balanced_pairs(
    tier_a: List[decision_mod.PreferencePair],
    tier_b: List[decision_mod.PreferencePair],
    epoch: int,
) -> List[decision_mod.PreferencePair]:
    max_new = int(PAIR_MAX_NEW_PAIRS_PER_EPOCH)
    if max_new <= 0:
        return sorted(tier_a + tier_b, key=lambda p: float(p.meta.get("quality_score", 0.0)), reverse=True)
    a_ratio = float(PAIR_TIER_A_RATIO_WARMUP) if int(epoch) <= int(PAIR_TIER_WARMUP_EPOCHS) else float(PAIR_TIER_A_RATIO_AFTER_WARMUP)
    a_quota = max(0, min(max_new, int(round(max_new * a_ratio))))
    b_quota = max_new - a_quota
    tier_a = sorted(tier_a, key=lambda p: float(p.meta.get("quality_score", 0.0)), reverse=True)
    tier_b = sorted(tier_b, key=lambda p: float(p.meta.get("quality_score", 0.0)), reverse=True)
    selected = tier_a[:a_quota] + tier_b[:b_quota]
    if len(selected) < max_new:
        chosen_ids = {p.pair_id for p in selected}
        leftovers_b = [p for p in tier_b[b_quota:] if p.pair_id not in chosen_ids]
        leftovers_a = [p for p in tier_a[a_quota:] if p.pair_id not in chosen_ids]
        leftovers_b = sorted(leftovers_b, key=lambda p: float(p.meta.get("quality_score", 0.0)), reverse=True)
        leftovers_a = sorted(leftovers_a, key=lambda p: float(p.meta.get("quality_score", 0.0)), reverse=True)
        # After warmup, prefer any available Tier-B leftovers before filling with Tier-A,
        # so the real cross-episode signal does not disappear behind high-scoring sandbox pairs.
        leftovers = (leftovers_b + leftovers_a) if int(epoch) > int(PAIR_TIER_WARMUP_EPOCHS) else (leftovers_a + leftovers_b)
        selected.extend(leftovers[: max_new - len(selected)])
    return selected


def build_pairs_from_traces(
    traces: Sequence[decision_mod.DecisionTrace],
    history_by_type: Dict[int, List[decision_mod.DecisionTrace]],
    epoch: int,
    success_action_counts: Dict[Tuple[int, int], int],
) -> Tuple[List[decision_mod.PreferencePair], Dict[str, int]]:
    raw_tier_a: List[decision_mod.PreferencePair] = []
    raw_legality: List[decision_mod.PreferencePair] = []
    for tr in traces:
        lp = decision_mod.build_legality_pair(tr)
        if lp is not None:
            raw_legality.append(lp)
        p = decision_mod.build_tier_a_pair(tr, confidence_min=decision_mod.TDPO_CONFIDENCE_MIN)
        if p is not None:
            raw_tier_a.append(p)
    raw_tier_a = raw_legality + raw_tier_a
    raw_tier_b = decision_mod.build_tier_b_pairs(
        traces=traces,
        history_by_type=history_by_type,
        confidence_min=float(PAIR_TIER_B_MIN_CONFIDENCE),
        max_pairs=int(TDPO_MAX_NEW_TIER_B_PAIRS_PER_EPOCH),
    )

    tier_a: List[decision_mod.PreferencePair] = []
    tier_b: List[decision_mod.PreferencePair] = []
    rejected = 0
    for p in raw_tier_a:
        ok, _, reason = _score_pair_quality(p, int(epoch), success_action_counts)
        if ok:
            tier_a.append(p)
        else:
            p.meta["quality_reason"] = reason
            rejected += 1
    for p in raw_tier_b:
        ok, _, reason = _score_pair_quality(p, int(epoch), success_action_counts)
        if ok:
            tier_b.append(p)
        else:
            p.meta["quality_reason"] = reason
            rejected += 1

    pairs = _select_tier_balanced_pairs(tier_a, tier_b, int(epoch))
    for p in pairs:
        meta = p.meta or {}
        if int(meta.get("positive_task_completed", 0) or 0):
            key = (int(meta.get("node_type_id", 0) or 0), int(p.positive_action))
            success_action_counts[key] = int(success_action_counts.get(key, 0)) + 1

    for tr in traces:
        if tr.realized_utility is None:
            continue
        buf = history_by_type[int(tr.node_type_id)]
        buf.append(tr)
        if len(buf) > int(decision_mod.TIER_B_MAX_HISTORY_PER_TYPE):
            del buf[: len(buf) - int(decision_mod.TIER_B_MAX_HISTORY_PER_TYPE)]

    return pairs, {
        "tier_a": sum(1 for p in pairs if str(p.tier).upper() == "A"),
        "tier_b": sum(1 for p in pairs if str(p.tier).upper() == "B"),
        "tier_a_raw": len(raw_tier_a),
        "tier_b_raw": len(raw_tier_b),
        "legality_raw": len(raw_legality),
        "legality_selected": sum(1 for p in pairs if int((p.meta or {}).get("legality_pair", 0) or 0) == 1),
        "rejected": int(rejected),
    }


def _trim_quality_buffer(buf: List[decision_mod.PreferencePair], max_size: int) -> List[decision_mod.PreferencePair]:
    max_size = int(max_size)
    if max_size <= 0:
        return []
    if len(buf) <= max_size:
        return buf
    ranked = sorted(
        list(buf),
        key=lambda p: (float((p.meta or {}).get("quality_score", 0.0)), float(getattr(p, "confidence", 0.0))),
        reverse=True,
    )
    top_k = max(1, int(round(max_size * float(PAIR_BUFFER_TOP_KEEP_RATIO))))
    recent_k = max(0, int(round(max_size * float(PAIR_BUFFER_RECENT_KEEP_RATIO))))
    keep: List[decision_mod.PreferencePair] = []
    seen = set()
    for p in ranked[:top_k]:
        keep.append(p); seen.add(p.pair_id)
    # Always keep a recent slice so new Tier-B / boundary pairs can enter even when
    # their current quality is lower than old saturated pairs.
    for p in list(buf)[-recent_k:]:
        if p.pair_id not in seen and len(keep) < max_size:
            keep.append(p); seen.add(p.pair_id)
    remaining = [p for p in ranked[top_k:] if p.pair_id not in seen]
    rng = random.Random(SEED + len(buf) + max_size)
    rng.shuffle(remaining)
    for p in remaining:
        if len(keep) >= max_size:
            break
        keep.append(p); seen.add(p.pair_id)
    return keep


def update_quality_pair_buffers(
    success_buffer: List[decision_mod.PreferencePair],
    explore_buffer: List[decision_mod.PreferencePair],
    new_pairs: Sequence[decision_mod.PreferencePair],
) -> Tuple[List[decision_mod.PreferencePair], List[decision_mod.PreferencePair]]:
    for p in new_pairs:
        if str((p.meta or {}).get("buffer_type", "explore")) == "success":
            success_buffer.append(p)
        else:
            explore_buffer.append(p)
    total_cap = max(1, int(TDPO_MAX_PREFERENCE_BUFFER))
    success_cap = max(1, int(round(total_cap * float(PAIR_SUCCESS_BUFFER_RATIO))))
    explore_cap = max(1, total_cap - success_cap)
    return _trim_quality_buffer(success_buffer, success_cap), _trim_quality_buffer(explore_buffer, explore_cap)


def build_quality_training_buffer(
    success_buffer: Sequence[decision_mod.PreferencePair],
    explore_buffer: Sequence[decision_mod.PreferencePair],
    global_update_step: int,
) -> List[decision_mod.PreferencePair]:
    total_available = len(success_buffer) + len(explore_buffer)
    if total_available == 0:
        return []
    sample_size = int(TDPO_BUFFER_SAMPLE_SIZE) if bool(TDPO_SAMPLE_FROM_BUFFER) and int(TDPO_BUFFER_SAMPLE_SIZE) > 0 else total_available
    sample_size = min(sample_size, total_available)
    success_target = min(len(success_buffer), int(round(sample_size * float(PAIR_TRAIN_SUCCESS_RATIO))))
    explore_target = min(len(explore_buffer), sample_size - success_target)
    if success_target + explore_target < sample_size:
        success_target = min(len(success_buffer), success_target + sample_size - success_target - explore_target)
    if success_target + explore_target < sample_size:
        explore_target = min(len(explore_buffer), explore_target + sample_size - success_target - explore_target)

    def _take(buf: Sequence[decision_mod.PreferencePair], k: int) -> List[decision_mod.PreferencePair]:
        ranked = sorted(
            list(buf),
            key=lambda p: (float((p.meta or {}).get("quality_score", 0.0)), float(getattr(p, "confidence", 0.0))),
            reverse=True,
        )
        if k <= 0:
            return []
        top_k = min(len(ranked), int(round(k * float(PAIR_TRAIN_TOP_SAMPLE_RATIO))))
        chosen = list(ranked[:top_k])
        rest = ranked[top_k:]
        rng = random.Random(SEED + int(global_update_step) + k + len(buf))
        rng.shuffle(rest)
        chosen.extend(rest[: max(0, k - len(chosen))])
        rng.shuffle(chosen)
        return chosen

    out = _take(success_buffer, success_target) + _take(explore_buffer, explore_target)
    rng = random.Random(SEED + int(global_update_step) + 991)
    rng.shuffle(out)
    return out


def tdpo_batch_signal_stats(batch: Sequence[decision_mod.PreferencePair]) -> Dict[str, float]:
    if not batch:
        return {
            "completion_pair_ratio": 0.0,
            "completion_advantage_ratio": 0.0,
            "node_success_advantage_ratio": 0.0,
            "pair_gap_mean": 0.0,
        }
    pos_completed = []
    neg_completed = []
    pos_node_success = []
    neg_node_success = []
    gaps = []
    for p in batch:
        meta = getattr(p, "meta", {}) or {}
        pc = int(meta.get("positive_task_completed", 0) or 0)
        nc = int(meta.get("negative_task_completed", 0) or 0)
        pn = int(meta.get("positive_node_success", 0) or 0)
        nn = int(meta.get("negative_node_success", 0) or 0)
        pos_completed.append(pc)
        neg_completed.append(nc)
        pos_node_success.append(pn)
        neg_node_success.append(nn)
        gaps.append(float(meta.get("utility_gap", meta.get("gap", 0.0)) or 0.0))
    n = float(len(batch))
    return {
        "completion_pair_ratio": float(sum(pos_completed) / n),
        "completion_advantage_ratio": float(sum(1 for pc, nc in zip(pos_completed, neg_completed) if pc > nc) / n),
        "node_success_advantage_ratio": float(sum(1 for pn, nn in zip(pos_node_success, neg_node_success) if pn > nn) / n),
        "pair_gap_mean": float(sum(gaps) / n),
    }


def train_tdpo_on_buffer(
    policy: decision_mod.TDPOPolicy,
    optimizer: torch.optim.Optimizer,
    preference_buffer: List[decision_mod.PreferencePair],
    global_update_step: int,
) -> Tuple[Dict[str, float], int]:
    if len(preference_buffer) < int(TDPO_MIN_PAIRS_TO_TRAIN):
        return {
            "loss": 0.0,
            "pref_acc": 0.0,
            "logp_margin": 0.0,
            "confidence_mean": 0.0,
            "grad_norm": 0.0,
            "grad_norm_clipped": 0.0,
            "completion_pair_ratio": 0.0,
            "completion_advantage_ratio": 0.0,
            "node_success_advantage_ratio": 0.0,
            "pair_gap_mean": 0.0,
            "num_updates": 0,
            "tdpo_seconds": 0.0,
            "ref_seconds": 0.0,
            "update_seconds": 0.0,
        }, global_update_step

    train_t0 = time.time()
    ref_seconds = 0.0
    update_seconds = 0.0

    rng = random.Random(SEED + global_update_step)
    # preference_buffer is already quality-sampled from success/exploration buffers.
    train_buffer = list(preference_buffer)

    missing_ref = [p for p in train_buffer if p.ref_logp_positive is None or p.ref_logp_negative is None]
    if missing_ref:
        t_ref = time.time()
        print(f"[tdpo-ref] filling reference logps | pairs={len(missing_ref)} batch={TDPO_BATCH_SIZE}", flush=True)
        policy.fill_reference_logps(missing_ref, batch_size=max(1, int(TDPO_BATCH_SIZE)), show_progress=True, stage="[tdpo-ref:missing]", print_every_batches=max(1, int(TDPO_REF_PRINT_EVERY_BATCHES)))
        ref_seconds += time.time() - t_ref
        print(f"[tdpo-ref] done | time={fmt_seconds(ref_seconds)}", flush=True)

    stats_acc = {"loss": 0.0, "pref_acc": 0.0, "logp_margin": 0.0, "confidence_mean": 0.0, "grad_norm": 0.0, "grad_norm_clipped": 0.0, "completion_pair_ratio": 0.0, "completion_advantage_ratio": 0.0, "node_success_advantage_ratio": 0.0, "pair_gap_mean": 0.0}
    num_updates = 0
    policy.train()
    order = list(range(len(train_buffer)))
    batch_size = max(1, int(TDPO_BATCH_SIZE))
    inner_epochs = max(1, int(TDPO_INNER_EPOCHS))
    total_updates = inner_epochs * int(math.ceil(len(order) / float(batch_size)))
    last_print_t = time.time()

    for inner_ep in range(1, inner_epochs + 1):
        rng.shuffle(order)
        for start in range(0, len(order), batch_size):
            upd_t0 = time.time()
            idx = order[start:start + batch_size]
            batch = [train_buffer[i] for i in idx]
            loss, stats = policy.dpo_loss(batch)
            signal_stats = tdpo_batch_signal_stats(batch)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm_tensor = torch.nn.utils.clip_grad_norm_(policy.trainable_parameters(), float(decision_mod.TDPO_GRAD_CLIP_NORM))
            grad_norm = float(grad_norm_tensor.detach().cpu().item()) if hasattr(grad_norm_tensor, "detach") else float(grad_norm_tensor)
            if not math.isfinite(grad_norm):
                grad_norm = 0.0
            grad_norm_clipped = float(min(grad_norm, float(decision_mod.TDPO_GRAD_CLIP_NORM)))
            stats["grad_norm"] = float(grad_norm)
            stats["grad_norm_clipped"] = float(grad_norm_clipped)
            stats.update(signal_stats)
            optimizer.step()
            upd_dt = time.time() - upd_t0
            update_seconds += upd_dt
            global_update_step += 1
            num_updates += 1
            for k in stats_acc:
                stats_acc[k] += float(stats.get(k, 0.0))

            now = time.time()
            if (
                TDPO_UPDATE_PRINT_EVERY and num_updates % int(TDPO_UPDATE_PRINT_EVERY) == 0
            ) or (now - last_print_t >= float(TDPO_UPDATE_PRINT_EVERY_SECONDS)) or num_updates == total_updates:
                last_print_t = now
                avg_update = update_seconds / max(1, num_updates)
                eta = avg_update * max(0, total_updates - num_updates)
                bar = text_progress_bar(num_updates, total_updates, int(ROLLOUT_PROGRESS_BAR_WIDTH))
                print(
                    f"[tdpo-update] {bar} {num_updates}/{total_updates} "
                    f"inner={inner_ep}/{inner_epochs} buffer={len(preference_buffer)} train_pairs={len(train_buffer)} "
                    f"loss={float(stats.get('loss', 0.0)):.4f} pref_acc={float(stats.get('pref_acc', 0.0)):.4f} "
                    f"grad={float(stats.get('grad_norm', 0.0)):.3e}/{float(stats.get('grad_norm_clipped', 0.0)):.3e} "
                    f"comp_pair={float(stats.get('completion_pair_ratio', 0.0)):.3f} comp_adv={float(stats.get('completion_advantage_ratio', 0.0)):.3f} "
                    f"last_batch={fmt_seconds(upd_dt)} avg_update={avg_update:.3f}s eta~{fmt_seconds(eta)}",
                    flush=True,
                )

            if ROLLING_REF_UPDATE_STEPS and global_update_step % int(ROLLING_REF_UPDATE_STEPS) == 0:
                t_ref = time.time()
                print(f"[tdpo-ref] rolling refresh | buffer={len(preference_buffer)} train_pairs={len(train_buffer)}", flush=True)
                policy.fill_reference_logps(train_buffer, batch_size=batch_size, show_progress=True, stage="[tdpo-ref:rolling]", print_every_batches=max(1, int(TDPO_REF_PRINT_EVERY_BATCHES)))
                ref_seconds += time.time() - t_ref
                print(f"[tdpo-ref] rolling refresh done | total_ref={fmt_seconds(ref_seconds)}", flush=True)

    if num_updates > 0:
        for k in stats_acc:
            stats_acc[k] /= float(num_updates)
    stats_acc["num_updates"] = int(num_updates)
    stats_acc["tdpo_seconds"] = float(time.time() - train_t0)
    stats_acc["ref_seconds"] = float(ref_seconds)
    stats_acc["update_seconds"] = float(update_seconds)
    return stats_acc, global_update_step


# ============================================================
# Main
# ============================================================


def main() -> None:
    random.seed(SEED)
    torch.manual_seed(SEED)
    print_cuda_layout()
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    run_dir = OUTPUT_ROOT / f"{RUN_NAME}_{now_tag()}"
    ensure_dir(run_dir)
    ensure_dir(run_dir / CHECKPOINT_DIR)
    ensure_dir(run_dir / RECORDS_DIR)
    ensure_dir(run_dir / PAIRS_DIR)
    ensure_dir(run_dir / TRACES_DIR)
    ensure_dir(run_dir / KNOWLEDGE_DIR)

    dataset, scan_info = load_dataset_from_dir(DATA_DIR, seed=SEED)
    tool_library_for_check = load_dynamic_tool_library()
    tool_library_coverage = validate_dynamic_tool_library_coverage(dataset, tool_library_for_check)
    if SAVE_AUXILIARY_JSON:
        save_json(run_dir / DATASET_SCAN_FILE, scan_info)
    save_json(run_dir / RUN_CONFIG_FILE, {
        "DATA_DIR": str(DATA_DIR),
        "SAMPLE_NUM_GRAPHS": int(SAMPLE_NUM_GRAPHS),
        "TDPO_ROLLOUT_CHUNK_GRAPHS": int(TDPO_ROLLOUT_CHUNK_GRAPHS),
        "SAMPLE_IN_ORDER": bool(SAMPLE_IN_ORDER),
        "SAMPLE_RANDOM_POOL_LIMIT": int(SAMPLE_RANDOM_POOL_LIMIT),
        "dataset_scan": scan_info,
        "USE_DYNAMIC_TOOL_LIBRARY": bool(USE_DYNAMIC_TOOL_LIBRARY),
        "USE_MULTIPLE_TOOL_LIBRARIES": bool(USE_MULTIPLE_TOOL_LIBRARIES),
        "TOOL_LIBRARY_PATHS": [str(p) for p in _selected_tool_library_paths()],
        "ALLOW_TOOL_LIBRARY_FALLBACK": bool(ALLOW_TOOL_LIBRARY_FALLBACK),
        "tool_library_coverage": tool_library_coverage,
        "ENV_RELAXED_RESOURCE_MODE": bool(ENV_RELAXED_RESOURCE_MODE),
        "ENV_Q_MAX": int(ENV_Q_MAX),
        "ENV_TOOL_INSTANCE_MULTIPLIER": float(ENV_TOOL_INSTANCE_MULTIPLIER),
        "ENV_TOOL_INSTANCE_ADDITIVE": int(ENV_TOOL_INSTANCE_ADDITIVE),
        "ENV_MIN_TOOL_INSTANCES": int(ENV_MIN_TOOL_INSTANCES),
        "ENV_ARRIVAL_GAP_RANGE": list(ENV_ARRIVAL_GAP_RANGE),
        "ENV_NODE_DEADLINE_MULTIPLIER": float(ENV_NODE_DEADLINE_MULTIPLIER),
        "ENV_TASK_DEADLINE_MULTIPLIER": float(ENV_TASK_DEADLINE_MULTIPLIER),
        "ENV_LOCAL_MAX_CONCURRENCY": int(ENV_LOCAL_MAX_CONCURRENCY),
        "ENV_LOCAL_QUEUE_CAPACITY": int(ENV_LOCAL_QUEUE_CAPACITY),
        "QWEN_MODEL_PATH": QWEN_MODEL_PATH,
        "PRETRAINED_KNOWLEDGE_MODEL_PATH": str(PRETRAINED_KNOWLEDGE_MODEL_PATH),
        "USE_KNOWLEDGE_MEMORY": USE_KNOWLEDGE_MEMORY,
        "GPU_PROFILE": GPU_PROFILE,
        "CUDA_VISIBLE_DEVICES_DEFAULT": CUDA_VISIBLE_DEVICES_DEFAULT,
        "LLM_ENABLE_MULTI_GPU": LLM_ENABLE_MULTI_GPU,
        "LLM_DEVICE_MAP": LLM_DEVICE_MAP,
        "LLM_MAX_MEMORY_PER_GPU": LLM_MAX_MEMORY_PER_GPU,
        "RUN_GREEDY_EVAL_EACH_EPOCH": RUN_GREEDY_EVAL_EACH_EPOCH,
        "EVAL_USE_FIXED_SEED": EVAL_USE_FIXED_SEED,
        "EVAL_SEED_OFFSET": EVAL_SEED_OFFSET,
        "TRAIN_WITH_SAMPLING": bool(TRAIN_WITH_SAMPLING),
        "TRAIN_SAMPLE_PROBABILITY": float(TRAIN_SAMPLE_PROBABILITY),
        "ACTION_TEMPERATURE": float(decision_mod.ACTION_TEMPERATURE),
        "TDPO_BETA": float(decision_mod.TDPO_BETA),
        "PROMPT_TOOL_TABLE_MODE": str(getattr(decision_mod, "PROMPT_TOOL_TABLE_MODE", "")),
        "PROMPT_INCLUDE_FULL_TOOL_TABLE": bool(getattr(decision_mod, "PROMPT_INCLUDE_FULL_TOOL_TABLE", False)),
        "PROMPT_INCLUDE_LOCAL_OPTION_DETAILS": bool(getattr(decision_mod, "PROMPT_INCLUDE_LOCAL_OPTION_DETAILS", False)),
        "LLM_USE_LORA": LLM_USE_LORA,
        "LLM_LORA_NUM_LAYERS": int(LLM_LORA_NUM_LAYERS),
        "LLM_LORA_LAYER_IDS": list(LLM_LORA_LAYER_IDS),
        "LLM_LORA_TARGET_MODULES": list(LLM_LORA_TARGET_MODULES),
        "LLM_LORA_R": LLM_LORA_R,
        "LLM_LORA_ALPHA": LLM_LORA_ALPHA,
        "LLM_LORA_DROPOUT": LLM_LORA_DROPOUT,
        "LLM_TUNE_LAST_N_BLOCKS": LLM_TUNE_LAST_N_BLOCKS,
        "NUM_EPOCHS": NUM_EPOCHS,
        "TDPO_BATCH_SIZE": TDPO_BATCH_SIZE,
        "TDPO_MAX_PREFERENCE_BUFFER": int(TDPO_MAX_PREFERENCE_BUFFER),
        "TDPO_SAMPLE_FROM_BUFFER": bool(TDPO_SAMPLE_FROM_BUFFER),
        "TDPO_BUFFER_SAMPLE_SIZE": int(TDPO_BUFFER_SAMPLE_SIZE),
        "TDPO_MAX_NEW_TIER_B_PAIRS_PER_EPOCH": int(TDPO_MAX_NEW_TIER_B_PAIRS_PER_EPOCH),
        "PAIR_MAX_NEW_PAIRS_PER_EPOCH": int(PAIR_MAX_NEW_PAIRS_PER_EPOCH),
        "TDPO_LEARNING_RATE": TDPO_LEARNING_RATE,
        "COMPLETION_FIRST_OBJECTIVE": bool(getattr(decision_mod, "COMPLETION_FIRST_OBJECTIVE", False)),
        "COMPLETION_FIRST_MISSION_WEIGHT": float(getattr(decision_mod, "COMPLETION_FIRST_MISSION_WEIGHT", 0.0)),
        "COMPLETION_FIRST_NODE_WEIGHT": float(getattr(decision_mod, "COMPLETION_FIRST_NODE_WEIGHT", 0.0)),
        "COMPLETION_FIRST_VIOLATION_WEIGHT": float(getattr(decision_mod, "COMPLETION_FIRST_VIOLATION_WEIGHT", 0.0)),
        "COMPLETION_FIRST_LATENCY_WEIGHT": float(getattr(decision_mod, "COMPLETION_FIRST_LATENCY_WEIGHT", 0.0)),
        "COMPLETION_FIRST_LATENCY_CAP": float(getattr(decision_mod, "COMPLETION_FIRST_LATENCY_CAP", 0.0)),
        "KNOWLEDGE_UPDATE_MEMORY_DURING_TDPO": KNOWLEDGE_UPDATE_MEMORY_DURING_TDPO,
        "KNOWLEDGE_UPDATE_WEIGHTS_DURING_TDPO": KNOWLEDGE_UPDATE_WEIGHTS_DURING_TDPO,
        "KNOWLEDGE_UPDATE_STRATEGY": int(KNOWLEDGE_UPDATE_STRATEGY),
        "KNOWLEDGE_FAST_UPDATE_TEXT_ENCODER": bool(KNOWLEDGE_FAST_UPDATE_TEXT_ENCODER),
        "KNOWLEDGE_FREEZE_LLM_ADAPTERS_DURING_TDPO_UPDATE": bool(KNOWLEDGE_FREEZE_LLM_ADAPTERS_DURING_TDPO_UPDATE),
        "KNOWLEDGE_FIT_MAX_SEGMENTS_PER_UPDATE": int(KNOWLEDGE_FIT_MAX_SEGMENTS_PER_UPDATE),
        "KNOWLEDGE_FIT_MAX_NEW_ATOMS_PER_UPDATE": int(KNOWLEDGE_FIT_MAX_NEW_ATOMS_PER_UPDATE),
        "KNOWLEDGE_DELAYED_START_EPOCHS": int(KNOWLEDGE_DELAYED_START_EPOCHS),
        "RUN_MEMORY_MAX_ATOMS": int(RUN_MEMORY_MAX_ATOMS),
        "RUN_MEMORY_MAX_PROTOTYPES": int(RUN_MEMORY_MAX_PROTOTYPES),
        "CONSERVATIVE_MAX_NEW_ATOMS_PER_UPDATE": int(CONSERVATIVE_MAX_NEW_ATOMS_PER_UPDATE),
        "CONSERVATIVE_MAX_TASK_RECORDS_PER_UPDATE": int(CONSERVATIVE_MAX_TASK_RECORDS_PER_UPDATE),
        "CONSERVATIVE_COMPLETED_ONLY": bool(CONSERVATIVE_COMPLETED_ONLY),
        "CONSERVATIVE_MIN_COMPLETION_RATE_TO_UPDATE": float(CONSERVATIVE_MIN_COMPLETION_RATE_TO_UPDATE),
        "USE_EXTERNAL_MEMORY_LIBRARY": bool(USE_EXTERNAL_MEMORY_LIBRARY),
        "EXTERNAL_MEMORY_LIBRARY_PATH": str(EXTERNAL_MEMORY_LIBRARY_PATH) if EXTERNAL_MEMORY_LIBRARY_PATH is not None else None,
        "EXTERNAL_MEMORY_USE_WHEN_RUN_PROTOTYPES_LT": int(EXTERNAL_MEMORY_USE_WHEN_RUN_PROTOTYPES_LT),
        "EXTERNAL_MEMORY_BLEND_WEIGHT": float(EXTERNAL_MEMORY_BLEND_WEIGHT),
        "COPY_PRETRAINED_KNOWLEDGE_TO_RUN_DIR": bool(COPY_PRETRAINED_KNOWLEDGE_TO_RUN_DIR),
    })

    num_tools = len(dataset.get("tool_catalog", []))
    action_dim = decision_mod.action_dim_from_num_tools(num_tools)
    graph_meta = build_graph_meta_map(dataset)
    knowledge_rows: List[Dict[str, object]] = []

    if should_load_knowledge_trainer_for_run():
        knowledge_trainer = load_knowledge_trainer(dataset, action_dim=action_dim, run_dir=run_dir)
    else:
        knowledge_trainer = None
        print("[knowledge-load] skipped: no-skill baseline; no memory/skill module is loaded.", flush=True)

    if knowledge_trainer is not None and GENERATE_NEW_MEMORY_BEFORE_TDPO:
        memory_dataset = subset_dataset(dataset, int(MAX_MEMORY_BOOTSTRAP_GRAPHS))
        print(f"[memory-bootstrap] policy={MEMORY_BOOTSTRAP_POLICY} | graphs={len(memory_dataset.get('graphs', []))}", flush=True)
        mem_summary, mem_records = run_heuristic_for_records(memory_dataset, seed=SEED + 31, policy_name=MEMORY_BOOTSTRAP_POLICY)
        if SAVE_AUXILIARY_JSON:
            save_json(run_dir / KNOWLEDGE_DIR / MEMORY_BOOTSTRAP_SUMMARY_FILE, mem_summary)
        if SAVE_MEMORY_BOOTSTRAP_TASK_RECORDS:
            save_json(run_dir / KNOWLEDGE_DIR / MEMORY_BOOTSTRAP_RECORDS_FILE, mem_records)
        mem_stats = update_knowledge_from_records(knowledge_trainer, mem_records, epoch=0)
        if SAVE_AUXILIARY_JSON:
            save_json(run_dir / KNOWLEDGE_DIR / MEMORY_BOOTSTRAP_STATS_FILE, mem_stats)
        save_knowledge_artifacts(knowledge_trainer, run_dir, tag="bootstrap")
        knowledge_rows.append(build_knowledge_row("bootstrap", 0, mem_stats, knowledge_trainer.knowledge_digest()))
        write_knowledge_csv(run_dir / KNOWLEDGE_METRICS_FILE, knowledge_rows)

    if knowledge_trainer is not None and RELEASE_KNOWLEDGE_BACKBONE_BEFORE_TDPO:
        if (
            int(KNOWLEDGE_UPDATE_STRATEGY) == 2
            and bool(KNOWLEDGE_UPDATE_WEIGHTS_DURING_TDPO)
            and not bool(KNOWLEDGE_FAST_UPDATE_TEXT_ENCODER)
        ):
            print(
                "[knowledge-release] WARNING: Qwen-based weight update conflicts with releasing the Qwen backbone. "
                "Switching to memory-only normal update for this run.",
                flush=True,
            )
            globals()["KNOWLEDGE_UPDATE_WEIGHTS_DURING_TDPO"] = False
        release_info = release_knowledge_backbone_for_tdpo(knowledge_trainer)
        if SAVE_AUXILIARY_JSON:
            save_json(run_dir / KNOWLEDGE_DIR / RELEASE_BEFORE_TDPO_FILE, release_info)
        print(f"[knowledge-release] {release_info}", flush=True)

    safe_cuda_empty_cache()

    policy_cfg = decision_mod.TDPOPolicyConfig(
        model_path=QWEN_MODEL_PATH if Path(QWEN_MODEL_PATH).exists() else QWEN_MODEL_PATH_FALLBACK,
        tokenizer_path=None,
        device=LLM_DEVICE,
        dtype=LLM_DTYPE,
        max_length=int(LLM_MAX_LENGTH),
        use_lora=bool(LLM_USE_LORA),
        lora_layer_ids=tuple(LLM_LORA_LAYER_IDS),
        lora_target_modules=tuple(LLM_LORA_TARGET_MODULES),
        lora_r=int(LLM_LORA_R),
        lora_alpha=float(LLM_LORA_ALPHA),
        lora_dropout=float(LLM_LORA_DROPOUT),
        tune_last_n_blocks=int(LLM_TUNE_LAST_N_BLOCKS),
        train_final_norm=bool(LLM_TRAIN_FINAL_NORM),
        gradient_checkpointing=bool(LLM_GRADIENT_CHECKPOINTING),
        enable_multi_gpu=bool(LLM_ENABLE_MULTI_GPU),
        device_map=str(LLM_DEVICE_MAP),
        max_memory_per_gpu=str(LLM_MAX_MEMORY_PER_GPU),
        low_cpu_mem_usage=bool(LLM_LOW_CPU_MEM_USAGE),
        offload_folder=str(run_dir / LLM_OFFLOAD_FOLDER_NAME),
        memory_vector_dim=int(action_dim + 4),
        beta=float(decision_mod.TDPO_BETA),
        seed=int(SEED),
        verbose=True,
    )
    policy = decision_mod.TDPOPolicy(action_dim=action_dim, cfg=policy_cfg)
    optimizer = torch.optim.AdamW(
        policy.trainable_parameters(),
        lr=float(TDPO_LEARNING_RATE),
        weight_decay=float(TDPO_WEIGHT_DECAY),
    )

    success_pair_buffer: List[decision_mod.PreferencePair] = []
    explore_pair_buffer: List[decision_mod.PreferencePair] = []
    preference_buffer: List[decision_mod.PreferencePair] = []
    success_action_counts: Dict[Tuple[int, int], int] = defaultdict(int)
    history_by_type: Dict[int, List[decision_mod.DecisionTrace]] = defaultdict(list)
    epoch_rows: List[Dict[str, object]] = []
    eval_rows: List[Dict[str, object]] = []
    energy_detail_rows: List[Dict[str, object]] = []
    delay_detail_rows: List[Dict[str, object]] = []
    global_update_step = 0
    best_score = -1.0e18
    best_ckpt = ""
    greedy_gap_bad_streak = 0

    for epoch in range(1, int(NUM_EPOCHS) + 1):
        epoch_t0 = time.time()
        print(f"[tdpo] epoch {epoch}/{NUM_EPOCHS} start", flush=True)
        rollout_summary, task_records, traces = run_tdpo_rollout_chunked(
            dataset=dataset,
            graph_meta=graph_meta,
            policy=policy,
            knowledge_trainer=knowledge_trainer,
            epoch=epoch,
            seed=SEED + epoch,
            use_memory=bool(USE_KNOWLEDGE_MEMORY),
            sample=bool(TRAIN_WITH_SAMPLING),
        )

        rollout_wall = time.time() - epoch_t0
        print(f"[tdpo-stage] epoch={epoch} rollout_done time={fmt_seconds(rollout_wall)} traces={len(traces)}", flush=True)
        pair_t0 = time.time()
        new_pairs, pair_counts = build_pairs_from_traces(traces, history_by_type, epoch=epoch, success_action_counts=success_action_counts)
        print(f"[tdpo-stage] epoch={epoch} pair_build_done time={fmt_seconds(time.time() - pair_t0)} pairs={len(new_pairs)}", flush=True)
        new_ref_seconds = 0.0
        if new_pairs:
            ref_t0 = time.time()
            print(f"[tdpo-ref:new] epoch={epoch} pairs={len(new_pairs)}", flush=True)
            policy.fill_reference_logps(new_pairs, batch_size=max(1, int(TDPO_BATCH_SIZE)), show_progress=True, stage=f"[tdpo-ref:new] epoch={epoch}", print_every_batches=max(1, int(TDPO_REF_PRINT_EVERY_BATCHES)))
            new_ref_seconds = time.time() - ref_t0
            print(f"[tdpo-ref:new] epoch={epoch} done time={fmt_seconds(new_ref_seconds)}", flush=True)
            success_pair_buffer, explore_pair_buffer = update_quality_pair_buffers(success_pair_buffer, explore_pair_buffer, new_pairs)

        preference_buffer = build_quality_training_buffer(success_pair_buffer, explore_pair_buffer, global_update_step)
        train_stage_t0 = time.time()
        tdpo_stats, global_update_step = train_tdpo_on_buffer(
            policy=policy,
            optimizer=optimizer,
            preference_buffer=preference_buffer,
            global_update_step=global_update_step,
        )
        print(f"[tdpo-stage] epoch={epoch} update_done time={fmt_seconds(time.time() - train_stage_t0)}", flush=True)
        safe_cuda_empty_cache()

        eval_row = None
        if bool(RUN_GREEDY_EVAL_EACH_EPOCH):
            eval_t0 = time.time()
            eval_seed = int(SEED + EVAL_SEED_OFFSET) if bool(EVAL_USE_FIXED_SEED) else int(SEED + EVAL_SEED_OFFSET + epoch)
            if bool(EVAL_PRINT_PROGRESS):
                print(f"[tdpo-eval] epoch={epoch} greedy_eval_start seed={eval_seed}", flush=True)
            eval_summary, _eval_records, eval_traces = run_tdpo_rollout_chunked(
                dataset=dataset,
                graph_meta=graph_meta,
                policy=policy,
                knowledge_trainer=knowledge_trainer,
                epoch=epoch,
                seed=eval_seed,
                use_memory=bool(USE_KNOWLEDGE_MEMORY),
                sample=False,
            )
            eval_seconds = time.time() - eval_t0
            eval_row = {
                "epoch": int(epoch),
                "completion_rate": float(eval_summary.get("completion_rate", 0.0)),
                "avg_latency_s": float(eval_summary.get("avg_latency_s", 0.0)),
                "energy_proxy": float(eval_summary.get("energy_proxy", 0.0)),
                "completed_total_energy_proxy": float(eval_summary.get("completed_total_energy_proxy", 0.0)),
                "avg_completed_energy_proxy_per_graph": float(eval_summary.get("avg_completed_energy_proxy_per_graph", 0.0)),
                "completed_total_delay_s": float(eval_summary.get("completed_total_delay_s", 0.0)),
                "avg_completed_delay_s_per_graph": float(eval_summary.get("avg_completed_delay_s_per_graph", 0.0)),
                "num_completed_nodes_logged": int(eval_summary.get("num_completed_nodes_logged", 0)),
                "num_completed": int(eval_summary.get("num_completed", 0)),
                "num_failed": int(eval_summary.get("num_failed", 0)),
                "num_unfinished": int(eval_summary.get("num_unfinished", 0)),
                "steps_taken": int(eval_summary.get("steps_taken", 0)),
                "hit_step_limit": bool(eval_summary.get("hit_step_limit", False)),
                "num_traces": int(len(eval_traces)),
                "rollout_seconds": float(eval_summary.get("rollout_seconds", 0.0)),
                "eval_seconds": float(eval_seconds),
                "eval_seed": int(eval_seed),
                "use_memory": bool(USE_KNOWLEDGE_MEMORY),
            }
            eval_rows.append(eval_row)
            write_eval_csv(run_dir / EVAL_METRICS_FILE, eval_rows)
            eval_energy_rows, eval_delay_rows = build_energy_delay_detail_rows(epoch, "eval", _eval_records)
            energy_detail_rows.extend(eval_energy_rows)
            delay_detail_rows.extend(eval_delay_rows)
            write_energy_csv(run_dir / ENERGY_FILE, energy_detail_rows)
            write_delay_csv(run_dir / DELAY_FILE, delay_detail_rows)
            print(
                f"[tdpo-eval] epoch={epoch} greedy_comp={eval_row['completion_rate']:.4f} "
                f"greedy_lat={eval_row['avg_latency_s']:.4f} "
                f"completed={eval_row['num_completed']} failed={eval_row['num_failed']} "
                f"time={fmt_seconds(eval_seconds)}",
                flush=True,
            )
            del _eval_records, eval_traces
            safe_cuda_empty_cache()

        if bool(GREEDY_GAP_GUARD_ENABLED) and eval_row is not None:
            rollout_comp = float(rollout_summary.get("completion_rate", 0.0))
            eval_comp = float(eval_row.get("completion_rate", 0.0))
            greedy_gap = rollout_comp - eval_comp
            if greedy_gap >= float(GREEDY_GAP_THRESHOLD) and eval_comp <= float(GREEDY_GAP_LOW_EVAL):
                greedy_gap_bad_streak += 1
            else:
                greedy_gap_bad_streak = max(0, greedy_gap_bad_streak - 1)
            if greedy_gap_bad_streak >= max(1, int(GREEDY_GAP_PATIENCE)):
                old_bias = float(policy.cfg.memory_action_bias_scale)
                new_bias = max(float(GREEDY_GAP_MIN_MEMORY_ACTION_BIAS_SCALE), old_bias * float(GREEDY_GAP_BIAS_DECAY))
                policy.cfg.memory_action_bias_scale = float(new_bias)
                decision_mod.MEMORY_ACTION_BIAS_SCALE = float(new_bias)
                old_temp = float(decision_mod.ACTION_TEMPERATURE)
                new_temp = min(float(GREEDY_GAP_MAX_TEMPERATURE), old_temp + float(GREEDY_GAP_TEMPERATURE_STEP))
                decision_mod.ACTION_TEMPERATURE = float(new_temp)
                print(
                    f"[tdpo-guard] epoch={epoch} rollout_comp={rollout_comp:.4f} eval_comp={eval_comp:.4f} "
                    f"gap={greedy_gap:.4f} | memory_action_bias {old_bias:.4f}->{new_bias:.4f} "
                    f"temperature {old_temp:.4f}->{new_temp:.4f}",
                    flush=True,
                )
                greedy_gap_bad_streak = 0

        knowledge_stats = {}
        update_enabled = knowledge_trainer is not None and int(KNOWLEDGE_UPDATE_STRATEGY) in {1, 2, 4}
        if update_enabled and epoch % max(1, int(KNOWLEDGE_UPDATE_EVERY_EPOCHS)) == 0:
            print(f"[knowledge-update] epoch={epoch} strategy={KNOWLEDGE_UPDATE_STRATEGY} records={len(task_records)}", flush=True)
            knowledge_stats = update_knowledge_from_records(knowledge_trainer, task_records, epoch=epoch)
            if SAVE_AUXILIARY_JSON:
                save_json(run_dir / KNOWLEDGE_DIR / knowledge_update_filename(epoch), knowledge_stats)
            save_knowledge_artifacts(knowledge_trainer, run_dir, tag="best")
            knowledge_rows.append(build_knowledge_row("epoch_update", epoch, knowledge_stats, knowledge_trainer.knowledge_digest()))
            write_knowledge_csv(run_dir / KNOWLEDGE_METRICS_FILE, knowledge_rows)

        digest = knowledge_digest_or_empty(knowledge_trainer)
        if bool(BEST_SCORE_USE_GREEDY_EVAL) and eval_row is not None:
            score = float(eval_row.get("completion_rate", 0.0)) * 1000.0 - float(eval_row.get("avg_latency_s", 0.0))
            best_metric_source = "greedy_eval"
        else:
            score = float(rollout_summary.get("completion_rate", 0.0)) * 1000.0 - float(rollout_summary.get("avg_latency_s", 0.0))
            best_metric_source = "rollout"
        is_best = False
        if SAVE_EVERY_EPOCH:
            policy.save_checkpoint(
                str(run_dir / CHECKPOINT_DIR / tdpo_checkpoint_filename(epoch)),
                extra={
                    "epoch": epoch,
                    "summary": rollout_summary,
                    "tdpo_stats": tdpo_stats,
                    "knowledge_digest": digest,
                },
            )
        if score > best_score:
            is_best = True
            best_score = score
            best_ckpt = policy.save_checkpoint(
                str(run_dir / CHECKPOINT_DIR / BEST_MODEL_FILE),
                extra={
                    "epoch": epoch,
                    "score": score,
                    "best_metric_source": best_metric_source,
                    "summary": rollout_summary,
                    "tdpo_stats": tdpo_stats,
                    "knowledge_digest": digest,
                },
            )

        if SAVE_TASK_RECORDS_EVERY_EPOCH:
            save_json(run_dir / RECORDS_DIR / task_records_filename(epoch), task_records)
        if SAVE_TRACE_JSONL:
            append_jsonl(run_dir / TRACES_DIR / TRACES_FILE, [trace_to_json(t) for t in traces])
            append_jsonl(run_dir / PAIRS_DIR / PAIRS_FILE, [pair_to_json(p) for p in new_pairs])
        if WRITE_PROMPT_SAMPLES and epoch == 1:
            samples = [
                {
                    "trace_id": t.trace_id,
                    "task_id": t.task_id,
                    "node_id": t.node_id,
                    "exec_action": t.exec_action,
                    "prompt": t.prompt,
                    "knowledge_prompt": t.knowledge_prompt,
                }
                for t in traces[: int(MAX_PROMPT_SAMPLES)]
            ]
            save_json(run_dir / PROMPT_SAMPLES_FILE, samples)

        row = {
            "epoch": int(epoch),
            "completion_rate": float(rollout_summary.get("completion_rate", 0.0)),
            "avg_latency_s": float(rollout_summary.get("avg_latency_s", 0.0)),
            "energy_proxy": float(rollout_summary.get("energy_proxy", 0.0)),
            "completed_total_energy_proxy": float(rollout_summary.get("completed_total_energy_proxy", 0.0)),
            "avg_completed_energy_proxy_per_graph": float(rollout_summary.get("avg_completed_energy_proxy_per_graph", 0.0)),
            "completed_total_delay_s": float(rollout_summary.get("completed_total_delay_s", 0.0)),
            "avg_completed_delay_s_per_graph": float(rollout_summary.get("avg_completed_delay_s_per_graph", 0.0)),
            "num_completed_nodes_logged": int(rollout_summary.get("num_completed_nodes_logged", 0)),
            "num_completed": int(rollout_summary.get("num_completed", 0)),
            "num_failed": int(rollout_summary.get("num_failed", 0)),
            "num_unfinished": int(rollout_summary.get("num_unfinished", 0)),
            "steps_taken": int(rollout_summary.get("steps_taken", 0)),
            "hit_step_limit": bool(rollout_summary.get("hit_step_limit", False)),
            "num_traces": int(len(traces)),
            "num_pairs_added": int(len(new_pairs)),
            "num_pairs_buffer": int(len(success_pair_buffer) + len(explore_pair_buffer)),
            "num_success_buffer": int(len(success_pair_buffer)),
            "num_explore_buffer": int(len(explore_pair_buffer)),
            "num_pairs_train": int(len(preference_buffer)),
            "num_tier_a_pairs": int(pair_counts.get("tier_a", 0)),
            "num_tier_b_pairs": int(pair_counts.get("tier_b", 0)),
            "num_tier_a_raw": int(pair_counts.get("tier_a_raw", 0)),
            "num_tier_b_raw": int(pair_counts.get("tier_b_raw", 0)),
            "tdpo_loss": float(tdpo_stats.get("loss", 0.0)),
            "tdpo_pref_acc": float(tdpo_stats.get("pref_acc", 0.0)),
            "tdpo_margin": float(tdpo_stats.get("logp_margin", 0.0)),
            "tdpo_confidence": float(tdpo_stats.get("confidence_mean", 0.0)),
            "tdpo_grad_norm": float(tdpo_stats.get("grad_norm", 0.0)),
            "tdpo_grad_norm_clipped": float(tdpo_stats.get("grad_norm_clipped", 0.0)),
            "tdpo_completion_pair_ratio": float(tdpo_stats.get("completion_pair_ratio", 0.0)),
            "tdpo_completion_advantage_ratio": float(tdpo_stats.get("completion_advantage_ratio", 0.0)),
            "tdpo_node_success_advantage_ratio": float(tdpo_stats.get("node_success_advantage_ratio", 0.0)),
            "tdpo_pair_gap_mean": float(tdpo_stats.get("pair_gap_mean", 0.0)),
            "rollout_seconds": float(rollout_summary.get("rollout_seconds", 0.0)),
            "pair_seconds": float(time.time() - pair_t0),
            "new_ref_seconds": float(new_ref_seconds),
            "tdpo_seconds": float(tdpo_stats.get("tdpo_seconds", 0.0)),
            "tdpo_ref_seconds": float(tdpo_stats.get("ref_seconds", 0.0)),
            "tdpo_update_seconds": float(tdpo_stats.get("update_seconds", 0.0)),
            "knowledge_update_seconds": float(knowledge_stats.get("fit_seconds", knowledge_stats.get("update_seconds", 0.0)) if knowledge_stats else 0.0),
            "epoch_seconds": float(time.time() - epoch_t0),
            "knowledge_num_atoms": int(digest.get("num_atoms", 0)),
            "knowledge_num_prototypes": int(digest.get("num_prototypes", 0)),
            "action_temperature": float(decision_mod.ACTION_TEMPERATURE),
            "memory_action_bias_scale": float(policy.cfg.memory_action_bias_scale),
            "greedy_gap_bad_streak": int(greedy_gap_bad_streak),
            "best_metric_source": str(best_metric_source),
            "is_best": bool(is_best),
        }
        epoch_rows.append(row)
        write_epoch_csv(run_dir / EPOCH_METRICS_FILE, epoch_rows)
        rollout_energy_rows, rollout_delay_rows = build_energy_delay_detail_rows(epoch, "rollout", task_records)
        energy_detail_rows.extend(rollout_energy_rows)
        delay_detail_rows.extend(rollout_delay_rows)
        write_energy_csv(run_dir / ENERGY_FILE, energy_detail_rows)
        write_delay_csv(run_dir / DELAY_FILE, delay_detail_rows)
        if SAVE_LATEST_SUMMARY_JSON:
            save_json(run_dir / LATEST_SUMMARY_FILE, {
                "epoch": epoch,
                "row": row,
                "rollout_summary": rollout_summary,
                "tdpo_stats": tdpo_stats,
                "knowledge_stats": knowledge_stats,
                "knowledge_digest": digest,
                "best_ckpt": best_ckpt,
            })
        print(
            f"[tdpo] epoch={epoch} comp={row['completion_rate']:.4f} lat={row['avg_latency_s']:.4f} "
            f"pairs+={row['num_pairs_added']} buffer={row['num_pairs_buffer']} "
            f"loss={row['tdpo_loss']:.4f} pref_acc={row['tdpo_pref_acc']:.4f} "
            f"grad={row['tdpo_grad_norm']:.3e}/{row['tdpo_grad_norm_clipped']:.3e} "
            f"rollout={fmt_seconds(row['rollout_seconds'])} update={fmt_seconds(row['tdpo_seconds'])} "
            f"knowledge_proto={row['knowledge_num_prototypes']}",
            flush=True,
        )

    final_knowledge = save_knowledge_artifacts_if_enabled(knowledge_trainer, run_dir, tag="final")
    if SAVE_FINAL_RESULT_JSON:
        save_json(run_dir / FINAL_RESULT_FILE, {
            "run_dir": str(run_dir.resolve()),
            "best_ckpt": best_ckpt,
            "final_knowledge": final_knowledge,
            "num_epochs": int(NUM_EPOCHS),
            "best_score": float(best_score),
            "eval_csv": str((run_dir / EVAL_METRICS_FILE).resolve()) if bool(RUN_GREEDY_EVAL_EACH_EPOCH) else "",
        })

    print(f"[tdpo] done | run_dir={run_dir.resolve()}", flush=True)
    print(f"[tdpo] best_ckpt={best_ckpt}", flush=True)


if __name__ == "__main__":
    main()
