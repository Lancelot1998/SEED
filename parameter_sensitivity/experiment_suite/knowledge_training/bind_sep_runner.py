# -*- coding: utf-8 -*-
"""
train_knowledge_dynamic_tools.py

Standalone knowledge training script for the dynamic tool-library environment.

Saved files per run directory are limited to:
  1) loss.csv
  2) best.pt
  3) memory.json

The script keeps the same task_env.py and knowledge_aspect_adapter.py interfaces.
It uses a stronger heuristic policy to generate closed-loop task records, then
fits KnowledgeTrainer on those records.  The checkpoint is updated after every epoch.
Memory saving is delayed for the first MEMORY_SAVE_SKIP_EPOCHS epochs, and memory.json
is saved only after the warm-up memory has been cleared.
"""

from __future__ import annotations

from experiment_suite.config.paths import (
    KNOWLEDGE_BIND_DATA_DIR as DATA_DIR,
    KNOWLEDGE_BIND_DATASET_FILE_PATTERNS as DATASET_FILE_PATTERNS,
    KNOWLEDGE_BIND_DATASET_METADATA_FILES as DATASET_METADATA_FILES,
    KNOWLEDGE_BIND_QWEN_MODEL_PATH as QWEN_MODEL_PATH,
    KNOWLEDGE_BIND_QWEN_MODEL_PATH_FALLBACK as QWEN_MODEL_PATH_FALLBACK,
    KNOWLEDGE_BIND_PRETRAINED_KNOWLEDGE_MODEL_PATH as PRETRAINED_KNOWLEDGE_MODEL_PATH,
    KNOWLEDGE_BIND_OPTIONAL_PRETRAINED_KNOWLEDGE_MEMORY_PATH as OPTIONAL_PRETRAINED_KNOWLEDGE_MEMORY_PATH,
    KNOWLEDGE_BIND_OUTPUT_ROOT as OUTPUT_ROOT,
    KNOWLEDGE_BIND_TOOL_LIBRARY_PATHS as TOOL_LIBRARY_PATHS,
    KNOWLEDGE_LOCAL_TOOL_LIBRARY_PATH as TOOL_LIBRARY_PATH,
    PROJECT_ROOT as THIS_DIR,
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
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch

# ============================================================
# User-editable globals: paths
# ============================================================



LOAD_PRETRAINED_KNOWLEDGE_IF_AVAILABLE = False
CLEAR_MEMORY_AFTER_LOAD = True

RUN_NAME = "knowledge_dynamic_tools_heuristic"

# ============================================================
# User-editable globals: data sampling
# ============================================================

SEED = 20260425
# Keep knowledge pretraining on the same graph-selection regime as TDPO training.
# Dataset files are first sorted by filename; SAMPLE_RANDOM_POOL_LIMIT restricts
# the selectable range to the first N sorted files.
# SAMPLE_IN_ORDER=True means use the first SAMPLE_NUM_GRAPHS files from that pool.
# SAMPLE_IN_ORDER=False means randomly sample SAMPLE_NUM_GRAPHS files from that pool.
SAMPLE_NUM_GRAPHS = 20
SAMPLE_IN_ORDER = False
SAMPLE_RANDOM_POOL_LIMIT = 1500

# ============================================================
# User-editable globals: dynamic tool libraries
# ============================================================

USE_DYNAMIC_TOOL_LIBRARY = True
USE_MULTIPLE_TOOL_LIBRARIES = True
ALLOW_TOOL_LIBRARY_FALLBACK = False

# ============================================================
# User-editable globals: environment
# ============================================================

ENV_DT = 0.10
ENV_Q_MAX = 10
ENV_ARRIVAL_GAP_RANGE = (1.0, 1.0)
ENV_METRICS_RECORD_PERIOD_STEPS = 1
ENV_NODE_DEADLINE_MULTIPLIER = 3.2
ENV_TASK_DEADLINE_MULTIPLIER = 2.6
ENV_MIN_NODE_DEADLINE_S = 2.0
ENV_MIN_TASK_DEADLINE_S = 6.0
ENV_LOCAL_MAX_CONCURRENCY = 4
ENV_LOCAL_QUEUE_CAPACITY = 64
ENV_RETRY_RESTART_PENALTY_S = 0.10
ENV_TARGET_TOOL_COMPUTE_S = 1.0
ENV_TARGET_TOOL_UPLINK_S = 0.25
ENV_TARGET_TOOL_DOWNLINK_S = 0.25
FORCE_MAX_ENV_STEPS = 0

# ============================================================
# User-editable globals: knowledge training
# ============================================================
KNOWLEDGE_BATCH_SIZE = 16
KNOWLEDGE_LR = 3e-5
KNOWLEDGE_WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 300
# KNOWLEDGE_BATCH_SIZE = 4
KNOWLEDGE_FIT_EPOCHS_PER_OUTER_EPOCH = 1
# KNOWLEDGE_LR = 2e-4
# KNOWLEDGE_WEIGHT_DECAY = 1e-5
KNOWLEDGE_DTYPE = "bfloat16"
KNOWLEDGE_DEVICE = "auto"
KNOWLEDGE_VERBOSE = True
KNOWLEDGE_USE_LLM_VERBALIZER = False
KNOWLEDGE_GRADIENT_CHECKPOINTING = True
KNOWLEDGE_MAX_LENGTH = 768

# Loss balancing switches. These keep the knowledge pretraining objective from being
# dominated by topology/wire heads and make action/tool evidence visible in gradients.
KNOWLEDGE_NORMALIZE_TOPO_WIRE_LOSSES = True
KNOWLEDGE_USE_MASKED_TOOL_LOSS = True
KNOWLEDGE_LOSS_TOPO_WEIGHT = 1.0
KNOWLEDGE_LOSS_WIRE_WEIGHT = 1.0
KNOWLEDGE_LOSS_TOOL_WEIGHT = 3.0
KNOWLEDGE_LOSS_OUT_WEIGHT = 1.0
KNOWLEDGE_LOSS_NEED_WEIGHT = 1.0

# Bind/sep disentanglement losses.
# Set block/div to 0.0 so enabling the adapter disentanglement switch only adds
# bind and sep into loss_total, without introducing extra block/div regularizers.
KNOWLEDGE_ENABLE_BIND_SEP_IN_TOTAL_LOSS = True
KNOWLEDGE_LAMBDA_BIND_POS = 0.25
KNOWLEDGE_LAMBDA_SEP = 0.20
KNOWLEDGE_LAMBDA_BLOCK = 0.0
KNOWLEDGE_LAMBDA_DIV = 0.0

# Explicit saved-memory caps. 0 or negative means no explicit cap.
# The newest atoms/prototypes are kept before each save to memory.json.
KNOWLEDGE_MAX_MEMORY_ATOMS = 2500
KNOWLEDGE_MAX_MEMORY_PROTOTYPES = 2500
# Do not save early unstable memory. Epochs 1..MEMORY_SAVE_SKIP_EPOCHS update the model
# and loss.csv only. At the boundary, run-local memory is cleared once; memory.json is
# first written from epoch MEMORY_SAVE_SKIP_EPOCHS + 1 onward.
MEMORY_SAVE_SKIP_EPOCHS = 100
CLEAR_MEMORY_AFTER_SAVE_WARMUP = True

# Heuristic policy parameters.  This is deliberately stronger than a single
# fastest-tool rule: it jointly considers latency, risk, deadline slack,
# queue pressure, packet loss, and local fallback risk.
HEURISTIC_EPSILON = 0.03
HEURISTIC_SOFTMAX_TEMPERATURE = 0.05
HEURISTIC_W_TIME = 1.00
HEURISTIC_W_RISK = 1.60
HEURISTIC_W_QUEUE = 0.45
HEURISTIC_W_PACKET_LOSS = 0.35
HEURISTIC_W_CONGESTION = 0.30
HEURISTIC_W_DEADLINE_VIOLATION = 2.00
HEURISTIC_W_LOCAL_FAIL = 1.35
HEURISTIC_ALLOW_LOCAL = True

# ============================================================
# Imports from local codebase
# ============================================================

from experiment_suite.knowledge_training import environment as task_env_mod
from experiment_suite.shared import knowledge_adapter as knowledge_mod


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
    if not bool(USE_DYNAMIC_TOOL_LIBRARY):
        if not bool(ALLOW_TOOL_LIBRARY_FALLBACK):
            raise RuntimeError("USE_DYNAMIC_TOOL_LIBRARY=False while ALLOW_TOOL_LIBRARY_FALLBACK=False.")
        return None
    paths = _selected_tool_library_paths()
    merged_entries: List[Dict[str, object]] = []
    seen = set()
    loaded_paths: List[str] = []
    for path in paths:
        if not path.exists():
            msg = f"dynamic tool library file not found: {path}"
            if bool(ALLOW_TOOL_LIBRARY_FALLBACK):
                print(f"[knowledge-train] WARNING: {msg}; continuing with fallback allowed", flush=True)
                continue
            raise FileNotFoundError(msg)
        data = load_json(path)
        if not isinstance(data, dict) or not isinstance(data.get("entries", None), list):
            raise ValueError(f"Invalid tool library format. Expected a dict with an entries list: {path}")
        for entry in data.get("entries", []) or []:
            if not isinstance(entry, dict):
                continue
            key = (str(entry.get("graph_id", "")), int(entry.get("node_id", -1)))
            if key in seen:
                continue
            seen.add(key)
            merged_entries.append(entry)
        loaded_paths.append(str(path))
    if not merged_entries and not bool(ALLOW_TOOL_LIBRARY_FALLBACK):
        raise RuntimeError("No dynamic tool library entries loaded and fallback is disabled.")
    out = {
        "schema_version": "merged_api_tool_library_v2",
        "created_by": "train_knowledge_dynamic_tools.py",
        "source_paths": loaded_paths,
        "num_entries": len(merged_entries),
        "entries": merged_entries,
    }
    print(f"[knowledge-train] dynamic tool libraries loaded: files={len(loaded_paths)} entries={len(merged_entries)} fallback_allowed={bool(ALLOW_TOOL_LIBRARY_FALLBACK)}", flush=True)
    return out


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
    required = 0
    missing_count = 0
    missing_examples: List[Dict[str, object]] = []
    for graph in dataset.get("graphs", []) or []:
        gid = str(graph.get("graph_id"))
        for node in graph.get("nodes", []) or []:
            nid = int(node.get("node_id", 0))
            for slot, allowed in enumerate(list(node.get("allowed_tools_mask", []))):
                if int(allowed) != 1:
                    continue
                required += 1
                if (gid, nid, int(slot)) not in keys:
                    missing_count += 1
                    if len(missing_examples) < 20:
                        missing_examples.append({"graph_id": gid, "node_id": nid, "slot": int(slot)})
    info = {
        "required_allowed_tool_slots": int(required),
        "covered_allowed_tool_slots": int(required - missing_count),
        "missing_allowed_tool_slots": int(missing_count),
        "coverage_rate": float((required - missing_count) / max(1, required)),
        "missing_examples": missing_examples,
    }
    print(f"[tool-library:coverage] required={required} covered={required - missing_count} missing={missing_count} coverage={info['coverage_rate']:.4f}", flush=True)
    if missing_count > 0 and not bool(ALLOW_TOOL_LIBRARY_FALLBACK):
        raise RuntimeError(
            "Dynamic tool library coverage check failed while ALLOW_TOOL_LIBRARY_FALLBACK=False. "
            f"missing={missing_count}, examples={missing_examples}"
        )
    return info


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
    sample_n = int(SAMPLE_NUM_GRAPHS) if SAMPLE_NUM_GRAPHS is not None else 0
    if sample_n and sample_n > 0:
        if bool(SAMPLE_IN_ORDER):
            dataset_files = candidate_files[: sample_n]
        else:
            dataset_files = rng.sample(candidate_files, k=min(sample_n, len(candidate_files)))
    else:
        dataset_files = list(candidate_files)

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
        "dataset_name": f"knowledge_dataset_from_{data_dir.name}",
        "config": {
            "source": "disk_pool",
            "data_dir": str(data_dir),
            "sample_count": len(raw_items),
            "seed": int(seed),
            "sample_in_order": bool(SAMPLE_IN_ORDER),
            "sample_num_graphs": int(SAMPLE_NUM_GRAPHS),
            "sample_random_pool_limit": int(SAMPLE_RANDOM_POOL_LIMIT),
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


def build_env(dataset: Dict[str, object], seed: int, tool_library: Optional[Dict[str, object]]):
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
        target_tool_compute_s=float(ENV_TARGET_TOOL_COMPUTE_S),
        target_tool_uplink_s=float(ENV_TARGET_TOOL_UPLINK_S),
        target_tool_downlink_s=float(ENV_TARGET_TOOL_DOWNLINK_S),
    )
    return task_env_mod.MissionEnvironment(dataset=dataset, env_cfg=env_cfg, tool_library=tool_library)


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
            "restart_count": task.restart_count,
            "failure_reason": task.failure_reason,
            "log": list(task.log),
        }
    return [finalized[k] for k in sorted(finalized.keys())]


def safe_mean(values: Sequence[float], default: float = 0.0) -> float:
    vals = [float(v) for v in values if v is not None]
    return float(sum(vals) / len(vals)) if vals else float(default)


def summarize_task_records(task_records: Sequence[Dict[str, object]], steps_taken: int, hit_step_limit: bool) -> Dict[str, object]:
    num_tasks = len(task_records)
    completed = [r for r in task_records if str(r.get("status")) == "completed"]
    failed = [r for r in task_records if str(r.get("status")) == "failed"]
    unfinished = [r for r in task_records if str(r.get("status")) not in {"completed", "failed"}]
    latencies = [float(r.get("latency_s")) for r in completed if r.get("latency_s") is not None]
    return {
        "num_tasks": int(num_tasks),
        "num_completed": int(len(completed)),
        "num_failed": int(len(failed)),
        "num_unfinished": int(len(unfinished)),
        "completion_rate": float(len(completed) / max(1, num_tasks)),
        "avg_latency_s": safe_mean(latencies, 0.0),
        "steps_taken": int(steps_taken),
        "hit_step_limit": bool(hit_step_limit),
    }


def _deadline_scale(dp: Dict[str, object]) -> float:
    node_remain = float(dp.get("node_remaining_deadline_s", 1.0) or 1.0)
    task_remain = float(dp.get("task_remaining_deadline_s", node_remain) or node_remain)
    return max(0.25, min(node_remain, task_remain))


def _tool_score(opt: Dict[str, object], dp: Dict[str, object]) -> float:
    if bool(opt.get("queue_blocked", False)):
        return 1.0e9
    deadline = _deadline_scale(dp)
    total_s = float(opt.get("predicted_total_s", 1e9) or 1e9)
    queue_s = float(opt.get("base_queue_delay_s", 0.0) or 0.0)
    risk = float(opt.get("risk_estimate", 1.0) or 1.0)
    pkt = float(opt.get("packet_loss_rate", 0.0) or 0.0)
    cong = float(opt.get("congestion_ratio", 0.0) or 0.0)
    violation = max(0.0, total_s - deadline) / max(deadline, 1e-9)
    return (
        HEURISTIC_W_TIME * total_s / max(deadline, 1e-9)
        + HEURISTIC_W_RISK * risk
        + HEURISTIC_W_QUEUE * queue_s / max(deadline, 1e-9)
        + HEURISTIC_W_PACKET_LOSS * pkt
        + HEURISTIC_W_CONGESTION * cong
        + HEURISTIC_W_DEADLINE_VIOLATION * violation
    )


def _local_score(local_opt: Dict[str, object], dp: Dict[str, object]) -> float:
    if not bool(HEURISTIC_ALLOW_LOCAL) or bool(local_opt.get("queue_blocked", False)):
        return 1.0e9
    deadline = _deadline_scale(dp)
    total_s = float(local_opt.get("predicted_total_s", local_opt.get("compute_s", 1e9)) or 1e9)
    fail = float(local_opt.get("fail_probability", 1.0) or 1.0)
    wait_s = float(local_opt.get("estimated_wait_s", 0.0) or 0.0)
    violation = max(0.0, total_s - deadline) / max(deadline, 1e-9)
    return (
        HEURISTIC_W_TIME * total_s / max(deadline, 1e-9)
        + HEURISTIC_W_LOCAL_FAIL * fail
        + HEURISTIC_W_QUEUE * wait_s / max(deadline, 1e-9)
        + HEURISTIC_W_DEADLINE_VIOLATION * violation
    )


def choose_structured_heuristic_action(dp: Dict[str, object], rng: random.Random):
    candidates: List[Tuple[float, object]] = []
    for opt in dp.get("tool_options", []) or []:
        score = _tool_score(opt, dp)
        if score < 1.0e8:
            candidates.append((score, task_env_mod.DecisionAction(action_type="tool", tool_type_id=int(opt["tool_type_id"]))))
    local_opt = dp.get("local_option", {}) or {}
    local_score = _local_score(local_opt, dp)
    if local_score < 1.0e8:
        candidates.append((local_score, task_env_mod.DecisionAction(action_type="local")))
    if not candidates:
        return task_env_mod.DecisionAction(action_type="pause", pause_duration_s=0.20)
    candidates.sort(key=lambda x: x[0])
    if rng.random() < float(HEURISTIC_EPSILON):
        return rng.choice([a for _, a in candidates[: min(3, len(candidates))]])
    if len(candidates) == 1 or float(HEURISTIC_SOFTMAX_TEMPERATURE) <= 0.0:
        return candidates[0][1]
    best = candidates[0][0]
    weights = [math.exp(-(s - best) / max(1e-6, float(HEURISTIC_SOFTMAX_TEMPERATURE))) for s, _ in candidates]
    total = sum(weights)
    r = rng.random() * total
    acc = 0.0
    for w, (_, action) in zip(weights, candidates):
        acc += w
        if acc >= r:
            return action
    return candidates[0][1]


def run_heuristic_closed_loop(dataset: Dict[str, object], tool_library: Optional[Dict[str, object]], seed: int) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    env = build_env(dataset, seed=seed, tool_library=tool_library)
    env.reset()
    rng = random.Random(seed + 1009)
    step_limit = compute_env_step_limit(dataset)
    steps_taken = 0
    while not env.done() and steps_taken < step_limit:
        dps = env.collect_decision_points()
        for dp in dps:
            action = choose_structured_heuristic_action(dp, rng)
            env.apply_decision(str(dp["task_id"]), int(dp["node_id"]), action)
        env.step()
        steps_taken += 1
    records = build_complete_task_records(env)
    summary = summarize_task_records(records, steps_taken=steps_taken, hit_step_limit=(not env.done()))
    return summary, records


def write_loss_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    fieldnames = [
        "epoch", "completion_rate", "avg_latency_s", "num_completed", "num_failed", "num_unfinished",
        "steps_taken", "hit_step_limit", "loss_total", "loss_topo", "loss_wire", "loss_tool",
        "loss_out", "loss_need", "loss_atom", "loss_bind_pos", "loss_sep", "loss_bind_pos_weighted", "loss_sep_weighted", "num_segments", "num_new_atoms", "num_new_prototypes",
        "num_atoms", "num_prototypes", "prototype_risk_mean", "prototype_risk_drift", "fit_seconds", "epoch_seconds",
        "memory_saved", "memory_warmup_cleared", "memory_max_atoms", "memory_max_prototypes",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})



def save_model_only(trainer, model_path: Path) -> Dict[str, str]:
    """Save knowledge weights without writing memory.json."""
    mp = Path(model_path).expanduser().resolve()
    mp.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "distiller_state_dict": trainer.distiller.state_dict(),
            "llm_adapter_state_dict": trainer.llm.adapter_state_dict(),
            "cfg": trainer.cfg.__dict__ if not hasattr(trainer.cfg, "__dataclass_fields__") else __import__("dataclasses").asdict(trainer.cfg),
            "action_dim": int(trainer.action_dim),
        },
        mp,
    )
    return {"model_path": str(mp), "memory_path": ""}


def clear_knowledge_memory(trainer, reason: str = "") -> Dict[str, int]:
    """Drop run-local memory once after the warm-up epochs so early memory is not saved."""
    before_atoms = len(getattr(trainer.memory, "buffer", []) or [])
    before_prototypes = len(getattr(trainer.memory, "prototypes", []) or [])
    trainer.memory.buffer = []
    trainer.memory.prototypes = []
    if hasattr(trainer, "last_snapshot"):
        trainer.last_snapshot = {"prototype_risk_mean": 0.0, "prototype_count": 0.0}
    suffix = f" reason={reason}" if reason else ""
    print(
        f"[knowledge-memory] warmup clear atoms {before_atoms}->0, prototypes {before_prototypes}->0{suffix}",
        flush=True,
    )
    return {
        "before_atoms": int(before_atoms),
        "after_atoms": 0,
        "before_prototypes": int(before_prototypes),
        "after_prototypes": 0,
    }


def trim_knowledge_memory(trainer, reason: str = "") -> Dict[str, int]:
    """Hard-cap in-memory atoms/prototypes before saving memory.json."""
    max_atoms = int(KNOWLEDGE_MAX_MEMORY_ATOMS)
    max_prototypes = int(KNOWLEDGE_MAX_MEMORY_PROTOTYPES)
    before_atoms = len(getattr(trainer.memory, "buffer", []) or [])
    before_prototypes = len(getattr(trainer.memory, "prototypes", []) or [])

    if max_atoms > 0 and before_atoms > max_atoms:
        trainer.memory.buffer = list(trainer.memory.buffer)[-max_atoms:]
    if max_prototypes > 0 and before_prototypes > max_prototypes:
        trainer.memory.prototypes = list(trainer.memory.prototypes)[-max_prototypes:]

    after_atoms = len(getattr(trainer.memory, "buffer", []) or [])
    after_prototypes = len(getattr(trainer.memory, "prototypes", []) or [])
    if before_atoms != after_atoms or before_prototypes != after_prototypes:
        suffix = f" reason={reason}" if reason else ""
        print(
            f"[knowledge-memory] trimmed atoms {before_atoms}->{after_atoms}, "
            f"prototypes {before_prototypes}->{after_prototypes}{suffix}",
            flush=True,
        )
    return {
        "before_atoms": int(before_atoms),
        "after_atoms": int(after_atoms),
        "before_prototypes": int(before_prototypes),
        "after_prototypes": int(after_prototypes),
    }


def build_knowledge_trainer(dataset: Dict[str, object], action_dim: int):
    if LOAD_PRETRAINED_KNOWLEDGE_IF_AVAILABLE and PRETRAINED_KNOWLEDGE_MODEL_PATH.exists():
        memory_path = None
        if OPTIONAL_PRETRAINED_KNOWLEDGE_MEMORY_PATH is not None and Path(OPTIONAL_PRETRAINED_KNOWLEDGE_MEMORY_PATH).exists():
            memory_path = str(OPTIONAL_PRETRAINED_KNOWLEDGE_MEMORY_PATH)
        trainer = knowledge_mod.KnowledgeTrainer.load(
            dataset=dataset,
            model_path=str(PRETRAINED_KNOWLEDGE_MODEL_PATH),
            memory_path=memory_path,
            action_dim=action_dim,
        )
        print(f"[knowledge-train] loaded pretrained knowledge: {PRETRAINED_KNOWLEDGE_MODEL_PATH}", flush=True)
        if CLEAR_MEMORY_AFTER_LOAD:
            trainer.memory.buffer = []
            trainer.memory.prototypes = []
            trainer.last_snapshot = {"prototype_risk_mean": 0.0, "prototype_count": 0.0}
    else:
        cfg = knowledge_mod.KnowledgeConfig(
            model_path=QWEN_MODEL_PATH if Path(QWEN_MODEL_PATH).exists() else QWEN_MODEL_PATH_FALLBACK,
            dtype=str(KNOWLEDGE_DTYPE),
            llm_max_length=int(KNOWLEDGE_MAX_LENGTH),
            batch_size=int(KNOWLEDGE_BATCH_SIZE),
            epochs_per_fit=int(KNOWLEDGE_FIT_EPOCHS_PER_OUTER_EPOCH),
            lr=float(KNOWLEDGE_LR),
            weight_decay=float(KNOWLEDGE_WEIGHT_DECAY),
            device=str(KNOWLEDGE_DEVICE),
            seed=int(SEED),
            verbose=bool(KNOWLEDGE_VERBOSE),
            use_llm_verbalizer=bool(KNOWLEDGE_USE_LLM_VERBALIZER),
            verbalize_on_update=bool(KNOWLEDGE_USE_LLM_VERBALIZER),
            max_buffer_atoms=int(KNOWLEDGE_MAX_MEMORY_ATOMS) if int(KNOWLEDGE_MAX_MEMORY_ATOMS) > 0 else 10**12,
            max_prototypes=int(KNOWLEDGE_MAX_MEMORY_PROTOTYPES) if int(KNOWLEDGE_MAX_MEMORY_PROTOTYPES) > 0 else 10**12,
            normalize_topo_wire_losses=bool(KNOWLEDGE_NORMALIZE_TOPO_WIRE_LOSSES),
            use_masked_tool_loss=bool(KNOWLEDGE_USE_MASKED_TOOL_LOSS),
            loss_topo_weight=float(KNOWLEDGE_LOSS_TOPO_WEIGHT),
            loss_wire_weight=float(KNOWLEDGE_LOSS_WIRE_WEIGHT),
            loss_tool_weight=float(KNOWLEDGE_LOSS_TOOL_WEIGHT),
            loss_out_weight=float(KNOWLEDGE_LOSS_OUT_WEIGHT),
            loss_need_weight=float(KNOWLEDGE_LOSS_NEED_WEIGHT),
            enable_disentangle_losses=bool(KNOWLEDGE_ENABLE_BIND_SEP_IN_TOTAL_LOSS),
            lambda_bind_pos=float(KNOWLEDGE_LAMBDA_BIND_POS),
            lambda_sep=float(KNOWLEDGE_LAMBDA_SEP),
            lambda_block=float(KNOWLEDGE_LAMBDA_BLOCK),
            lambda_div=float(KNOWLEDGE_LAMBDA_DIV),
        )
        cfg.llm_gradient_checkpointing = bool(KNOWLEDGE_GRADIENT_CHECKPOINTING) if hasattr(cfg, "llm_gradient_checkpointing") else False
        trainer = knowledge_mod.KnowledgeTrainer(dataset=dataset, action_dim=action_dim, cfg=cfg)
        print("[knowledge-train] initialized knowledge trainer from Qwen/config", flush=True)

    trainer.cfg.batch_size = int(KNOWLEDGE_BATCH_SIZE)
    trainer.cfg.epochs_per_fit = int(KNOWLEDGE_FIT_EPOCHS_PER_OUTER_EPOCH)
    trainer.cfg.lr = float(KNOWLEDGE_LR)
    trainer.cfg.weight_decay = float(KNOWLEDGE_WEIGHT_DECAY)
    trainer.cfg.verbose = bool(KNOWLEDGE_VERBOSE)
    trainer.cfg.use_llm_verbalizer = bool(KNOWLEDGE_USE_LLM_VERBALIZER)
    trainer.cfg.verbalize_on_update = bool(KNOWLEDGE_USE_LLM_VERBALIZER)
    if hasattr(trainer.cfg, "max_buffer_atoms"):
        trainer.cfg.max_buffer_atoms = int(KNOWLEDGE_MAX_MEMORY_ATOMS) if int(KNOWLEDGE_MAX_MEMORY_ATOMS) > 0 else 10**12
    if hasattr(trainer.cfg, "max_prototypes"):
        trainer.cfg.max_prototypes = int(KNOWLEDGE_MAX_MEMORY_PROTOTYPES) if int(KNOWLEDGE_MAX_MEMORY_PROTOTYPES) > 0 else 10**12
    trainer.cfg.normalize_topo_wire_losses = bool(KNOWLEDGE_NORMALIZE_TOPO_WIRE_LOSSES)
    trainer.cfg.use_masked_tool_loss = bool(KNOWLEDGE_USE_MASKED_TOOL_LOSS)
    trainer.cfg.loss_topo_weight = float(KNOWLEDGE_LOSS_TOPO_WEIGHT)
    trainer.cfg.loss_wire_weight = float(KNOWLEDGE_LOSS_WIRE_WEIGHT)
    trainer.cfg.loss_tool_weight = float(KNOWLEDGE_LOSS_TOOL_WEIGHT)
    trainer.cfg.loss_out_weight = float(KNOWLEDGE_LOSS_OUT_WEIGHT)
    trainer.cfg.loss_need_weight = float(KNOWLEDGE_LOSS_NEED_WEIGHT)
    trainer.cfg.enable_disentangle_losses = bool(KNOWLEDGE_ENABLE_BIND_SEP_IN_TOTAL_LOSS)
    trainer.cfg.lambda_bind_pos = float(KNOWLEDGE_LAMBDA_BIND_POS)
    trainer.cfg.lambda_sep = float(KNOWLEDGE_LAMBDA_SEP)
    trainer.cfg.lambda_block = float(KNOWLEDGE_LAMBDA_BLOCK)
    trainer.cfg.lambda_div = float(KNOWLEDGE_LAMBDA_DIV)
    trim_knowledge_memory(trainer, reason="after_trainer_init")
    return trainer


def main() -> None:
    random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    run_dir = OUTPUT_ROOT / f"{RUN_NAME}_{now_tag()}"
    ensure_dir(run_dir)
    loss_csv_path = run_dir / LOSS_FILE
    best_model_path = run_dir / BEST_MODEL_FILE
    memory_path = run_dir / MEMORY_FILE

    dataset, scan_info = load_dataset_from_dir(DATA_DIR, seed=SEED)
    print(f"[knowledge-train] dataset graphs={len(dataset.get('graphs', []))} tools={len(dataset.get('tool_catalog', []))} files_head={scan_info.get('file_names_head')}", flush=True)
    tool_library = load_dynamic_tool_library()
    validate_dynamic_tool_library_coverage(dataset, tool_library)

    num_tools = len(dataset.get("tool_catalog", []))
    action_dim = int(num_tools + 2)
    trainer = build_knowledge_trainer(dataset, action_dim=action_dim)

    rows: List[Dict[str, object]] = []
    for epoch in range(1, int(NUM_EPOCHS) + 1):
        epoch_t0 = time.time()
        print(f"[knowledge-train] epoch {epoch}/{NUM_EPOCHS} start", flush=True)
        summary, records = run_heuristic_closed_loop(dataset, tool_library=tool_library, seed=SEED + epoch)
        stats = trainer.fit_on_task_records(
            list(records),
            batch_size=int(KNOWLEDGE_BATCH_SIZE),
            epochs=int(KNOWLEDGE_FIT_EPOCHS_PER_OUTER_EPOCH),
        )
        memory_warmup_cleared = False
        if (
            bool(CLEAR_MEMORY_AFTER_SAVE_WARMUP)
            and int(MEMORY_SAVE_SKIP_EPOCHS) > 0
            and int(epoch) == int(MEMORY_SAVE_SKIP_EPOCHS)
        ):
            clear_knowledge_memory(trainer, reason=f"after_epoch_{epoch}_warmup_before_memory_save")
            memory_warmup_cleared = True

        trim_knowledge_memory(trainer, reason=f"epoch_{epoch}_before_save")
        digest = trainer.knowledge_digest()
        memory_saved = int(epoch) > int(MEMORY_SAVE_SKIP_EPOCHS)
        if memory_saved:
            trainer.save(str(best_model_path), str(memory_path))
        else:
            save_model_only(trainer, best_model_path)
        row = {
            "epoch": int(epoch),
            "completion_rate": float(summary.get("completion_rate", 0.0)),
            "avg_latency_s": float(summary.get("avg_latency_s", 0.0)),
            "num_completed": int(summary.get("num_completed", 0)),
            "num_failed": int(summary.get("num_failed", 0)),
            "num_unfinished": int(summary.get("num_unfinished", 0)),
            "steps_taken": int(summary.get("steps_taken", 0)),
            "hit_step_limit": bool(summary.get("hit_step_limit", False)),
            "loss_total": float(stats.get("loss_total", 0.0)),
            "loss_topo": float(stats.get("loss_topo", 0.0)),
            "loss_wire": float(stats.get("loss_wire", 0.0)),
            "loss_tool": float(stats.get("loss_tool", 0.0)),
            "loss_out": float(stats.get("loss_out", 0.0)),
            "loss_need": float(stats.get("loss_need", 0.0)),
            "loss_atom": float(stats.get("loss_atom", 0.0)),
            "loss_bind_pos": float(stats.get("loss_bind_pos", 0.0)),
            "loss_sep": float(stats.get("loss_sep", 0.0)),
            "loss_bind_pos_weighted": (
                float(stats.get("loss_bind_pos", 0.0)) * float(KNOWLEDGE_LAMBDA_BIND_POS)
                if bool(KNOWLEDGE_ENABLE_BIND_SEP_IN_TOTAL_LOSS) else 0.0
            ),
            "loss_sep_weighted": (
                float(stats.get("loss_sep", 0.0)) * float(KNOWLEDGE_LAMBDA_SEP)
                if bool(KNOWLEDGE_ENABLE_BIND_SEP_IN_TOTAL_LOSS) else 0.0
            ),
            "loss_topo_depth": float(stats.get("loss_topo_depth", 0.0)),
            "loss_topo_role": float(stats.get("loss_topo_role", 0.0)),
            "loss_topo_critical": float(stats.get("loss_topo_critical", 0.0)),
            "loss_wire_queue": float(stats.get("loss_wire_queue", 0.0)),
            "loss_wire_slack": float(stats.get("loss_wire_slack", 0.0)),
            "loss_wire_violation": float(stats.get("loss_wire_violation", 0.0)),
            "loss_atom_risk": float(stats.get("loss_atom_risk", 0.0)),
            "loss_atom_gain": float(stats.get("loss_atom_gain", 0.0)),
            "loss_atom_conf": float(stats.get("loss_atom_conf", 0.0)),
            "tool_loss_observed_actions": float(stats.get("tool_loss_observed_actions", 0.0)),
            "num_segments": int(stats.get("num_segments", 0)),
            "num_new_atoms": int(stats.get("num_new_atoms", 0)),
            "num_new_prototypes": int(stats.get("num_new_prototypes", 0)),
            "num_atoms": int(digest.get("num_atoms", 0)),
            "num_prototypes": int(digest.get("num_prototypes", 0)),
            "prototype_risk_mean": float(stats.get("prototype_risk_mean", 0.0)),
            "prototype_risk_drift": float(stats.get("prototype_risk_drift", 0.0)),
            "fit_seconds": float(stats.get("fit_seconds", 0.0)),
            "epoch_seconds": float(time.time() - epoch_t0),
            "memory_saved": bool(memory_saved),
            "memory_warmup_cleared": bool(memory_warmup_cleared),
            "memory_max_atoms": int(KNOWLEDGE_MAX_MEMORY_ATOMS),
            "memory_max_prototypes": int(KNOWLEDGE_MAX_MEMORY_PROTOTYPES),
        }
        rows.append(row)
        write_loss_csv(loss_csv_path, rows)
        print(
            f"[knowledge-train] epoch={epoch} comp={row['completion_rate']:.4f} lat={row['avg_latency_s']:.4f} "
            f"loss={row['loss_total']:.4f} segments={row['num_segments']} atoms={row['num_atoms']} protos={row['num_prototypes']} "
            f"memory_saved={bool(memory_saved)} saved=({loss_csv_path.name}, {best_model_path.name}"
            f"{', ' + memory_path.name if memory_saved else ''})",
            flush=True,
        )

    print(f"[knowledge-train] done | run_dir={run_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
