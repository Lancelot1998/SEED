# -*- coding: utf-8 -*-
"""
baseline_runtime.py

Independent baseline implementation for knowledge + Qwen action policy + PPO/SFT/DPO.

This module owns the shared SFT/DPO/PPO policy, rollout, metric, and training
runtime. It imports the environment and knowledge layers through package-qualified
interfaces.

Action convention:
  0 ... NUM_TOOL_TYPES-1  -> tool action with tool_type_id=index
  NUM_TOOL_TYPES          -> local action
  NUM_TOOL_TYPES + 1      -> pause action
"""

from __future__ import annotations

import csv
import json
import math
import os
import random
import shutil
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from baseline_suite.config.paths import (
    DATA_DIR,
    DATASET_FILE_PATTERNS,
    DATASET_METADATA_FILES,
    DEFAULT_POLICY_OFFLOAD_PATH,
    EXTERNAL_MEMORY_LIBRARY_PATH,
    LOCAL_TOOL_LIBRARY_PATH,
    OPTIONAL_PRETRAINED_KNOWLEDGE_MEMORY_PATH,
    OUTPUT_ROOT,
    PRETRAINED_KNOWLEDGE_MODEL_PATH,
    PROJECT_ROOT,
    QWEN_MODEL_PATH,
    QWEN_MODEL_PATH_FALLBACK,
    RunArtifacts,
    TOOL_LIBRARY_PATHS,
)

THIS_DIR = PROJECT_ROOT

from baseline_suite.config.settings import *  # noqa: F403 - shared experiment settings

_DYNAMIC_TOOL_LIBRARY_CACHE: Optional[Dict[str, object]] = None
_DYNAMIC_TOOL_LIBRARY_CACHE_KEY: Optional[Tuple[str, ...]] = None

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except Exception as exc:
    raise RuntimeError("transformers is required. Please install transformers>=4.38.") from exc

from baseline_suite.environment import task_env as task_env_mod
from baseline_suite.knowledge import adapter as knowledge_mod

# ============================================================
# Dataclasses
# ============================================================

@dataclass
class PolicyConfig:
    model_path: str
    tokenizer_path: Optional[str] = None
    dtype: str = LLM_DTYPE
    max_length: int = LLM_MAX_LENGTH
    use_lora: bool = LLM_USE_LORA
    lora_layer_ids: Tuple[int, ...] = LLM_LORA_LAYER_IDS
    lora_target_modules: Tuple[str, ...] = LLM_LORA_TARGET_MODULES
    lora_r: int = LLM_LORA_R
    lora_alpha: float = LLM_LORA_ALPHA
    lora_dropout: float = LLM_LORA_DROPOUT
    tune_last_n_blocks: int = LLM_TUNE_LAST_N_BLOCKS
    train_final_norm: bool = LLM_TRAIN_FINAL_NORM
    gradient_checkpointing: bool = LLM_GRADIENT_CHECKPOINTING
    enable_multi_gpu: bool = LLM_ENABLE_MULTI_GPU
    device_map: str = LLM_DEVICE_MAP
    max_memory_per_gpu: str = LLM_MAX_MEMORY_PER_GPU
    low_cpu_mem_usage: bool = LLM_LOW_CPU_MEM_USAGE
    offload_folder: str = str(DEFAULT_POLICY_OFFLOAD_PATH)
    seed: int = SEED
    verbose: bool = True

@dataclass
class ActionInfo:
    action_index: int
    action_type: str
    tool_type_id: Optional[int]
    log_prob: float
    probability: float
    value: float
    valid_action_indices: List[int]
    prompt: str
    knowledge_prompt: str = ""
    action_mask: List[int] = field(default_factory=list)
    action_names: List[str] = field(default_factory=list)

@dataclass
class RolloutTrace:
    trace_id: str
    epoch: int
    step_count: int
    env_time: float
    task_id: str
    node_id: int
    node_type_id: int
    prompt: str
    knowledge_prompt: str
    action_mask: List[int]
    exec_action: int
    valid_actions: List[int]
    old_log_prob: float
    probability: float
    value: float
    action_names: List[str]
    node_remaining_deadline_s: float
    task_remaining_deadline_s: float
    realized_utility: float = 0.0
    reward: float = 0.0
    node_success: int = 0
    mission_success: int = 0
    violation: int = 0
    actual_latency_s: float = 0.0
    outcome_event_type: str = ""

@dataclass
class PreferenceExample:
    prompt: str
    positive_action: int
    negative_action: int
    ref_positive_logp: float = 0.0
    ref_negative_logp: float = 0.0
    reward_gap: float = 0.0
    action_names: List[str] = field(default_factory=list)
    meta: Dict[str, object] = field(default_factory=dict)

@dataclass
class SFTExample:
    prompt: str
    target_action: int
    action_names: List[str] = field(default_factory=list)
    meta: Dict[str, object] = field(default_factory=dict)

# ============================================================
# Basic helpers
# ============================================================

def now_tag() -> str:
    return time.strftime("%Y%m%d_%H%M%S")

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)

def load_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

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
    x = float(x)
    if x < 60.0:
        return f"{x:.1f}s"
    if x < 3600.0:
        return f"{x / 60.0:.1f}m"
    return f"{x / 3600.0:.2f}h"

def text_progress_bar(done: int, total: int, width: int = 28) -> str:
    total = max(1, int(total))
    done = max(0, min(int(done), total))
    filled = int(round(width * done / total))
    return "[" + "#" * filled + "." * (width - filled) + "]"

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
            print(f"{prefix} cuda:{i} {prop.name} total={prop.total_memory / (1024 ** 3):.2f}GiB", flush=True)
        except Exception:
            pass

def dtype_from_name(name: str) -> torch.dtype:
    name = str(name).lower()
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if name in {"fp16", "float16", "half"}:
        return torch.float16
    return torch.float32

def resolve_llm_path(path: str = QWEN_MODEL_PATH, fallback: str = QWEN_MODEL_PATH_FALLBACK) -> str:
    p = Path(path).expanduser()
    if p.exists():
        return str(p)
    pf = Path(fallback).expanduser()
    if pf.exists():
        return str(pf)
    return path

def build_max_memory_map(max_memory_per_gpu: str) -> Optional[Dict[int, str]]:
    if not torch.cuda.is_available():
        return None
    return {i: str(max_memory_per_gpu) for i in range(torch.cuda.device_count())}

def first_parameter_device(module: nn.Module, default: Optional[torch.device] = None) -> torch.device:
    for p in module.parameters():
        return p.device
    return default or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def last_parameter_device(module: nn.Module, default: Optional[torch.device] = None) -> torch.device:
    last = None
    for p in module.parameters():
        last = p.device
    return last or default or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ============================================================
# Dataset and dynamic tool library
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
        "dataset_name": f"baseline_dataset_from_{data_dir.name}",
        "config": {"source": "disk_pool", "data_dir": str(data_dir), "sample_count": len(raw_items), "seed": int(seed)},
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
    return task_env_mod._normalize_dataset_structure(out, fallback_dataset_name=str(dataset.get("dataset_name", "baseline_subset")))

def build_graph_meta_map(dataset: Dict[str, object]) -> Dict[str, Dict[str, object]]:
    return {str(g.get("graph_id")): g for g in dataset.get("graphs", [])}

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
            raise RuntimeError("USE_DYNAMIC_TOOL_LIBRARY=False while ALLOW_TOOL_LIBRARY_FALLBACK=False.")
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
                print(f"[baseline] WARNING: {msg}; continuing with fallback allowed", flush=True)
                continue
            raise FileNotFoundError(msg)
        data = load_json(path)
        if not isinstance(data, dict) or not isinstance(data.get("entries", None), list):
            raise ValueError(f"Invalid tool library format: {path}")
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
        "created_by": "baseline_common.py",
        "source_paths": loaded_paths,
        "num_entries": len(merged_entries),
        "entries": merged_entries,
    }
    _DYNAMIC_TOOL_LIBRARY_CACHE = merged
    _DYNAMIC_TOOL_LIBRARY_CACHE_KEY = cache_key
    print(f"[baseline] dynamic tool libraries loaded: files={len(loaded_paths)} | entries={len(merged_entries)} | fallback_allowed={bool(ALLOW_TOOL_LIBRARY_FALLBACK)}", flush=True)
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
    missing_count = 0
    for graph in dataset.get("graphs", []) or []:
        graph_id = str(graph.get("graph_id"))
        for node in graph.get("nodes", []) or []:
            node_id = int(node.get("node_id", 0))
            for slot, allowed in enumerate(list(node.get("allowed_tools_mask", []))):
                if int(allowed) != 1:
                    continue
                required += 1
                if (graph_id, node_id, int(slot)) not in keys:
                    missing_count += 1
                    if len(missing) < 20:
                        missing.append({"graph_id": graph_id, "node_id": node_id, "slot": int(slot)})
    info = {
        "required_allowed_tool_slots": int(required),
        "covered_allowed_tool_slots": int(required - missing_count),
        "missing_allowed_tool_slots": int(missing_count),
        "coverage_rate": float((required - missing_count) / max(1, required)),
        "missing_examples": missing,
    }
    print(f"[tool-library:coverage] required={info['required_allowed_tool_slots']} covered={info['covered_allowed_tool_slots']} missing={info['missing_allowed_tool_slots']} coverage={info['coverage_rate']:.4f}", flush=True)
    if missing_count > 0 and not bool(ALLOW_TOOL_LIBRARY_FALLBACK):
        raise RuntimeError("Dynamic tool library coverage check failed while ALLOW_TOOL_LIBRARY_FALLBACK=False. " + f"missing={missing_count} examples={missing[:5]}")
    return info

# ============================================================
# Environment helpers
# ============================================================

def build_relaxed_tool_profiles(dataset: Dict[str, object]) -> Optional[List[object]]:
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
    return task_env_mod.MissionEnvironment(
        dataset=dataset,
        env_cfg=env_cfg,
        tool_profiles=build_relaxed_tool_profiles(dataset),
        tool_library=load_dynamic_tool_library(),
    )

def compute_env_step_limit(dataset: Dict[str, object]) -> int:
    if int(FORCE_MAX_ENV_STEPS) > 0:
        return int(FORCE_MAX_ENV_STEPS)
    num_graphs = len(dataset.get("graphs", []))
    num_nodes = sum(int(g.get("num_nodes", len(g.get("nodes", [])))) for g in dataset.get("graphs", []))
    return max(200, int(40 * max(1, num_nodes) + 80 * max(1, num_graphs)))

def build_complete_task_records(env) -> List[Dict[str, object]]:
    records = list(getattr(env, "completed_task_records", []) or [])
    existing = {str(r.get("task_id")) for r in records}
    for task_id, task in getattr(env, "tasks", {}).items():
        if str(task_id) in existing:
            continue
        if task.status in {"completed", "failed"}:
            records.append(
                {
                    "task_id": task.graph_id,
                    "status": task.status,
                    "arrival_time": task.arrival_time,
                    "end_time": task.end_time,
                    "latency_s": None if task.end_time is None else round(task.end_time - task.arrival_time, 4),
                    "task_deadline_s": float(task.task_deadline),
                    "restart_count": task.restart_count,
                    "failure_reason": task.failure_reason,
                    "log": list(task.log),
                }
            )
    return records

def env_runtime_counts(env) -> Dict[str, int]:
    tasks = getattr(env, "tasks", {})
    nodes = []
    for task in tasks.values():
        nodes.extend(list(getattr(task, "nodes", {}).values()))
    return {
        "task_waiting": sum(1 for t in tasks.values() if t.status == "waiting"),
        "task_active": sum(1 for t in tasks.values() if t.status == "active"),
        "task_completed": sum(1 for t in tasks.values() if t.status == "completed"),
        "task_failed": sum(1 for t in tasks.values() if t.status == "failed"),
        "node_ready": sum(1 for n in nodes if n.status in {"ready", "idle"}),
        "queued_tool": sum(len(q) for q in getattr(env, "tool_queues", {}).values()),
        "running_tool": sum(len(q) for q in getattr(env, "tool_running", {}).values()),
        "queued_local": len(getattr(env, "local_queue", [])),
        "running_local": len(getattr(env, "local_running", [])),
    }

def compact_env_counts(counts: Dict[str, int]) -> str:
    keys = ["task_waiting", "task_active", "task_completed", "task_failed", "node_ready", "queued_tool", "running_tool", "queued_local", "running_local"]
    return " ".join(f"{k}={int(counts.get(k, 0))}" for k in keys)

# ============================================================
# Metrics and CSV writers.  Epoch CSV field names intentionally match TDPO run.
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
        if str(ev.get("event_type")) in {"decision_tool", "decision_local", "decision_pause", "tool_execution_started", "local_execution_started"}
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
    return float(float(payload.get("queue_delay_s", 0.0) or 0.0) + float(payload.get("compute_s", payload.get("predicted_compute_s", 0.0)) or 0.0))

def _completed_node_metric_rows(record: Dict[str, object]) -> List[Dict[str, object]]:
    if str(record.get("status")) != "completed":
        return []
    by_node: Dict[int, List[Dict[str, object]]] = {}
    for ev in list(record.get("log", []) or []):
        node_id = ev.get("node_id")
        if node_id is None:
            continue
        try:
            by_node.setdefault(int(node_id), []).append(ev)
        except Exception:
            continue
    rows: List[Dict[str, object]] = []
    for node_id in sorted(by_node):
        events = by_node[node_id]
        if not any(str(ev.get("event_type")) in {"tool_success", "local_success"} for ev in events):
            continue
        rows.append({"task_id": str(record.get("task_id", "")), "node_id": int(node_id), "energy_proxy": float(sum(_event_energy_proxy(ev) for ev in events)), "delay_s": float(_node_delay_from_events(events))})
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

def build_energy_delay_detail_rows(epoch: int, stage: str, task_records: Sequence[Dict[str, object]]) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    completed = _completed_task_records(task_records)
    node_rows: List[Dict[str, object]] = []
    for rec in completed:
        node_rows.extend(_completed_node_metric_rows(rec))
    total_energy = float(sum(_record_energy_proxy(rec) for rec in completed))
    total_delay = float(sum(_completed_task_latency(rec) for rec in completed))
    base = {"stage": str(stage), "epoch": int(epoch), "num_completed_tasks": int(len(completed)), "num_completed_nodes": int(len(node_rows))}
    energy_rows = [{**base, "record_type": "epoch_total", "task_id": "", "node_id": "", "status": "completed_only", "energy_proxy": total_energy, "epoch_completed_total_energy_proxy": total_energy}]
    delay_rows = [{**base, "record_type": "epoch_total", "task_id": "", "node_id": "", "status": "completed_only", "delay_s": total_delay, "epoch_completed_total_delay_s": total_delay}]
    for rec in completed:
        task_id = str(rec.get("task_id", ""))
        task_energy = float(_record_energy_proxy(rec))
        task_delay = float(_completed_task_latency(rec))
        energy_rows.append({**base, "record_type": "task", "task_id": task_id, "node_id": "", "status": "completed", "energy_proxy": task_energy, "epoch_completed_total_energy_proxy": total_energy})
        delay_rows.append({**base, "record_type": "task", "task_id": task_id, "node_id": "", "status": "completed", "delay_s": task_delay, "epoch_completed_total_delay_s": total_delay})
    for item in node_rows:
        energy_rows.append({**base, "record_type": "node", "task_id": item.get("task_id", ""), "node_id": item.get("node_id", ""), "status": "completed", "energy_proxy": float(item.get("energy_proxy", 0.0)), "epoch_completed_total_energy_proxy": total_energy})
        delay_rows.append({**base, "record_type": "node", "task_id": item.get("task_id", ""), "node_id": item.get("node_id", ""), "status": "completed", "delay_s": float(item.get("delay_s", 0.0)), "epoch_completed_total_delay_s": total_delay})
    return energy_rows, delay_rows

def _write_csv(path: Path, rows: Sequence[Dict[str, object]], fieldnames: Sequence[str]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})

def write_epoch_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    fieldnames = [
        "epoch", "completion_rate", "avg_latency_s", "energy_proxy",
        "completed_total_energy_proxy", "avg_completed_energy_proxy_per_graph",
        "completed_total_delay_s", "avg_completed_delay_s_per_graph", "num_completed_nodes_logged",
        "num_completed", "num_failed", "num_unfinished", "steps_taken", "hit_step_limit", "num_traces",
        "num_pairs_added", "num_pairs_buffer", "num_success_buffer", "num_explore_buffer", "num_pairs_train",
        "num_tier_a_pairs", "num_tier_b_pairs", "num_tier_a_raw", "num_tier_b_raw",
        "tdpo_loss", "tdpo_pref_acc", "tdpo_margin", "tdpo_confidence", "tdpo_grad_norm", "tdpo_grad_norm_clipped",
        "tdpo_completion_pair_ratio", "tdpo_completion_advantage_ratio", "tdpo_node_success_advantage_ratio", "tdpo_pair_gap_mean",
        "rollout_seconds", "pair_seconds", "new_ref_seconds", "tdpo_seconds", "tdpo_ref_seconds", "tdpo_update_seconds", "knowledge_update_seconds", "epoch_seconds",
        "knowledge_num_atoms", "knowledge_num_prototypes", "is_best",
    ]
    _write_csv(path, rows, fieldnames)

def write_eval_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    fieldnames = [
        "epoch", "completion_rate", "avg_latency_s", "energy_proxy", "completed_total_energy_proxy", "avg_completed_energy_proxy_per_graph",
        "completed_total_delay_s", "avg_completed_delay_s_per_graph", "num_completed_nodes_logged", "num_completed", "num_failed", "num_unfinished",
        "steps_taken", "hit_step_limit", "num_traces", "rollout_seconds", "eval_seconds", "eval_seed", "use_memory",
    ]
    _write_csv(path, rows, fieldnames)

def write_energy_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    _write_csv(path, rows, ["stage", "epoch", "record_type", "task_id", "node_id", "status", "energy_proxy", "epoch_completed_total_energy_proxy", "num_completed_tasks", "num_completed_nodes"])

def write_delay_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    _write_csv(path, rows, ["stage", "epoch", "record_type", "task_id", "node_id", "status", "delay_s", "epoch_completed_total_delay_s", "num_completed_tasks", "num_completed_nodes"])

def write_knowledge_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    fieldnames = ["stage", "epoch", "mode", "loss_total", "loss_topo", "loss_wire", "loss_tool", "loss_out", "loss_need", "loss_atom", "num_segments", "num_new_atoms", "num_new_prototypes", "num_atoms", "num_prototypes", "prototype_risk_mean", "prototype_risk_drift", "fit_seconds", "total_updates"]
    _write_csv(path, rows, fieldnames)

def build_knowledge_row(stage: str, epoch: int, stats: Dict[str, object], digest: Dict[str, object]) -> Dict[str, object]:
    stats = stats or {}
    digest = digest or {}
    return {
        "stage": str(stage), "epoch": int(epoch), "mode": str(stats.get("mode", "")),
        "loss_total": float(stats.get("loss_total", 0.0) or 0.0),
        "loss_topo": float(stats.get("loss_topo", 0.0) or 0.0),
        "loss_wire": float(stats.get("loss_wire", 0.0) or 0.0),
        "loss_tool": float(stats.get("loss_tool", 0.0) or 0.0),
        "loss_out": float(stats.get("loss_out", 0.0) or 0.0),
        "loss_need": float(stats.get("loss_need", 0.0) or 0.0),
        "loss_atom": float(stats.get("loss_atom", 0.0) or 0.0),
        "num_segments": int(stats.get("num_segments", 0) or 0),
        "num_new_atoms": int(stats.get("num_new_atoms", 0) or 0),
        "num_new_prototypes": int(stats.get("num_new_prototypes", 0) or 0),
        "num_atoms": int(digest.get("num_atoms", 0) or 0),
        "num_prototypes": int(digest.get("num_prototypes", 0) or 0),
        "prototype_risk_mean": float(digest.get("prototype_risk_mean", stats.get("prototype_risk_mean", 0.0)) or 0.0),
        "prototype_risk_drift": float(stats.get("prototype_risk_drift", 0.0) or 0.0),
        "fit_seconds": float(stats.get("fit_seconds", stats.get("update_seconds", 0.0)) or 0.0),
        "total_updates": int(stats.get("total_updates", 0) or 0),
    }

# ============================================================
# Action, prompt, knowledge helpers
# ============================================================

def action_dim_from_num_tools(num_tools: int) -> int:
    return int(num_tools) + 2

def local_action_index(num_tools: int) -> int:
    return int(num_tools)

def pause_action_index(num_tools: int) -> int:
    return int(num_tools) + 1

def action_index_to_env_action(action_index: int, num_tools: int, pause_duration_s: float = PAUSE_DURATION_S):
    idx = int(action_index)
    if idx < int(num_tools):
        return task_env_mod.DecisionAction(action_type="tool", tool_type_id=idx)
    if idx == local_action_index(num_tools):
        return task_env_mod.DecisionAction(action_type="local")
    return task_env_mod.DecisionAction(action_type="pause", pause_duration_s=float(pause_duration_s))

def action_index_to_label(action_index: int, num_tools: int) -> str:
    idx = int(action_index)
    if idx < int(num_tools):
        return f"tool:{idx}"
    if idx == local_action_index(num_tools):
        return "local"
    if idx == pause_action_index(num_tools):
        return "pause"
    return f"invalid:{idx}"

def valid_action_mask_from_dp(dp: Dict[str, object], num_tools: int, allow_pause: bool = True) -> List[int]:
    mask = [1] * action_dim_from_num_tools(num_tools)
    return mask

def action_mask_to_indices(mask: Sequence[int]) -> List[int]:
    return [i for i, v in enumerate(mask) if int(v) == 1]

def _fmt(x: object, digits: int = 4) -> str:
    try:
        return f"{float(x):.{digits}f}"
    except Exception:
        return str(x)

def _short_text(text: object, limit: int = 160) -> str:
    s = str(text or "").replace("\n", " ").strip()
    return s if len(s) <= limit else s[: max(0, limit - 3)] + "..."

def infer_queue_class(dp: Dict[str, object]) -> int:
    q_tool = int(dp.get("queued_tool_jobs_total", 0) or 0)
    q_local = int(dp.get("queued_local_jobs", 0) or 0)
    q = q_tool + q_local
    if q <= 1:
        return 0
    if q <= 5:
        return 1
    return 2

def infer_slack_class(dp: Dict[str, object]) -> int:
    rem = float(dp.get("node_remaining_deadline_s", 0.0) or 0.0)
    fastest = fastest_predicted_latency(dp, default=1.0)
    ratio = rem / max(1e-6, fastest)
    if ratio < 1.2:
        return 0
    if ratio < 2.0:
        return 1
    return 2

def build_state_query_embedding(dp: Dict[str, object], num_tools: int, target_dim: int = 64) -> List[float]:
    vals: List[float] = [
        float(dp.get("node_type_id", 0) or 0) / 16.0,
        float(dp.get("task_remaining_deadline_s", 0.0) or 0.0) / 20.0,
        float(dp.get("node_remaining_deadline_s", 0.0) or 0.0) / 10.0,
        float(dp.get("ready_frontier_count", 0) or 0) / 8.0,
        float(dp.get("completed_node_count", 0) or 0) / max(1.0, float(dp.get("num_nodes", 1) or 1)),
        float(dp.get("num_predecessors", 0) or 0) / 8.0,
        float(dp.get("num_successors", 0) or 0) / 8.0,
        float(dp.get("running_tool_jobs_total", 0) or 0) / 16.0,
        float(dp.get("queued_tool_jobs_total", 0) or 0) / 32.0,
        float(dp.get("running_local_jobs", 0) or 0) / 8.0,
        float(dp.get("queued_local_jobs", 0) or 0) / 32.0,
    ]
    local = dp.get("local_option", {}) or {}
    vals.extend([
        float(local.get("predicted_total_s", 0.0) or 0.0) / 10.0,
        float(local.get("fail_probability", 0.0) or 0.0),
        1.0 if bool(local.get("queue_blocked", False)) else 0.0,
    ])
    tool_opts = list(dp.get("tool_options", []) or [])
    if tool_opts:
        vals.extend([
            min(float(o.get("predicted_total_s", 0.0) or 0.0) for o in tool_opts) / 10.0,
            safe_mean([float(o.get("risk_estimate", 0.0) or 0.0) for o in tool_opts], 0.0),
            safe_mean([float(o.get("packet_loss_rate", 0.0) or 0.0) for o in tool_opts], 0.0),
            safe_mean([1.0 if bool(o.get("queue_blocked", False)) else 0.0 for o in tool_opts], 0.0),
        ])
    vals.extend([float(x) for x in list(dp.get("allowed_tools_mask", []))[:num_tools]])
    if len(vals) < target_dim:
        vals += [0.0] * (target_dim - len(vals))
    return [max(-10.0, min(10.0, float(x))) for x in vals[:target_dim]]

def query_knowledge_for_dp(dp: Dict[str, object], num_tools: int, knowledge_trainer, use_memory: bool) -> Dict[str, object]:
    action_dim = action_dim_from_num_tools(num_tools)
    if knowledge_trainer is None or not bool(use_memory):
        return {"memory_vector": [0.0] * (action_dim + 4), "action_bias": [0.0] * action_dim, "knowledge_prompt": "", "selected": []}
    try:
        q = build_state_query_embedding(dp, num_tools, target_dim=64)
        out = knowledge_trainer.query_memory(q, int(dp.get("node_type_id", 0) or 0), infer_queue_class(dp), infer_slack_class(dp))
        text = str(out.get("knowledge_prompt", "") or "")
        if len(text) > int(KNOWLEDGE_PROMPT_MAX_CHARS):
            text = text[: int(KNOWLEDGE_PROMPT_MAX_CHARS)]
        out = dict(out)
        out["knowledge_prompt"] = text
        return out
    except Exception as exc:
        return {"memory_vector": [0.0] * (action_dim + 4), "action_bias": [0.0] * action_dim, "knowledge_prompt": f"[knowledge unavailable: {type(exc).__name__}]", "selected": []}

def _current_action_name_map(dp: Dict[str, object], num_tools: int) -> List[str]:
    names = [f"tool_{i}" for i in range(num_tools)] + ["local", "pause"]
    for opt in list(dp.get("tool_options", []) or []):
        try:
            idx = int(opt.get("tool_type_id", opt.get("slot", -1)))
        except Exception:
            continue
        if 0 <= idx < num_tools:
            names[idx] = str(opt.get("tool_name") or opt.get("name") or f"tool_{idx}")
    return names

def build_prompt_from_dp(dp: Dict[str, object], num_tools: int, knowledge_prompt: str = "", graph_meta: Optional[Dict[str, object]] = None) -> Tuple[str, List[str]]:
    graph_meta = graph_meta or {}
    action_names = _current_action_name_map(dp, num_tools)
    allowed_mask = list(dp.get("allowed_tools_mask", []) or [])[:num_tools]

    if not bool(FAST_COMPACT_PROMPT):
        lines: List[str] = []
        lines.append("You are a node-level scheduler for a vehicular edge-computing mission DAG.")
        lines.append("Choose exactly one action for the current ready node. Output only ACTION_INDEX=<integer>.")
        lines.append("[Action convention]")
        lines.append(f"Tool actions: 0..{max(0, num_tools - 1)}; local action: {local_action_index(num_tools)}; pause action: {pause_action_index(num_tools)}.")
        lines.append("[Task state]")
        lines.append(
            f"task_id={dp.get('task_id')} node_id={dp.get('node_id')} node_type={dp.get('node_type_id')} "
            f"ready_frontier={dp.get('ready_frontier_count')} completed_nodes={dp.get('completed_node_count')}/{dp.get('num_nodes')} restart={dp.get('restart_count')}"
        )
        lines.append(
            f"task_remaining={_fmt(dp.get('task_remaining_deadline_s'))}s node_remaining={_fmt(dp.get('node_remaining_deadline_s'))}s "
            f"node_deadline={_fmt(dp.get('node_deadline_s'))}s task_deadline={_fmt(dp.get('task_deadline_s'))}s"
        )
        intent = str(dp.get("intent_text") or "")
        if intent:
            lines.append(f"node_intent={_short_text(intent, 180)}")
        lines.append("allowed_tool_mask=" + ",".join(str(int(x)) for x in allowed_mask))
        lines.append("Rule: a tool index is semantically allowed only when its allowed_tool_mask value is 1. Local and pause may still be selected, but pause should be rare.")
        lines.append("[Candidate actions]")
        for i in range(num_tools):
            opt = next((row for row in list(dp.get("tool_options", []) or []) if int(row.get("tool_type_id", row.get("slot", -1))) == i), None)
            allowed = int(allowed_mask[i]) if i < len(allowed_mask) else 0
            if opt is None:
                lines.append(f"ACTION_INDEX={i}: tool name={action_names[i]} allowed={allowed} unavailable=1")
            else:
                lines.append(
                    f"ACTION_INDEX={i}: tool name={action_names[i]} allowed={allowed} queue_blocked={bool(opt.get('queue_blocked', False))} "
                    f"pred_total={_fmt(opt.get('predicted_total_s'))}s queue={_fmt(opt.get('base_queue_delay_s', opt.get('estimated_wait_s', 0.0)))}s "
                    f"exec={_fmt(opt.get('exec_s'))}s uplink={_fmt(opt.get('uplink_s'))}s downlink={_fmt(opt.get('downlink_s'))}s "
                    f"risk={_fmt(opt.get('risk_estimate'))} pkt_loss={_fmt(opt.get('packet_loss_rate'))}"
                )
        local = dp.get("local_option", {}) or {}
        lines.append(
            f"ACTION_INDEX={local_action_index(num_tools)}: local name=local queue_blocked={bool(local.get('queue_blocked', False))} "
            f"pred_total={_fmt(local.get('predicted_total_s'))}s compute={_fmt(local.get('compute_s'))}s fail_prob={_fmt(local.get('fail_probability'))}"
        )
        lines.append(f"ACTION_INDEX={pause_action_index(num_tools)}: pause name=pause duration={_fmt(PAUSE_DURATION_S)}s")
        if knowledge_prompt:
            lines.append("[Retrieved knowledge]")
            lines.append(str(knowledge_prompt))
        lines.append("[Decision]")
        lines.append("Return one line only: ACTION_INDEX=<integer>")
        return "\n".join(lines) + "\n", action_names

    # Compact mode: fewer tokens per decision, but still exposes the same action
    # convention, current DAG pressure, allowed-tool mask, local option, available
    # tool timing, and retrieved knowledge text.
    intent = str(dp.get("intent_text") or "")
    if not intent and isinstance(graph_meta, dict):
        node_id = int(dp.get("node_id", 0) or 0)
        for n in graph_meta.get("nodes", []) or []:
            if int(n.get("node_id", -1)) == node_id:
                intent = str(n.get("text") or n.get("intent") or "")
                break
    lines: List[str] = [
        "Choose one DAG scheduling action. Output only ACTION_INDEX=<integer>.",
        f"Actions: tools 0..{max(0, num_tools - 1)}, local={local_action_index(num_tools)}, pause={pause_action_index(num_tools)}.",
        f"State: task={dp.get('task_id')} node={dp.get('node_id')} type={dp.get('node_type_id')} ready={dp.get('ready_frontier_count')} done={dp.get('completed_node_count')}/{dp.get('num_nodes')}",
        f"Deadlines: task_rem={_fmt(dp.get('task_remaining_deadline_s'))}s node_rem={_fmt(dp.get('node_remaining_deadline_s'))}s node_dl={_fmt(dp.get('node_deadline_s'))}s.",
        f"Queues: tool_run={dp.get('running_tool_jobs_total')} tool_q={dp.get('queued_tool_jobs_total')} local_run={dp.get('running_local_jobs')} local_q={dp.get('queued_local_jobs')}.",
        "allowed_tool_mask=" + ",".join(str(int(x)) for x in allowed_mask),
    ]
    if intent:
        lines.append(f"Intent: {_short_text(intent, 120)}")
    lines.append("Candidates:")
    tool_options = list(dp.get("tool_options", []) or [])
    option_by_idx = {}
    for row in tool_options:
        try:
            option_by_idx[int(row.get("tool_type_id", row.get("slot", -1)))] = row
        except Exception:
            continue
    for i in range(num_tools):
        allowed = int(allowed_mask[i]) if i < len(allowed_mask) else 0
        opt = option_by_idx.get(i)
        if allowed != 1 and opt is None:
            continue
        if opt is None:
            lines.append(f"{i}: tool {action_names[i]} allowed={allowed} unavailable=1")
            continue
        lines.append(
            f"{i}: tool {action_names[i]} allowed={allowed} block={int(bool(opt.get('queue_blocked', False)))} "
            f"t={_fmt(opt.get('predicted_total_s'), 3)} q={_fmt(opt.get('base_queue_delay_s', opt.get('estimated_wait_s', 0.0)), 3)} "
            f"risk={_fmt(opt.get('risk_estimate'), 3)} loss={_fmt(opt.get('packet_loss_rate'), 3)}"
        )
    local = dp.get("local_option", {}) or {}
    lines.append(
        f"{local_action_index(num_tools)}: local block={int(bool(local.get('queue_blocked', False)))} "
        f"t={_fmt(local.get('predicted_total_s'), 3)} fail={_fmt(local.get('fail_probability'), 3)}"
    )
    lines.append(f"{pause_action_index(num_tools)}: pause {PAUSE_DURATION_S:.2f}s; use rarely.")
    if knowledge_prompt:
        lines.append("Knowledge: " + _short_text(str(knowledge_prompt), int(KNOWLEDGE_PROMPT_MAX_CHARS)))
    lines.append("Answer: ACTION_INDEX=")
    return "\n".join(lines) + "\n", action_names

def fastest_predicted_latency(dp: Dict[str, object], default: float = 1.0) -> float:
    vals: List[float] = []
    local = dp.get("local_option", {}) or {}
    if not bool(local.get("queue_blocked", False)):
        vals.append(float(local.get("predicted_total_s", local.get("compute_s", default)) or default))
    for opt in list(dp.get("tool_options", []) or []):
        if not bool(opt.get("queue_blocked", False)):
            vals.append(float(opt.get("predicted_total_s", default) or default))
    return min(vals) if vals else float(default)

def sandbox_utility_for_action(dp: Dict[str, object], action_index: int, num_tools: int) -> float:
    idx = int(action_index)
    rem = float(dp.get("node_remaining_deadline_s", 0.0) or 0.0)
    if idx < num_tools:
        opt = None
        for row in list(dp.get("tool_options", []) or []):
            try:
                if int(row.get("tool_type_id", row.get("slot", -1))) == idx:
                    opt = row
                    break
            except Exception:
                pass
        if opt is None:
            return -100.0
        pred = float(opt.get("predicted_total_s", 0.0) or 0.0)
        queue_blocked = bool(opt.get("queue_blocked", False))
        allowed_mask = list(dp.get("allowed_tools_mask", []) or [])
        allowed = idx < len(allowed_mask) and int(allowed_mask[idx]) == 1
        node_success = 1.0 if allowed and not queue_blocked and pred <= max(1e-6, rem) else 0.0
        violation = 1.0 - node_success
        latency = min(float(UTILITY_LATENCY_CAP), pred / max(1e-6, rem))
        mild_risk = 0.25 * float(opt.get("risk_estimate", 0.0) or 0.0) + 0.25 * float(opt.get("packet_loss_rate", 0.0) or 0.0)
        return UTILITY_W_NODE * node_success - UTILITY_W_VIOL * violation - UTILITY_W_LATENCY * latency - mild_risk
    if idx == local_action_index(num_tools):
        local = dp.get("local_option", {}) or {}
        pred = float(local.get("predicted_total_s", local.get("compute_s", 0.0)) or 0.0)
        fail_prob = float(local.get("fail_probability", 0.0) or 0.0)
        queue_blocked = bool(local.get("queue_blocked", False))
        node_success = 1.0 if (not queue_blocked and pred <= max(1e-6, rem)) else 0.0
        violation = 1.0 - node_success
        latency = min(float(UTILITY_LATENCY_CAP), pred / max(1e-6, rem))
        return UTILITY_W_NODE * node_success - UTILITY_W_VIOL * violation - UTILITY_W_LATENCY * latency - 2.0 * fail_prob
    if idx == pause_action_index(num_tools):
        return -float(PAUSE_UTILITY_PENALTY)
    return -100.0

def heuristic_teacher_action(dp: Dict[str, object], num_tools: int) -> int:
    scores = [(sandbox_utility_for_action(dp, i, num_tools), i) for i in range(action_dim_from_num_tools(num_tools))]
    scores.sort(key=lambda x: (x[0], -x[1]), reverse=True)
    return int(scores[0][1])

def choose_counterfactual_action(dp: Dict[str, object], exec_action: int, num_tools: int, rng: random.Random) -> int:
    actions = list(range(action_dim_from_num_tools(num_tools)))
    actions = [a for a in actions if int(a) != int(exec_action)]
    if not actions:
        return int(exec_action)
    if bool(DPO_RANDOM_COUNTERFACTUAL):
        return int(rng.choice(actions))
    utils = [(sandbox_utility_for_action(dp, a, num_tools), a) for a in actions]
    utils.sort(key=lambda x: x[0], reverse=True)
    return int(utils[0][1])

def realized_utility(node_success: int, mission_success: int, violation: int, actual_latency_s: float, node_remaining_deadline_s: float) -> float:
    lat_norm = min(float(UTILITY_LATENCY_CAP), float(actual_latency_s) / max(1e-6, float(node_remaining_deadline_s)))
    return float(UTILITY_W_MISSION * int(mission_success) + UTILITY_W_NODE * int(node_success) - UTILITY_W_VIOL * int(violation) - UTILITY_W_LATENCY * lat_norm)

# ============================================================
# Minimal LoRA Qwen action policy
# ============================================================

class LoRALinear(nn.Module):
    def __init__(self, base_layer: nn.Linear, r: int, alpha: float, dropout: float) -> None:
        super().__init__()
        self.base_layer = base_layer
        for p in self.base_layer.parameters():
            p.requires_grad = False
        self.in_features = int(base_layer.in_features)
        self.out_features = int(base_layer.out_features)
        self.r = int(max(1, r))
        self.alpha = float(alpha)
        self.scaling = float(alpha) / float(self.r)
        self.lora_dropout = nn.Dropout(float(dropout)) if float(dropout) > 0.0 else nn.Identity()
        device = base_layer.weight.device
        self.lora_A = nn.Parameter(torch.empty(self.r, self.in_features, device=device, dtype=torch.float32))
        self.lora_B = nn.Parameter(torch.empty(self.out_features, self.r, device=device, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
    @property
    def weight(self) -> torch.Tensor:
        return self.base_layer.weight
    @property
    def bias(self) -> Optional[torch.Tensor]:
        return self.base_layer.bias
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base_layer(x)
        lora_in = self.lora_dropout(x).to(dtype=self.lora_A.dtype)
        lora_out = F.linear(F.linear(lora_in, self.lora_A), self.lora_B) * self.scaling
        return base_out + lora_out.to(dtype=base_out.dtype)

def normalize_layer_ids(layer_ids: Sequence[int], num_layers: int) -> List[int]:
    out: List[int] = []
    for raw in layer_ids:
        idx = int(raw)
        if idx < 0:
            idx = int(num_layers) + idx
        if 0 <= idx < int(num_layers) and idx not in out:
            out.append(idx)
    return out

def inject_lora_into_block(block: nn.Module, target_module_names: Sequence[str], r: int, alpha: float, dropout: float) -> List[str]:
    target_set = {str(x) for x in target_module_names}
    replaced: List[str] = []
    for module_path, module in list(block.named_modules()):
        for child_name, child in list(module.named_children()):
            if child_name in target_set and isinstance(child, nn.Linear) and not isinstance(child, LoRALinear):
                setattr(module, child_name, LoRALinear(child, r=int(r), alpha=float(alpha), dropout=float(dropout)))
                replaced.append(f"{module_path}.{child_name}" if module_path else child_name)
    return replaced

class QwenActionPolicy(nn.Module):
    def __init__(self, action_dim: int, cfg: PolicyConfig) -> None:
        super().__init__()
        self.action_dim = int(action_dim)
        self.cfg = cfg
        torch.manual_seed(int(cfg.seed))
        model_path = resolve_llm_path(cfg.model_path)
        tokenizer_path = cfg.tokenizer_path or model_path
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True, use_fast=False)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        self.tokenizer.truncation_side = "left"
        dtype = dtype_from_name(cfg.dtype)
        load_kwargs = {"trust_remote_code": True, "torch_dtype": dtype, "low_cpu_mem_usage": bool(cfg.low_cpu_mem_usage)}
        if torch.cuda.is_available() and bool(cfg.enable_multi_gpu) and str(cfg.device_map).lower() != "none":
            load_kwargs["device_map"] = str(cfg.device_map)
            load_kwargs["max_memory"] = build_max_memory_map(cfg.max_memory_per_gpu)
            load_kwargs["offload_folder"] = str(cfg.offload_folder)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
        if not (torch.cuda.is_available() and bool(cfg.enable_multi_gpu) and str(cfg.device_map).lower() != "none"):
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(device)
        if hasattr(self.model, "config"):
            self.model.config.use_cache = False
        if bool(cfg.gradient_checkpointing) and hasattr(self.model, "gradient_checkpointing_enable"):
            try:
                self.model.gradient_checkpointing_enable()
            except Exception:
                pass
        for p in self.model.parameters():
            p.requires_grad = False
        self._inject_lora_or_unfreeze()
        hidden_size = int(getattr(self.model.config, "hidden_size", getattr(self.model.config, "n_embd", 4096)))
        value_device = last_parameter_device(self.model)
        head_hidden = max(64, min(512, hidden_size // 4))
        self.action_head = nn.Sequential(
            nn.Linear(hidden_size, head_hidden),
            nn.Tanh(),
            nn.Linear(head_hidden, self.action_dim),
        ).to(value_device)
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, head_hidden),
            nn.Tanh(),
            nn.Linear(head_hidden, 1),
        ).to(value_device)
        self._feature_cache: Dict[str, torch.Tensor] = {}
        self._feature_cache_order: List[str] = []
        if bool(LIGHTWEIGHT_HEAD_ONLY_POLICY):
            self.model.eval()
        if cfg.verbose:
            trainable = sum(p.numel() for p in self.trainable_parameters() if p.requires_grad)
            mode = "frozen-qwen-head" if bool(LIGHTWEIGHT_HEAD_ONLY_POLICY) else "completion-logp"
            print(f"[policy] loaded Qwen from {model_path} | action_dim={self.action_dim} | mode={mode} | trainable_params={trainable}", flush=True)

    def _get_transformer_layers(self) -> Optional[nn.ModuleList]:
        base = getattr(self.model, "model", self.model)
        layers = getattr(base, "layers", None)
        if layers is None:
            layers = getattr(base, "h", None)
        return layers

    def _inject_lora_or_unfreeze(self) -> None:
        layers = self._get_transformer_layers()
        if layers is None:
            return
        if bool(self.cfg.use_lora):
            selected = normalize_layer_ids(self.cfg.lora_layer_ids, len(layers))
            replaced: List[str] = []
            for idx in selected:
                replaced.extend(inject_lora_into_block(layers[idx], self.cfg.lora_target_modules, self.cfg.lora_r, self.cfg.lora_alpha, self.cfg.lora_dropout))
            if self.cfg.verbose:
                print(f"[policy] LoRA injected | layers={selected} | modules={len(replaced)}", flush=True)
        else:
            n = int(self.cfg.tune_last_n_blocks)
            if n > 0:
                for block in list(layers)[-n:]:
                    for p in block.parameters():
                        p.requires_grad = True
        if bool(self.cfg.train_final_norm):
            base = getattr(self.model, "model", self.model)
            norm = getattr(base, "norm", None)
            if norm is not None:
                for p in norm.parameters():
                    p.requires_grad = True

    def trainable_parameters(self) -> List[nn.Parameter]:
        return [p for p in self.parameters() if p.requires_grad]

    def action_completion(self, action_index: int, action_names: Optional[Sequence[str]] = None) -> str:
        return f"{LLM_ACTION_OUTPUT_PREFIX}{int(action_index)}"

    def _token_len(self, text: str) -> int:
        return len(self.tokenizer(text, add_special_tokens=False)["input_ids"])

    def _cache_feature(self, prompt: str, rep_cpu: torch.Tensor) -> None:
        max_items = int(LIGHTWEIGHT_FEATURE_CACHE_MAX)
        if max_items <= 0:
            return
        key = str(prompt)
        if key not in self._feature_cache:
            self._feature_cache_order.append(key)
        self._feature_cache[key] = rep_cpu.detach().to("cpu", dtype=torch.float16)
        while len(self._feature_cache_order) > max_items:
            old_key = self._feature_cache_order.pop(0)
            self._feature_cache.pop(old_key, None)

    def encode_prompts(self, prompts: Sequence[str]) -> torch.Tensor:
        prompts = [str(x) for x in prompts]
        if not prompts:
            return torch.zeros((0, int(getattr(self.model.config, "hidden_size", getattr(self.model.config, "n_embd", 4096)))), dtype=torch.float32, device=next(self.action_head.parameters()).device)
        head_device = next(self.action_head.parameters()).device
        reps: List[Optional[torch.Tensor]] = [None] * len(prompts)
        missing: List[Tuple[int, str]] = []
        for i, prompt in enumerate(prompts):
            cached = self._feature_cache.get(prompt)
            if cached is None:
                missing.append((i, prompt))
            else:
                reps[i] = cached
        if missing:
            self.model.eval()
            batch_size = max(1, int(BATCH_DECISION_MAX_SIZE))
            device = first_parameter_device(self.model)
            with torch.inference_mode():
                for start in range(0, len(missing), batch_size):
                    chunk = missing[start:start + batch_size]
                    chunk_prompts = [x[1] for x in chunk]
                    enc = self.tokenizer(chunk_prompts, padding=True, truncation=True, max_length=int(self.cfg.max_length), return_tensors="pt")
                    enc = {k: v.to(device) for k, v in enc.items()}
                    base_model = getattr(self.model, "model", None)
                    if base_model is not None:
                        out = base_model(**enc, use_cache=False)
                        hidden = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
                    else:
                        out = self.model(**enc, output_hidden_states=True, use_cache=False)
                        hidden = out.hidden_states[-1]
                    idx = torch.full((hidden.shape[0],), hidden.shape[1] - 1, dtype=torch.long, device=hidden.device)
                    chunk_reps = hidden[torch.arange(hidden.shape[0], device=hidden.device), idx].detach()
                    for (orig_i, prompt), rep in zip(chunk, chunk_reps):
                        rep_cpu = rep.to("cpu", dtype=torch.float16)
                        self._cache_feature(prompt, rep_cpu)
                        reps[orig_i] = rep_cpu
                    del out, hidden, chunk_reps, enc
        stacked = torch.stack([r for r in reps if r is not None], dim=0).to(device=head_device, dtype=torch.float32)
        return stacked

    def _completion_logps_flat(self, prompts: Sequence[str], actions: Sequence[int], action_names: Optional[Sequence[Sequence[str]]] = None, require_grad: bool = True) -> torch.Tensor:
        prompts = list(prompts)
        actions = [int(a) for a in actions]
        texts: List[str] = []
        comp_lens: List[int] = []
        for i, (prompt, action) in enumerate(zip(prompts, actions)):
            names = list(action_names[i]) if action_names is not None and i < len(action_names) else []
            comp = self.action_completion(action, names)
            texts.append(str(prompt) + comp)
            comp_lens.append(max(1, self._token_len(comp)))
        enc = self.tokenizer(texts, padding=True, truncation=True, max_length=int(self.cfg.max_length), return_tensors="pt")
        device = first_parameter_device(self.model)
        enc = {k: v.to(device) for k, v in enc.items()}
        ctx = torch.enable_grad() if require_grad else torch.no_grad()
        with ctx:
            out = self.model(**enc, use_cache=False)
            logits = out.logits
            logp = F.log_softmax(logits[:, :-1, :].float(), dim=-1)
            input_ids = enc["input_ids"]
            attn = enc["attention_mask"]
            seq_width = input_ids.shape[1]
            vals: List[torch.Tensor] = []
            for row_idx, c_len in enumerate(comp_lens):
                L = int(attn[row_idx].sum().item())
                pad = seq_width - L
                c_len = min(int(c_len), max(1, L - 1))
                start = max(1, L - c_len)
                token_positions = list(range(pad + start, pad + L))
                row_vals = []
                for token_pos in token_positions:
                    logit_pos = token_pos - 1
                    target_id = input_ids[row_idx, token_pos]
                    row_vals.append(logp[row_idx, logit_pos, target_id])
                vals.append(torch.stack(row_vals).mean())
            return torch.stack(vals, dim=0)

    def log_probs_for_actions(self, prompts: Sequence[str], actions: Sequence[int], action_names: Optional[Sequence[Sequence[str]]] = None, batch_size: int = LLM_COMPLETION_SCORE_BATCH_SIZE, require_grad: bool = True) -> torch.Tensor:
        if bool(LIGHTWEIGHT_HEAD_ONLY_POLICY):
            logits = self.action_logp_matrix(prompts, action_names=action_names, require_grad=require_grad)
            if logits.numel() == 0:
                return torch.zeros((0,), dtype=torch.float32, device=next(self.action_head.parameters()).device)
            logp = F.log_softmax(logits.float(), dim=-1)
            action_t = torch.tensor([int(a) for a in actions], dtype=torch.long, device=logp.device)
            return logp.gather(1, action_t.view(-1, 1)).squeeze(1)
        outs: List[torch.Tensor] = []
        for start in range(0, len(prompts), max(1, int(batch_size))):
            sub_prompts = list(prompts)[start:start + int(batch_size)]
            sub_actions = list(actions)[start:start + int(batch_size)]
            sub_names = list(action_names)[start:start + int(batch_size)] if action_names is not None else None
            outs.append(self._completion_logps_flat(sub_prompts, sub_actions, sub_names, require_grad=require_grad))
        if not outs:
            return torch.zeros((0,), dtype=torch.float32, device=last_parameter_device(self.model))
        return torch.cat(outs, dim=0)

    def action_logp_matrix(self, prompts: Sequence[str], action_names: Optional[Sequence[Sequence[str]]] = None, require_grad: bool = True) -> torch.Tensor:
        prompts = list(prompts)
        if not prompts:
            device = next(self.action_head.parameters()).device if bool(LIGHTWEIGHT_HEAD_ONLY_POLICY) else last_parameter_device(self.model)
            return torch.zeros((0, self.action_dim), dtype=torch.float32, device=device)
        if bool(LIGHTWEIGHT_HEAD_ONLY_POLICY):
            reps = self.encode_prompts(prompts)
            return self.action_head(reps.float())
        flat_prompts: List[str] = []
        flat_actions: List[int] = []
        flat_names: List[List[str]] = []
        for i, prompt in enumerate(prompts):
            names = list(action_names[i]) if action_names is not None and i < len(action_names) else []
            for a in range(self.action_dim):
                flat_prompts.append(prompt)
                flat_actions.append(a)
                flat_names.append(names)
        flat = self.log_probs_for_actions(flat_prompts, flat_actions, flat_names, require_grad=require_grad)
        return flat.view(len(prompts), self.action_dim)

    def values_for_prompts(self, prompts: Sequence[str], require_grad: bool = True) -> torch.Tensor:
        prompts = list(prompts)
        if not prompts:
            device = next(self.value_head.parameters()).device if bool(LIGHTWEIGHT_HEAD_ONLY_POLICY) else last_parameter_device(self.model)
            return torch.zeros((0,), dtype=torch.float32, device=device)
        if bool(LIGHTWEIGHT_HEAD_ONLY_POLICY):
            reps = self.encode_prompts(prompts)
            return self.value_head(reps.float()).squeeze(-1)
        enc = self.tokenizer(prompts, padding=True, truncation=True, max_length=int(self.cfg.max_length), return_tensors="pt")
        device = first_parameter_device(self.model)
        enc = {k: v.to(device) for k, v in enc.items()}
        ctx = torch.enable_grad() if require_grad else torch.no_grad()
        with ctx:
            out = self.model(**enc, output_hidden_states=True, use_cache=False)
            hidden = out.hidden_states[-1]
            attn = enc["attention_mask"]
            # With left padding, the last non-pad token is at the final column for every row.
            idx = torch.full((hidden.shape[0],), hidden.shape[1] - 1, dtype=torch.long, device=hidden.device)
            reps = hidden[torch.arange(hidden.shape[0], device=hidden.device), idx]
            reps = reps.to(next(self.value_head.parameters()).device)
            return self.value_head(reps.float()).squeeze(-1)

    @torch.no_grad()
    def select_actions(self, dps: Sequence[Dict[str, object]], num_tools: int, use_memory: bool, knowledge_trainer=None, sample: bool = True, temperature: float = ACTION_TEMPERATURE, graph_metas: Optional[Sequence[Dict[str, object]]] = None) -> List[Tuple[object, ActionInfo]]:
        dps = list(dps)
        graph_metas = list(graph_metas or [{} for _ in dps])
        prompts: List[str] = []
        knowledge_prompts: List[str] = []
        action_names: List[List[str]] = []
        masks: List[List[int]] = []
        for dp, gm in zip(dps, graph_metas):
            ko = query_knowledge_for_dp(dp, num_tools, knowledge_trainer, use_memory)
            prompt, names = build_prompt_from_dp(dp, num_tools, str(ko.get("knowledge_prompt", "") or ""), gm)
            prompts.append(prompt)
            knowledge_prompts.append(str(ko.get("knowledge_prompt", "") or ""))
            action_names.append(list(names))
            masks.append(valid_action_mask_from_dp(dp, num_tools))
        logp_scores = self.action_logp_matrix(prompts, action_names=action_names, require_grad=False)
        if bool(ROLLOUT_COMPUTE_VALUE_HEAD):
            values = self.values_for_prompts(prompts, require_grad=False)
        else:
            values = torch.zeros((len(prompts),), dtype=torch.float32, device=logp_scores.device)
        results: List[Tuple[object, ActionInfo]] = []
        for i, dp in enumerate(dps):
            row = logp_scores[i]
            # The environment remains the final legality gate.  This mirrors prompt-only legality mode.
            probs = torch.softmax(row / max(1e-6, float(temperature)), dim=-1)
            if bool(sample):
                idx = int(torch.multinomial(probs, num_samples=1).item())
            else:
                idx = int(torch.argmax(row).item())
            env_action = action_index_to_env_action(idx, num_tools)
            info = ActionInfo(
                action_index=int(idx),
                action_type=str(env_action.action_type),
                tool_type_id=None if env_action.tool_type_id is None else int(env_action.tool_type_id),
                log_prob=float(torch.log(torch.clamp(probs[idx], min=1e-12)).detach().cpu().item()),
                probability=float(probs[idx].detach().cpu().item()),
                value=float(values[i].detach().cpu().item()),
                valid_action_indices=action_mask_to_indices(masks[i]),
                prompt=prompts[i],
                knowledge_prompt=knowledge_prompts[i],
                action_mask=masks[i],
                action_names=action_names[i],
            )
            results.append((env_action, info))
        return results

    def save_checkpoint(self, path: str, extra: Optional[Dict[str, object]] = None) -> str:
        p = Path(path).expanduser().resolve()
        p.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "cfg": asdict(self.cfg),
            "action_dim": int(self.action_dim),
            "trainable_state_dict": {k: v.detach().cpu() for k, v in self.state_dict().items() if any(x in k for x in ["lora_A", "lora_B", "action_head", "value_head"])},
            "extra": extra or {},
        }
        torch.save(state, p)
        return str(p)

# ============================================================
# Outcome attachment
# ============================================================

def index_records_by_task(task_records: Sequence[Dict[str, object]]) -> Dict[str, Dict[str, object]]:
    return {str(r.get("task_id")): r for r in task_records}

def find_trace_outcome(trace: RolloutTrace, task_record: Dict[str, object]) -> Dict[str, object]:
    events = list(task_record.get("log", []) or [])
    node_events = [ev for ev in events if ev.get("node_id") is not None and int(ev.get("node_id")) == int(trace.node_id)]
    mission_success = 1 if str(task_record.get("status")) == "completed" else 0
    start_t = float(trace.env_time)
    invalid_or_rejected = {
        "decision_tool_invalid_missing_id", "decision_tool_invalid_unknown_tool", "decision_tool_invalid_not_allowed",
        "decision_tool_rejected_queue_full", "decision_local_rejected_queue_full", "decision_invalid_generated_action",
    }
    terminal = {"tool_success", "local_success", "local_failed", "node_timeout", "task_restarted", "task_failed", *invalid_or_rejected}
    candidates = []
    for ev in node_events:
        t = float(ev.get("time", 0.0) or 0.0)
        if t + 1e-12 < start_t:
            continue
        if str(ev.get("event_type")) in terminal:
            candidates.append(ev)
    if not candidates:
        violation = 1 if str(task_record.get("status")) == "failed" else 0
        actual_latency = max(0.0, float(PAUSE_DURATION_S))
        util = realized_utility(0, mission_success, violation, actual_latency, trace.node_remaining_deadline_s)
        return {"realized_utility": util, "reward": util, "node_success": 0, "mission_success": mission_success, "violation": violation, "actual_latency_s": actual_latency, "event_type": "fallback"}
    ev = sorted(candidates, key=lambda x: float(x.get("time", 0.0) or 0.0))[0]
    et = str(ev.get("event_type"))
    node_success = 1 if et in {"tool_success", "local_success"} else 0
    violation = 1 if et in {"local_failed", "node_timeout", "task_restarted", "task_failed"} or et in invalid_or_rejected else 0
    actual_latency = max(0.0, float(ev.get("time", start_t) or start_t) - start_t)
    util = realized_utility(node_success, mission_success, violation, actual_latency, trace.node_remaining_deadline_s)
    return {"realized_utility": util, "reward": util, "node_success": node_success, "mission_success": mission_success, "violation": violation, "actual_latency_s": actual_latency, "event_type": et}

def attach_realized_utilities(traces: Sequence[RolloutTrace], task_records: Sequence[Dict[str, object]]) -> None:
    by_task = index_records_by_task(task_records)
    for tr in traces:
        rec = by_task.get(str(tr.task_id))
        if rec is None:
            continue
        out = find_trace_outcome(tr, rec)
        tr.realized_utility = float(out["realized_utility"])
        tr.reward = float(out["reward"])
        tr.node_success = int(out["node_success"])
        tr.mission_success = int(out["mission_success"])
        tr.violation = int(out["violation"])
        tr.actual_latency_s = float(out["actual_latency_s"])
        tr.outcome_event_type = str(out.get("event_type", ""))

def trace_to_json(trace: RolloutTrace) -> Dict[str, object]:
    return dict(trace.__dict__)

def pair_to_json(pair: PreferenceExample) -> Dict[str, object]:
    d = dict(pair.__dict__)
    names = list(pair.action_names or [])
    d["positive_action_name"] = names[pair.positive_action] if 0 <= int(pair.positive_action) < len(names) else str(pair.positive_action)
    d["negative_action_name"] = names[pair.negative_action] if 0 <= int(pair.negative_action) < len(names) else str(pair.negative_action)
    return d

def sft_to_json(ex: SFTExample) -> Dict[str, object]:
    d = dict(ex.__dict__)
    names = list(ex.action_names or [])
    d["target_action_name"] = names[ex.target_action] if 0 <= int(ex.target_action) < len(names) else str(ex.target_action)
    return d

# ============================================================
# Rollout
# ============================================================

def _select_rollout_actions(policy: QwenActionPolicy, chunk: Sequence[Dict[str, object]], graph_metas: Sequence[Dict[str, object]], num_tools: int, use_memory: bool, knowledge_trainer, sample: bool, rng: random.Random) -> List[Tuple[object, ActionInfo]]:
    if not chunk:
        return []
    if not bool(sample):
        return policy.select_actions(chunk, num_tools=num_tools, use_memory=use_memory, knowledge_trainer=knowledge_trainer, sample=False, temperature=ACTION_TEMPERATURE, graph_metas=graph_metas)
    p_sample = max(0.0, min(1.0, float(TRAIN_SAMPLE_PROBABILITY)))
    flags = [bool(rng.random() < p_sample) for _ in chunk]
    results: List[Optional[Tuple[object, ActionInfo]]] = [None] * len(chunk)
    for flag in (False, True):
        idxs = [i for i, v in enumerate(flags) if bool(v) == bool(flag)]
        if not idxs:
            continue
        sub = [chunk[i] for i in idxs]
        metas = [graph_metas[i] for i in idxs]
        sub_results = policy.select_actions(sub, num_tools=num_tools, use_memory=use_memory, knowledge_trainer=knowledge_trainer, sample=bool(flag), temperature=ACTION_TEMPERATURE, graph_metas=metas)
        for i, r in zip(idxs, sub_results):
            results[i] = r
    return [r for r in results if r is not None]

def run_policy_rollout(dataset: Dict[str, object], graph_meta: Dict[str, Dict[str, object]], policy: QwenActionPolicy, knowledge_trainer, epoch: int, seed: int, use_memory: bool, sample: bool, policy_name: str) -> Tuple[Dict[str, object], List[Dict[str, object]], List[RolloutTrace]]:
    env = build_env(dataset, seed=seed)
    env.reset()
    rng = random.Random(seed + epoch * 17)
    num_tools = len(dataset.get("tool_catalog", []))
    step_limit = compute_env_step_limit(dataset)
    traces: List[RolloutTrace] = []
    steps_taken = 0
    rollout_t0 = time.time()
    last_print_t = rollout_t0
    last_decision_print_t = rollout_t0
    collect_seconds = select_seconds = apply_seconds = env_step_seconds = 0.0
    decision_count = 0
    max_select_seconds = 0.0
    slow_select_count = 0
    while not env.done() and steps_taken < step_limit:
        step_t0 = time.time()
        t0 = time.time()
        dps = env.collect_decision_points()
        collect_seconds += time.time() - t0
        if dps:
            chunk_size = int(BATCH_DECISION_MAX_SIZE) if bool(BATCH_DECISION_ENABLED) and int(BATCH_DECISION_MAX_SIZE) > 0 else (len(dps) if bool(BATCH_DECISION_ENABLED) else 1)
            chunk_size = max(1, min(chunk_size, len(dps)))
            for chunk_start in range(0, len(dps), chunk_size):
                chunk = list(dps[chunk_start:chunk_start + chunk_size])
                metas = [graph_meta.get(str(dp["task_id"]), {}) for dp in chunk]
                t_select = time.time()
                action_info_list = _select_rollout_actions(policy, chunk, metas, num_tools, use_memory, knowledge_trainer, sample, rng)
                select_dt = time.time() - t_select
                batch_n = max(1, len(chunk))
                select_seconds += select_dt
                decision_count += batch_n
                max_select_seconds = max(max_select_seconds, select_dt / batch_n)
                if select_dt >= float(ROLLOUT_SLOW_SELECT_SECONDS):
                    slow_select_count += 1
                    print(f"[{policy_name}-rollout:slow-select] epoch={epoch} step={steps_taken}/{step_limit} batch={chunk_start + 1}-{chunk_start + len(chunk)}/{len(dps)} select={fmt_seconds(select_dt)} counts=({compact_env_counts(env_runtime_counts(env))})", flush=True)
                now = time.time()
                if (ROLLOUT_DECISION_HEARTBEAT_EVERY and decision_count % int(ROLLOUT_DECISION_HEARTBEAT_EVERY) < batch_n) or (now - last_decision_print_t >= float(ROLLOUT_PRINT_EVERY_SECONDS) and len(dps) > 1):
                    last_decision_print_t = now
                    print(f"[{policy_name}-rollout:decision] epoch={epoch} step={steps_taken}/{step_limit} decisions={decision_count} elapsed={fmt_seconds(now - rollout_t0)} avg_select={select_seconds / max(1, decision_count):.3f}s counts=({compact_env_counts(env_runtime_counts(env))})", flush=True)
                for dp, (action, info) in zip(chunk, action_info_list):
                    tr = RolloutTrace(
                        trace_id=str(uuid.uuid4()), epoch=int(epoch), step_count=int(env.step_count), env_time=float(env.time),
                        task_id=str(dp["task_id"]), node_id=int(dp["node_id"]), node_type_id=int(dp.get("node_type_id", 0)),
                        prompt=info.prompt, knowledge_prompt=info.knowledge_prompt, action_mask=list(info.action_mask), exec_action=int(info.action_index),
                        valid_actions=list(info.valid_action_indices), old_log_prob=float(info.log_prob), probability=float(info.probability), value=float(info.value),
                        action_names=list(info.action_names), node_remaining_deadline_s=float(dp.get("node_remaining_deadline_s", 0.0) or 0.0), task_remaining_deadline_s=float(dp.get("task_remaining_deadline_s", 0.0) or 0.0),
                    )
                    traces.append(tr)
                    t_apply = time.time()
                    env.apply_decision(str(dp["task_id"]), int(dp["node_id"]), action)
                    apply_seconds += time.time() - t_apply
        t_env = time.time()
        env.step()
        env_step_seconds += time.time() - t_env
        steps_taken += 1
        now = time.time()
        if (PRINT_EVERY_ENV_STEPS and steps_taken % int(PRINT_EVERY_ENV_STEPS) == 0) or (now - last_print_t >= float(ROLLOUT_PRINT_EVERY_SECONDS)):
            last_print_t = now
            elapsed = now - rollout_t0
            pct = 100.0 * steps_taken / max(1, step_limit)
            bar = text_progress_bar(steps_taken, step_limit, ROLLOUT_PROGRESS_BAR_WIDTH)
            eta = elapsed / max(1, steps_taken) * max(0, step_limit - steps_taken)
            print(f"[{policy_name}-rollout] epoch={epoch} {bar} {steps_taken}/{step_limit} ({pct:.1f}%) dps={len(dps)} traces={len(traces)} elapsed={fmt_seconds(elapsed)} eta~{fmt_seconds(eta)} counts=({compact_env_counts(env_runtime_counts(env))})", flush=True)
    task_records = build_complete_task_records(env)
    attach_realized_utilities(traces, task_records)
    summary = summarize_task_records(task_records, policy_name=policy_name, steps_taken=steps_taken, hit_step_limit=(not env.done()))
    summary.update({"rollout_seconds": float(time.time() - rollout_t0), "collect_seconds": collect_seconds, "select_seconds": select_seconds, "apply_seconds": apply_seconds, "env_step_seconds": env_step_seconds, "decision_count": decision_count, "max_select_seconds": max_select_seconds, "slow_select_count": slow_select_count})
    return summary, task_records, traces

def chunk_dataset_for_rollout(dataset: Dict[str, object], chunk_size: int) -> List[Dict[str, object]]:
    graphs = list(dataset.get("graphs", []) or [])
    chunk_size = int(chunk_size)
    if chunk_size <= 0 or len(graphs) <= chunk_size:
        return [dataset]
    chunks: List[Dict[str, object]] = []
    for start in range(0, len(graphs), chunk_size):
        out = dict(dataset)
        out["graphs"] = graphs[start:start + chunk_size]
        out["config"] = dict(dataset.get("config", {}))
        out["config"]["rollout_chunk_start"] = int(start)
        chunks.append(task_env_mod._normalize_dataset_structure(out, fallback_dataset_name=str(dataset.get("dataset_name", "baseline_chunk"))))
    return chunks

def run_policy_rollout_chunked(dataset: Dict[str, object], graph_meta: Dict[str, Dict[str, object]], policy: QwenActionPolicy, knowledge_trainer, epoch: int, seed: int, use_memory: bool, sample: bool, policy_name: str) -> Tuple[Dict[str, object], List[Dict[str, object]], List[RolloutTrace]]:
    chunks = chunk_dataset_for_rollout(dataset, int(ROLLOUT_CHUNK_GRAPHS))
    if len(chunks) <= 1:
        return run_policy_rollout(dataset, graph_meta, policy, knowledge_trainer, epoch, seed, use_memory, sample, policy_name)
    all_records: List[Dict[str, object]] = []
    all_traces: List[RolloutTrace] = []
    total_steps = 0
    hit_limit = False
    rollout_t0 = time.time()
    for ci, chunk in enumerate(chunks, start=1):
        print(f"[{policy_name}-rollout:chunk] epoch={epoch} chunk={ci}/{len(chunks)} graphs={len(chunk.get('graphs', []))}", flush=True)
        summary, records, traces = run_policy_rollout(chunk, graph_meta, policy, knowledge_trainer, epoch, seed + ci * 1009, use_memory, sample, policy_name)
        all_records.extend(records)
        all_traces.extend(traces)
        total_steps += int(summary.get("steps_taken", 0))
        hit_limit = hit_limit or bool(summary.get("hit_step_limit", False))
    merged = summarize_task_records(all_records, policy_name=policy_name, steps_taken=total_steps, hit_step_limit=hit_limit)
    merged["rollout_seconds"] = float(time.time() - rollout_t0)
    return merged, all_records, all_traces

# ============================================================
# Training objectives
# ============================================================

def build_sft_examples_from_traces(traces: Sequence[RolloutTrace], num_tools: int, max_new: int = SFT_MAX_NEW_EXAMPLES_PER_EPOCH) -> List[SFTExample]:
    out: List[SFTExample] = []
    for tr in traces:
        # Reconstruct a minimal dp is not available here, so use the best action among action names only if teacher was stored in meta.
        # During collection we instead derive SFT labels from stored prompt candidates via simple sandbox is impossible, so use successful rollout action when node succeeded.
        target = int(tr.exec_action) if int(tr.node_success) == 1 else int(local_action_index(num_tools))
        out.append(SFTExample(prompt=tr.prompt, target_action=target, action_names=list(tr.action_names), meta={"trace_id": tr.trace_id, "node_success": tr.node_success, "mission_success": tr.mission_success}))
        if len(out) >= int(max_new):
            break
    return out

def build_sft_examples_from_dps(dps: Sequence[Dict[str, object]], graph_meta: Dict[str, Dict[str, object]], num_tools: int, knowledge_trainer, use_memory: bool, max_new: int) -> List[SFTExample]:
    out: List[SFTExample] = []
    for dp in dps:
        ko = query_knowledge_for_dp(dp, num_tools, knowledge_trainer, use_memory)
        prompt, names = build_prompt_from_dp(dp, num_tools, str(ko.get("knowledge_prompt", "") or ""), graph_meta.get(str(dp.get("task_id")), {}))
        target = heuristic_teacher_action(dp, num_tools)
        out.append(SFTExample(prompt=prompt, target_action=int(target), action_names=list(names), meta={"task_id": str(dp.get("task_id")), "node_id": int(dp.get("node_id", 0))}))
        if len(out) >= int(max_new):
            break
    return out

def collect_teacher_states(dataset: Dict[str, object], graph_meta: Dict[str, Dict[str, object]], knowledge_trainer, seed: int, use_memory: bool, max_examples: int) -> List[SFTExample]:
    env = build_env(dataset, seed=seed)
    env.reset()
    num_tools = len(dataset.get("tool_catalog", []))
    step_limit = compute_env_step_limit(dataset)
    steps = 0
    examples: List[SFTExample] = []
    while not env.done() and steps < step_limit and len(examples) < max_examples:
        dps = env.collect_decision_points()
        if dps:
            examples.extend(build_sft_examples_from_dps(dps, graph_meta, num_tools, knowledge_trainer, use_memory, max_new=max_examples - len(examples)))
            for dp in dps:
                action_idx = heuristic_teacher_action(dp, num_tools)
                env.apply_decision(str(dp["task_id"]), int(dp["node_id"]), action_index_to_env_action(action_idx, num_tools))
        env.step()
        steps += 1
    return examples

def train_sft(policy: QwenActionPolicy, optimizer: torch.optim.Optimizer, examples: Sequence[SFTExample]) -> Dict[str, float]:
    if len(examples) < int(SFT_MIN_EXAMPLES_TO_TRAIN):
        return {"loss": 0.0, "pref_acc": 0.0, "logp_margin": 0.0, "confidence_mean": 0.0, "grad_norm": 0.0, "grad_norm_clipped": 0.0, "update_seconds": 0.0, "tdpo_seconds": 0.0}
    t0 = time.time()
    rows = list(examples)
    if len(rows) > int(SFT_BUFFER_SAMPLE_SIZE):
        rows = random.sample(rows, int(SFT_BUFFER_SAMPLE_SIZE))
    stats: List[Dict[str, float]] = []
    for inner in range(max(1, int(INNER_EPOCHS))):
        random.shuffle(rows)
        for start in range(0, len(rows), max(1, int(BATCH_SIZE))):
            batch = rows[start:start + int(BATCH_SIZE)]
            prompts = [x.prompt for x in batch]
            targets = torch.tensor([int(x.target_action) for x in batch], dtype=torch.long, device=last_parameter_device(policy.model))
            names = [x.action_names for x in batch]
            logits = policy.action_logp_matrix(prompts, action_names=names, require_grad=True)
            loss = F.cross_entropy(logits.float(), targets.to(logits.device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(policy.trainable_parameters(), float(PPO_MAX_GRAD_NORM))
            optimizer.step()
            pred = torch.argmax(logits.detach(), dim=-1)
            acc = (pred == targets.to(pred.device)).float().mean().item()
            stats.append({"loss": float(loss.detach().cpu().item()), "acc": float(acc), "grad": float(grad)})
    return {"loss": safe_mean([s["loss"] for s in stats]), "pref_acc": safe_mean([s["acc"] for s in stats]), "logp_margin": 0.0, "confidence_mean": 0.0, "grad_norm": safe_mean([s["grad"] for s in stats]), "grad_norm_clipped": min(float(PPO_MAX_GRAD_NORM), safe_mean([s["grad"] for s in stats])), "update_seconds": time.time() - t0, "tdpo_seconds": time.time() - t0}

def build_dpo_pairs_from_traces(traces: Sequence[RolloutTrace], num_tools: int, rng: random.Random, max_new: int = DPO_MAX_NEW_PAIRS_PER_EPOCH) -> List[PreferenceExample]:
    pairs: List[PreferenceExample] = []
    for tr in traces:
        cf = rng.choice([a for a in range(action_dim_from_num_tools(num_tools)) if a != int(tr.exec_action)])
        exec_u = float(tr.realized_utility)
        # Basic DPO baseline: compare executed outcome against a simple counterfactual prior.
        cf_u = -float(PAUSE_UTILITY_PENALTY) if cf == pause_action_index(num_tools) else 0.0
        if abs(exec_u - cf_u) < float(DPO_MIN_UTILITY_GAP):
            continue
        if exec_u >= cf_u:
            pos, neg = int(tr.exec_action), int(cf)
        else:
            pos, neg = int(cf), int(tr.exec_action)
        pairs.append(PreferenceExample(prompt=tr.prompt, positive_action=pos, negative_action=neg, reward_gap=abs(exec_u - cf_u), action_names=list(tr.action_names), meta={"trace_id": tr.trace_id, "exec_utility": exec_u, "cf_utility": cf_u}))
        if len(pairs) >= int(max_new):
            break
    return pairs

def fill_reference_logps(policy: QwenActionPolicy, pairs: Sequence[PreferenceExample]) -> None:
    if not pairs:
        return
    prompts = [p.prompt for p in pairs]
    names = [p.action_names for p in pairs]
    pos = [p.positive_action for p in pairs]
    neg = [p.negative_action for p in pairs]
    with torch.no_grad():
        pos_logp = policy.log_probs_for_actions(prompts, pos, names, require_grad=False).detach().cpu().tolist()
        neg_logp = policy.log_probs_for_actions(prompts, neg, names, require_grad=False).detach().cpu().tolist()
    for p, lp, ln in zip(pairs, pos_logp, neg_logp):
        p.ref_positive_logp = float(lp)
        p.ref_negative_logp = float(ln)

def train_dpo(policy: QwenActionPolicy, optimizer: torch.optim.Optimizer, pairs: Sequence[PreferenceExample]) -> Dict[str, float]:
    if len(pairs) < int(DPO_MIN_PAIRS_TO_TRAIN):
        return {"loss": 0.0, "pref_acc": 0.0, "logp_margin": 0.0, "confidence_mean": 0.0, "grad_norm": 0.0, "grad_norm_clipped": 0.0, "update_seconds": 0.0, "tdpo_seconds": 0.0, "pair_gap_mean": 0.0}
    t0 = time.time()
    rows = list(pairs)
    if len(rows) > int(DPO_BUFFER_SAMPLE_SIZE):
        rows = random.sample(rows, int(DPO_BUFFER_SAMPLE_SIZE))
    stats: List[Dict[str, float]] = []
    for inner in range(max(1, int(INNER_EPOCHS))):
        random.shuffle(rows)
        for start in range(0, len(rows), max(1, int(BATCH_SIZE))):
            batch = rows[start:start + int(BATCH_SIZE)]
            prompts = [p.prompt for p in batch]
            names = [p.action_names for p in batch]
            pos = [p.positive_action for p in batch]
            neg = [p.negative_action for p in batch]
            pos_logp = policy.log_probs_for_actions(prompts, pos, names, require_grad=True)
            neg_logp = policy.log_probs_for_actions(prompts, neg, names, require_grad=True)
            ref_pos = torch.tensor([p.ref_positive_logp for p in batch], dtype=torch.float32, device=pos_logp.device)
            ref_neg = torch.tensor([p.ref_negative_logp for p in batch], dtype=torch.float32, device=pos_logp.device)
            margin = (pos_logp - neg_logp) - (ref_pos - ref_neg)
            loss = -F.logsigmoid(float(DPO_BETA) * margin).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(policy.trainable_parameters(), float(PPO_MAX_GRAD_NORM))
            optimizer.step()
            stats.append({"loss": float(loss.detach().cpu().item()), "acc": float((margin.detach() > 0).float().mean().cpu().item()), "margin": float(margin.detach().mean().cpu().item()), "grad": float(grad)})
    return {"loss": safe_mean([s["loss"] for s in stats]), "pref_acc": safe_mean([s["acc"] for s in stats]), "logp_margin": safe_mean([s["margin"] for s in stats]), "confidence_mean": 1.0, "grad_norm": safe_mean([s["grad"] for s in stats]), "grad_norm_clipped": min(float(PPO_MAX_GRAD_NORM), safe_mean([s["grad"] for s in stats])), "update_seconds": time.time() - t0, "tdpo_seconds": time.time() - t0, "pair_gap_mean": safe_mean([p.reward_gap for p in rows])}

def train_ppo(policy: QwenActionPolicy, optimizer: torch.optim.Optimizer, traces: Sequence[RolloutTrace]) -> Dict[str, float]:
    rows = [tr for tr in traces if tr.prompt]
    if not rows:
        return {"loss": 0.0, "pref_acc": 0.0, "logp_margin": 0.0, "confidence_mean": 0.0, "grad_norm": 0.0, "grad_norm_clipped": 0.0, "update_seconds": 0.0, "tdpo_seconds": 0.0}
    t0 = time.time()
    rewards = torch.tensor([float(tr.reward) for tr in rows], dtype=torch.float32)
    if bool(PPO_REWARD_NORMALIZE) and len(rewards) > 1:
        rewards = (rewards - rewards.mean()) / (rewards.std(unbiased=False) + 1e-6)
    for tr, r in zip(rows, rewards.tolist()):
        tr.reward = float(r)
    stats: List[Dict[str, float]] = []
    for inner in range(max(1, int(PPO_INNER_EPOCHS))):
        random.shuffle(rows)
        for start in range(0, len(rows), max(1, int(PPO_BATCH_SIZE))):
            batch = rows[start:start + int(PPO_BATCH_SIZE)]
            prompts = [tr.prompt for tr in batch]
            actions = [tr.exec_action for tr in batch]
            names = [tr.action_names for tr in batch]
            old_logp = torch.tensor([tr.old_log_prob for tr in batch], dtype=torch.float32, device=last_parameter_device(policy.model))
            returns = torch.tensor([tr.reward for tr in batch], dtype=torch.float32, device=old_logp.device)
            values = policy.values_for_prompts(prompts, require_grad=True).to(old_logp.device)
            mat = policy.action_logp_matrix(prompts, action_names=names, require_grad=True).to(old_logp.device)
            dist_logp = F.log_softmax(mat / max(1e-6, float(ACTION_TEMPERATURE)), dim=-1)
            action_t = torch.tensor(actions, dtype=torch.long, device=old_logp.device)
            logp = dist_logp.gather(1, action_t.view(-1, 1)).squeeze(1)
            adv = returns - values.detach()
            if len(adv) > 1:
                adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-6)
            ratio = torch.exp(logp - old_logp)
            unclipped = ratio * adv
            clipped = torch.clamp(ratio, 1.0 - float(PPO_CLIP_RANGE), 1.0 + float(PPO_CLIP_RANGE)) * adv
            policy_loss = -torch.min(unclipped, clipped).mean()
            value_loss = F.mse_loss(values, returns)
            probs = torch.softmax(mat / max(1e-6, float(ACTION_TEMPERATURE)), dim=-1)
            entropy = -(probs * torch.log(torch.clamp(probs, min=1e-12))).sum(dim=-1).mean()
            loss = policy_loss + float(PPO_VALUE_COEF) * value_loss - float(PPO_ENTROPY_COEF) * entropy
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(policy.trainable_parameters(), float(PPO_MAX_GRAD_NORM))
            optimizer.step()
            approx_kl = (old_logp - logp.detach()).mean().cpu().item()
            clip_frac = ((ratio.detach() - 1.0).abs() > float(PPO_CLIP_RANGE)).float().mean().cpu().item()
            stats.append({"loss": float(loss.detach().cpu().item()), "policy_loss": float(policy_loss.detach().cpu().item()), "value_loss": float(value_loss.detach().cpu().item()), "entropy": float(entropy.detach().cpu().item()), "grad": float(grad), "kl": float(approx_kl), "clip_frac": float(clip_frac)})
    return {"loss": safe_mean([s["loss"] for s in stats]), "pref_acc": safe_mean([1.0 - s["clip_frac"] for s in stats]), "logp_margin": -safe_mean([s["kl"] for s in stats]), "confidence_mean": safe_mean([s["entropy"] for s in stats]), "grad_norm": safe_mean([s["grad"] for s in stats]), "grad_norm_clipped": min(float(PPO_MAX_GRAD_NORM), safe_mean([s["grad"] for s in stats])), "update_seconds": time.time() - t0, "tdpo_seconds": time.time() - t0, "policy_loss": safe_mean([s["policy_loss"] for s in stats]), "value_loss": safe_mean([s["value_loss"] for s in stats])}

# ============================================================
# Knowledge load/update
# ============================================================

def enforce_run_memory_limits(knowledge_trainer) -> Dict[str, int]:
    if knowledge_trainer is None:
        return {"num_atoms": 0, "num_prototypes": 0}
    memory = getattr(knowledge_trainer, "memory", None)
    if memory is None:
        return {"num_atoms": 0, "num_prototypes": 0}
    max_atoms = int(RUN_MEMORY_MAX_ATOMS)
    max_protos = int(RUN_MEMORY_MAX_PROTOTYPES)
    if max_atoms > 0 and len(memory.buffer) > max_atoms:
        memory.buffer = memory.buffer[-max_atoms:]
    if max_protos > 0 and len(memory.prototypes) > max_protos:
        memory.prototypes = memory.prototypes[-max_protos:]
    return {"num_atoms": len(memory.buffer), "num_prototypes": len(memory.prototypes)}

def copy_pretrained_knowledge_to_run_dir(run_dir: Path) -> Optional[str]:
    if not bool(COPY_PRETRAINED_KNOWLEDGE_TO_RUN_DIR):
        return None
    if not PRETRAINED_KNOWLEDGE_MODEL_PATH.exists():
        return None
    dst = RunArtifacts(run_dir).pretrained_knowledge_copy
    ensure_dir(dst.parent)
    shutil.copy2(PRETRAINED_KNOWLEDGE_MODEL_PATH, dst)
    return str(dst)

def load_knowledge_trainer(dataset: Dict[str, object], action_dim: int, run_dir: Path):
    if not bool(USE_KNOWLEDGE_MEMORY) and int(KNOWLEDGE_UPDATE_STRATEGY) == 3:
        return None
    if not PRETRAINED_KNOWLEDGE_MODEL_PATH.exists():
        raise FileNotFoundError(f"pretrained knowledge model not found: {PRETRAINED_KNOWLEDGE_MODEL_PATH}")
    copied_path = copy_pretrained_knowledge_to_run_dir(run_dir)
    memory_path = None
    if bool(KNOWLEDGE_USE_EXISTING_MEMORY_JSON) and OPTIONAL_PRETRAINED_KNOWLEDGE_MEMORY_PATH is not None and Path(OPTIONAL_PRETRAINED_KNOWLEDGE_MEMORY_PATH).exists():
        memory_path = str(OPTIONAL_PRETRAINED_KNOWLEDGE_MEMORY_PATH)
    trainer = knowledge_mod.KnowledgeTrainer.load(dataset=dataset, model_path=str(PRETRAINED_KNOWLEDGE_MODEL_PATH), memory_path=memory_path, action_dim=int(action_dim))
    trainer.cfg.batch_size = int(KNOWLEDGE_BATCH_SIZE)
    trainer.cfg.epochs_per_fit = int(KNOWLEDGE_FIT_EPOCHS_PER_UPDATE)
    trainer.cfg.verbose = bool(KNOWLEDGE_VERBOSE)
    trainer.cfg.train_log_mode = str(KNOWLEDGE_TRAIN_LOG_MODE)
    trainer.cfg.log_every_fit_step = int(KNOWLEDGE_PROGRESS_EVERY_BATCHES)
    trainer.cfg.use_llm_verbalizer = bool(KNOWLEDGE_USE_LLM_VERBALIZER)
    trainer.cfg.fast_update_text_encoder = bool(KNOWLEDGE_FAST_UPDATE_TEXT_ENCODER)
    trainer.cfg.freeze_llm_adapters_during_fit = bool(KNOWLEDGE_FREEZE_LLM_ADAPTERS_DURING_UPDATE)
    trainer.cfg.fit_max_segments = int(KNOWLEDGE_FIT_MAX_SEGMENTS_PER_UPDATE)
    trainer.cfg.fit_max_new_atoms = int(KNOWLEDGE_FIT_MAX_NEW_ATOMS_PER_UPDATE)
    trainer.cfg.max_buffer_atoms = int(RUN_MEMORY_MAX_ATOMS)
    trainer.cfg.max_prototypes = int(RUN_MEMORY_MAX_PROTOTYPES)
    if bool(KNOWLEDGE_CLEAR_MEMORY_AFTER_LOAD):
        trainer.memory.buffer = []
        trainer.memory.prototypes = []
    enforce_run_memory_limits(trainer)
    if SAVE_AUXILIARY_JSON:
        save_json(RunArtifacts(run_dir).loaded_knowledge_info, {"pretrained_model_path": str(PRETRAINED_KNOWLEDGE_MODEL_PATH), "pretrained_input_copy_path": copied_path, "loaded_memory_path": memory_path, "action_dim": int(action_dim), "run_memory_max_atoms": int(RUN_MEMORY_MAX_ATOMS), "run_memory_max_prototypes": int(RUN_MEMORY_MAX_PROTOTYPES)})
    return trainer

def release_knowledge_backbone_for_policy(knowledge_trainer) -> Dict[str, object]:
    if knowledge_trainer is None:
        return {"released": False, "reason": "knowledge_trainer_none"}
    info = {"released": False}
    try:
        if hasattr(knowledge_trainer, "llm"):
            knowledge_trainer.llm.model = None
            knowledge_trainer.llm.tokenizer = None
            info["released"] = True
        safe_cuda_empty_cache()
    except Exception as exc:
        info["error"] = repr(exc)
    return info

def knowledge_digest_or_empty(knowledge_trainer) -> Dict[str, object]:
    if knowledge_trainer is None:
        return {"num_atoms": 0, "num_prototypes": 0, "disabled": True}
    return knowledge_trainer.knowledge_digest()

def save_knowledge_artifacts(knowledge_trainer, run_dir: Path, tag: str = "best") -> Dict[str, str]:
    if knowledge_trainer is None:
        return {"disabled": "true"}
    artifacts = RunArtifacts(run_dir)
    return knowledge_trainer.save(str(artifacts.knowledge_model(tag)), str(artifacts.knowledge_memory(tag)))

def _task_records_completion_rate(task_records: Sequence[Dict[str, object]]) -> float:
    records = list(task_records)
    if not records:
        return 0.0
    return float(sum(1 for r in records if str(r.get("status")) == "completed") / max(1, len(records)))

def _filter_conservative_task_records(task_records: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    records = list(task_records)
    if bool(CONSERVATIVE_COMPLETED_ONLY):
        completed = [r for r in records if str(r.get("status")) == "completed"]
        completion_rate = len(completed) / max(1, len(records))
        if completion_rate < float(CONSERVATIVE_MIN_COMPLETION_RATE_TO_UPDATE):
            return []
        records = completed
    if int(CONSERVATIVE_MAX_TASK_RECORDS_PER_UPDATE) > 0:
        records = records[: int(CONSERVATIVE_MAX_TASK_RECORDS_PER_UPDATE)]
    return records

def update_knowledge_from_records(knowledge_trainer, task_records: Sequence[Dict[str, object]], epoch: Optional[int] = None) -> Dict[str, object]:
    if knowledge_trainer is None:
        return {"mode": "disabled", "fit_seconds": 0.0}
    strategy = int(KNOWLEDGE_UPDATE_STRATEGY)
    if strategy == 3:
        return {"mode": "off", "fit_seconds": 0.0}
    if strategy == 4 and epoch is not None and int(epoch) <= int(KNOWLEDGE_DELAYED_START_EPOCHS):
        return {"mode": "delayed_skip", "fit_seconds": 0.0}
    records = list(task_records)
    if strategy in {1, 4}:
        records = _filter_conservative_task_records(records)
    if not records:
        return {"mode": "no_records_after_filter", "fit_seconds": 0.0, "num_segments": 0, "num_new_atoms": 0, "num_new_prototypes": 0}

    t0 = time.time()
    if not bool(KNOWLEDGE_UPDATE_WEIGHTS_DURING_TRAINING):
        # Memory-only update: no optimizer step and no weight movement.  This keeps the
        # pretrained knowledge checkpoint read-only while allowing run-local prototypes
        # to be appended from successful rollout records.
        segments = knowledge_trainer.extractor.extract(records)
        max_segments = int(KNOWLEDGE_FIT_MAX_SEGMENTS_PER_UPDATE or 0)
        if max_segments > 0:
            segments = segments[:max_segments]
        max_atoms = int(CONSERVATIVE_MAX_NEW_ATOMS_PER_UPDATE if strategy in {1, 4} else KNOWLEDGE_FIT_MAX_NEW_ATOMS_PER_UPDATE)
        if max_atoms > 0:
            segments = segments[:max_atoms]
        proto_before = len(knowledge_trainer.memory.prototypes)
        atoms = knowledge_trainer._segments_to_atoms(
            segments,
            progress_prefix=("[knowledge-memory]" if bool(KNOWLEDGE_VERBOSE) else None),
            progress_every_batches=max(1, int(KNOWLEDGE_PROGRESS_EVERY_BATCHES)),
        )
        for atom in atoms:
            knowledge_trainer.memory.add_atom(atom)
        enforce_run_memory_limits(knowledge_trainer)
        proto_after = len(knowledge_trainer.memory.prototypes)
        risk_mean = float(sum(p.no_tool_risk_mean for p in knowledge_trainer.memory.prototypes) / max(1, len(knowledge_trainer.memory.prototypes)))
        knowledge_trainer.last_snapshot["prototype_risk_mean"] = risk_mean
        knowledge_trainer.last_snapshot["prototype_count"] = float(len(knowledge_trainer.memory.prototypes))
        return {
            "mode": "memory_only_conservative" if strategy in {1, 4} else "memory_only_normal",
            "loss_total": 0.0,
            "loss_topo": 0.0,
            "loss_wire": 0.0,
            "loss_tool": 0.0,
            "loss_out": 0.0,
            "loss_need": 0.0,
            "loss_atom": 0.0,
            "num_segments": int(len(segments)),
            "num_new_atoms": int(len(atoms)),
            "num_new_prototypes": int(proto_after - proto_before),
            "prototype_risk_mean": risk_mean,
            "prototype_risk_drift": 0.0,
            "fit_seconds": float(time.time() - t0),
            "total_updates": 0,
        }

    stats = knowledge_trainer.fit_on_task_records(records, batch_size=int(KNOWLEDGE_BATCH_SIZE), epochs=int(KNOWLEDGE_FIT_EPOCHS_PER_UPDATE))
    stats["mode"] = "conservative" if strategy in {1, 4} else "normal"
    enforce_run_memory_limits(knowledge_trainer)
    return stats

# ============================================================
# Main training loop
# ============================================================

def _row_from_epoch(epoch: int, rollout_summary: Dict[str, object], traces: Sequence[RolloutTrace], train_stats: Dict[str, float], counts: Dict[str, int], times: Dict[str, float], digest: Dict[str, object], is_best: bool) -> Dict[str, object]:
    return {
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
        "num_pairs_added": int(counts.get("num_pairs_added", 0)),
        "num_pairs_buffer": int(counts.get("num_pairs_buffer", 0)),
        "num_success_buffer": int(counts.get("num_success_buffer", 0)),
        "num_explore_buffer": int(counts.get("num_explore_buffer", 0)),
        "num_pairs_train": int(counts.get("num_pairs_train", 0)),
        "num_tier_a_pairs": int(counts.get("num_tier_a_pairs", 0)),
        "num_tier_b_pairs": int(counts.get("num_tier_b_pairs", 0)),
        "num_tier_a_raw": int(counts.get("num_tier_a_raw", 0)),
        "num_tier_b_raw": int(counts.get("num_tier_b_raw", 0)),
        "tdpo_loss": float(train_stats.get("loss", 0.0)),
        "tdpo_pref_acc": float(train_stats.get("pref_acc", 0.0)),
        "tdpo_margin": float(train_stats.get("logp_margin", 0.0)),
        "tdpo_confidence": float(train_stats.get("confidence_mean", 0.0)),
        "tdpo_grad_norm": float(train_stats.get("grad_norm", 0.0)),
        "tdpo_grad_norm_clipped": float(train_stats.get("grad_norm_clipped", 0.0)),
        "tdpo_completion_pair_ratio": float(train_stats.get("completion_pair_ratio", 0.0)),
        "tdpo_completion_advantage_ratio": float(train_stats.get("completion_advantage_ratio", 0.0)),
        "tdpo_node_success_advantage_ratio": float(train_stats.get("node_success_advantage_ratio", 0.0)),
        "tdpo_pair_gap_mean": float(train_stats.get("pair_gap_mean", 0.0)),
        "rollout_seconds": float(rollout_summary.get("rollout_seconds", 0.0)),
        "pair_seconds": float(times.get("pair_seconds", 0.0)),
        "new_ref_seconds": float(times.get("new_ref_seconds", 0.0)),
        "tdpo_seconds": float(train_stats.get("tdpo_seconds", train_stats.get("update_seconds", 0.0))),
        "tdpo_ref_seconds": float(times.get("ref_seconds", 0.0)),
        "tdpo_update_seconds": float(train_stats.get("update_seconds", 0.0)),
        "knowledge_update_seconds": float(times.get("knowledge_update_seconds", 0.0)),
        "epoch_seconds": float(times.get("epoch_seconds", 0.0)),
        "knowledge_num_atoms": int(digest.get("num_atoms", 0)),
        "knowledge_num_prototypes": int(digest.get("num_prototypes", 0)),
        "is_best": bool(is_best),
    }

def _eval_after_epoch(dataset, graph_meta, policy, knowledge_trainer, epoch: int, policy_name: str, eval_rows: List[Dict[str, object]], energy_rows: List[Dict[str, object]], delay_rows: List[Dict[str, object]], run_dir: Path) -> Optional[Dict[str, object]]:
    if not bool(RUN_GREEDY_EVAL_EACH_EPOCH):
        return None
    eval_t0 = time.time()
    eval_seed = int(SEED + EVAL_SEED_OFFSET) if bool(EVAL_USE_FIXED_SEED) else int(SEED + EVAL_SEED_OFFSET + epoch)
    if bool(EVAL_PRINT_PROGRESS):
        print(f"[{policy_name}-eval] epoch={epoch} greedy_eval_start seed={eval_seed}", flush=True)
    eval_summary, eval_records, eval_traces = run_policy_rollout_chunked(dataset, graph_meta, policy, knowledge_trainer, epoch, eval_seed, bool(USE_KNOWLEDGE_MEMORY), False, policy_name)
    row = {
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
        "eval_seconds": float(time.time() - eval_t0),
        "eval_seed": int(eval_seed),
        "use_memory": bool(USE_KNOWLEDGE_MEMORY),
    }
    eval_rows.append(row)
    artifacts = RunArtifacts(run_dir)
    write_eval_csv(artifacts.eval_metrics, eval_rows)
    e_rows, d_rows = build_energy_delay_detail_rows(epoch, "eval", eval_records)
    energy_rows.extend(e_rows)
    delay_rows.extend(d_rows)
    write_energy_csv(artifacts.energy_metrics, energy_rows)
    write_delay_csv(artifacts.delay_metrics, delay_rows)
    print(f"[{policy_name}-eval] epoch={epoch} comp={row['completion_rate']:.4f} lat={row['avg_latency_s']:.4f} time={fmt_seconds(row['eval_seconds'])}", flush=True)
    return row

def run_training(algorithm: str) -> None:
    algorithm = str(algorithm).lower().strip()
    if algorithm not in {"ppo", "sft", "dpo"}:
        raise ValueError("algorithm must be one of: ppo, sft, dpo")
    random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    print_cuda_layout()
    policy_name = algorithm.upper()
    run_name = f"qwen7b_knowledge_{algorithm}_baseline"
    run_dir = OUTPUT_ROOT / f"{run_name}_{now_tag()}"
    artifacts = RunArtifacts(run_dir)
    for subdirectory in artifacts.subdirectories:
        ensure_dir(subdirectory)
    dataset, scan_info = load_dataset_from_dir(DATA_DIR, seed=SEED)
    tool_library = load_dynamic_tool_library()
    coverage = validate_dynamic_tool_library_coverage(dataset, tool_library)
    if SAVE_AUXILIARY_JSON:
        save_json(artifacts.dataset_scan, scan_info)
    num_tools = len(dataset.get("tool_catalog", []))
    action_dim = action_dim_from_num_tools(num_tools)
    graph_meta = build_graph_meta_map(dataset)
    save_json(artifacts.run_config, {
        "algorithm": algorithm, "DATA_DIR": str(DATA_DIR), "SAMPLE_NUM_GRAPHS": int(SAMPLE_NUM_GRAPHS), "ROLLOUT_CHUNK_GRAPHS": int(ROLLOUT_CHUNK_GRAPHS),
        "SAMPLE_IN_ORDER": bool(SAMPLE_IN_ORDER), "SAMPLE_RANDOM_POOL_LIMIT": int(SAMPLE_RANDOM_POOL_LIMIT), "dataset_scan": scan_info,
        "USE_DYNAMIC_TOOL_LIBRARY": bool(USE_DYNAMIC_TOOL_LIBRARY), "USE_MULTIPLE_TOOL_LIBRARIES": bool(USE_MULTIPLE_TOOL_LIBRARIES), "TOOL_LIBRARY_PATHS": [str(p) for p in _selected_tool_library_paths()], "ALLOW_TOOL_LIBRARY_FALLBACK": bool(ALLOW_TOOL_LIBRARY_FALLBACK), "tool_library_coverage": coverage,
        "ENV_RELAXED_RESOURCE_MODE": bool(ENV_RELAXED_RESOURCE_MODE), "ENV_Q_MAX": int(ENV_Q_MAX), "ENV_TOOL_INSTANCE_MULTIPLIER": float(ENV_TOOL_INSTANCE_MULTIPLIER), "ENV_TOOL_INSTANCE_ADDITIVE": int(ENV_TOOL_INSTANCE_ADDITIVE), "ENV_MIN_TOOL_INSTANCES": int(ENV_MIN_TOOL_INSTANCES), "ENV_ARRIVAL_GAP_RANGE": list(ENV_ARRIVAL_GAP_RANGE), "ENV_NODE_DEADLINE_MULTIPLIER": float(ENV_NODE_DEADLINE_MULTIPLIER), "ENV_TASK_DEADLINE_MULTIPLIER": float(ENV_TASK_DEADLINE_MULTIPLIER), "ENV_LOCAL_MAX_CONCURRENCY": int(ENV_LOCAL_MAX_CONCURRENCY), "ENV_LOCAL_QUEUE_CAPACITY": int(ENV_LOCAL_QUEUE_CAPACITY),
        "QWEN_MODEL_PATH": QWEN_MODEL_PATH, "PRETRAINED_KNOWLEDGE_MODEL_PATH": str(PRETRAINED_KNOWLEDGE_MODEL_PATH), "USE_KNOWLEDGE_MEMORY": bool(USE_KNOWLEDGE_MEMORY),
        "GPU_PROFILE": GPU_PROFILE, "CUDA_VISIBLE_DEVICES_DEFAULT": CUDA_VISIBLE_DEVICES_DEFAULT, "LLM_ENABLE_MULTI_GPU": bool(LLM_ENABLE_MULTI_GPU), "LLM_DEVICE_MAP": LLM_DEVICE_MAP, "LLM_MAX_MEMORY_PER_GPU": LLM_MAX_MEMORY_PER_GPU,
        "NUM_EPOCHS": int(NUM_EPOCHS), "BATCH_SIZE": int(BATCH_SIZE), "LEARNING_RATE": float(LEARNING_RATE), "ACTION_TEMPERATURE": float(ACTION_TEMPERATURE), "TRAIN_WITH_SAMPLING": bool(TRAIN_WITH_SAMPLING), "TRAIN_SAMPLE_PROBABILITY": float(TRAIN_SAMPLE_PROBABILITY), "FAST_COMPACT_PROMPT": bool(FAST_COMPACT_PROMPT), "KNOWLEDGE_PROMPT_MAX_CHARS": int(KNOWLEDGE_PROMPT_MAX_CHARS), "ROLLOUT_COMPUTE_VALUE_HEAD": bool(ROLLOUT_COMPUTE_VALUE_HEAD), "LIGHTWEIGHT_HEAD_ONLY_POLICY": bool(LIGHTWEIGHT_HEAD_ONLY_POLICY), "LIGHTWEIGHT_FEATURE_CACHE_MAX": int(LIGHTWEIGHT_FEATURE_CACHE_MAX), "LLM_USE_LORA": bool(LLM_USE_LORA), "LLM_MAX_LENGTH": int(LLM_MAX_LENGTH),
        "DPO_BETA": float(DPO_BETA), "PPO_CLIP_RANGE": float(PPO_CLIP_RANGE), "SFT_MAX_EXAMPLE_BUFFER": int(SFT_MAX_EXAMPLE_BUFFER),
        "KNOWLEDGE_UPDATE_STRATEGY": int(KNOWLEDGE_UPDATE_STRATEGY), "KNOWLEDGE_UPDATE_EVERY_EPOCHS": int(KNOWLEDGE_UPDATE_EVERY_EPOCHS), "KNOWLEDGE_FAST_UPDATE_TEXT_ENCODER": bool(KNOWLEDGE_FAST_UPDATE_TEXT_ENCODER),
    })
    knowledge_rows: List[Dict[str, object]] = []
    knowledge_trainer = load_knowledge_trainer(dataset, action_dim, run_dir) if bool(USE_KNOWLEDGE_MEMORY) or int(KNOWLEDGE_UPDATE_STRATEGY) in {1, 2, 4} else None
    if knowledge_trainer is not None and bool(RELEASE_KNOWLEDGE_BACKBONE_BEFORE_POLICY):
        release_info = release_knowledge_backbone_for_policy(knowledge_trainer)
        if SAVE_AUXILIARY_JSON:
            save_json(artifacts.knowledge_release, release_info)
        print(f"[knowledge-release] {release_info}", flush=True)
    safe_cuda_empty_cache()
    policy_cfg = PolicyConfig(
        model_path=resolve_llm_path(QWEN_MODEL_PATH, QWEN_MODEL_PATH_FALLBACK), tokenizer_path=None, dtype=LLM_DTYPE, max_length=int(LLM_MAX_LENGTH),
        use_lora=bool(LLM_USE_LORA), lora_layer_ids=tuple(LLM_LORA_LAYER_IDS), lora_target_modules=tuple(LLM_LORA_TARGET_MODULES), lora_r=int(LLM_LORA_R), lora_alpha=float(LLM_LORA_ALPHA), lora_dropout=float(LLM_LORA_DROPOUT),
        tune_last_n_blocks=int(LLM_TUNE_LAST_N_BLOCKS), train_final_norm=bool(LLM_TRAIN_FINAL_NORM), gradient_checkpointing=bool(LLM_GRADIENT_CHECKPOINTING), enable_multi_gpu=bool(LLM_ENABLE_MULTI_GPU), device_map=str(LLM_DEVICE_MAP), max_memory_per_gpu=str(LLM_MAX_MEMORY_PER_GPU), low_cpu_mem_usage=bool(LLM_LOW_CPU_MEM_USAGE), offload_folder=str(artifacts.policy_offload), seed=int(SEED), verbose=True,
    )
    policy = QwenActionPolicy(action_dim=action_dim, cfg=policy_cfg)
    optimizer = torch.optim.AdamW(policy.trainable_parameters(), lr=float(LEARNING_RATE), weight_decay=float(WEIGHT_DECAY))
    dpo_buffer: List[PreferenceExample] = []
    sft_buffer: List[SFTExample] = []
    epoch_rows: List[Dict[str, object]] = []
    eval_rows: List[Dict[str, object]] = []
    energy_rows: List[Dict[str, object]] = []
    delay_rows: List[Dict[str, object]] = []
    best_score = -1.0e18
    best_ckpt = ""
    for epoch in range(1, int(NUM_EPOCHS) + 1):
        epoch_t0 = time.time()
        print(f"[{algorithm}] epoch {epoch}/{NUM_EPOCHS} start", flush=True)
        rollout_summary, task_records, traces = run_policy_rollout_chunked(dataset, graph_meta, policy, knowledge_trainer, epoch, SEED + epoch, bool(USE_KNOWLEDGE_MEMORY), bool(TRAIN_WITH_SAMPLING), policy_name)
        print(f"[{algorithm}-stage] epoch={epoch} rollout_done time={fmt_seconds(float(rollout_summary.get('rollout_seconds', 0.0)))} traces={len(traces)}", flush=True)
        pair_t0 = time.time()
        new_ref_seconds = 0.0
        train_stats: Dict[str, float] = {}
        counts: Dict[str, int] = {}
        new_pairs: List[PreferenceExample] = []
        new_sft: List[SFTExample] = []
        if algorithm == "ppo":
            train_stats = train_ppo(policy, optimizer, traces)
            counts = {"num_pairs_added": len(traces), "num_pairs_buffer": len(traces), "num_pairs_train": len(traces), "num_tier_a_pairs": len(traces), "num_tier_a_raw": len(traces)}
        elif algorithm == "dpo":
            rng = random.Random(SEED + epoch * 10007)
            new_pairs = build_dpo_pairs_from_traces(traces, num_tools, rng, max_new=int(DPO_MAX_NEW_PAIRS_PER_EPOCH))
            if new_pairs:
                ref_t0 = time.time()
                fill_reference_logps(policy, new_pairs)
                new_ref_seconds = time.time() - ref_t0
            dpo_buffer.extend(new_pairs)
            if len(dpo_buffer) > int(DPO_MAX_PAIR_BUFFER):
                dpo_buffer = dpo_buffer[-int(DPO_MAX_PAIR_BUFFER):]
            train_stats = train_dpo(policy, optimizer, dpo_buffer)
            counts = {"num_pairs_added": len(new_pairs), "num_pairs_buffer": len(dpo_buffer), "num_pairs_train": min(len(dpo_buffer), int(DPO_BUFFER_SAMPLE_SIZE)), "num_tier_a_pairs": len(new_pairs), "num_tier_a_raw": len(new_pairs)}
        elif algorithm == "sft":
            # Use a basic heuristic teacher under the same env and knowledge prompt format.
            new_sft = collect_teacher_states(dataset, graph_meta, knowledge_trainer, seed=SEED + epoch * 31, use_memory=bool(USE_KNOWLEDGE_MEMORY), max_examples=int(SFT_MAX_NEW_EXAMPLES_PER_EPOCH))
            sft_buffer.extend(new_sft)
            if len(sft_buffer) > int(SFT_MAX_EXAMPLE_BUFFER):
                sft_buffer = sft_buffer[-int(SFT_MAX_EXAMPLE_BUFFER):]
            train_stats = train_sft(policy, optimizer, sft_buffer)
            counts = {"num_pairs_added": len(new_sft), "num_pairs_buffer": len(sft_buffer), "num_pairs_train": min(len(sft_buffer), int(SFT_BUFFER_SAMPLE_SIZE)), "num_tier_a_pairs": len(new_sft), "num_tier_a_raw": len(new_sft)}
        pair_seconds = time.time() - pair_t0
        safe_cuda_empty_cache()
        eval_row = _eval_after_epoch(dataset, graph_meta, policy, knowledge_trainer, epoch, policy_name, eval_rows, energy_rows, delay_rows, run_dir)
        knowledge_stats = {}
        if knowledge_trainer is not None and int(KNOWLEDGE_UPDATE_STRATEGY) in {1, 2, 4} and epoch % max(1, int(KNOWLEDGE_UPDATE_EVERY_EPOCHS)) == 0:
            print(f"[knowledge-update] epoch={epoch} strategy={KNOWLEDGE_UPDATE_STRATEGY} records={len(task_records)}", flush=True)
            knowledge_stats = update_knowledge_from_records(knowledge_trainer, task_records, epoch=epoch)
            if SAVE_AUXILIARY_JSON:
                save_json(artifacts.knowledge_update(epoch), knowledge_stats)
            save_knowledge_artifacts(knowledge_trainer, run_dir, tag="best")
            knowledge_rows.append(build_knowledge_row("epoch_update", epoch, knowledge_stats, knowledge_trainer.knowledge_digest()))
            write_knowledge_csv(artifacts.knowledge_metrics, knowledge_rows)
        digest = knowledge_digest_or_empty(knowledge_trainer)
        score_source = eval_row if eval_row is not None else rollout_summary
        score = float(score_source.get("completion_rate", 0.0)) * 1000.0 - float(score_source.get("avg_latency_s", 0.0))
        is_best = False
        if bool(SAVE_EVERY_EPOCH):
            policy.save_checkpoint(str(artifacts.policy_checkpoint(algorithm, epoch)), extra={"epoch": epoch, "summary": rollout_summary, "train_stats": train_stats, "knowledge_digest": digest})
        if score > best_score:
            is_best = True
            best_score = score
            best_ckpt = policy.save_checkpoint(str(artifacts.best_checkpoint), extra={"epoch": epoch, "score": score, "summary": rollout_summary, "train_stats": train_stats, "knowledge_digest": digest})
        if SAVE_TASK_RECORDS_EVERY_EPOCH:
            save_json(artifacts.task_records(epoch), task_records)
        if SAVE_TRACE_JSONL:
            append_jsonl(artifacts.traces, [trace_to_json(t) for t in traces])
            if algorithm == "dpo":
                append_jsonl(artifacts.dpo_pairs, [pair_to_json(p) for p in new_pairs])
            if algorithm == "sft":
                append_jsonl(artifacts.sft_examples, [sft_to_json(x) for x in new_sft])
        if WRITE_PROMPT_SAMPLES and epoch == 1:
            save_json(artifacts.prompt_samples, [{"trace_id": t.trace_id, "task_id": t.task_id, "node_id": t.node_id, "exec_action": t.exec_action, "prompt": t.prompt, "knowledge_prompt": t.knowledge_prompt, "action_names": t.action_names} for t in traces[: int(MAX_PROMPT_SAMPLES)]])
        times = {"pair_seconds": pair_seconds, "new_ref_seconds": new_ref_seconds, "ref_seconds": new_ref_seconds, "knowledge_update_seconds": float(knowledge_stats.get("fit_seconds", knowledge_stats.get("update_seconds", 0.0)) if knowledge_stats else 0.0), "epoch_seconds": time.time() - epoch_t0}
        row = _row_from_epoch(epoch, rollout_summary, traces, train_stats, counts, times, digest, is_best)
        epoch_rows.append(row)
        write_epoch_csv(artifacts.epoch_metrics, epoch_rows)
        e_rows, d_rows = build_energy_delay_detail_rows(epoch, "rollout", task_records)
        energy_rows.extend(e_rows)
        delay_rows.extend(d_rows)
        write_energy_csv(artifacts.energy_metrics, energy_rows)
        write_delay_csv(artifacts.delay_metrics, delay_rows)
        if SAVE_LATEST_SUMMARY_JSON:
            save_json(artifacts.latest_summary, {"epoch": epoch, "row": row, "rollout_summary": rollout_summary, "train_stats": train_stats, "knowledge_stats": knowledge_stats, "knowledge_digest": digest, "best_ckpt": best_ckpt})
        print(f"[{algorithm}] epoch={epoch} comp={row['completion_rate']:.4f} lat={row['avg_latency_s']:.4f} samples+={row['num_pairs_added']} buffer={row['num_pairs_buffer']} loss={row['tdpo_loss']:.4f} score_acc={row['tdpo_pref_acc']:.4f} grad={row['tdpo_grad_norm']:.3e}/{row['tdpo_grad_norm_clipped']:.3e} rollout={fmt_seconds(row['rollout_seconds'])} update={fmt_seconds(row['tdpo_update_seconds'])} knowledge_proto={row['knowledge_num_prototypes']}", flush=True)
    final_knowledge = save_knowledge_artifacts(knowledge_trainer, run_dir, tag="final") if knowledge_trainer is not None else {"disabled": "true"}
    if SAVE_FINAL_RESULT_JSON:
        save_json(artifacts.final_result, {"algorithm": algorithm, "run_dir": str(run_dir.resolve()), "best_ckpt": best_ckpt, "final_knowledge": final_knowledge, "num_epochs": int(NUM_EPOCHS), "best_score": float(best_score), "eval_csv": str(artifacts.eval_metrics.resolve()) if bool(RUN_GREEDY_EVAL_EACH_EPOCH) else ""})
    print(f"[{algorithm}] done | run_dir={run_dir.resolve()}", flush=True)
    print(f"[{algorithm}] best_ckpt={best_ckpt}", flush=True)
