# -*- coding: utf-8 -*-
"""
eval_tdpo_no_skill_with_knowledge_env_sweep.py

Purpose
-------
Use an already-trained NO-SKILL TDPO/Qwen policy checkpoint to evaluate multiple
knowledge-trained environment cases.  The knowledge best.pt/memory.json paths are
used to identify the corresponding case and to report memory statistics, while the
default evaluation path keeps the policy no-skill by not injecting memory into the
decision prompt.

For each knowledge case and each selected TDPO checkpoint epoch, this script runs
NO TDPO update and NO knowledge update. It only evaluates the fixed policy with
the specified knowledge best.pt + memory.json and writes files compatible with the
original train_TDPO_no_skill.py output format:

  - epoch_metrics.csv
  - epoch.csv                 # duplicate of epoch_metrics.csv for convenience
  - energy.csv
  - delay.csv
  - case_config.json

Important
---------
This script loads knowledge in a memory-only way only for case bookkeeping and
optional retrieval diagnostics. It reads the knowledge cfg from best.pt and reads
memory.json into PrototypeMemory, but it does NOT instantiate the knowledge Qwen
backbone.  With EVAL_USE_KNOWLEDGE_MEMORY_IN_POLICY=False, the TDPO policy remains
a strict no-skill/no-memory baseline during eval.

How to use
----------
1) Put this file in the same directory as:
      train_TDPO_no_skill.py
      decision_TDPO_ye_no_skill.py
      knowledge_aspect_adapter.py
      task_env.py
2) Fill TDPO_CHECKPOINTS or TDPO_CKPT_DIR + EVAL_POLICY_EPOCHS.
3) Fill KNOWLEDGE_CASES directly, or fill KNOWLEDGE_SWEEP_ROOT and the two env lists.
4) Run:
      python3 eval_tdpo_no_skill_with_knowledge_env_sweep.py
"""

from __future__ import annotations

from experiment_suite.config.paths import (
    EVAL_OVERRIDE_QWEN_MODEL_PATH as OVERRIDE_QWEN_MODEL_PATH,
    EVAL_OVERRIDE_QWEN_TOKENIZER_PATH as OVERRIDE_QWEN_TOKENIZER_PATH,
    KNOWLEDGE_CASE_DIR_TEMPLATE,
    SAME_SKILL_KNOWLEDGE_BEST_PATH,
    SAME_SKILL_KNOWLEDGE_MEMORY_PATH,
)

from experiment_suite.config.paths import (
    TDPO_NO_SKILL_EVAL_TIME_OUTPUT_ROOT as OUTPUT_ROOT,
    TDPO_NO_SKILL_EVAL_TIME_DATA_DIR as DATA_DIR,
    TDPO_NO_SKILL_EVAL_TIME_TDPO_CHECKPOINTS as TDPO_CHECKPOINTS,
    TDPO_NO_SKILL_EVAL_TIME_TDPO_CKPT_DIR as TDPO_CKPT_DIR,
    TDPO_NO_SKILL_EVAL_TIME_TDPO_CKPT_TEMPLATE as TDPO_CKPT_TEMPLATE,
    TDPO_NO_SKILL_EVAL_TIME_TDPO_BEST_CKPT_NAME as TDPO_BEST_CKPT_NAME,
    TDPO_NO_SKILL_EVAL_TIME_KNOWLEDGE_CASES as KNOWLEDGE_CASES,
    TDPO_NO_SKILL_EVAL_TIME_KNOWLEDGE_SWEEP_ROOT as KNOWLEDGE_SWEEP_ROOT,
    TDPO_NO_SKILL_EVAL_TIME_KNOWLEDGE_BEST_FILENAME as KNOWLEDGE_BEST_FILENAME,
    TDPO_NO_SKILL_EVAL_TIME_KNOWLEDGE_MEMORY_FILENAME as KNOWLEDGE_MEMORY_FILENAME,
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
import gc
import importlib
import json
import math
import os
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

# ============================================================
# User-editable globals: module names
# ============================================================


TRAIN_MODULE_NAME = "experiment_suite.tdpo_no_skill.trainer"
DECISION_MODULE_NAME = "experiment_suite.tdpo_no_skill.decision"
KNOWLEDGE_MODULE_NAME = "experiment_suite.shared.knowledge_adapter"

# ============================================================
# User-editable globals: output and dataset
# ============================================================

RUN_NAME = "tdpo_no_skill_fixed_policy_eval_knowledge_sweep"

SAMPLE_NUM_GRAPHS = 20
SAMPLE_IN_ORDER = False
SAMPLE_RANDOM_POOL_LIMIT = 1500
SEED = 20260425

# Keep the eval dataset selection exactly consistent with your TDPO run unless
# you intentionally want a different test set.
MAX_TRAIN_GRAPHS = SAMPLE_NUM_GRAPHS
TDPO_ROLLOUT_CHUNK_GRAPHS = 20

# ============================================================
# User-editable globals: trained TDPO/Qwen checkpoints
# ============================================================

# Option A: fill explicit policy checkpoints. The epoch value is written to
# epoch_metrics.csv as the "epoch" column.

# Option B: leave TDPO_CHECKPOINTS empty and fill these.
EVAL_POLICY_EPOCHS: List[Union[int, str]] = []  # e.g., [20, 40, 60] or ["best"]

# If non-empty, override the model_path stored inside the TDPO checkpoint cfg.
# Useful when the checkpoint was trained on another machine.

# ============================================================
# User-editable globals: knowledge checkpoints and matching env
# ============================================================

# Option A: fill explicit knowledge cases. This is the safest mode.
# arrival_gap can be a scalar v -> (v, v), or a tuple/list (min, max).

# Option B: leave KNOWLEDGE_CASES empty and use a sweep root plus two lists.
# The script assumes each case directory contains best.pt and memory.json.
ENV_ARRIVAL_GAP_SWEEP_LIST: List[Union[float, Tuple[float, float]]] = [1.0]
ENV_TASK_DEADLINE_MULTIPLIER_SWEEP_LIST: List[float] = [2.6]

# ============================================================
# User-editable globals: eval behavior
# ============================================================

# Keep False for a strict no-skill baseline.  The script still reads each
# knowledge best.pt/memory.json so the case paths and memory statistics are recorded,
# but the no-skill policy receives no memory/skill signal during action selection.
EVAL_USE_KNOWLEDGE_MEMORY_IN_POLICY = False
EVAL_GREEDY = True
EVAL_USE_FIXED_SEED = True
EVAL_SEED_OFFSET = 100000
DETAIL_STAGE_NAME = "rollout"  # keep "rollout" for downstream scripts that filter rollout rows

# If True, write one combined CSV across all knowledge cases and TDPO epochs.
WRITE_COMBINED_SUMMARY = True

# GPU / loading overrides. Leave empty to use the values from train_fast_mem.py or
# the values stored in the policy checkpoint.
CUDA_VISIBLE_DEVICES_DEFAULT = ""
LLM_DEVICE = "auto"
LLM_DTYPE = ""
LLM_MAX_LENGTH = 0
LLM_ENABLE_MULTI_GPU: Optional[bool] = None
LLM_DEVICE_MAP = ""
LLM_MAX_MEMORY_PER_GPU = ""

# ============================================================
# Helpers
# ============================================================


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def now_tag() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def save_json(path: Path, data: object) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def safe_cuda_empty_cache() -> None:
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass


def _format_sweep_value(value: float) -> str:
    text = f"{float(value):.6g}"
    return text.replace("-", "m").replace(".", "p")


def _as_arrival_gap(value: object) -> Tuple[float, float]:
    if isinstance(value, (list, tuple)):
        vals = list(value)
        if len(vals) == 0:
            raise ValueError("arrival_gap range cannot be empty")
        if len(vals) == 1:
            v = float(vals[0])
            return (v, v)
        a, b = float(vals[0]), float(vals[1])
        return (min(a, b), max(a, b))
    v = float(value)
    return (v, v)


def _arrival_tag(arrival_gap: Tuple[float, float]) -> str:
    a, b = float(arrival_gap[0]), float(arrival_gap[1])
    if abs(a - b) < 1e-12:
        return _format_sweep_value(a)
    return f"{_format_sweep_value(a)}to{_format_sweep_value(b)}"


def _clean_name(text: str) -> str:
    s = str(text).strip().replace("\\", "/").split("/")[-1]
    keep = []
    for ch in s:
        if ch.isalnum() or ch in {"-", "_", "."}:
            keep.append(ch)
        else:
            keep.append("_")
    out = "".join(keep).strip("_")
    return out or "case"


def apply_train_module_overrides(train_mod) -> None:
    train_mod.DATA_DIR = Path(DATA_DIR)
    train_mod.SAMPLE_NUM_GRAPHS = int(SAMPLE_NUM_GRAPHS)
    train_mod.MAX_TRAIN_GRAPHS = int(MAX_TRAIN_GRAPHS)
    train_mod.SAMPLE_IN_ORDER = bool(SAMPLE_IN_ORDER)
    train_mod.SAMPLE_RANDOM_POOL_LIMIT = int(SAMPLE_RANDOM_POOL_LIMIT)
    train_mod.SEED = int(SEED)
    train_mod.TDPO_ROLLOUT_CHUNK_GRAPHS = int(TDPO_ROLLOUT_CHUNK_GRAPHS)
    train_mod.USE_KNOWLEDGE_MEMORY = bool(EVAL_USE_KNOWLEDGE_MEMORY_IN_POLICY)
    train_mod.TRAIN_WITH_SAMPLING = False
    # Eval-only: never update TDPO or knowledge/memory.
    if hasattr(train_mod, "KNOWLEDGE_UPDATE_STRATEGY"):
        train_mod.KNOWLEDGE_UPDATE_STRATEGY = 3
    if hasattr(train_mod, "KNOWLEDGE_UPDATE_MEMORY_DURING_TDPO"):
        train_mod.KNOWLEDGE_UPDATE_MEMORY_DURING_TDPO = False
    if hasattr(train_mod, "KNOWLEDGE_UPDATE_WEIGHTS_DURING_TDPO"):
        train_mod.KNOWLEDGE_UPDATE_WEIGHTS_DURING_TDPO = False
    if LLM_DEVICE:
        train_mod.LLM_DEVICE = str(LLM_DEVICE)
    if LLM_DTYPE:
        train_mod.LLM_DTYPE = str(LLM_DTYPE)
    if int(LLM_MAX_LENGTH or 0) > 0:
        train_mod.LLM_MAX_LENGTH = int(LLM_MAX_LENGTH)
    if LLM_ENABLE_MULTI_GPU is not None:
        train_mod.LLM_ENABLE_MULTI_GPU = bool(LLM_ENABLE_MULTI_GPU)
    if LLM_DEVICE_MAP:
        train_mod.LLM_DEVICE_MAP = str(LLM_DEVICE_MAP)
    if LLM_MAX_MEMORY_PER_GPU:
        train_mod.LLM_MAX_MEMORY_PER_GPU = str(LLM_MAX_MEMORY_PER_GPU)


def apply_env_case(train_mod, arrival_gap: Tuple[float, float], task_deadline_multiplier: float) -> None:
    train_mod.ENV_ARRIVAL_GAP_RANGE = (float(arrival_gap[0]), float(arrival_gap[1]))
    train_mod.ENV_TASK_DEADLINE_MULTIPLIER = float(task_deadline_multiplier)


def resolve_policy_checkpoints() -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    for item in TDPO_CHECKPOINTS:
        path = Path(str(item.get("path", ""))).expanduser()
        if not str(path).strip():
            continue
        epoch = item.get("epoch", None)
        out.append({"epoch": epoch, "path": path})
    if out:
        return out

    if not str(TDPO_CKPT_DIR).strip():
        raise ValueError("Fill TDPO_CHECKPOINTS or set TDPO_CKPT_DIR + EVAL_POLICY_EPOCHS.")
    root = Path(TDPO_CKPT_DIR).expanduser()
    for raw_epoch in EVAL_POLICY_EPOCHS:
        if str(raw_epoch).lower() == "best":
            out.append({"epoch": "best", "path": root / TDPO_BEST_CKPT_NAME})
        else:
            epoch = int(raw_epoch)
            out.append({"epoch": epoch, "path": root / TDPO_CKPT_TEMPLATE.format(epoch=epoch)})
    if not out:
        raise ValueError("EVAL_POLICY_EPOCHS is empty. Fill at least one epoch or explicit TDPO_CHECKPOINTS.")
    return out


def resolve_knowledge_cases() -> List[Dict[str, object]]:
    cases: List[Dict[str, object]] = []
    if KNOWLEDGE_CASES:
        for idx, item in enumerate(KNOWLEDGE_CASES, start=1):
            arrival_gap = _as_arrival_gap(item.get("arrival_gap", 1.0))
            deadline = float(item.get("task_deadline_multiplier", 2.6))
            name = str(item.get("case_name", "") or f"case_{idx:03d}_arrival_{_arrival_tag(arrival_gap)}_deadline_{_format_sweep_value(deadline)}")
            cases.append({
                "case_idx": int(idx),
                "case_name": _clean_name(name),
                "knowledge_best_path": Path(str(item.get("knowledge_best_path", ""))).expanduser(),
                "knowledge_memory_path": Path(str(item.get("knowledge_memory_path", ""))).expanduser(),
                "arrival_gap": arrival_gap,
                "task_deadline_multiplier": deadline,
            })
        return cases

    if not str(KNOWLEDGE_SWEEP_ROOT).strip():
        raise ValueError("Fill KNOWLEDGE_CASES or set KNOWLEDGE_SWEEP_ROOT with env sweep lists.")
    root = Path(KNOWLEDGE_SWEEP_ROOT).expanduser()
    case_idx = 0
    for deadline in [float(x) for x in ENV_TASK_DEADLINE_MULTIPLIER_SWEEP_LIST]:
        for raw_gap in ENV_ARRIVAL_GAP_SWEEP_LIST:
            case_idx += 1
            gap = _as_arrival_gap(raw_gap)
            dirname = KNOWLEDGE_CASE_DIR_TEMPLATE.format(
                case_idx=case_idx,
                arrival_tag=_arrival_tag(gap),
                deadline_tag=_format_sweep_value(deadline),
            )
            cases.append({
                "case_idx": int(case_idx),
                "case_name": _clean_name(dirname),
                "knowledge_best_path": root / dirname / KNOWLEDGE_BEST_FILENAME,
                "knowledge_memory_path": root / dirname / KNOWLEDGE_MEMORY_FILENAME,
                "arrival_gap": gap,
                "task_deadline_multiplier": float(deadline),
            })
    return cases


class MemoryOnlyKnowledge:
    """A lightweight knowledge object compatible with decision_TDPO_ye retrieval.

    It provides .memory, .external_memory, .query_memory(), and .knowledge_digest()
    but never loads the knowledge-side Qwen backbone.
    """

    def __init__(self, knowledge_mod, cfg, action_dim: int, memory_path: Optional[Path]) -> None:
        self.cfg = cfg
        self.action_dim = int(action_dim)
        self.memory = knowledge_mod.PrototypeMemory(action_dim=int(action_dim), cfg=cfg)
        self.external_memory = None
        self.last_snapshot = {"prototype_risk_mean": 0.0, "prototype_count": 0.0}
        if memory_path is not None and Path(memory_path).exists():
            data = json.loads(Path(memory_path).read_text(encoding="utf-8"))
            self.memory.load_json(data)
        if self.memory.prototypes:
            self.last_snapshot["prototype_risk_mean"] = float(
                sum(float(p.no_tool_risk_mean) for p in self.memory.prototypes) / max(1, len(self.memory.prototypes))
            )
            self.last_snapshot["prototype_count"] = float(len(self.memory.prototypes))

    def query_memory(self, state_query_embedding: List[float], node_type_id: int, queue_class: int, slack_class: int) -> Dict[str, object]:
        retrieved = self.memory.query(
            query_embedding=state_query_embedding,
            node_type_id=int(node_type_id),
            queue_class=int(queue_class),
            slack_class=int(slack_class),
            k=int(self.cfg.k_retrieve),
        )
        if not retrieved:
            return {
                "memory_vector": [0.0] * (self.action_dim + 4),
                "action_bias": [0.0] * self.action_dim,
                "selected": [],
                "knowledge_prompt": "",
            }
        total_weight = sum(max(1.0e-6, float(score)) for _, score in retrieved)
        risk = 0.0
        conf = 0.0
        support = 0.0
        action_bias = [0.0] * self.action_dim
        selected = []
        prompt_lines = []
        for proto, score in retrieved:
            w = max(1.0e-6, float(score)) / max(1.0e-6, total_weight)
            risk += w * float(proto.no_tool_risk_mean)
            conf += w * float(proto.confidence_mean)
            support += w * float(proto.support)
            means = list(proto.action_gain_means or [])
            for i in range(self.action_dim):
                if i < len(means):
                    action_bias[i] += w * float(means[i])
            selected.append({
                "prototype_id": proto.prototype_id,
                "score": float(score),
                "support": int(proto.support),
                "node_type_id": int(proto.node_type_id),
                "queue_class": int(proto.queue_class),
                "slack_class": int(proto.slack_class),
            })
            if proto.prompt_text:
                prompt_lines.append(str(proto.prompt_text))
        memory_vector = [risk, conf, support / 10.0, float(len(retrieved)) / max(1.0, float(self.cfg.k_retrieve))] + list(action_bias)
        return {
            "memory_vector": memory_vector,
            "action_bias": action_bias,
            "selected": selected,
            "knowledge_prompt": "\n".join(prompt_lines[: int(self.cfg.k_retrieve)]),
            "query_embedding_dim": len(state_query_embedding),
            "proto_embedding_dim": self.memory.embedding_dim(),
        }

    def knowledge_digest(self) -> Dict[str, object]:
        top_prompts = [p.prompt_text for p in self.memory.prototypes[:5] if getattr(p, "prompt_text", "")]
        return {
            "num_atoms": len(self.memory.buffer),
            "num_prototypes": len(self.memory.prototypes),
            "prototype_risk_mean": float(self.last_snapshot.get("prototype_risk_mean", 0.0)),
            "sample_prompts": top_prompts,
            "embedding_dim": self.memory.embedding_dim(),
            "memory_only": True,
        }


def load_memory_only_knowledge(knowledge_mod, case: Dict[str, object], action_dim: int) -> MemoryOnlyKnowledge:
    import torch
    best_path = Path(case["knowledge_best_path"]).expanduser()
    memory_path = Path(case["knowledge_memory_path"]).expanduser()
    if not best_path.exists():
        raise FileNotFoundError(f"knowledge best.pt does not exist: {best_path}")
    if not memory_path.exists():
        raise FileNotFoundError(f"knowledge memory.json does not exist: {memory_path}")
    ckpt = torch.load(str(best_path), map_location="cpu")
    cfg = knowledge_mod.KnowledgeConfig(**ckpt.get("cfg", {}))
    cfg.verbose = False
    cfg.use_llm_verbalizer = False
    cfg.verbalize_on_update = False
    cfg.k_retrieve = int(getattr(cfg, "k_retrieve", 3))
    ckpt_action_dim = int(ckpt.get("action_dim", action_dim))
    if ckpt_action_dim != int(action_dim):
        print(
            f"[knowledge-load] WARNING: checkpoint action_dim={ckpt_action_dim}, current action_dim={action_dim}. "
            "Using current action_dim for eval.",
            flush=True,
        )
    obj = MemoryOnlyKnowledge(knowledge_mod=knowledge_mod, cfg=cfg, action_dim=int(action_dim), memory_path=memory_path)
    print(
        f"[knowledge-load] memory-only case={case['case_name']} atoms={len(obj.memory.buffer)} "
        f"prototypes={len(obj.memory.prototypes)} best={best_path} memory={memory_path}",
        flush=True,
    )
    return obj


def build_policy_from_checkpoint(decision_mod, policy_item: Dict[str, object], action_dim: int, run_dir: Path):
    import torch
    ckpt_path = Path(policy_item["path"]).expanduser()
    if not ckpt_path.exists():
        raise FileNotFoundError(f"TDPO checkpoint does not exist: {ckpt_path}")
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    ckpt_action_dim = int(ckpt.get("action_dim", action_dim))
    if ckpt_action_dim != int(action_dim):
        raise RuntimeError(f"TDPO checkpoint action_dim={ckpt_action_dim}, but current dataset action_dim={action_dim}")
    cfg_dict = dict(ckpt.get("cfg", {}) or {})
    if OVERRIDE_QWEN_MODEL_PATH:
        cfg_dict["model_path"] = str(OVERRIDE_QWEN_MODEL_PATH)
    if OVERRIDE_QWEN_TOKENIZER_PATH:
        cfg_dict["tokenizer_path"] = str(OVERRIDE_QWEN_TOKENIZER_PATH)
    if LLM_DEVICE:
        cfg_dict["device"] = str(LLM_DEVICE)
    if LLM_DTYPE:
        cfg_dict["dtype"] = str(LLM_DTYPE)
    if int(LLM_MAX_LENGTH or 0) > 0:
        cfg_dict["max_length"] = int(LLM_MAX_LENGTH)
    if LLM_ENABLE_MULTI_GPU is not None:
        cfg_dict["enable_multi_gpu"] = bool(LLM_ENABLE_MULTI_GPU)
    if LLM_DEVICE_MAP:
        cfg_dict["device_map"] = str(LLM_DEVICE_MAP)
    if LLM_MAX_MEMORY_PER_GPU:
        cfg_dict["max_memory_per_gpu"] = str(LLM_MAX_MEMORY_PER_GPU)
    cfg_dict["offload_folder"] = str(run_dir / TDPO_OFFLOAD_DIR)
    cfg_dict["verbose"] = True
    allowed = set(decision_mod.TDPOPolicyConfig.__dataclass_fields__.keys())
    cfg_dict = {k: v for k, v in cfg_dict.items() if k in allowed}
    cfg = decision_mod.TDPOPolicyConfig(**cfg_dict)
    policy = decision_mod.TDPOPolicy(action_dim=int(action_dim), cfg=cfg)
    load_info = policy.load_trainable_checkpoint(str(ckpt_path), strict=False)
    policy.eval()
    print(
        f"[policy-load] checkpoint={ckpt_path} loaded={len(load_info.get('loaded', []))} "
        f"missing={len(load_info.get('missing', []))}",
        flush=True,
    )
    return policy, ckpt


def policy_epoch_value(policy_item: Dict[str, object], ckpt: Dict[str, object]) -> int:
    raw_epoch = policy_item.get("epoch", None)
    if isinstance(raw_epoch, int):
        return int(raw_epoch)
    if raw_epoch is not None and str(raw_epoch).lower() != "best":
        try:
            return int(raw_epoch)
        except Exception:
            pass
    extra = ckpt.get("extra", {}) if isinstance(ckpt, dict) else {}
    try:
        return int(extra.get("epoch", 0))
    except Exception:
        return 0


def build_epoch_row(train_mod, epoch: int, summary: Dict[str, object], traces: Sequence[object], digest: Dict[str, object], rollout_seconds: float, is_best: bool) -> Dict[str, object]:
    return {
        "epoch": int(epoch),
        "completion_rate": float(summary.get("completion_rate", 0.0)),
        "avg_latency_s": float(summary.get("avg_latency_s", 0.0)),
        "energy_proxy": float(summary.get("energy_proxy", 0.0)),
        "completed_total_energy_proxy": float(summary.get("completed_total_energy_proxy", 0.0)),
        "avg_completed_energy_proxy_per_graph": float(summary.get("avg_completed_energy_proxy_per_graph", 0.0)),
        "completed_total_delay_s": float(summary.get("completed_total_delay_s", 0.0)),
        "avg_completed_delay_s_per_graph": float(summary.get("avg_completed_delay_s_per_graph", 0.0)),
        "num_completed_nodes_logged": int(summary.get("num_completed_nodes_logged", 0)),
        "num_completed": int(summary.get("num_completed", 0)),
        "num_failed": int(summary.get("num_failed", 0)),
        "num_unfinished": int(summary.get("num_unfinished", 0)),
        "steps_taken": int(summary.get("steps_taken", 0)),
        "hit_step_limit": bool(summary.get("hit_step_limit", False)),
        "num_traces": int(len(traces)),
        "num_pairs_added": 0,
        "num_pairs_buffer": 0,
        "num_success_buffer": 0,
        "num_explore_buffer": 0,
        "num_pairs_train": 0,
        "num_tier_a_pairs": 0,
        "num_tier_b_pairs": 0,
        "num_tier_a_raw": 0,
        "num_tier_b_raw": 0,
        "tdpo_loss": 0.0,
        "tdpo_pref_acc": 0.0,
        "tdpo_margin": 0.0,
        "tdpo_confidence": 0.0,
        "tdpo_grad_norm": 0.0,
        "tdpo_grad_norm_clipped": 0.0,
        "tdpo_completion_pair_ratio": 0.0,
        "tdpo_completion_advantage_ratio": 0.0,
        "tdpo_node_success_advantage_ratio": 0.0,
        "tdpo_pair_gap_mean": 0.0,
        "rollout_seconds": float(summary.get("rollout_seconds", rollout_seconds)),
        "pair_seconds": 0.0,
        "new_ref_seconds": 0.0,
        "tdpo_seconds": 0.0,
        "tdpo_ref_seconds": 0.0,
        "tdpo_update_seconds": 0.0,
        "knowledge_update_seconds": 0.0,
        "epoch_seconds": float(rollout_seconds),
        "knowledge_num_atoms": int(digest.get("num_atoms", 0)),
        "knowledge_num_prototypes": int(digest.get("num_prototypes", 0)),
        "is_best": bool(is_best),
    }


def write_combined_summary(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    ensure_dir(path.parent)
    fieldnames = [
        "case_name", "policy_epoch", "policy_checkpoint", "eval_use_knowledge_memory_in_policy", "knowledge_best_path", "knowledge_memory_path",
        "arrival_gap_min_s", "arrival_gap_max_s", "task_deadline_multiplier",
        "completion_rate", "avg_latency_s", "energy_proxy", "avg_completed_delay_s_per_graph",
        "avg_completed_energy_proxy_per_graph", "num_completed_nodes_logged", "num_completed", "num_failed",
        "num_unfinished", "steps_taken", "hit_step_limit", "num_traces", "knowledge_num_atoms",
        "knowledge_num_prototypes", "run_dir",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k) for k in fieldnames})


def main() -> None:
    if CUDA_VISIBLE_DEVICES_DEFAULT and "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(CUDA_VISIBLE_DEVICES_DEFAULT)

    train_mod = importlib.import_module(TRAIN_MODULE_NAME)
    decision_mod = importlib.import_module(DECISION_MODULE_NAME)
    knowledge_mod = importlib.import_module(KNOWLEDGE_MODULE_NAME)
    apply_train_module_overrides(train_mod)

    random.seed(int(SEED))
    try:
        import torch
        torch.manual_seed(int(SEED))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(SEED))
    except Exception:
        pass

    run_root = OUTPUT_ROOT / f"{RUN_NAME}_{now_tag()}"
    ensure_dir(run_root)

    dataset, scan_info = train_mod.load_dataset_from_dir(Path(DATA_DIR), seed=int(SEED))
    dataset = train_mod.subset_dataset(dataset, int(MAX_TRAIN_GRAPHS))
    graph_meta = train_mod.build_graph_meta_map(dataset)
    tool_library = train_mod.load_dynamic_tool_library()
    train_mod.validate_dynamic_tool_library_coverage(dataset, tool_library)

    num_tools = len(dataset.get("tool_catalog", []))
    action_dim = int(num_tools + 2)
    policy_items = resolve_policy_checkpoints()
    knowledge_cases = resolve_knowledge_cases()

    save_json(run_root / EVAL_CONFIG_FILE, {
        "data_dir": str(DATA_DIR),
        "scan_info": scan_info,
        "num_graphs": len(dataset.get("graphs", [])),
        "action_dim": int(action_dim),
        "decision_module": str(DECISION_MODULE_NAME),
        "train_module": str(TRAIN_MODULE_NAME),
        "eval_use_knowledge_memory_in_policy": bool(EVAL_USE_KNOWLEDGE_MEMORY_IN_POLICY),
        "policy_checkpoints": [{"epoch": str(x.get("epoch")), "path": str(x.get("path"))} for x in policy_items],
        "knowledge_cases": [
            {
                "case_name": c["case_name"],
                "knowledge_best_path": str(c["knowledge_best_path"]),
                "knowledge_memory_path": str(c["knowledge_memory_path"]),
                "arrival_gap": list(c["arrival_gap"]),
                "task_deadline_multiplier": c["task_deadline_multiplier"],
            }
            for c in knowledge_cases
        ],
    })

    print(
        f"[eval] start | policies={len(policy_items)} knowledge_cases={len(knowledge_cases)} "
        f"graphs={len(dataset.get('graphs', []))} action_dim={action_dim} root={run_root.resolve()}",
        flush=True,
    )

    case_rows: Dict[str, List[Dict[str, object]]] = {str(c["case_name"]): [] for c in knowledge_cases}
    case_energy_rows: Dict[str, List[Dict[str, object]]] = {str(c["case_name"]): [] for c in knowledge_cases}
    case_delay_rows: Dict[str, List[Dict[str, object]]] = {str(c["case_name"]): [] for c in knowledge_cases}
    case_best_scores: Dict[str, float] = {str(c["case_name"]): -1.0e18 for c in knowledge_cases}
    combined_rows: List[Dict[str, object]] = []

    for policy_idx, policy_item in enumerate(policy_items, start=1):
        for case_idx, case in enumerate(knowledge_cases, start=1):
            case_name = str(case["case_name"])
            case_dir = run_root / case_name
            ensure_dir(case_dir)
            arrival_gap = tuple(case["arrival_gap"])
            deadline_mul = float(case["task_deadline_multiplier"])
            apply_env_case(train_mod, arrival_gap, deadline_mul)

            save_json(case_dir / CASE_CONFIG_FILE, {
                "case_name": case_name,
                "arrival_gap": list(arrival_gap),
                "task_deadline_multiplier": deadline_mul,
                "eval_use_knowledge_memory_in_policy": bool(EVAL_USE_KNOWLEDGE_MEMORY_IN_POLICY),
                "knowledge_best_path": str(case["knowledge_best_path"]),
                "knowledge_memory_path": str(case["knowledge_memory_path"]),
                "detail_stage_name": DETAIL_STAGE_NAME,
            })

            print(
                f"[eval] policy {policy_idx}/{len(policy_items)} case {case_idx}/{len(knowledge_cases)} | "
                f"case={case_name} arrival_gap={arrival_gap} deadline_mul={deadline_mul}",
                flush=True,
            )

            knowledge = load_memory_only_knowledge(knowledge_mod, case=case, action_dim=action_dim)
            policy, ckpt = build_policy_from_checkpoint(decision_mod, policy_item=policy_item, action_dim=action_dim, run_dir=case_dir)
            epoch_value = policy_epoch_value(policy_item, ckpt)
            eval_seed = int(SEED + EVAL_SEED_OFFSET + max(0, int(epoch_value))) if bool(EVAL_USE_FIXED_SEED) else int(SEED + time.time())

            t0 = time.time()
            with __import__("torch").no_grad():
                summary, task_records, traces = train_mod.run_tdpo_rollout_chunked(
                    dataset=dataset,
                    graph_meta=graph_meta,
                    policy=policy,
                    knowledge_trainer=knowledge,
                    epoch=int(epoch_value),
                    seed=int(eval_seed),
                    use_memory=bool(EVAL_USE_KNOWLEDGE_MEMORY_IN_POLICY),
                    sample=not bool(EVAL_GREEDY),
                )
            eval_seconds = time.time() - t0
            digest = knowledge.knowledge_digest()
            score = float(summary.get("completion_rate", 0.0)) * 1000.0 - float(summary.get("avg_latency_s", 0.0))
            is_best = bool(score > case_best_scores[case_name])
            if is_best:
                case_best_scores[case_name] = float(score)
            row = build_epoch_row(
                train_mod=train_mod,
                epoch=int(epoch_value),
                summary=summary,
                traces=traces,
                digest=digest,
                rollout_seconds=eval_seconds,
                is_best=is_best,
            )
            case_rows[case_name].append(row)
            train_mod.write_epoch_csv(case_dir / EPOCH_METRICS_FILE, case_rows[case_name])
            train_mod.write_epoch_csv(case_dir / EPOCH_ALIAS_FILE, case_rows[case_name])

            energy_rows, delay_rows = train_mod.build_energy_delay_detail_rows(int(epoch_value), str(DETAIL_STAGE_NAME), task_records)
            case_energy_rows[case_name].extend(energy_rows)
            case_delay_rows[case_name].extend(delay_rows)
            train_mod.write_energy_csv(case_dir / ENERGY_FILE, case_energy_rows[case_name])
            train_mod.write_delay_csv(case_dir / DELAY_FILE, case_delay_rows[case_name])

            combined = {
                "case_name": case_name,
                "policy_epoch": int(epoch_value),
                "policy_checkpoint": str(Path(policy_item["path"]).expanduser()),
                "eval_use_knowledge_memory_in_policy": bool(EVAL_USE_KNOWLEDGE_MEMORY_IN_POLICY),
                "knowledge_best_path": str(case["knowledge_best_path"]),
                "knowledge_memory_path": str(case["knowledge_memory_path"]),
                "arrival_gap_min_s": float(arrival_gap[0]),
                "arrival_gap_max_s": float(arrival_gap[1]),
                "task_deadline_multiplier": float(deadline_mul),
                "completion_rate": row["completion_rate"],
                "avg_latency_s": row["avg_latency_s"],
                "energy_proxy": row["energy_proxy"],
                "avg_completed_delay_s_per_graph": row["avg_completed_delay_s_per_graph"],
                "avg_completed_energy_proxy_per_graph": row["avg_completed_energy_proxy_per_graph"],
                "num_completed_nodes_logged": row["num_completed_nodes_logged"],
                "num_completed": row["num_completed"],
                "num_failed": row["num_failed"],
                "num_unfinished": row["num_unfinished"],
                "steps_taken": row["steps_taken"],
                "hit_step_limit": row["hit_step_limit"],
                "num_traces": row["num_traces"],
                "knowledge_num_atoms": row["knowledge_num_atoms"],
                "knowledge_num_prototypes": row["knowledge_num_prototypes"],
                "run_dir": str(case_dir.resolve()),
            }
            combined_rows.append(combined)
            if bool(WRITE_COMBINED_SUMMARY):
                write_combined_summary(run_root / COMBINED_SUMMARY_FILE, combined_rows)

            print(
                f"[eval] done | case={case_name} epoch={epoch_value} "
                f"comp={row['completion_rate']:.4f} lat={row['avg_latency_s']:.4f} "
                f"energy={row['energy_proxy']:.4f} nodes={row['num_completed_nodes_logged']} "
                f"time={eval_seconds:.1f}s out={case_dir.resolve()}",
                flush=True,
            )

            del policy, ckpt, knowledge, task_records, traces
            safe_cuda_empty_cache()

    if bool(WRITE_COMBINED_SUMMARY):
        write_combined_summary(run_root / COMBINED_SUMMARY_FILE, combined_rows)
    print(f"[eval] all done | root={run_root.resolve()}", flush=True)


if __name__ == "__main__":
    main()
