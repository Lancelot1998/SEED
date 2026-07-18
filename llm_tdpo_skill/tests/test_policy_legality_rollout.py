# -*- coding: utf-8 -*-
"""
test_policy_legality_rollout.py

Standalone diagnostic test for LLM direct-action TDPO policy.

Purpose:
  1) Load the current train / decision / task_env / knowledge files.
  2) Run a short rollout with the policy, without TDPO update.
  3) Check whether generated actions are parseable, in-range, in candidate table,
     legal by policy mask, legal by allowed-tool indicator, and accepted by env.
  4) Export detailed JSON / JSONL diagnostics for manual inspection.

Typical usage:
  python test_policy_legality_rollout.py \
    --train-file ./train_fast_mem.py \
    --decision-file ./llm_tdpo/policy/tdpo.py \
    --task-env-file ./llm_tdpo/environment/task_env.py \
    --knowledge-file ./llm_tdpo/knowledge/adapter.py \
    --qwen-model-path /models/Qwen \
    --sample-num-graphs 20 \
    --max-decisions 200 \
    --use-memory 1 \
    --sample 0 \
    --save-dir ./policy_legality_test_runs

Optional trained checkpoint:
  python test_policy_legality_rollout.py ... \
    --checkpoint ./tdpo_decision_runs_batch_fast_skill/.../checkpoints/best.pt
"""

from __future__ import annotations

import argparse
import collections
import importlib.util
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from llm_tdpo.config.paths import (
    CURRENT_POLICY_ENTRY_FILE,
    KNOWLEDGE_ENTRY_FILE,
    POLICY_DECISIONS_FILE,
    POLICY_LEGALITY_OUTPUT_DIR,
    POLICY_LEGALITY_REPORT_FILE,
    POLICY_TASK_RECORDS_FILE,
    QWEN_MODEL_PATH,
    QWEN_MODEL_PATH_FALLBACK,
    RUN_OFFLOAD_DIR_NAME,
    TASK_ENV_ENTRY_FILE,
    TRAIN_ENTRY_FILE,
)


def _json_write(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _jsonl_write(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _load_module_from_file(module_name: str, path: Path):
    path = Path(path).expanduser().resolve()
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {module_name} from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return sys.modules.get(module_name, mod)


def _unwrap_trainer(ret):
    if isinstance(ret, (tuple, list)):
        return ret[0] if ret else None
    return ret


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return float(default)
        return float(x)
    except Exception:
        return float(default)


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        if x is None:
            return int(default)
        return int(x)
    except Exception:
        return int(default)


def _mean(values: Sequence[float], default: float = 0.0) -> float:
    vals = [float(v) for v in values if v is not None]
    return float(sum(vals) / len(vals)) if vals else float(default)


def _pct(n: int, d: int) -> float:
    return float(n / max(1, d))


def _counter_to_dict(c: collections.Counter) -> Dict[str, int]:
    return {str(k): int(v) for k, v in c.items()}


def _summarize_numbers(values: Sequence[float]) -> Dict[str, float]:
    vals = sorted(float(v) for v in values if v is not None)
    if not vals:
        return {"count": 0, "mean": 0.0, "median": 0.0, "p90": 0.0, "p95": 0.0, "max": 0.0, "min": 0.0}
    def q(p: float) -> float:
        if len(vals) == 1:
            return float(vals[0])
        idx = min(len(vals) - 1, max(0, int(math.ceil(p * len(vals))) - 1))
        return float(vals[idx])
    return {
        "count": int(len(vals)),
        "mean": float(sum(vals) / len(vals)),
        "median": q(0.50),
        "p90": q(0.90),
        "p95": q(0.95),
        "max": float(vals[-1]),
        "min": float(vals[0]),
    }


def _token_count(tokenizer, text: str) -> int:
    try:
        return int(len(tokenizer.encode(str(text or ""), add_special_tokens=True, truncation=False)))
    except Exception:
        return int(len(str(text or "").split()))


def _action_label(action_index: int, action_names: Sequence[str], num_tools: int) -> str:
    idx = int(action_index)
    if 0 <= idx < len(action_names):
        return str(action_names[idx])
    if idx < int(num_tools):
        return f"tool_{idx}"
    if idx == int(num_tools):
        return "LOCAL"
    if idx == int(num_tools) + 1:
        return "PAUSE"
    return f"INVALID_INDEX_{idx}"


def _legal_by_indicator(decision_mod, action_index: int, allowed_mask: Sequence[int], num_tools: int) -> bool:
    try:
        return bool(decision_mod.is_action_legal_by_indicator(int(action_index), list(allowed_mask), int(num_tools)))
    except Exception:
        idx = int(action_index)
        if idx < 0:
            return False
        if idx < int(num_tools):
            return idx < len(allowed_mask) and int(allowed_mask[idx]) == 1
        if idx in {int(num_tools), int(num_tools) + 1}:
            return True
        return False


def _load_checkpoint_if_requested(policy, checkpoint_path: Optional[str]) -> Dict[str, Any]:
    if not checkpoint_path:
        return {"loaded": False, "path": "", "reason": "no checkpoint provided"}
    path = Path(checkpoint_path).expanduser().resolve()
    if not path.exists():
        return {"loaded": False, "path": str(path), "reason": "checkpoint does not exist"}

    ckpt = torch.load(str(path), map_location="cpu")
    state = None
    if isinstance(ckpt, dict):
        for key in ("policy_state_dict", "model_state_dict", "state_dict", "policy", "model"):
            if key in ckpt and isinstance(ckpt[key], dict):
                state = ckpt[key]
                break
        if state is None:
            maybe_tensor_values = any(hasattr(v, "shape") for v in ckpt.values())
            if maybe_tensor_values:
                state = ckpt
    if state is None:
        return {"loaded": False, "path": str(path), "reason": f"unsupported checkpoint keys={list(ckpt.keys()) if isinstance(ckpt, dict) else type(ckpt)}"}

    missing, unexpected = policy.load_state_dict(state, strict=False)
    return {
        "loaded": True,
        "path": str(path),
        "missing_keys": list(missing),
        "unexpected_keys": list(unexpected),
        "num_missing": int(len(missing)),
        "num_unexpected": int(len(unexpected)),
    }


def _make_policy(train_mod, decision_mod, action_dim: int, run_dir: Path, qwen_model_path: str, seed: int):
    model_path = str(qwen_model_path or getattr(train_mod, "QWEN_MODEL_PATH", QWEN_MODEL_PATH))
    fallback = str(getattr(train_mod, "QWEN_MODEL_PATH_FALLBACK", QWEN_MODEL_PATH_FALLBACK))
    if not Path(model_path).exists() and Path(fallback).exists():
        model_path = fallback

    cfg = decision_mod.TDPOPolicyConfig(
        model_path=model_path,
        tokenizer_path=None,
        device=str(getattr(train_mod, "LLM_DEVICE", "auto")),
        dtype=str(getattr(train_mod, "LLM_DTYPE", "bfloat16")),
        max_length=int(getattr(train_mod, "LLM_MAX_LENGTH", getattr(decision_mod, "LLM_MAX_LENGTH", 1024))),
        use_lora=bool(getattr(train_mod, "LLM_USE_LORA", getattr(decision_mod, "LLM_USE_LORA", True))),
        lora_layer_ids=tuple(getattr(train_mod, "LLM_LORA_LAYER_IDS", getattr(decision_mod, "LLM_LORA_LAYER_IDS", ()))),
        lora_target_modules=tuple(getattr(train_mod, "LLM_LORA_TARGET_MODULES", getattr(decision_mod, "LLM_LORA_TARGET_MODULES", ()))),
        lora_r=int(getattr(train_mod, "LLM_LORA_R", getattr(decision_mod, "LLM_LORA_R", 8))),
        lora_alpha=float(getattr(train_mod, "LLM_LORA_ALPHA", getattr(decision_mod, "LLM_LORA_ALPHA", 16.0))),
        lora_dropout=float(getattr(train_mod, "LLM_LORA_DROPOUT", getattr(decision_mod, "LLM_LORA_DROPOUT", 0.05))),
        tune_last_n_blocks=int(getattr(train_mod, "LLM_TUNE_LAST_N_BLOCKS", getattr(decision_mod, "LLM_TUNE_LAST_N_BLOCKS", 0))),
        train_final_norm=bool(getattr(train_mod, "LLM_TRAIN_FINAL_NORM", getattr(decision_mod, "LLM_TRAIN_FINAL_NORM", False))),
        gradient_checkpointing=bool(getattr(train_mod, "LLM_GRADIENT_CHECKPOINTING", getattr(decision_mod, "LLM_GRADIENT_CHECKPOINTING", True))),
        enable_multi_gpu=bool(getattr(train_mod, "LLM_ENABLE_MULTI_GPU", getattr(decision_mod, "LLM_ENABLE_MULTI_GPU", False))),
        device_map=str(getattr(train_mod, "LLM_DEVICE_MAP", getattr(decision_mod, "LLM_DEVICE_MAP", "none"))),
        max_memory_per_gpu=str(getattr(train_mod, "LLM_MAX_MEMORY_PER_GPU", getattr(decision_mod, "LLM_MAX_MEMORY_PER_GPU", "76GiB"))),
        low_cpu_mem_usage=bool(getattr(train_mod, "LLM_LOW_CPU_MEM_USAGE", getattr(decision_mod, "LLM_LOW_CPU_MEM_USAGE", True))),
        offload_folder=str(run_dir / RUN_OFFLOAD_DIR_NAME),
        memory_vector_dim=int(action_dim + 4),
        beta=float(getattr(decision_mod, "TDPO_BETA", 0.05)),
        seed=int(seed),
        verbose=True,
    )
    return decision_mod.TDPOPolicy(action_dim=action_dim, cfg=cfg)


def _load_knowledge_if_requested(train_mod, dataset: Dict[str, Any], action_dim: int, run_dir: Path, use_memory: bool):
    if not use_memory:
        return None, {"loaded": False, "reason": "use_memory=0"}
    try:
        ret = train_mod.load_knowledge_trainer(dataset, action_dim, run_dir)
        trainer = _unwrap_trainer(ret)
        release_info = {}
        if trainer is not None and hasattr(train_mod, "release_knowledge_backbone_for_tdpo"):
            release_info = train_mod.release_knowledge_backbone_for_tdpo(trainer)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return trainer, {"loaded": trainer is not None, "release_info": release_info}
    except Exception as exc:
        return None, {"loaded": False, "reason": repr(exc)}


def _task_records_from_env(train_mod, env) -> List[Dict[str, Any]]:
    if hasattr(train_mod, "build_complete_task_records"):
        try:
            return list(train_mod.build_complete_task_records(env))
        except Exception:
            pass
    records = list(getattr(env, "completed_task_records", []) or [])
    seen = {str(r.get("task_id")) for r in records}
    for task_id, task in getattr(env, "tasks", {}).items():
        if str(task_id) in seen:
            continue
        end_time = getattr(task, "end_time", None)
        arr = float(getattr(task, "arrival_time", 0.0))
        records.append({
            "task_id": str(getattr(task, "graph_id", task_id)),
            "status": str(getattr(task, "status", "unknown")),
            "arrival_time": arr,
            "end_time": end_time,
            "latency_s": None if end_time is None else float(end_time) - arr,
            "task_deadline_s": float(getattr(task, "task_deadline", 0.0)),
            "restart_count": int(getattr(task, "restart_count", 0)),
            "failure_reason": str(getattr(task, "failure_reason", "")),
            "log": list(getattr(task, "log", []) or []),
        })
    return records


def _summarize_task_records(task_records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    records = list(task_records)
    total = len(records)
    completed = [r for r in records if str(r.get("status")) == "completed"]
    failed = [r for r in records if str(r.get("status")) == "failed"]
    lat_completed = [_safe_float(r.get("latency_s"), 0.0) for r in completed if r.get("latency_s") is not None]
    failure_reasons = collections.Counter(str(r.get("failure_reason") or "") for r in failed)
    return {
        "num_tasks": int(total),
        "num_completed": int(len(completed)),
        "num_failed": int(len(failed)),
        "completion_rate": _pct(len(completed), total),
        "avg_completed_latency_s": _mean(lat_completed, 0.0),
        "failure_reasons": _counter_to_dict(failure_reasons),
    }


def _collect_env_event_counts(task_records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    event_counts = collections.Counter()
    invalid_counts = collections.Counter()
    for rec in task_records:
        for ev in list(rec.get("log", []) or []):
            et = str(ev.get("event_type", ""))
            event_counts[et] += 1
            if ("invalid" in et) or ("rejected" in et) or ("failed" in et):
                reason = ""
                payload = ev.get("payload", {})
                if isinstance(payload, dict):
                    reason = str(payload.get("reason", payload.get("source_event", "")) or "")
                invalid_counts[f"{et}:{reason}"] += 1
    return {
        "event_counts": _counter_to_dict(event_counts),
        "invalid_or_rejected_event_counts": _counter_to_dict(invalid_counts),
    }


def run_legality_rollout(args) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]]]:
    run_dir = Path(args.save_dir).expanduser().resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    # Preload dependency modules under the names used by train/decision imports.
    task_env_mod = _load_module_from_file("task_env", Path(args.task_env_file))
    knowledge_mod = _load_module_from_file("knowledge_aspect_adapter", Path(args.knowledge_file))
    decision_mod = _load_module_from_file("decision_TDPO_ye", Path(args.decision_file))
    train_mod = _load_module_from_file("train_fast_mem_under_test", Path(args.train_file))

    # Make sure train uses the exact loaded modules.
    train_mod.task_env_mod = task_env_mod
    train_mod.knowledge_mod = knowledge_mod
    train_mod.decision_mod = decision_mod

    if int(args.sample_num_graphs) > 0:
        setattr(train_mod, "SAMPLE_NUM_GRAPHS", int(args.sample_num_graphs))
        setattr(train_mod, "MAX_TRAIN_GRAPHS", int(args.sample_num_graphs))
    if args.data_dir:
        setattr(train_mod, "DATA_DIR", Path(args.data_dir).expanduser())

    dataset, graph_meta = train_mod.load_dataset_from_dir(Path(getattr(train_mod, "DATA_DIR")), seed=int(args.seed))
    num_tools = int(len(dataset.get("tool_catalog", []) or []))
    action_dim = int(decision_mod.action_dim_from_num_tools(num_tools))

    knowledge_trainer, knowledge_info = _load_knowledge_if_requested(
        train_mod=train_mod,
        dataset=dataset,
        action_dim=action_dim,
        run_dir=run_dir,
        use_memory=bool(int(args.use_memory)),
    )

    policy = _make_policy(
        train_mod=train_mod,
        decision_mod=decision_mod,
        action_dim=action_dim,
        run_dir=run_dir,
        qwen_model_path=str(args.qwen_model_path or ""),
        seed=int(args.seed),
    )
    ckpt_info = _load_checkpoint_if_requested(policy, args.checkpoint)

    env = train_mod.build_env(dataset, seed=int(args.seed))
    env.reset()

    step_limit = int(args.max_steps)
    if step_limit <= 0:
        step_limit = int(train_mod.compute_env_step_limit(dataset))
    max_decisions = int(args.max_decisions)
    use_memory = bool(int(args.use_memory))
    sample = bool(int(args.sample))
    temperature = float(args.temperature)

    decision_rows: List[Dict[str, Any]] = []
    prompt_lengths: List[int] = []
    knowledge_lengths: List[int] = []
    selection_times: List[float] = []

    action_type_counts = collections.Counter()
    action_name_counts = collections.Counter()
    parse_failed_count = 0
    in_range_count = 0
    candidate_name_count = 0
    legal_mask_count = 0
    legal_indicator_count = 0
    invalid_generated_count = 0
    total_decisions = 0
    stopped_by_max_decisions = False

    t0 = time.time()
    steps = 0

    while (not env.done()) and steps < step_limit:
        dps = list(env.collect_decision_points() or [])
        if dps:
            remaining = max_decisions - total_decisions if max_decisions > 0 else len(dps)
            if max_decisions > 0 and remaining <= 0:
                stopped_by_max_decisions = True
                break
            if max_decisions > 0 and len(dps) > remaining:
                dps = dps[:remaining]

            metas = [graph_meta.get(str(dp.get("task_id")), {}) for dp in dps]
            ts = time.time()
            results = policy.select_actions(
                dps=dps,
                num_tools=num_tools,
                use_memory=use_memory,
                knowledge_trainer=knowledge_trainer,
                sample=sample,
                temperature=temperature,
                graph_metas=metas,
            )
            select_s = time.time() - ts
            per_select_s = select_s / max(1, len(dps))

            for dp, meta, (action, info) in zip(dps, metas, results):
                idx = int(info.action_index)
                action_names = list(getattr(info, "action_names", []) or [])
                action_label = _action_label(idx, action_names, num_tools)
                action_type = str(getattr(action, "action_type", getattr(info, "action_type", "")))
                allowed_mask = list(getattr(info, "prompt_allowed_tool_mask", []) or dp.get("allowed_tools_mask", []) or [])
                action_mask = list(getattr(info, "action_mask", []) or [])

                parse_failed = bool("parse_failed" in str(getattr(info, "generated_action_text", ""))) or action_type == "invalid_generated" or idx >= action_dim
                in_range = bool(0 <= idx < action_dim)
                in_candidate_table = bool(0 <= idx < len(action_names))
                legal_by_mask = bool(in_range and idx < len(action_mask) and int(action_mask[idx]) == 1)
                legal_by_indicator = bool(in_range and _legal_by_indicator(decision_mod, idx, allowed_mask, num_tools))

                parse_failed_count += int(parse_failed)
                in_range_count += int(in_range)
                candidate_name_count += int(in_candidate_table)
                legal_mask_count += int(legal_by_mask)
                legal_indicator_count += int(legal_by_indicator)
                invalid_generated_count += int(action_type == "invalid_generated")
                action_type_counts[action_type] += 1
                action_name_counts[action_label] += 1
                total_decisions += 1
                selection_times.append(float(per_select_s))

                p_len = _token_count(policy.tokenizer, str(info.prompt))
                k_len = _token_count(policy.tokenizer, str(info.knowledge_prompt))
                prompt_lengths.append(p_len)
                knowledge_lengths.append(k_len)

                row = {
                    "decision_id": int(total_decisions),
                    "env_step": int(getattr(env, "step_count", steps)),
                    "env_time": float(getattr(env, "time", 0.0)),
                    "task_id": str(dp.get("task_id")),
                    "node_id": int(dp.get("node_id", -1)),
                    "node_type_id": int(dp.get("node_type_id", -1)),
                    "action_index": int(idx),
                    "action_type": action_type,
                    "tool_type_id": None if getattr(action, "tool_type_id", None) is None else int(action.tool_type_id),
                    "action_name": action_label,
                    "generated_action_text": str(getattr(info, "generated_action_text", "")),
                    "parse_failed": int(parse_failed),
                    "in_range": int(in_range),
                    "in_candidate_table": int(in_candidate_table),
                    "legal_by_policy_mask": int(legal_by_mask),
                    "legal_by_allowed_indicator": int(legal_by_indicator),
                    "allowed_tool_mask": [int(x) for x in allowed_mask],
                    "action_mask": [int(x) for x in action_mask],
                    "action_names": action_names,
                    "valid_action_indices": list(getattr(info, "valid_action_indices", []) or []),
                    "prompt_tokens": int(p_len),
                    "knowledge_prompt_tokens": int(k_len),
                    "prompt_head": str(info.prompt)[:800],
                    "prompt_tail": str(info.prompt)[-800:],
                    "knowledge_prompt": str(info.knowledge_prompt),
                    "node_remaining_deadline_s": _safe_float(dp.get("node_remaining_deadline_s"), 0.0),
                    "task_remaining_deadline_s": _safe_float(dp.get("task_remaining_deadline_s"), 0.0),
                    "tool_options": dp.get("tool_options", []),
                    "local_option": dp.get("local_option", {}),
                }
                decision_rows.append(row)
                env.apply_decision(str(dp.get("task_id")), int(dp.get("node_id")), action)

                if max_decisions > 0 and total_decisions >= max_decisions:
                    stopped_by_max_decisions = True
                    break

        if stopped_by_max_decisions:
            break
        env.step()
        steps += 1

    task_records = _task_records_from_env(train_mod, env)
    task_summary = _summarize_task_records(task_records)
    event_summary = _collect_env_event_counts(task_records)

    report = {
        "config": {
            "train_file": str(Path(args.train_file).expanduser().resolve()),
            "decision_file": str(Path(args.decision_file).expanduser().resolve()),
            "task_env_file": str(Path(args.task_env_file).expanduser().resolve()),
            "knowledge_file": str(Path(args.knowledge_file).expanduser().resolve()),
            "qwen_model_path": str(args.qwen_model_path),
            "checkpoint": str(args.checkpoint or ""),
            "sample_num_graphs": int(args.sample_num_graphs),
            "use_memory": int(args.use_memory),
            "sample": int(args.sample),
            "temperature": float(args.temperature),
            "max_decisions": int(args.max_decisions),
            "step_limit": int(step_limit),
            "seed": int(args.seed),
            "num_tools": int(num_tools),
            "action_dim": int(action_dim),
        },
        "checkpoint_info": ckpt_info,
        "knowledge_info": knowledge_info,
        "runtime": {
            "wall_seconds": float(time.time() - t0),
            "steps_taken": int(steps),
            "stopped_by_max_decisions": bool(stopped_by_max_decisions),
            "env_done": bool(env.done()),
        },
        "decision_summary": {
            "num_decisions": int(total_decisions),
            "parse_failed_count": int(parse_failed_count),
            "parse_failed_rate": _pct(parse_failed_count, total_decisions),
            "invalid_generated_count": int(invalid_generated_count),
            "invalid_generated_rate": _pct(invalid_generated_count, total_decisions),
            "in_range_count": int(in_range_count),
            "in_range_rate": _pct(in_range_count, total_decisions),
            "in_candidate_table_count": int(candidate_name_count),
            "in_candidate_table_rate": _pct(candidate_name_count, total_decisions),
            "legal_by_policy_mask_count": int(legal_mask_count),
            "legal_by_policy_mask_rate": _pct(legal_mask_count, total_decisions),
            "legal_by_allowed_indicator_count": int(legal_indicator_count),
            "legal_by_allowed_indicator_rate": _pct(legal_indicator_count, total_decisions),
            "action_type_counts": _counter_to_dict(action_type_counts),
            "action_name_counts": _counter_to_dict(action_name_counts),
            "selection_seconds": _summarize_numbers(selection_times),
            "prompt_tokens": _summarize_numbers(prompt_lengths),
            "knowledge_prompt_tokens": _summarize_numbers(knowledge_lengths),
        },
        "task_summary": task_summary,
        "event_summary": event_summary,
        "first_bad_examples": [
            row for row in decision_rows
            if int(row["parse_failed"]) or not int(row["legal_by_policy_mask"]) or not int(row["legal_by_allowed_indicator"])
        ][:20],
        "first_examples": decision_rows[: min(20, len(decision_rows))],
    }
    return report, decision_rows, task_records


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train-file", type=str, default=TRAIN_ENTRY_FILE)
    p.add_argument("--decision-file", type=str, default=CURRENT_POLICY_ENTRY_FILE)
    p.add_argument("--task-env-file", type=str, default=TASK_ENV_ENTRY_FILE)
    p.add_argument("--knowledge-file", type=str, default=KNOWLEDGE_ENTRY_FILE)
    p.add_argument("--qwen-model-path", type=str, default=QWEN_MODEL_PATH)
    p.add_argument("--data-dir", type=str, default="")
    p.add_argument("--checkpoint", type=str, default="")
    p.add_argument("--sample-num-graphs", type=int, default=20)
    p.add_argument("--max-decisions", type=int, default=200)
    p.add_argument("--max-steps", type=int, default=0)
    p.add_argument("--use-memory", type=int, default=1)
    p.add_argument("--sample", type=int, default=0)
    p.add_argument("--temperature", type=float, default=0.60)
    p.add_argument("--seed", type=int, default=20260425)
    p.add_argument("--save-dir", type=str, default=str(POLICY_LEGALITY_OUTPUT_DIR))
    p.add_argument("--save-json", type=str, default=POLICY_LEGALITY_REPORT_FILE)
    p.add_argument("--save-decisions-jsonl", type=str, default=POLICY_DECISIONS_FILE)
    p.add_argument("--save-task-records", type=str, default=POLICY_TASK_RECORDS_FILE)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    save_dir = Path(args.save_dir).expanduser().resolve()
    save_dir.mkdir(parents=True, exist_ok=True)

    report, decision_rows, task_records = run_legality_rollout(args)

    report_path = save_dir / args.save_json
    decisions_path = save_dir / args.save_decisions_jsonl
    task_records_path = save_dir / args.save_task_records

    _json_write(report_path, report)
    _jsonl_write(decisions_path, decision_rows)
    _json_write(task_records_path, task_records)

    ds = report["decision_summary"]
    ts = report["task_summary"]
    print("=" * 100)
    print("[policy-legality-test] done")
    print(f"report={report_path}")
    print(f"decisions={decisions_path}")
    print(f"task_records={task_records_path}")
    print("-" * 100)
    print(f"decisions={ds['num_decisions']} parse_failed={ds['parse_failed_count']} ({ds['parse_failed_rate']:.2%}) "
          f"invalid_generated={ds['invalid_generated_count']} ({ds['invalid_generated_rate']:.2%})")
    print(f"legal_by_mask={ds['legal_by_policy_mask_count']} ({ds['legal_by_policy_mask_rate']:.2%}) "
          f"legal_by_indicator={ds['legal_by_allowed_indicator_count']} ({ds['legal_by_allowed_indicator_rate']:.2%})")
    print(f"tasks={ts['num_tasks']} completed={ts['num_completed']} completion_rate={ts['completion_rate']:.2%} "
          f"avg_completed_latency={ts['avg_completed_latency_s']:.4f}s")
    print(f"action_type_counts={ds['action_type_counts']}")
    print(f"prompt_tokens={ds['prompt_tokens']}")
    print("=" * 100)


if __name__ == "__main__":
    main()
