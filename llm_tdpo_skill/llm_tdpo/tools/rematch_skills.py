#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Rematch the skill/memory used by each node in one traced task.

Default target:
  epoch = 75
  task id contains sample index 1090

The script does not load Qwen or the TDPO checkpoint.  It reproduces the
memory-retrieval and mapped-memory prompt construction used in decision_TDPO.py:
  1. reconstruct a decision point from each trace prompt;
  2. build the same compact state query embedding;
  3. query memory prototypes with the same similarity formula;
  4. build the same compact mapped memory text used in the LLM prompt;
  5. attribute the matched memory object to new_memory / old_memory and record
     its collection/index/prototype_id or atom_id.

Output JSON contains, for every node decision, both:
  - trace_applied_knowledge_prompt: the prompt text already stored in trace if present;
  - rematched_knowledge_prompt: the prompt text reconstructed from memory files.

If the trace already has knowledge_prompt, that is the safest record of what was
actually injected at runtime.  The rematched text is used to recover source and
position in memory.json.
"""

from __future__ import annotations

import argparse
import ast
import difflib
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from llm_tdpo.config.paths import (
    REMATCH_NEW_MEMORY_PATH,
    REMATCH_OLD_MEMORY_PATH,
    REMATCH_OUTPUT_PATH,
    REMATCH_TRACE_PATH,
)


# ============================================================
# Default paths supplied by the user
# ============================================================

DEFAULT_TRACE_PATH = REMATCH_TRACE_PATH
DEFAULT_NEW_MEMORY_PATH = REMATCH_NEW_MEMORY_PATH
DEFAULT_OLD_MEMORY_PATH = REMATCH_OLD_MEMORY_PATH
DEFAULT_OUTPUT_PATH = REMATCH_OUTPUT_PATH


# ============================================================
# Constants copied from the uploaded decision_TDPO.py behavior
# ============================================================

MEMORY_VECTOR_DIM_FALLBACK = 64
KNOWLEDGE_PROMPT_TOP_K = 1
KNOWLEDGE_PROMPT_TOTAL_MAX_CHARS = 1200
MEMORY_TOOL_NAME_MIN_SIMILARITY = 0.65
K_RETRIEVE = 3


# ============================================================
# Generic helpers
# ============================================================


def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return float(default)
        return float(x)
    except Exception:
        return float(default)


def pad_or_trim(values: Sequence[float], dim: int) -> List[float]:
    vals = [float(v) for v in list(values or [])]
    if len(vals) >= int(dim):
        return vals[: int(dim)]
    return vals + [0.0] * (int(dim) - len(vals))


def local_action_index(num_tools: int) -> int:
    return int(num_tools)


def pause_action_index(num_tools: int) -> int:
    return int(num_tools) + 1


def mean_or_zero(values: Sequence[float]) -> float:
    vals = [float(v) for v in values if v is not None]
    return float(sum(vals) / len(vals)) if vals else 0.0


def normalize_tool_string(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text


def tool_name_similarity(a: Any, b: Any) -> float:
    a_norm = normalize_tool_string(a)
    b_norm = normalize_tool_string(b)
    if not a_norm or not b_norm:
        return 0.0
    if a_norm == b_norm:
        return 1.0
    seq = difflib.SequenceMatcher(None, a_norm, b_norm).ratio()
    a_tokens = {t for t in a_norm.split("_") if t}
    b_tokens = {t for t in b_norm.split("_") if t}
    jac = len(a_tokens & b_tokens) / max(1, len(a_tokens | b_tokens))
    if a_norm in b_norm or b_norm in a_norm:
        seq = max(seq, 0.75)
    return float(max(seq, jac))


def is_placeholder_tool_name(value: Any) -> bool:
    norm = normalize_tool_string(value)
    return bool(re.fullmatch(r"tool_?\d+", norm or "") or re.fullmatch(r"tool_type_?\d+", norm or ""))


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    va = [float(x) for x in list(a or [])]
    vb = [float(x) for x in list(b or [])]
    dim = max(len(va), len(vb))
    if dim <= 0:
        return 0.0
    va = pad_or_trim(va, dim)
    vb = pad_or_trim(vb, dim)
    na = math.sqrt(sum(x * x for x in va))
    nb = math.sqrt(sum(x * x for x in vb))
    denom = na * nb
    if denom <= 1e-12:
        return 0.0
    return float(sum(x * y for x, y in zip(va, vb)) / denom)


def prototype_similarity(
    query_embedding: Sequence[float],
    node_type_id: int,
    queue_class: int,
    slack_class: int,
    proto: Dict[str, Any],
) -> float:
    type_term = 1.0 if int(node_type_id) == int(proto.get("node_type_id", -999)) else 0.0
    cos = cosine(query_embedding, proto.get("centroid", []))
    regime_diff = abs(int(queue_class) - int(proto.get("queue_class", 0))) + abs(
        int(slack_class) - int(proto.get("slack_class", 0))
    )
    regime = math.exp(-regime_diff / 2.0)
    return float(0.4 * type_term + 0.4 * cos + 0.2 * regime)


def json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


# ============================================================
# Trace parsing and decision-point reconstruction
# ============================================================


def _parse_key_value_tail(tail: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for part in tail.split(";"):
        part = part.strip()
        if not part or "=" not in part:
            continue
        key, val = part.split("=", 1)
        key = key.strip()
        val = val.strip().rstrip(",")
        if val.lower() == "true":
            out[key] = True
        elif val.lower() == "false":
            out[key] = False
        else:
            try:
                if re.fullmatch(r"[-+]?\d+", val):
                    out[key] = int(val)
                else:
                    out[key] = float(val)
            except Exception:
                out[key] = val
    return out


def _parse_prompt_state(prompt: str) -> Dict[str, Any]:
    parsed: Dict[str, Any] = {}
    text = str(prompt or "")

    m = re.search(
        r"task_rem=([0-9.+\-eE]+)s;\s*task_elapsed=([0-9.+\-eE]+)s;\s*completed=(\d+)/(\d+);\s*frontier=(\d+)",
        text,
    )
    if m:
        parsed.update(
            {
                "task_remaining_deadline_s": float(m.group(1)),
                "task_elapsed_time_s": float(m.group(2)),
                "completed_node_count": int(m.group(3)),
                "num_nodes": int(m.group(4)),
                "ready_frontier_count": int(m.group(5)),
            }
        )

    m = re.search(
        r"node_id=(\d+);\s*type=(\d+);\s*pred=(\d+);\s*succ=(\d+);\s*node_rem=([0-9.+\-eE]+)s;\s*node_deadline=([0-9.+\-eE]+)s",
        text,
    )
    if m:
        parsed.update(
            {
                "node_id": int(m.group(1)),
                "node_type_id": int(m.group(2)),
                "num_predecessors": int(m.group(3)),
                "num_successors": int(m.group(4)),
                "node_remaining_deadline_s": float(m.group(5)),
                "node_deadline_s": float(m.group(6)),
            }
        )

    m = re.search(
        r"cycles=([0-9.+\-eE]+);\s*uplink_bits=(\d+);\s*downlink_bits=(\d+)",
        text,
    )
    if m:
        parsed.update(
            {
                "computation_cycles": float(m.group(1)),
                "uplink_size_bits": int(m.group(2)),
                "downlink_size_bits": int(m.group(3)),
            }
        )

    for key in (
        "running_tool_jobs_total",
        "queued_tool_jobs_total",
        "running_local_jobs",
        "queued_local_jobs",
    ):
        m = re.search(rf"{key}:\s*(\d+)", text)
        if m:
            parsed[key] = int(m.group(1))

    m = re.search(r"allowed_tool_indicator_text_only\s*=\s*(\[[^\]]*\])", text)
    if m:
        try:
            parsed["allowed_tools_mask"] = list(ast.literal_eval(m.group(1)))
        except Exception:
            pass

    return parsed


def _parse_candidate_actions_from_prompt(prompt: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any], List[int]]:
    tool_options: List[Dict[str, Any]] = []
    local_option: Dict[str, Any] = {}
    valid_indices: List[int] = []

    for raw_line in str(prompt or "").splitlines():
        line = raw_line.strip()
        if not line.startswith("- ACTION_INDEX="):
            continue
        m = re.match(r"-\s*ACTION_INDEX=(\d+);\s*name:\s*([^;]+);\s*type=([^;]+);?\s*(.*)$", line)
        if not m:
            continue
        idx = int(m.group(1))
        name = m.group(2).strip()
        typ = m.group(3).strip().lower()
        tail = m.group(4).strip()
        kv = _parse_key_value_tail(tail)
        valid_indices.append(idx)

        if typ == "tool":
            opt = {
                "tool_type_id": idx,
                "slot": idx,
                "action_index": idx,
                "name": name,
                "tool_name": name,
                "queue_blocked": bool(kv.get("blocked", False)),
                "predicted_total_s": safe_float(kv.get("pred_total"), 0.0),
                "base_queue_delay_s": safe_float(kv.get("queue_s"), 0.0),
                "uplink_s": safe_float(kv.get("uplink"), 0.0),
                "exec_s": safe_float(kv.get("exec"), 0.0),
                "downlink_s": safe_float(kv.get("downlink"), 0.0),
                "risk_estimate": safe_float(kv.get("risk"), 0.0),
                "packet_loss_rate": safe_float(kv.get("pkt_loss"), 0.0),
                "queue_length": int(kv.get("queue", 0) or 0),
                "running_instances": int(kv.get("running", 0) or 0),
                "queue_capacity": int(kv.get("qcap", 0) or 0),
                "instances": int(kv.get("instances", 0) or 0),
            }
            tool_options.append(opt)
        elif typ == "local":
            local_option = {
                "queue_blocked": bool(kv.get("blocked", False)),
                "predicted_total_s": safe_float(kv.get("pred_total"), 0.0),
                "estimated_wait_s": safe_float(kv.get("queue_s"), 0.0),
                "compute_s": safe_float(kv.get("compute"), 0.0),
                "fail_probability": safe_float(kv.get("fail_prob"), 0.0),
                "queue_length": int(kv.get("queue", 0) or 0),
                "running_local_jobs": int(kv.get("running", 0) or 0),
            }

    return tool_options, local_option, valid_indices


def reconstruct_decision_point(trace: Dict[str, Any]) -> Dict[str, Any]:
    prompt = str(trace.get("prompt", "") or "")
    dp = _parse_prompt_state(prompt)
    tool_options, local_option, valid_indices = _parse_candidate_actions_from_prompt(prompt)

    dp.update(
        {
            "task_id": str(trace.get("task_id", "")),
            "node_id": int(trace.get("node_id", dp.get("node_id", 0)) or 0),
            "node_type_id": int(trace.get("node_type_id", dp.get("node_type_id", 0)) or 0),
            "node_remaining_deadline_s": safe_float(
                trace.get("node_remaining_deadline_s", dp.get("node_remaining_deadline_s", 0.0)), 0.0
            ),
            "task_remaining_deadline_s": safe_float(
                trace.get("task_remaining_deadline_s", dp.get("task_remaining_deadline_s", 0.0)), 0.0
            ),
            "tool_options": tool_options,
            "local_option": local_option,
        }
    )

    if "allowed_tools_mask" not in dp:
        mask = trace.get("prompt_allowed_tool_mask", [])
        if isinstance(mask, list):
            dp["allowed_tools_mask"] = [int(x) for x in mask]
    if "allowed_tools_mask" not in dp:
        action_names = list(trace.get("action_names", []) or [])
        inferred_num_tools = max(0, len(action_names) - 2)
        dp["allowed_tools_mask"] = [0] * inferred_num_tools
        for opt in tool_options:
            idx = int(opt.get("tool_type_id", -1))
            if 0 <= idx < inferred_num_tools:
                dp["allowed_tools_mask"][idx] = 1

    # Fall back to trace values when the prompt does not expose these fields.
    for key, default in (
        ("task_elapsed_time_s", safe_float(trace.get("env_time"), 0.0)),
        ("completed_node_count", 0),
        ("num_nodes", 0),
        ("ready_frontier_count", 0),
        ("num_predecessors", 0),
        ("num_successors", 0),
        ("node_deadline_s", safe_float(trace.get("node_remaining_deadline_s"), 0.0)),
        ("computation_cycles", 0.0),
        ("uplink_size_bits", 0),
        ("downlink_size_bits", 0),
        ("running_tool_jobs_total", 0),
        ("queued_tool_jobs_total", 0),
        ("running_local_jobs", 0),
        ("queued_local_jobs", 0),
    ):
        dp.setdefault(key, default)

    dp["_valid_candidate_indices_from_prompt"] = valid_indices
    return dp


def build_state_query_embedding(dp: Dict[str, Any], num_tools: int, target_dim: int = MEMORY_VECTOR_DIM_FALLBACK) -> List[float]:
    values: List[float] = []
    values.append(safe_float(dp.get("node_type_id"), 0.0) / 16.0)
    values.append(safe_float(dp.get("task_remaining_deadline_s"), 0.0) / 100.0)
    values.append(safe_float(dp.get("node_remaining_deadline_s"), 0.0) / 50.0)

    tool_by_id = {int(opt.get("tool_type_id", -1)): opt for opt in list(dp.get("tool_options", []) or [])}
    for tid in range(int(num_tools)):
        opt = tool_by_id.get(tid, {})
        values.extend(
            [
                0.0 if not opt else 1.0,
                1.0 if bool(opt.get("queue_blocked", False)) else 0.0,
                safe_float(opt.get("predicted_total_s"), 0.0) / 20.0,
                safe_float(opt.get("risk_estimate"), 0.0),
                safe_float(opt.get("packet_loss_rate"), 0.0),
                safe_float(opt.get("base_queue_delay_s"), 0.0) / 10.0,
            ]
        )

    local = dp.get("local_option", {}) or {}
    values.extend(
        [
            0.0 if bool(local.get("queue_blocked", False)) else 1.0,
            safe_float(local.get("predicted_total_s", local.get("compute_s", 0.0)), 0.0) / 20.0,
            safe_float(local.get("fail_probability"), 0.0),
            safe_float(local.get("estimated_wait_s"), 0.0) / 10.0,
        ]
    )
    return pad_or_trim(values, target_dim)


def fastest_predicted_latency(dp: Dict[str, Any], default: float = 1.0) -> float:
    vals: List[float] = []
    for opt in list(dp.get("tool_options", []) or []):
        if not bool(opt.get("queue_blocked", False)):
            vals.append(safe_float(opt.get("predicted_total_s"), default))
    local = dp.get("local_option", {}) or {}
    if not bool(local.get("queue_blocked", False)):
        vals.append(safe_float(local.get("predicted_total_s", local.get("compute_s", default)), default))
    return min(vals) if vals else float(default)


def infer_queue_class(dp: Dict[str, Any]) -> int:
    queue_values: List[float] = []
    for opt in list(dp.get("tool_options", []) or []):
        queue_values.append(safe_float(opt.get("base_queue_delay_s"), 0.0))
    local = dp.get("local_option", {}) or {}
    queue_values.append(safe_float(local.get("estimated_wait_s"), 0.0))
    q = max(queue_values or [0.0])
    if q < 0.25:
        return 0
    if q < 1.0:
        return 1
    return 2


def infer_slack_class(dp: Dict[str, Any]) -> int:
    remain = safe_float(dp.get("node_remaining_deadline_s"), 0.0)
    fastest = fastest_predicted_latency(dp, default=1.0)
    ratio = remain / max(fastest, 1.0e-6)
    if ratio < 1.2:
        return 0
    if ratio < 2.5:
        return 1
    return 2


def current_tool_name_action_map(dp: Dict[str, Any], num_tools: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    seen = set()
    for opt in list(dp.get("tool_options", []) or []):
        if not isinstance(opt, dict):
            continue
        raw_idx = opt.get("action_index", opt.get("tool_type_id", opt.get("slot", None)))
        try:
            idx = int(raw_idx)
        except Exception:
            continue
        if idx < 0 or idx >= int(num_tools):
            continue
        names: List[str] = []
        for key in ("tool_name", "name"):
            val = str(opt.get(key, "") or "").strip()
            if val and not is_placeholder_tool_name(val) and val not in names:
                names.append(val)
        if not names:
            continue
        display_name = names[0]
        for name in names:
            norm = normalize_tool_string(name)
            if not norm or (norm, idx) in seen:
                continue
            seen.add((norm, idx))
            rows.append(
                {
                    "name": str(name),
                    "display_name": str(display_name),
                    "norm": norm,
                    "current_action_index": int(idx),
                }
            )
    return rows


# ============================================================
# Memory loading and attribution
# ============================================================


def _extract_memory_root(data: Dict[str, Any]) -> Dict[str, Any]:
    if "buffer" in data or "prototypes" in data:
        return data
    if isinstance(data.get("memory"), dict):
        return data["memory"]
    if isinstance(data.get("knowledge_memory"), dict):
        return data["knowledge_memory"]
    return data


def load_memory_rows(path: Path, label: str) -> List[Dict[str, Any]]:
    path = Path(path).expanduser()
    data = json.loads(path.read_text(encoding="utf-8"))
    root = _extract_memory_root(data if isinstance(data, dict) else {})
    rows: List[Dict[str, Any]] = []
    for collection in ("prototypes", "buffer"):
        values = root.get(collection, []) if isinstance(root, dict) else []
        if not isinstance(values, list):
            continue
        for idx, obj in enumerate(values):
            if not isinstance(obj, dict):
                continue
            row = dict(obj)
            uid = str(row.get("prototype_id") or row.get("atom_id") or f"{label}:{collection}:{idx}")
            row["_memory_uid"] = uid
            row["_source_label"] = label
            row["_source_path"] = str(path)
            row["_collection"] = collection
            row["_index"] = int(idx)
            row["_location"] = {
                "source_label": label,
                "source_path": str(path),
                "collection": collection,
                "index0": int(idx),
                "index1": int(idx + 1),
                "prototype_id": row.get("prototype_id"),
                "atom_id": row.get("atom_id"),
            }
            rows.append(row)
    return rows


def build_memory_index(new_memory_path: Path, old_memory_path: Optional[Path], retrieval_store: str) -> Tuple[List[Dict[str, Any]], Dict[str, List[Dict[str, Any]]]]:
    all_rows: List[Dict[str, Any]] = []
    new_rows = load_memory_rows(new_memory_path, "new_memory")
    all_rows.extend(new_rows)
    old_rows: List[Dict[str, Any]] = []
    if old_memory_path is not None and Path(old_memory_path).expanduser().exists():
        old_rows = load_memory_rows(old_memory_path, "old_memory")
        all_rows.extend(old_rows)

    locations_by_uid: Dict[str, List[Dict[str, Any]]] = {}
    for row in all_rows:
        locations_by_uid.setdefault(str(row.get("_memory_uid")), []).append(dict(row.get("_location", {})))

    if retrieval_store == "new":
        base_rows = new_rows
    elif retrieval_store == "old":
        base_rows = old_rows
    elif retrieval_store == "combined":
        preferred: Dict[str, Dict[str, Any]] = {}
        for row in old_rows:
            preferred[str(row.get("_memory_uid"))] = row
        for row in new_rows:
            preferred[str(row.get("_memory_uid"))] = row
        base_rows = list(preferred.values())
    else:
        raise ValueError(f"Unsupported retrieval_store={retrieval_store!r}; use new, old, or combined.")

    for row in base_rows:
        uid = str(row.get("_memory_uid"))
        row["_all_locations"] = locations_by_uid.get(uid, [dict(row.get("_location", {}))])
        labels = {loc.get("source_label") for loc in row["_all_locations"]}
        if labels == {"new_memory"}:
            row["_source_attribution"] = "new_memory_only"
        elif labels == {"old_memory"}:
            row["_source_attribution"] = "old_memory_only"
        elif "new_memory" in labels and "old_memory" in labels:
            row["_source_attribution"] = "old_memory_loaded_or_updated_in_new_memory"
        else:
            row["_source_attribution"] = "+".join(sorted(str(x) for x in labels if x))

    return base_rows, locations_by_uid


def compact_memory_ref(item: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if item is None:
        return None
    return {
        "memory_uid": item.get("_memory_uid"),
        "source_attribution": item.get("_source_attribution"),
        "selected_source_label": item.get("_source_label"),
        "selected_collection": item.get("_collection"),
        "selected_index0": item.get("_index"),
        "selected_index1": int(item.get("_index", -1)) + 1 if item.get("_index") is not None else None,
        "all_locations": item.get("_all_locations", []),
        "prototype_id": item.get("prototype_id"),
        "atom_id": item.get("atom_id"),
        "node_type_id": item.get("node_type_id"),
        "queue_class": item.get("queue_class"),
        "slack_class": item.get("slack_class"),
        "support": item.get("support"),
        "confidence": item.get("confidence"),
        "confidence_mean": item.get("confidence_mean"),
        "recommended_action": item.get("recommended_action", {}),
        "observed_outcome": item.get("observed_outcome", {}),
        "prompt_text": item.get("prompt_text", ""),
    }


# ============================================================
# Memory prompt reconstruction copied from decision_TDPO.py behavior
# ============================================================


def history_tool_names_from_memory(item: Dict[str, Any], include_candidates: bool = True) -> List[str]:
    names: List[str] = []

    def add(value: Any) -> None:
        text = str(value or "").strip()
        if text and text not in names:
            names.append(text)

    rec = item.get("recommended_action", {})
    if isinstance(rec, dict) and str(rec.get("action_type", "")).lower() == "tool":
        add(rec.get("tool_name"))
        add(rec.get("name"))

    if include_candidates:
        for act in list(item.get("candidate_actions", []) or []):
            if isinstance(act, dict) and str(act.get("action_type", "")).lower() == "tool":
                add(act.get("tool_name"))
                add(act.get("name"))

    if not names:
        text = str(item.get("prompt_text", "") or "")
        for pat in (
            r"tool_name\s*=\s*['\"]([^'\"]+)['\"]",
            r"Choose\s+TOOL\s+slot\s+\d+\s*:\s*([^\(\n\.]+)",
            r"Candidate actions:\s*TOOL\s+slot\s+\d+\s*:\s*([^\|\n]+)",
        ):
            for m in re.finditer(pat, text, flags=re.IGNORECASE):
                add(m.group(1).strip())
    return names


def recommended_action_type(item: Dict[str, Any], num_tools: Optional[int] = None) -> str:
    rec = item.get("recommended_action", {})
    if isinstance(rec, dict):
        typ = str(rec.get("action_type", "") or "").strip().lower()
        if typ in {"tool", "local", "pause"}:
            return typ
        idx_obj = rec.get("action_index", rec.get("action_id", rec.get("selected_action_index", None)))
        try:
            idx = int(idx_obj)
            if num_tools is not None:
                if 0 <= idx < int(num_tools):
                    return "tool"
                if idx == local_action_index(int(num_tools)):
                    return "local"
                if idx == pause_action_index(int(num_tools)):
                    return "pause"
        except Exception:
            pass
    obs = item.get("observed_outcome", {})
    if isinstance(obs, dict):
        for key in ("recommended_action_type", "action_type", "selected_action_type", "exec_action_type"):
            typ = str(obs.get(key, "") or "").strip().lower()
            if typ in {"tool", "local", "pause"}:
                return typ
    if num_tools is not None:
        gains = list(item.get("action_gain_means", item.get("action_gain_priors", [])) or [])
        if len(gains) >= int(num_tools) + 2:
            try:
                vals = [float(x) for x in gains]
                best_idx = max(range(len(vals)), key=lambda i: vals[i])
                if best_idx < int(num_tools):
                    return "tool"
                if best_idx == local_action_index(int(num_tools)):
                    return "local"
                if best_idx == pause_action_index(int(num_tools)):
                    return "pause"
            except Exception:
                pass
    text = str(item.get("prompt_text", "") or "").lower()
    if re.search(r"\bchoose\s+local\b", text) or ("local execution" in text and not re.search(r"\bchoose\s+tool\b", text)):
        return "local"
    if re.search(r"\bchoose\s+pause\b", text) or "pause;" in text:
        return "pause"
    if re.search(r"\bchoose\s+tool\b", text) or "use_tool" in text or re.search(r"\btool\s+slot\b", text):
        return "tool"
    return "unknown"


def best_current_tool_match(history_name: Any, current_tools: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    best = None
    best_score = -1.0
    for cur in current_tools:
        score = tool_name_similarity(history_name, cur.get("name", ""))
        if score > best_score:
            best_score = score
            best = cur
    if best is None:
        return None
    out = dict(best)
    out["similarity"] = float(best_score)
    return out


def first_finite_value(*values: Any) -> float:
    for value in values:
        val = safe_float(value, float("nan"))
        if not math.isnan(val):
            return val
    return float("nan")


def evidence_text(item: Dict[str, Any]) -> str:
    rec = item.get("recommended_action", {}) if isinstance(item.get("recommended_action", {}), dict) else {}
    obs = item.get("observed_outcome", {}) if isinstance(item.get("observed_outcome", {}), dict) else {}
    env_ctx = item.get("environment_context", {}) if isinstance(item.get("environment_context", {}), dict) else {}
    state_ctx = item.get("state_context", {}) if isinstance(item.get("state_context", {}), dict) else {}
    gains = list(item.get("action_gain_means", item.get("action_gain_priors", [])) or [])
    vals: List[str] = []

    gain = first_finite_value(rec.get("gain"), rec.get("completion_gain"), obs.get("recommended_action_gain"))
    if not math.isnan(gain):
        vals.append(f"gain={gain:.3f}")
    elif gains:
        vals.append(f"max_gain={max(float(x) for x in gains):.3f}")

    total = first_finite_value(obs.get("mean_total_s"), rec.get("predicted_total_s"), rec.get("total_s"))
    if not math.isnan(total):
        vals.append(f"total={total:.3f}s")

    queue = first_finite_value(obs.get("mean_queue_s"), rec.get("queue_s"), rec.get("queue_delay_s"), rec.get("base_queue_delay_s"))
    if not math.isnan(queue):
        vals.append(f"queue={queue:.3f}s")

    risk = first_finite_value(obs.get("mean_risk"), rec.get("risk_estimate"), rec.get("risk"))
    if not math.isnan(risk):
        vals.append(f"risk={risk:.3f}")

    pkt = first_finite_value(obs.get("mean_packet_loss"), rec.get("packet_loss_rate"), rec.get("pkt_loss"))
    if not math.isnan(pkt):
        vals.append(f"pkt_loss={pkt:.3f}")

    rel = first_finite_value(rec.get("reliability"), rec.get("server_reliability"), obs.get("mean_reliability"), env_ctx.get("reliability"))
    if not math.isnan(rel):
        vals.append(f"reliability={rel:.3f}")

    cgain = first_finite_value(rec.get("completion_gain"), obs.get("mean_completion_gain"), env_ctx.get("completion_gain"))
    if not math.isnan(cgain):
        vals.append(f"completion_gain={cgain:.3f}")

    failp = first_finite_value(rec.get("fail_probability"), rec.get("failure_probability"), obs.get("fail_probability"), state_ctx.get("fail_probability"))
    if not math.isnan(failp):
        vals.append(f"fail_probability={failp:.3f}")

    sr = first_finite_value(obs.get("success_rate"))
    if not math.isnan(sr):
        vals.append(f"success_rate={sr:.3f}")

    ts = obs.get("task_success", None)
    if ts is not None:
        try:
            vals.append(f"task_success={int(ts)}")
        except Exception:
            vals.append(f"task_success={ts}")

    conf = first_finite_value(item.get("confidence_mean"), item.get("confidence"))
    if not math.isnan(conf):
        vals.append(f"confidence={conf:.3f}")

    return ", ".join(vals) if vals else "no numeric evidence"


def tool_name_memory_rows(memory_items: List[Dict[str, Any]], current_tools: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
    scored: List[Tuple[float, float, Dict[str, Any]]] = []
    if not current_tools:
        return []
    min_sim = float(MEMORY_TOOL_NAME_MIN_SIMILARITY)
    for item in memory_items:
        history_names = history_tool_names_from_memory(item, include_candidates=True)
        if not history_names:
            continue
        best_hist = ""
        best_match = None
        best_score = -1.0
        for hist_name in history_names:
            match = best_current_tool_match(hist_name, current_tools)
            if match is not None and float(match.get("similarity", 0.0)) > best_score:
                best_score = float(match.get("similarity", 0.0))
                best_hist = hist_name
                best_match = match
        if best_match is None or best_score < min_sim:
            continue
        conf = safe_float(item.get("confidence_mean", item.get("confidence", 0.0)), 0.0)
        scored.append(
            (
                best_score,
                conf,
                {
                    "item": item,
                    "history_tool_name": best_hist,
                    "current_match": best_match,
                    "similarity": best_score,
                    "current_action_index": int(best_match.get("current_action_index")),
                },
            )
        )
    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return [row for _, _, row in scored[: max(0, int(top_k))]]


def state_memory_rows(
    selected_items: List[Dict[str, Any]],
    current_tools: List[Dict[str, Any]],
    num_tools: int,
    top_k: int,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    min_sim = float(MEMORY_TOOL_NAME_MIN_SIMILARITY)
    for item in selected_items[: max(0, int(top_k))]:
        typ = recommended_action_type(item, num_tools=num_tools)
        row: Dict[str, Any] = {"item": item, "action_type": typ, "current_action_index": None}
        if typ == "tool":
            names = history_tool_names_from_memory(item, include_candidates=False)
            if not names:
                names = history_tool_names_from_memory(item, include_candidates=True)
            hist_name = names[0] if names else "unknown_tool"
            match = best_current_tool_match(hist_name, current_tools)
            row["history_tool_name"] = hist_name
            row["current_match"] = match
            if match is not None and float(match.get("similarity", 0.0)) >= min_sim:
                row["current_action_index"] = int(match["current_action_index"])
        elif typ == "local":
            row["current_action_index"] = local_action_index(num_tools)
        elif typ == "pause":
            row["current_action_index"] = pause_action_index(num_tools)
        rows.append(row)
    return rows


def query_memory_prototypes(
    memory_rows: List[Dict[str, Any]],
    query_embedding: Sequence[float],
    node_type_id: int,
    queue_class: int,
    slack_class: int,
    k: int,
) -> List[Dict[str, Any]]:
    scored: List[Tuple[float, Dict[str, Any]]] = []
    for row in memory_rows:
        if str(row.get("_collection")) != "prototypes":
            continue
        if not row.get("centroid"):
            continue
        score = prototype_similarity(query_embedding, node_type_id, queue_class, slack_class, row)
        scored.append((score, row))
    scored.sort(key=lambda x: x[0], reverse=True)
    selected: List[Dict[str, Any]] = []
    for score, row in scored[: max(0, int(k))]:
        cp = dict(row)
        cp["_retrieval_score"] = float(score)
        selected.append(cp)
    return selected


def build_mapped_memory_guidance(
    memory_rows: List[Dict[str, Any]],
    selected_items: List[Dict[str, Any]],
    dp: Dict[str, Any],
    num_tools: int,
) -> Tuple[str, List[Dict[str, Any]], List[Dict[str, Any]]]:
    current_tools = current_tool_name_action_map(dp, num_tools)
    k = max(1, int(KNOWLEDGE_PROMPT_TOP_K))
    tool_rows = tool_name_memory_rows(memory_rows, current_tools, top_k=k)
    state_rows = state_memory_rows(selected_items, current_tools, num_tools=num_tools, top_k=k)

    lines: List[str] = []
    if not tool_rows and not state_rows:
        return "", tool_rows, state_rows

    lines.append("Memory weak support only; never override current deadline/load/allowed candidates.")

    if tool_rows:
        for rank, row in enumerate(tool_rows, start=1):
            item = row["item"]
            match = row["current_match"]
            hist_name = str(row.get("history_tool_name", "") or "unknown_tool")
            cur_name = str(match.get("display_name", match.get("name", "current_tool")))
            sim = float(row.get("similarity", 0.0))
            line = (
                f"Tool memory {rank}: hist_tool={hist_name} -> current_candidate={cur_name}; "
                f"name_sim={sim:.3f}; evidence=({evidence_text(item)})."
            )
            row["applied_prompt_line"] = line
            lines.append(line)
    elif current_tools:
        lines.append("Tool memory: no reliable tool-name match above threshold.")
    else:
        lines.append("Tool memory: no real current tool names available.")

    if state_rows:
        for rank, row in enumerate(state_rows, start=1):
            item = row["item"]
            typ = str(row.get("action_type", "unknown")).upper()
            if typ == "TOOL" and row.get("current_match") is not None and row.get("current_action_index") is not None:
                match = row["current_match"]
                hist_name = str(row.get("history_tool_name", "") or "unknown_tool")
                cur_name = str(match.get("display_name", match.get("name", "current_tool")))
                sim = float(match.get("similarity", 0.0))
                line = (
                    f"State memory {rank}: similar state favored TOOL {hist_name} -> {cur_name}; "
                    f"name_sim={sim:.3f}; evidence=({evidence_text(item)})."
                )
            elif typ == "LOCAL" and row.get("current_action_index") is not None:
                line = f"State memory {rank}: similar state favored LOCAL; evidence=({evidence_text(item)})."
            elif typ == "PAUSE" and row.get("current_action_index") is not None:
                line = f"State memory {rank}: similar state favored PAUSE; evidence=({evidence_text(item)})."
            elif typ == "TOOL":
                hist_name = str(row.get("history_tool_name", "") or "unknown_tool")
                line = (
                    f"State memory {rank}: similar state favored TOOL {hist_name}, but no reliable current mapping; "
                    f"evidence=({evidence_text(item)})."
                )
            else:
                line = f"State memory {rank}: similar state favored {typ}; evidence=({evidence_text(item)})."
            row["applied_prompt_line"] = line
            lines.append(line)

    text = "\n".join(lines)
    max_chars = max(256, int(KNOWLEDGE_PROMPT_TOTAL_MAX_CHARS))
    if len(text) > max_chars:
        cut = text[:max_chars].rstrip()
        if "\n" in cut:
            cut = cut.rsplit("\n", 1)[0].rstrip()
        text = cut + "\nMemory text truncated by compact prompt limit."
    return text, tool_rows, state_rows


def memory_support_block(knowledge_prompt: str) -> str:
    text = str(knowledge_prompt or "").strip()
    if not text:
        return "Memory support: none."
    lines = ["Memory support:"]
    for line in text.splitlines():
        line = line.strip()
        if line:
            lines.append(f"- {line}")
    return "\n".join(lines)


# ============================================================
# Trace filtering and output construction
# ============================================================


def task_id_matches(task_id: str, target_task: str) -> bool:
    task_id = str(task_id)
    target = str(target_task)
    if task_id == target:
        return True
    if target.isdigit():
        return f"_{int(target):06d}_" in task_id or f"_{int(target)}_" in task_id or task_id.endswith(f"_{int(target):06d}")
    return target in task_id


def read_target_traces(trace_path: Path, epoch: int, target_task: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with Path(trace_path).expanduser().open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if int(obj.get("epoch", -999999)) != int(epoch):
                continue
            if not task_id_matches(str(obj.get("task_id", "")), str(target_task)):
                continue
            obj["_trace_line_no"] = int(line_no)
            rows.append(obj)
    rows.sort(key=lambda x: (int(x.get("step_count", 0) or 0), int(x.get("node_id", 0) or 0), x.get("_trace_line_no", 0)))
    return rows


def _line_norm(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip())


def prompt_text_matches(a: str, b: str) -> bool:
    return _line_norm(a) == _line_norm(b)


def build_node_output(trace: Dict[str, Any], memory_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    action_names = list(trace.get("action_names", []) or [])
    if len(action_names) >= 2:
        num_tools = len(action_names) - 2
    else:
        mask = list(trace.get("prompt_allowed_tool_mask", []) or [])
        num_tools = len(mask)

    dp = reconstruct_decision_point(trace)
    if num_tools <= 0:
        num_tools = len(list(dp.get("allowed_tools_mask", []) or []))

    query_embedding = build_state_query_embedding(dp, num_tools=num_tools, target_dim=MEMORY_VECTOR_DIM_FALLBACK)
    queue_class = infer_queue_class(dp)
    slack_class = infer_slack_class(dp)
    selected = query_memory_prototypes(
        memory_rows=memory_rows,
        query_embedding=query_embedding,
        node_type_id=int(dp.get("node_type_id", 0)),
        queue_class=queue_class,
        slack_class=slack_class,
        k=K_RETRIEVE,
    )
    rematched_prompt, tool_rows, state_rows = build_mapped_memory_guidance(memory_rows, selected, dp, num_tools=num_tools)
    trace_prompt = str(trace.get("knowledge_prompt", "") or "").strip()
    unique_prompt = trace_prompt if trace_prompt else rematched_prompt

    out: Dict[str, Any] = {
        "trace_line_no": trace.get("_trace_line_no"),
        "trace_id": trace.get("trace_id"),
        "epoch": trace.get("epoch"),
        "step_count": trace.get("step_count"),
        "env_time": trace.get("env_time"),
        "task_id": trace.get("task_id"),
        "node_id": int(trace.get("node_id", 0) or 0),
        "node_type_id": int(trace.get("node_type_id", 0) or 0),
        "exec_action": trace.get("exec_action"),
        "exec_action_name": action_names[int(trace.get("exec_action", -1))] if str(trace.get("exec_action", "")).lstrip("-").isdigit() and 0 <= int(trace.get("exec_action", -1)) < len(action_names) else None,
        "generated_action_text": trace.get("generated_action_text"),
        "mission_success": trace.get("mission_success"),
        "node_success": trace.get("node_success"),
        "violation": trace.get("violation"),
        "num_tools": int(num_tools),
        "action_names": action_names,
        "valid_actions": trace.get("valid_actions", []),
        "prompt_allowed_tool_mask": trace.get("prompt_allowed_tool_mask", []),
        "reconstructed_dp_summary": {
            "node_type_id": dp.get("node_type_id"),
            "task_remaining_deadline_s": dp.get("task_remaining_deadline_s"),
            "node_remaining_deadline_s": dp.get("node_remaining_deadline_s"),
            "queue_class": int(queue_class),
            "slack_class": int(slack_class),
            "tool_options": dp.get("tool_options", []),
            "local_option": dp.get("local_option", {}),
        },
        "trace_applied_knowledge_prompt": trace_prompt,
        "rematched_knowledge_prompt": rematched_prompt,
        "trace_and_rematched_prompt_exact_match": bool(prompt_text_matches(trace_prompt, rematched_prompt)) if trace_prompt or rematched_prompt else True,
        "unique_skill_prompt_text": unique_prompt,
        "skill_actual_llm_prompt_format": memory_support_block(unique_prompt),
        "selected_state_prototypes_top_k": [
            {
                "rank": i + 1,
                "retrieval_score": item.get("_retrieval_score"),
                "memory": compact_memory_ref(item),
            }
            for i, item in enumerate(selected)
        ],
        "tool_memory_match": None,
        "state_memory_match": None,
    }

    if tool_rows:
        row = tool_rows[0]
        out["tool_memory_match"] = {
            "rank": 1,
            "applied_prompt_line": row.get("applied_prompt_line", ""),
            "history_tool_name": row.get("history_tool_name"),
            "current_candidate": (row.get("current_match") or {}).get("display_name"),
            "current_action_index": row.get("current_action_index"),
            "name_similarity": row.get("similarity"),
            "memory": compact_memory_ref(row.get("item")),
        }

    if state_rows:
        row = state_rows[0]
        out["state_memory_match"] = {
            "rank": 1,
            "applied_prompt_line": row.get("applied_prompt_line", ""),
            "action_type": row.get("action_type"),
            "history_tool_name": row.get("history_tool_name"),
            "current_candidate": (row.get("current_match") or {}).get("display_name") if isinstance(row.get("current_match"), dict) else None,
            "current_action_index": row.get("current_action_index"),
            "name_similarity": (row.get("current_match") or {}).get("similarity") if isinstance(row.get("current_match"), dict) else None,
            "retrieval_score": (row.get("item") or {}).get("_retrieval_score") if isinstance(row.get("item"), dict) else None,
            "memory": compact_memory_ref(row.get("item")),
        }

    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Rematch node-level skill/memory for a target traced task.")
    parser.add_argument("--trace-path", type=Path, default=DEFAULT_TRACE_PATH)
    parser.add_argument("--new-memory-path", type=Path, default=DEFAULT_NEW_MEMORY_PATH)
    parser.add_argument("--old-memory-path", type=Path, default=DEFAULT_OLD_MEMORY_PATH)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--epoch", type=int, default=75)
    parser.add_argument("--task-id", type=str, default="1090", help="Exact task_id or numeric id such as 1090.")
    parser.add_argument(
        "--retrieval-store",
        type=str,
        default="new",
        choices=["new", "old", "combined"],
        help=(
            "new = use the run memory.json as the training-equivalent store; old = use only old memory; "
            "combined = union old+new, preferring new rows on duplicate ids."
        ),
    )
    args = parser.parse_args()

    if not args.trace_path.expanduser().exists():
        raise FileNotFoundError(f"trace file not found: {args.trace_path}")
    if not args.new_memory_path.expanduser().exists():
        raise FileNotFoundError(f"new memory file not found: {args.new_memory_path}")
    if args.old_memory_path and not args.old_memory_path.expanduser().exists():
        print(f"[warn] old memory file not found; continuing without it: {args.old_memory_path}")
        old_path: Optional[Path] = None
    else:
        old_path = args.old_memory_path

    memory_rows, locations_by_uid = build_memory_index(args.new_memory_path, old_path, retrieval_store=args.retrieval_store)
    target_traces = read_target_traces(args.trace_path, epoch=args.epoch, target_task=args.task_id)
    if not target_traces:
        raise RuntimeError(f"No traces found for epoch={args.epoch}, task_id={args.task_id!r} in {args.trace_path}")

    node_outputs = [build_node_output(trace, memory_rows) for trace in target_traces]
    exact_matches = sum(1 for row in node_outputs if row.get("trace_and_rematched_prompt_exact_match"))

    result: Dict[str, Any] = {
        "target": {
            "epoch": int(args.epoch),
            "task_id_query": str(args.task_id),
            "matched_task_ids": sorted({str(x.get("task_id")) for x in target_traces}),
        },
        "input_paths": {
            "trace_path": str(args.trace_path),
            "new_memory_path": str(args.new_memory_path),
            "old_memory_path": str(args.old_memory_path) if args.old_memory_path else None,
            "output_path": str(args.output_path),
        },
        "matching_config": {
            "retrieval_store": args.retrieval_store,
            "memory_vector_dim": MEMORY_VECTOR_DIM_FALLBACK,
            "k_retrieve": K_RETRIEVE,
            "knowledge_prompt_top_k": KNOWLEDGE_PROMPT_TOP_K,
            "memory_tool_name_min_similarity": MEMORY_TOOL_NAME_MIN_SIMILARITY,
            "knowledge_prompt_total_max_chars": KNOWLEDGE_PROMPT_TOTAL_MAX_CHARS,
            "note": (
                "trace_applied_knowledge_prompt is the runtime prompt text if present in traces.jsonl; "
                "rematched_knowledge_prompt is reconstructed from the selected memory store."
            ),
        },
        "memory_inventory": {
            "retrieval_rows": len(memory_rows),
            "unique_memory_ids_with_locations": len(locations_by_uid),
            "new_memory_rows_in_retrieval_store": sum(1 for r in memory_rows if r.get("_source_label") == "new_memory"),
            "old_memory_rows_in_retrieval_store": sum(1 for r in memory_rows if r.get("_source_label") == "old_memory"),
        },
        "summary": {
            "num_trace_decisions": len(target_traces),
            "num_unique_nodes": len({int(x.get("node_id", -1)) for x in target_traces}),
            "trace_rematch_exact_prompt_matches": int(exact_matches),
            "trace_rematch_exact_prompt_match_rate": float(exact_matches / max(1, len(node_outputs))),
        },
        "nodes": node_outputs,
    }

    args.output_path.expanduser().parent.mkdir(parents=True, exist_ok=True)
    args.output_path.expanduser().write_text(json.dumps(result, ensure_ascii=False, indent=2, default=json_default), encoding="utf-8")
    print(f"[done] wrote {args.output_path.expanduser().resolve()}")
    print(
        f"[summary] decisions={len(target_traces)} unique_nodes={result['summary']['num_unique_nodes']} "
        f"exact_prompt_match={exact_matches}/{len(node_outputs)} retrieval_store={args.retrieval_store}"
    )


if __name__ == "__main__":
    main()
