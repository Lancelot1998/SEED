from __future__ import annotations

import copy
import json
import math
import random
import re
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from llm_tdpo_ablation.config.paths import QWEN_MODEL_PATH

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "transformers is required for this file. Please install transformers>=4.38."
    ) from exc


def _auto_device(user_device: str) -> str:
    if user_device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return user_device


def _safe_float(x: object, default: float = 0.0) -> float:
    if x is None:
        return default
    try:
        return float(x)
    except Exception:
        return default



def _is_graph_like_dict(obj: object) -> bool:
    return isinstance(obj, dict) and "nodes" in obj and "edges" in obj and (
        "graph_id" in obj or "sample_id" in obj or "num_nodes" in obj
    )


def _normalize_graph_dict(
    graph: Dict[str, object],
    fallback_graph_id: str,
    sample_meta: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    g = copy.deepcopy(graph)
    g["graph_id"] = str(g.get("graph_id") or fallback_graph_id)

    nodes = list(g.get("nodes", []))
    num_tool_types = 0
    for node in nodes:
        mask = list(node.get("allowed_tools_mask", []))
        num_tool_types = max(num_tool_types, len(mask))

    for idx, node in enumerate(nodes):
        node["node_id"] = int(node.get("node_id", idx))
        node["node_type_id"] = int(node.get("node_type_id", 0))
        node["computation_cycles"] = float(node.get("computation_cycles", 0.0))
        node["uplink_size_bits"] = int(node.get("uplink_size_bits", 0))
        node["downlink_size_bits"] = int(node.get("downlink_size_bits", 0))
        node["node_deadline"] = float(node.get("node_deadline", 0.0))
        mask = [int(x) for x in list(node.get("allowed_tools_mask", []))]
        if len(mask) < num_tool_types:
            mask = mask + [0] * (num_tool_types - len(mask))
        node["allowed_tools_mask"] = mask

    edges = []
    for edge in list(g.get("edges", [])):
        edges.append({"src": int(edge.get("src", 0)), "dst": int(edge.get("dst", 0))})

    g["nodes"] = nodes
    g["edges"] = edges
    g["num_nodes"] = int(g.get("num_nodes", len(nodes)))
    g["task_deadline"] = float(g.get("task_deadline", 0.0))

    if sample_meta is not None:
        for key in ("sample_id", "sample_index", "intent", "intent_meta", "segmented_intent"):
            if key in sample_meta and key not in g:
                g[key] = copy.deepcopy(sample_meta[key])
    return g


def _graph_from_raw_item(raw_item: Dict[str, object], fallback_graph_id: str) -> Dict[str, object]:
    if isinstance(raw_item, dict) and isinstance(raw_item.get("graph"), dict):
        sample_graph = raw_item["graph"]
        graph_id = str(raw_item.get("sample_id") or sample_graph.get("graph_id") or fallback_graph_id)
        return _normalize_graph_dict(sample_graph, fallback_graph_id=graph_id, sample_meta=raw_item)
    if _is_graph_like_dict(raw_item):
        return _normalize_graph_dict(raw_item, fallback_graph_id=fallback_graph_id)
    raise ValueError("Unsupported dataset item format. Expected a graph dict or a sample dict containing a 'graph' field.")


def _infer_tool_catalog_from_graphs(graphs: List[Dict[str, object]]) -> List[Dict[str, object]]:
    num_tool_types = 0
    for graph in graphs:
        for node in graph.get("nodes", []):
            num_tool_types = max(num_tool_types, len(node.get("allowed_tools_mask", [])))
    return [{"tool_type_id": i, "name": f"tool_type_{i}"} for i in range(num_tool_types)]


def _infer_task_type_specs_from_graphs(
    graphs: List[Dict[str, object]],
    num_tool_types: int,
) -> List[Dict[str, object]]:
    max_node_type_id = -1
    per_type_masks: Dict[int, List[int]] = {}
    for graph in graphs:
        for node in graph.get("nodes", []):
            node_type_id = int(node.get("node_type_id", 0))
            max_node_type_id = max(max_node_type_id, node_type_id)
            if node_type_id not in per_type_masks:
                per_type_masks[node_type_id] = [0] * num_tool_types
            mask = [int(x) for x in list(node.get("allowed_tools_mask", []))[:num_tool_types]]
            if len(mask) < num_tool_types:
                mask = mask + [0] * (num_tool_types - len(mask))
            per_type_masks[node_type_id] = [max(a, b) for a, b in zip(per_type_masks[node_type_id], mask)]
    return [
        {
            "task_type_id": i,
            "name": f"task_type_{i}",
            "allowed_tools_mask": copy.deepcopy(per_type_masks.get(i, [0] * num_tool_types)),
        }
        for i in range(max_node_type_id + 1)
    ]


def _normalize_dataset_structure(
    dataset_or_item: Dict[str, object],
    fallback_dataset_name: str = "normalized_dataset",
) -> Dict[str, object]:
    base = dataset_or_item if isinstance(dataset_or_item, dict) else {}
    if isinstance(base, dict) and "graphs" in base:
        graphs = [
            _graph_from_raw_item(graph, fallback_graph_id=f"graph_{i:06d}")
            for i, graph in enumerate(list(base.get("graphs", [])))
        ]
        schema_version = str(base.get("schema_version", "2.0.0"))
        dataset_name = str(base.get("dataset_name", fallback_dataset_name))
        config = copy.deepcopy(base.get("config", {}))
        tool_catalog = copy.deepcopy(base.get("tool_catalog", []))
        task_type_specs = copy.deepcopy(base.get("task_type_specs", []))
    else:
        graph = _graph_from_raw_item(base, fallback_graph_id="graph_000000")
        graphs = [graph]
        schema_version = str(base.get("schema_version", "2.0.0"))
        dataset_name = str(base.get("dataset_name", graph["graph_id"]))
        config = {}
        tool_catalog = []
        task_type_specs = []

    if not tool_catalog:
        tool_catalog = _infer_tool_catalog_from_graphs(graphs)
    num_tool_types = len(tool_catalog)

    for graph in graphs:
        for node in graph.get("nodes", []):
            mask = [int(x) for x in list(node.get("allowed_tools_mask", []))]
            if len(mask) < num_tool_types:
                mask = mask + [0] * (num_tool_types - len(mask))
            elif len(mask) > num_tool_types:
                mask = mask[:num_tool_types]
            node["allowed_tools_mask"] = mask

    if not task_type_specs:
        task_type_specs = _infer_task_type_specs_from_graphs(graphs, num_tool_types)

    return {
        "schema_version": schema_version,
        "dataset_name": dataset_name,
        "config": config,
        "task_type_specs": task_type_specs,
        "tool_catalog": tool_catalog,
        "graphs": graphs,
    }


@dataclass
class KnowledgeConfig:
    model_path: str = QWEN_MODEL_PATH
    tokenizer_path: Optional[str] = None
    dtype: str = "bfloat16"
    llm_max_length: int = 1024
    llm_batch_size: int = 1
    freeze_llm: bool = True
    use_llm_verbalizer: bool = True
    verbalize_on_update: bool = True
    verbalize_max_new_tokens: int = 128
    verbalize_use_chat_template: bool = True
    verbalize_clean_max_sentences: int = 4
    verbalize_fallback_to_template: bool = True
    verbalize_strip_role_markers: bool = True
    verbalize_top_k_actions: int = 5
    verbalize_include_all_aspects: bool = True
    verbalize_action_temperature: float = 1.0

    use_aspect_adapters: bool = True
    aspect_names: Tuple[str, str, str, str] = ("topo", "wire", "tool", "out")
    adapter_layer_ids: Tuple[int, ...] = (-8, -6, -4, -2)
    adapter_bottleneck: int = 64
    adapter_dropout: float = 0.05
    adapter_scale: float = 1.0
    adapter_init_zero_up: bool = True

    d_text_proj: int = 256
    d_g: int = 64
    d_s: int = 64
    d_a: int = 64
    d_o: int = 32
    d_k: int = 64
    d_v: int = 128
    d_atom: int = 256

    hidden_g1: int = 128
    hidden_g2: int = 64
    hidden_s1: int = 128
    hidden_s2: int = 64
    hidden_a1: int = 128
    hidden_a2: int = 64
    hidden_o1: int = 64

    lr: float = 2e-4
    weight_decay: float = 1e-5
    lambda_bind_pos: float = 0.25
    lambda_sep: float = 0.20
    lambda_block: float = 0.08
    lambda_div: float = 0.03
    lambda_atom: float = 0.50
    enable_disentangle_losses: bool = False

    # Loss balancing for dynamic-tool knowledge pretraining.
    # Topology/wire losses are multi-objective sums, so they are averaged by default.
    # Tool loss is masked to observed actions so unexecuted actions are not treated as zero-gain labels.
    normalize_topo_wire_losses: bool = True
    use_masked_tool_loss: bool = True
    loss_topo_weight: float = 1.0
    loss_wire_weight: float = 1.0
    loss_tool_weight: float = 3.0
    loss_out_weight: float = 1.0
    loss_need_weight: float = 1.0

    tau_assign: float = 0.75
    k_retrieve: int = 3
    max_buffer_atoms: int = 50000
    max_prototypes: int = 4096
    max_new_prototypes_to_verbalize_per_fit: int = 16

    batch_size: int = 4
    epochs_per_fit: int = 4
    shuffle_segments_each_epoch: bool = True
    device: str = "auto"
    seed: int = 20260327
    verbose: bool = True
    print_prefix: str = "[knowledge-adapter]"
    log_every_fit_step: int = 10
    train_log_mode: str = "epoch"  # epoch, step, both

    # Conservative high-speed TDPO-time update controls.
    # When fast_update_text_encoder=True, update-time training and atom construction
    # use deterministic numeric embeddings derived from the segment instead of four
    # Qwen text-encoding passes. This keeps the memory/prototype interface unchanged
    # while making periodic knowledge updates practical during TDPO.
    fast_update_text_encoder: bool = False
    fit_max_segments: int = 0
    fit_max_new_atoms: int = 0
    freeze_llm_adapters_during_fit: bool = False


@dataclass
class KnowledgeAtom:
    atom_id: str
    node_type_id: int
    depth_tier: int
    role_id: int
    critical_flag: int
    queue_class: int
    slack_class: int
    no_tool_risk_prior: float
    action_gain_priors: List[float]
    confidence: float
    embedding: List[float]
    prompt_text: str = ""
    action_counts: List[int] = field(default_factory=list)
    action_probs: List[float] = field(default_factory=list)
    action_detail_stats: List[Dict[str, float]] = field(default_factory=list)
    aspect_weights: List[float] = field(default_factory=list)
    wire_factor_scores: List[float] = field(default_factory=list)
    outcome_probs: List[float] = field(default_factory=list)
    state_context: Dict[str, object] = field(default_factory=dict)
    environment_context: Dict[str, object] = field(default_factory=dict)
    candidate_actions: List[Dict[str, object]] = field(default_factory=list)
    recommended_action: Dict[str, object] = field(default_factory=dict)
    observed_outcome: Dict[str, object] = field(default_factory=dict)


@dataclass
class Prototype:
    prototype_id: str
    node_type_id: int
    depth_tier: int
    queue_class: int
    slack_class: int
    centroid: List[float]
    support: int
    no_tool_risk_mean: float
    action_gain_means: List[float]
    action_counts: List[int]
    confidence_mean: float
    prompt_text: str = ""
    action_probs: List[float] = field(default_factory=list)
    action_detail_stats: List[Dict[str, float]] = field(default_factory=list)
    aspect_weights: List[float] = field(default_factory=list)
    wire_factor_scores: List[float] = field(default_factory=list)
    outcome_probs: List[float] = field(default_factory=list)
    state_context: Dict[str, object] = field(default_factory=dict)
    environment_context: Dict[str, object] = field(default_factory=dict)
    candidate_actions: List[Dict[str, object]] = field(default_factory=list)
    recommended_action: Dict[str, object] = field(default_factory=dict)
    observed_outcome: Dict[str, object] = field(default_factory=dict)


class DatasetIndex:
    def __init__(self, dataset: Dict[str, object]) -> None:
        self.dataset = _normalize_dataset_structure(dataset)
        dataset = self.dataset
        self.graph_map = {g["graph_id"]: g for g in dataset["graphs"]}
        self.num_tool_types = len(dataset["tool_catalog"])
        self.num_node_types = len(dataset["task_type_specs"])

        self.node_maps: Dict[str, Dict[int, Dict[str, object]]] = {}
        self.predecessors: Dict[str, Dict[int, List[int]]] = {}
        self.successors: Dict[str, Dict[int, List[int]]] = {}
        self.depths: Dict[str, Dict[int, int]] = {}
        self.critical_flags: Dict[str, Dict[int, int]] = {}
        self.roles: Dict[str, Dict[int, int]] = {}

        for graph in dataset["graphs"]:
            gid = graph["graph_id"]
            node_map = {int(n["node_id"]): n for n in graph["nodes"]}
            preds = {nid: [] for nid in node_map}
            succs = {nid: [] for nid in node_map}
            for e in graph["edges"]:
                u = int(e["src"])
                v = int(e["dst"])
                succs[u].append(v)
                preds[v].append(u)
            self.node_maps[gid] = node_map
            self.predecessors[gid] = preds
            self.successors[gid] = succs
            self.depths[gid] = self._compute_depths(gid)
            self.critical_flags[gid] = self._compute_critical_flags(gid)
            self.roles[gid] = self._compute_roles(gid)

    def _compute_depths(self, graph_id: str) -> Dict[int, int]:
        preds = self.predecessors[graph_id]
        nodes = sorted(preds.keys())
        depth = {nid: 0 for nid in nodes}
        for nid in nodes:
            if preds[nid]:
                depth[nid] = 1 + max(depth[p] for p in preds[nid])
        return depth

    def _compute_critical_flags(self, graph_id: str) -> Dict[int, int]:
        succs = self.successors[graph_id]
        nodes = sorted(succs.keys(), reverse=True)
        longest = {nid: 0 for nid in succs}
        for nid in nodes:
            if succs[nid]:
                longest[nid] = 1 + max(longest[s] for s in succs[nid])
        max_len = max(longest.values()) if longest else 0
        return {nid: int(longest[nid] == max_len) for nid in succs}

    def _compute_roles(self, graph_id: str) -> Dict[int, int]:
        preds = self.predecessors[graph_id]
        succs = self.successors[graph_id]
        roles = {}
        for nid in preds:
            indeg = len(preds[nid])
            outdeg = len(succs[nid])
            if indeg > 1:
                roles[nid] = 2
            elif outdeg > 1:
                roles[nid] = 1
            else:
                roles[nid] = 0
        return roles

    def node_static_features(self, graph_id: str, node_id: int) -> Dict[str, float]:
        node = self.node_maps[graph_id][node_id]
        depth = self.depths[graph_id][node_id]
        max_depth = max(self.depths[graph_id].values()) if self.depths[graph_id] else 1
        return {
            "node_type_id": int(node["node_type_id"]),
            "depth_ratio": float(depth / max(1, max_depth)),
            "indeg": float(len(self.predecessors[graph_id][node_id])),
            "outdeg": float(len(self.successors[graph_id][node_id])),
            "critical_flag": float(self.critical_flags[graph_id][node_id]),
            "role_id": float(self.roles[graph_id][node_id]),
            "comp_cycles": float(node["computation_cycles"]),
            "uplink_bits": float(node["uplink_size_bits"]),
            "downlink_bits": float(node["downlink_size_bits"]),
            "node_deadline": float(node["node_deadline"]),
        }


class SegmentExtractor:
    def __init__(self, dataset_index: DatasetIndex, action_dim: int) -> None:
        self.dataset_index = dataset_index
        self.action_dim = action_dim

    def extract(self, task_records: List[Dict[str, object]]) -> List[Dict[str, object]]:
        segments: List[Dict[str, object]] = []
        for task_rec in task_records:
            graph_id = str(task_rec["task_id"])
            task_latency = _safe_float(task_rec.get("latency_s"), 0.0)
            task_status = str(task_rec.get("status", "failed"))
            restart_count = int(task_rec.get("restart_count", 0))
            task_deadline = _safe_float(self.dataset_index.graph_map[graph_id]["task_deadline"], 1.0)
            task_success = 1 if task_status == "completed" else 0
            base_return = (1.0 - min(1.0, task_latency / max(task_deadline, 1e-6))) if task_success else -1.0
            task_return = base_return - 0.05 * restart_count
            by_node: Dict[int, List[Dict[str, object]]] = {}
            for ev in task_rec.get("log", []):
                if ev["node_id"] is None:
                    continue
                nid = int(ev["node_id"])
                by_node.setdefault(nid, []).append(ev)
            for node_id, events in by_node.items():
                events = sorted(events, key=lambda x: _safe_float(x.get("time"), 0.0))
                seg = self._build_segment(graph_id, node_id, events, task_return, task_success)
                if seg is not None:
                    segments.append(seg)
        return segments

    def _queue_class(self, q: float) -> int:
        if q < 0.5:
            return 0
        if q < 2.0:
            return 1
        return 2

    def _slack_class(self, s: float) -> int:
        if s < 0.33:
            return 0
        if s < 0.66:
            return 1
        return 2

    def _blank_action_detail_acc(self) -> List[Dict[str, float]]:
        return [
            {
                "decisions": 0.0,
                "successes": 0.0,
                "failures": 0.0,
                "queue_full_rejects": 0.0,
                "total_s_sum": 0.0,
                "exec_s_sum": 0.0,
                "uplink_s_sum": 0.0,
                "downlink_s_sum": 0.0,
                "queue_s_sum": 0.0,
                "risk_sum": 0.0,
                "packet_loss_sum": 0.0,
            }
            for _ in range(self.action_dim)
        ]

    def _record_action_decision(
        self,
        acc: List[Dict[str, float]],
        action_index: int,
        *,
        total_s: float = 0.0,
        exec_s: float = 0.0,
        uplink_s: float = 0.0,
        downlink_s: float = 0.0,
        queue_s: float = 0.0,
        risk: float = 0.0,
        packet_loss: float = 0.0,
    ) -> None:
        if not (0 <= int(action_index) < self.action_dim):
            return
        row = acc[int(action_index)]
        row["decisions"] += 1.0
        row["total_s_sum"] += max(0.0, float(total_s))
        row["exec_s_sum"] += max(0.0, float(exec_s))
        row["uplink_s_sum"] += max(0.0, float(uplink_s))
        row["downlink_s_sum"] += max(0.0, float(downlink_s))
        row["queue_s_sum"] += max(0.0, float(queue_s))
        row["risk_sum"] += max(0.0, min(1.0, float(risk)))
        row["packet_loss_sum"] += max(0.0, min(1.0, float(packet_loss)))

    def _record_action_success(self, acc: List[Dict[str, float]], action_index: int) -> None:
        if 0 <= int(action_index) < self.action_dim:
            acc[int(action_index)]["successes"] += 1.0

    def _record_action_failure(self, acc: List[Dict[str, float]], action_index: int) -> None:
        if 0 <= int(action_index) < self.action_dim:
            acc[int(action_index)]["failures"] += 1.0

    def _record_queue_reject(self, acc: List[Dict[str, float]], action_index: int) -> None:
        if 0 <= int(action_index) < self.action_dim:
            acc[int(action_index)]["queue_full_rejects"] += 1.0

    def _finalize_action_detail_stats(self, acc: List[Dict[str, float]]) -> List[Dict[str, float]]:
        out: List[Dict[str, float]] = []
        for row in acc:
            decisions = max(0.0, float(row.get("decisions", 0.0)))
            denom = max(1.0, decisions)
            successes = max(0.0, float(row.get("successes", 0.0)))
            failures = max(0.0, float(row.get("failures", 0.0)))
            out.append({
                "decisions": decisions,
                "successes": successes,
                "failures": failures,
                "queue_full_rejects": max(0.0, float(row.get("queue_full_rejects", 0.0))),
                "success_rate": successes / max(1.0, decisions),
                "failure_rate": failures / max(1.0, decisions),
                "mean_total_s": float(row.get("total_s_sum", 0.0)) / denom,
                "mean_exec_s": float(row.get("exec_s_sum", 0.0)) / denom,
                "mean_uplink_s": float(row.get("uplink_s_sum", 0.0)) / denom,
                "mean_downlink_s": float(row.get("downlink_s_sum", 0.0)) / denom,
                "mean_comm_s": (float(row.get("uplink_s_sum", 0.0)) + float(row.get("downlink_s_sum", 0.0))) / denom,
                "mean_queue_s": float(row.get("queue_s_sum", 0.0)) / denom,
                "mean_risk": float(row.get("risk_sum", 0.0)) / denom,
                "mean_packet_loss": float(row.get("packet_loss_sum", 0.0)) / denom,
            })
        return out

    def _json_safe(self, value: object):
        try:
            return json.loads(json.dumps(value, ensure_ascii=False, default=str))
        except Exception:
            return copy.deepcopy(value)

    def _first_decision_context(self, events: List[Dict[str, object]]) -> Dict[str, object]:
        for ev in events:
            et = str(ev.get("event_type", ""))
            if not et.startswith("decision_"):
                continue
            payload = ev.get("payload", {}) or {}
            ctx = payload.get("decision_context", {})
            if isinstance(ctx, dict) and ctx:
                return self._json_safe(ctx)
        return {}

    def _split_intent_text(self, intent_text: str) -> Dict[str, str]:
        text = str(intent_text or "")
        node_step = ""
        mission_context = ""
        if "Node step:" in text:
            after = text.split("Node step:", 1)[1]
            if "Mission context:" in after:
                node_step, rest = after.split("Mission context:", 1)
                mission_context = rest.split("Node type:", 1)[0].strip()
            else:
                node_step = after.split("Node type:", 1)[0].strip()
        elif text:
            node_step = text.split("\n", 1)[0].strip()
        return {"node_step": node_step.strip(), "mission_context": mission_context.strip()}

    def _action_index_from_action(self, action: Dict[str, object]) -> int:
        at = str(action.get("action_type", ""))
        if at == "local":
            return self.action_dim - 2
        if at == "pause":
            return self.action_dim - 1
        try:
            return int(action.get("slot", action.get("tool_type_id", -1)))
        except Exception:
            return -1

    def _action_label_from_action(self, action: Dict[str, object], fallback_idx: int = -1) -> str:
        at = str(action.get("action_type", ""))
        if at == "local":
            return "LOCAL execution"
        if at == "pause":
            return "PAUSE"
        slot = action.get("slot", action.get("tool_type_id", fallback_idx))
        name = action.get("tool_name") or action.get("name") or f"tool {slot}"
        cat = action.get("category", "unknown")
        return f"TOOL slot {slot}: {name} (category={cat})"

    def _candidate_actions_from_context(self, ctx: Dict[str, object]) -> List[Dict[str, object]]:
        raw_actions = ctx.get("candidate_actions")
        if not isinstance(raw_actions, list):
            raw_actions = []
            raw_actions.extend(ctx.get("tool_options", []) if isinstance(ctx.get("tool_options"), list) else [])
            if isinstance(ctx.get("local_option"), dict):
                raw_actions.append(ctx.get("local_option"))
            raw_actions.append({"action_type": "pause", "pause_duration_s": 0.2})
        keep_keys = {
            "action_type", "slot", "tool_type_id", "name", "tool_name", "category", "description",
            "compute_frequency_hz", "uplink_bandwidth_hz", "downlink_bandwidth_hz", "coverage_distance_m",
            "server_reliability", "info_richness", "validity_horizon_s", "semantic_gain", "completion_gain",
            "success_probability", "energy_j", "requires_tool_to_complete", "source", "queue_length", "running_jobs",
            "queue_capacity", "instances", "queue_blocked", "uplink_s", "downlink_s", "exec_s",
            "base_queue_delay_s", "estimated_wait_s", "predicted_total_s", "risk_estimate", "packet_loss_rate",
            "congestion_ratio", "compute_s", "fail_probability", "local_running_jobs", "local_max_concurrency",
            "local_queue_capacity", "allowed_reason",
        }
        out: List[Dict[str, object]] = []
        for row in raw_actions:
            if not isinstance(row, dict):
                continue
            clean = {k: self._json_safe(v) for k, v in row.items() if k in keep_keys}
            if "action_type" not in clean:
                clean["action_type"] = "tool" if "slot" in clean or "tool_type_id" in clean else "unknown"
            idx = self._action_index_from_action(clean)
            clean["action_index"] = int(idx)
            clean["action_label"] = self._action_label_from_action(clean, idx)
            out.append(clean)
        return out

    def _choose_recommended_action(
        self,
        candidate_actions: List[Dict[str, object]],
        action_counts: List[int],
        norm_action_gain: List[float],
        action_stats: List[Dict[str, float]],
    ) -> Dict[str, object]:
        observed_indices = [i for i, c in enumerate(action_counts) if int(c) > 0]
        if observed_indices:
            best_idx = max(
                observed_indices,
                key=lambda i: (
                    float(norm_action_gain[i]),
                    float(action_stats[i].get("success_rate", 0.0)) if i < len(action_stats) else 0.0,
                    -float(action_stats[i].get("mean_total_s", 0.0)) if i < len(action_stats) else 0.0,
                ),
            )
        else:
            best_idx = -1
        by_idx = {self._action_index_from_action(a): a for a in candidate_actions}
        base = copy.deepcopy(by_idx.get(best_idx, {})) if best_idx >= 0 else {}
        if not base and candidate_actions:
            feasible = [a for a in candidate_actions if not bool(a.get("queue_blocked", False))]
            feasible = feasible or candidate_actions
            base = min(feasible, key=lambda a: float(a.get("predicted_total_s", a.get("compute_s", 1e9)) or 1e9))
            best_idx = self._action_index_from_action(base)
        if not base:
            base = {"action_type": "pause", "pause_duration_s": 0.2, "action_index": self.action_dim - 1}
            best_idx = self.action_dim - 1
        base = copy.deepcopy(base)
        base["action_index"] = int(best_idx)
        base["action_label"] = self._action_label_from_action(base, best_idx)
        if str(base.get("action_type")) == "tool":
            slot = int(base.get("slot", base.get("tool_type_id", best_idx)))
            tool_name = str(base.get("tool_name") or base.get("name") or f"tool {slot}")
            base["decision_command"] = f"use_tool(slot={slot}, tool_name='{tool_name}')"
            reason = "best observed action gain among executed actions"
            if float(base.get("predicted_total_s", 0.0) or 0.0) > 0.0:
                reason += f"; predicted_total={float(base.get('predicted_total_s', 0.0)):.3f}s"
            if "server_reliability" in base:
                reason += f"; reliability={float(base.get('server_reliability', 0.0)):.2f}"
            if "completion_gain" in base:
                reason += f"; completion_gain={float(base.get('completion_gain', 0.0)):.2f}"
            base["why_best"] = reason
        elif str(base.get("action_type")) == "local":
            base["decision_command"] = "use_local()"
            base["why_best"] = "local execution had the best observed action gain or the lowest feasible predicted total time"
        else:
            base["decision_command"] = f"pause(duration_s={float(base.get('pause_duration_s', 0.2)):.2f})"
            base["why_best"] = "pause is only recommended when no tool/local candidate is feasible under queue or deadline constraints"
        return base

    def _build_memory_contexts(
        self,
        graph_id: str,
        node_id: int,
        static: Dict[str, float],
        ctx: Dict[str, object],
        queue_class: int,
        slack_class: int,
        candidate_actions: List[Dict[str, object]],
        recommended_action: Dict[str, object],
        action_stats: List[Dict[str, float]],
        node_success: int,
        task_success: int,
        violation: int,
        task_return: float,
    ) -> Tuple[Dict[str, object], Dict[str, object], Dict[str, object]]:
        parts = self._split_intent_text(str(ctx.get("intent_text", "")))
        allowed_slots = ctx.get("allowed_slots", [])
        if not allowed_slots:
            mask = ctx.get("allowed_tools_mask", [])
            allowed_slots = [int(i) for i, x in enumerate(mask) if int(x) == 1] if isinstance(mask, list) else []
        state_context = {
            "graph_id": str(graph_id),
            "node_id": int(node_id),
            "node_type_id": int(static.get("node_type_id", ctx.get("node_type_id", 0))),
            "intent_text": str(ctx.get("intent_text", "")),
            "node_step": parts.get("node_step", ""),
            "mission_context": parts.get("mission_context", ""),
            "allowed_slots": self._json_safe(allowed_slots),
            "requires_tool_to_complete": bool(ctx.get("requires_tool_to_complete", False)),
            "local_success_probability": ctx.get("local_success_probability"),
            "node_deadline_s": float(ctx.get("node_deadline_s", static.get("node_deadline", 0.0)) or 0.0),
            "task_remaining_deadline_s": float(ctx.get("task_remaining_deadline_s", 0.0) or 0.0),
            "node_remaining_deadline_s": float(ctx.get("node_remaining_deadline_s", 0.0) or 0.0),
            "queue_class": int(queue_class),
            "slack_class": int(slack_class),
            "critical_flag": int(static.get("critical_flag", 0)),
            "depth_ratio": float(static.get("depth_ratio", 0.0)),
            "predecessors_satisfied": True,
        }
        local_option = ctx.get("local_option", {}) if isinstance(ctx.get("local_option"), dict) else {}
        tool_risks = [float(a.get("risk_estimate", 0.0) or 0.0) for a in candidate_actions if a.get("action_type") == "tool"]
        tool_losses = [float(a.get("packet_loss_rate", 0.0) or 0.0) for a in candidate_actions if a.get("action_type") == "tool"]
        tool_queues = [float(a.get("base_queue_delay_s", 0.0) or 0.0) for a in candidate_actions if a.get("action_type") == "tool"]
        environment_context = {
            "local_queue_length": int(ctx.get("local_queue_length", local_option.get("queue_length", 0)) or 0),
            "local_running_jobs": int(ctx.get("local_running_jobs", local_option.get("local_running_jobs", 0)) or 0),
            "tool_queue_lengths": self._json_safe(ctx.get("tool_queue_lengths", {})),
            "tool_running_counts": self._json_safe(ctx.get("tool_running_counts", {})),
            "local_compute_s": float(local_option.get("compute_s", 0.0) or 0.0),
            "local_fail_probability": float(local_option.get("fail_probability", 0.0) or 0.0),
            "local_queue_blocked": bool(local_option.get("queue_blocked", False)),
            "local_predicted_total_s": float(local_option.get("predicted_total_s", 0.0) or 0.0),
            "max_tool_queue_delay_s": max(tool_queues) if tool_queues else 0.0,
            "max_tool_risk_estimate": max(tool_risks) if tool_risks else 0.0,
            "max_packet_loss_rate": max(tool_losses) if tool_losses else 0.0,
            "dominant_pressure": "deadline" if int(slack_class) == 0 else ("queue" if int(queue_class) >= 1 else "balanced"),
        }
        rec_idx = int(recommended_action.get("action_index", -1))
        rec_stats = action_stats[rec_idx] if 0 <= rec_idx < len(action_stats) and isinstance(action_stats[rec_idx], dict) else {}
        observed_outcome = {
            "node_success": int(node_success),
            "task_success": int(task_success),
            "violation": int(violation),
            "task_return": float(task_return),
            "recommended_action_index": int(rec_idx),
            "recommended_action_label": recommended_action.get("action_label", "unknown"),
            "recommended_action_gain": float(recommended_action.get("gain", 0.0) or 0.0),
            "mean_total_s": float(rec_stats.get("mean_total_s", 0.0) or 0.0),
            "mean_queue_s": float(rec_stats.get("mean_queue_s", 0.0) or 0.0),
            "mean_risk": float(rec_stats.get("mean_risk", 0.0) or 0.0),
            "mean_packet_loss": float(rec_stats.get("mean_packet_loss", 0.0) or 0.0),
            "success_rate": float(rec_stats.get("success_rate", float(node_success)) or 0.0),
            "decisions": float(rec_stats.get("decisions", 0.0) or 0.0),
        }
        return state_context, environment_context, observed_outcome

    def _build_segment(
        self,
        graph_id: str,
        node_id: int,
        events: List[Dict[str, object]],
        task_return: float,
        task_success: int,
    ) -> Optional[Dict[str, object]]:
        static = self.dataset_index.node_static_features(graph_id, node_id)
        node_deadline = static["node_deadline"]
        first_time = _safe_float(events[0].get("time"), 0.0)
        decision_ctx = self._first_decision_context(events)

        steps: List[Dict[str, object]] = []
        action_counts = [0] * self.action_dim
        action_gain = [0.0] * self.action_dim
        action_detail_acc = self._blank_action_detail_acc()
        node_success = 0
        violation = 0
        tool_invoked = 0

        for idx, ev in enumerate(events):
            et = str(ev.get("event_type", ""))
            p = ev.get("payload", {}) or {}
            t = _safe_float(ev.get("time"), 0.0)
            elapsed = max(0.0, t - first_time)
            slack_ratio = max(0.0, 1.0 - elapsed / max(node_deadline, 1e-6))

            packet_loss = 0.0
            queue_delay = 0.0
            base_queue_delay = 0.0
            risk_estimate = 0.0
            exec_s = 0.0
            progress = 0.0
            call_ok = 0.0
            replan = 0.0
            violation_flag = 0.0
            action_type_id = 0
            action_index = self.action_dim - 1
            tool_id = -1

            if et == "decision_tool_rejected_queue_full":
                rejected_tool_id = int(p.get("tool_type_id", -1))
                if 0 <= rejected_tool_id < self.action_dim:
                    self._record_queue_reject(action_detail_acc, rejected_tool_id)
            if et == "decision_tool":
                action_type_id = 1
                tool_id = int(p.get("tool_type_id", -1))
                if 0 <= tool_id < self.action_dim:
                    action_index = tool_id
                tool_invoked = 1
                packet_loss = _safe_float(p.get("packet_loss_rate"), 0.0)
                base_queue_delay = _safe_float(p.get("base_queue_delay_s"), 0.0)
                risk_estimate = _safe_float(p.get("risk_estimate"), 0.0)
                exec_s = _safe_float(p.get("exec_s"), 0.0)
                uplink_s = _safe_float(p.get("uplink_s"), 0.0)
                downlink_s = _safe_float(p.get("downlink_s"), 0.0)
                total_s = _safe_float(p.get("predicted_total_s"), base_queue_delay + uplink_s + exec_s + downlink_s)
                self._record_action_decision(
                    action_detail_acc,
                    action_index,
                    total_s=total_s,
                    exec_s=exec_s,
                    uplink_s=uplink_s,
                    downlink_s=downlink_s,
                    queue_s=base_queue_delay,
                    risk=risk_estimate,
                    packet_loss=packet_loss,
                )
            elif et == "decision_local":
                action_type_id = 2
                action_index = self.action_dim - 2
                risk_estimate = _safe_float(p.get("fail_probability"), 0.0)
                exec_s = _safe_float(p.get("compute_s"), 0.0)
                wait_s = _safe_float(p.get("estimated_wait_s"), _safe_float(p.get("queue_delay_s"), 0.0))
                total_s = _safe_float(p.get("predicted_total_s"), exec_s + wait_s)
                self._record_action_decision(
                    action_detail_acc,
                    action_index,
                    total_s=total_s,
                    exec_s=exec_s,
                    queue_s=wait_s,
                    risk=risk_estimate,
                    packet_loss=0.0,
                )
            elif et == "decision_pause":
                action_type_id = 3
                action_index = self.action_dim - 1
                exec_s = _safe_float(p.get("pause_duration_s"), 0.0)
                self._record_action_decision(
                    action_detail_acc,
                    action_index,
                    total_s=exec_s,
                    exec_s=exec_s,
                    queue_s=0.0,
                    risk=0.0,
                    packet_loss=0.0,
                )
            elif et == "tool_execution_started":
                queue_delay = _safe_float(p.get("queue_delay_s"), 0.0)
            elif et == "tool_success":
                node_success = 1
                progress = 1.0
                call_ok = 1.0
                success_tool_id = int(p.get("tool_type_id", -1))
                if 0 <= success_tool_id < self.action_dim:
                    self._record_action_success(action_detail_acc, success_tool_id)
                queue_delay = _safe_float(p.get("queue_delay_s"), 0.0)
                risk_estimate = _safe_float(p.get("risk_estimate"), 0.0)
                exec_s = _safe_float(p.get("exec_s"), 0.0)
            elif et == "local_success":
                node_success = 1
                progress = 1.0
                call_ok = 1.0
                self._record_action_success(action_detail_acc, self.action_dim - 2)
                exec_s = _safe_float(p.get("compute_s"), 0.0)
            elif et in {"local_failed", "node_timeout", "task_failed"}:
                violation = 1
                violation_flag = 1.0
                if et == "local_failed":
                    self._record_action_failure(action_detail_acc, self.action_dim - 2)
                if et != "task_failed":
                    replan = 1.0
            elif et == "task_restarted":
                replan = 1.0
                violation_flag = 1.0

            if 0 <= action_index < self.action_dim and et in {"decision_tool", "decision_local", "decision_pause"}:
                action_counts[action_index] += 1
                action_gain[action_index] += task_return

            g_feat = [
                static["depth_ratio"],
                static["indeg"] / 5.0,
                static["outdeg"] / 5.0,
                static["critical_flag"],
                static["role_id"] / 2.0,
                static["node_type_id"] / max(1.0, float(self.dataset_index.num_node_types)),
            ]
            s_feat = [
                slack_ratio,
                packet_loss,
                queue_delay,
                base_queue_delay,
                risk_estimate,
                exec_s / 10.0,
            ]
            a_feat = [
                float(action_type_id) / 3.0,
                float(max(action_index, 0)) / max(1.0, float(self.action_dim - 1)),
                1.0 if tool_id >= 0 else 0.0,
                call_ok,
                progress,
                float(idx) / max(1.0, float(len(events) - 1)),
            ]
            # Decision-time outcome features: do not expose final success/failure labels to the outcome head.
            # Non-decision events remain in the sequence for the other heads, but the outcome head attends only
            # to decision events through out_mask in _build_batch().
            o_feat = [
                slack_ratio,
                min(1.0, base_queue_delay + queue_delay),
                risk_estimate,
                float(action_type_id) / 3.0,
            ]

            y_topo = 0.35 * static["critical_flag"] + 0.35 * (1.0 if static["role_id"] > 0 else 0.0) + 0.30 * float(idx == 0)
            y_wire = max(packet_loss, min(1.0, base_queue_delay / 2.0), min(1.0, queue_delay / 2.0), risk_estimate, 1.0 - slack_ratio)
            y_tool = max(call_ok * progress, 1.0 if et in {"decision_tool", "tool_success"} else 0.0)
            y_out = max(violation_flag, replan, float(idx == len(events) - 1), float(node_success and idx == len(events) - 1))

            steps.append(
                {
                    "g_feat": g_feat,
                    "s_feat": s_feat,
                    "a_feat": a_feat,
                    "o_feat": o_feat,
                    "y": [float(y_topo), float(y_wire), float(y_tool), float(y_out)],
                    "event_type": et,
                    "is_decision_event": bool(et in {"decision_tool", "decision_local", "decision_pause"}),
                }
            )

        if not steps:
            return None

        depth_tier = min(2, int(round(static["depth_ratio"] * 2.0)))
        queue_class = self._queue_class(max([s["s_feat"][2] + s["s_feat"][3] for s in steps] + [0.0]))
        slack_class = self._slack_class(min([s["s_feat"][0] for s in steps] + [1.0]))
        confidence = 0.5 + 0.5 * min(1.0, len(steps) / 8.0)
        no_tool_risk_prior = 1.0 - max(0.0, task_return)
        norm_action_gain = [
            (action_gain[i] / max(1, action_counts[i])) if action_counts[i] > 0 else 0.0
            for i in range(self.action_dim)
        ]

        topo_text = (
            f"Node type {int(static['node_type_id'])}. Depth ratio {static['depth_ratio']:.2f}. "
            f"Role {int(static['role_id'])}. Critical path flag {int(static['critical_flag'])}."
        )
        wire_text = (
            f"Queue class {queue_class}. Slack class {slack_class}. "
            f"Observed max queue delay {max([s['s_feat'][2] + s['s_feat'][3] for s in steps]):.2f}. "
            f"Observed mean risk {sum(s['s_feat'][4] for s in steps)/max(1,len(steps)):.2f}."
        )
        tool_text = (
            f"Tool invoked {int(tool_invoked)}. Action counts {action_counts}. "
            f"Average action gains {[round(x, 3) for x in norm_action_gain[: min(6, len(norm_action_gain))]]}."
        )
        out_text = (
            f"Decision-time outcome context. Queue class {queue_class}. Slack class {slack_class}. "
            f"Tool invoked {int(tool_invoked)}. Action counts {action_counts}. "
            "Final success/failure labels are intentionally not included in this text."
        )

        action_detail_stats = self._finalize_action_detail_stats(action_detail_acc)
        candidate_actions = self._candidate_actions_from_context(decision_ctx)
        recommended_action = self._choose_recommended_action(candidate_actions, action_counts, norm_action_gain, action_detail_stats)
        if 0 <= int(recommended_action.get("action_index", -1)) < len(norm_action_gain):
            recommended_action["gain"] = float(norm_action_gain[int(recommended_action["action_index"])])
        state_context, environment_context, observed_outcome = self._build_memory_contexts(
            graph_id=graph_id,
            node_id=node_id,
            static=static,
            ctx=decision_ctx,
            queue_class=queue_class,
            slack_class=slack_class,
            candidate_actions=candidate_actions,
            recommended_action=recommended_action,
            action_stats=action_detail_stats,
            node_success=node_success,
            task_success=task_success,
            violation=violation,
            task_return=task_return,
        )

        return {
            "graph_id": graph_id,
            "node_id": node_id,
            "static": static,
            "steps": steps,
            "depth_tier": depth_tier,
            "role_id": int(static["role_id"]),
            "critical_flag": int(static["critical_flag"]),
            "queue_class": queue_class,
            "slack_class": slack_class,
            "node_success": node_success,
            "task_success": task_success,
            "violation": violation,
            "tool_invoked": tool_invoked,
            "task_return": task_return,
            "no_tool_risk_prior": no_tool_risk_prior,
            "action_gain": norm_action_gain,
            "action_counts": action_counts,
            "action_observed_mask": [1.0 if int(c) > 0 else 0.0 for c in action_counts],
            "action_detail_stats": action_detail_stats,
            "state_context": state_context,
            "environment_context": environment_context,
            "candidate_actions": candidate_actions,
            "recommended_action": recommended_action,
            "observed_outcome": observed_outcome,
            "confidence": confidence,
            "texts": {
                "topo": topo_text,
                "wire": wire_text,
                "tool": tool_text,
                "out": out_text,
            },
        }



class AspectAdapter(nn.Module):
    def __init__(self, hidden_size: int, bottleneck: int, dropout: float, scale: float, init_zero_up: bool) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.down = nn.Linear(hidden_size, bottleneck, bias=False)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
        self.up = nn.Linear(bottleneck, hidden_size, bias=False)
        self.scale = float(scale)
        nn.init.normal_(self.down.weight, mean=0.0, std=0.02)
        if init_zero_up:
            nn.init.zeros_(self.up.weight)
        else:
            nn.init.normal_(self.up.weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_dtype = x.dtype
        h = self.norm(x.float())
        h = self.down(h)
        h = self.act(h)
        h = self.dropout(h)
        h = self.up(h)
        return (h * self.scale).to(x_dtype)


class LLMBackbone:
    def __init__(self, cfg: KnowledgeConfig) -> None:
        self.cfg = cfg
        self.device = torch.device(_auto_device(cfg.device))
        self._adapter_train_mode = False
        if cfg.dtype == "bfloat16" and torch.cuda.is_available():
            torch_dtype = torch.bfloat16
        elif cfg.dtype == "float16" and torch.cuda.is_available():
            torch_dtype = torch.float16
        else:
            torch_dtype = torch.float32

        tok_path = cfg.tokenizer_path or cfg.model_path
        self.tokenizer = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=True, use_fast=False)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            cfg.model_path,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
        )
        self.model.to(self.device)
        self.model.eval()
        if cfg.freeze_llm:
            for p in self.model.parameters():
                p.requires_grad = False

        self.base_model = self._resolve_base_model(self.model)
        self.layers = self._resolve_layers(self.base_model)
        self.hidden_size = int(self.model.config.hidden_size)
        self.selected_layer_ids = self._resolve_adapter_layer_ids(cfg.adapter_layer_ids, len(self.layers))
        self.aspect_adapters = nn.ModuleDict({
            aspect: nn.ModuleDict({
                str(layer_id): AspectAdapter(
                    hidden_size=self.hidden_size,
                    bottleneck=cfg.adapter_bottleneck,
                    dropout=cfg.adapter_dropout,
                    scale=cfg.adapter_scale,
                    init_zero_up=cfg.adapter_init_zero_up,
                )
                for layer_id in self.selected_layer_ids
            })
            for aspect in cfg.aspect_names
        }).to(self.device)
        for p in self.aspect_adapters.parameters():
            p.requires_grad = True

        self._active_aspect: Optional[str] = None
        self._hook_handles: List[object] = []
        self._register_aspect_hooks()
        if self.cfg.verbose:
            print(
                f"{self.cfg.print_prefix} loaded Qwen backbone from {cfg.model_path} on {self.device} | "
                f"hidden={self.hidden_size} | adapter_layers={self.selected_layer_ids}",
                flush=True,
            )

    def _resolve_base_model(self, model: nn.Module) -> nn.Module:
        if hasattr(model, "model"):
            return model.model
        if hasattr(model, "base_model"):
            return model.base_model
        if hasattr(model, "transformer"):
            return model.transformer
        raise RuntimeError("Unable to resolve the decoder backbone from the loaded LLM.")

    def _resolve_layers(self, base_model: nn.Module) -> nn.ModuleList:
        if hasattr(base_model, "layers"):
            return base_model.layers
        if hasattr(base_model, "h"):
            return base_model.h
        if hasattr(base_model, "decoder") and hasattr(base_model.decoder, "layers"):
            return base_model.decoder.layers
        raise RuntimeError("Unable to locate decoder layers for adapter injection.")

    def _resolve_adapter_layer_ids(self, layer_ids: Sequence[int], n_layers: int) -> List[int]:
        resolved: List[int] = []
        for lid in layer_ids:
            idx = int(lid)
            if idx < 0:
                idx = n_layers + idx
            idx = max(0, min(n_layers - 1, idx))
            resolved.append(idx)
        resolved = sorted(set(resolved))
        if not resolved:
            resolved = [max(0, n_layers - 1)]
        return resolved

    def _register_aspect_hooks(self) -> None:
        for layer_id in self.selected_layer_ids:
            layer = self.layers[layer_id]
            handle = layer.register_forward_hook(self._make_layer_hook(layer_id))
            self._hook_handles.append(handle)

    def _make_layer_hook(self, layer_id: int):
        def _hook(_module: nn.Module, _inputs, output):
            if (not self.cfg.use_aspect_adapters) or (self._active_aspect is None):
                return output
            adapter = self.aspect_adapters[self._active_aspect][str(layer_id)]
            if isinstance(output, tuple):
                hidden = output[0]
                adapted = hidden + adapter(hidden)
                return (adapted,) + output[1:]
            return output + adapter(output)
        return _hook

    def set_adapter_train(self, mode: bool) -> None:
        self._adapter_train_mode = bool(mode)
        self.aspect_adapters.train(mode)
        self.model.eval()

    def trainable_parameters(self) -> List[nn.Parameter]:
        return [p for p in self.aspect_adapters.parameters() if p.requires_grad]

    def adapter_state_dict(self) -> Dict[str, torch.Tensor]:
        return self.aspect_adapters.state_dict()

    def load_adapter_state_dict(self, state_dict: Dict[str, torch.Tensor], strict: bool = True) -> None:
        if state_dict:
            self.aspect_adapters.load_state_dict(state_dict, strict=strict)

    @contextmanager
    def use_aspect(self, aspect: Optional[str]):
        prev = self._active_aspect
        self._active_aspect = aspect if self.cfg.use_aspect_adapters else None
        try:
            yield
        finally:
            self._active_aspect = prev

    def _forward_base(self, enc: Dict[str, torch.Tensor]) -> torch.Tensor:
        with self.use_aspect(self._active_aspect):
            out = self.base_model(
                **enc,
                output_hidden_states=True,
                return_dict=True,
                use_cache=False,
            )
        hidden = getattr(out, "last_hidden_state", None)
        if hidden is None:
            hidden = out.hidden_states[-1]
        return hidden

    def encode_texts(self, texts: Sequence[str], aspect: Optional[str] = None, detach: bool = False) -> torch.Tensor:
        if not texts:
            return torch.zeros((0, self.hidden_size), dtype=torch.float32, device=self.device)
        outputs: List[torch.Tensor] = []
        self.aspect_adapters.train(self._adapter_train_mode)
        for i in range(0, len(texts), self.cfg.llm_batch_size):
            batch = list(texts[i: i + self.cfg.llm_batch_size])
            enc = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.cfg.llm_max_length,
                return_tensors="pt",
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}
            self._active_aspect = aspect
            hidden = self._forward_base(enc).float()
            self._active_aspect = None
            mask = enc["attention_mask"].unsqueeze(-1).float()
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
            outputs.append(pooled.detach() if detach else pooled)
        return torch.cat(outputs, dim=0)

    def _build_verbalize_inputs(self, prompt: str) -> Dict[str, torch.Tensor]:
        if self.cfg.verbalize_use_chat_template and hasattr(self.tokenizer, "apply_chat_template"):
            try:
                messages = [
                    {
                        "role": "system",
                        "content": (
                            "You convert structured execution facts into at most two short plain sentences "
                            "for a downstream decision model. Return only the final answer sentences."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ]
                rendered = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                enc = self.tokenizer(
                    rendered,
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.cfg.llm_max_length,
                )
                return {k: v.to(self.device) for k, v in enc.items()}
            except Exception:
                pass
        enc = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=self.cfg.llm_max_length)
        return {k: v.to(self.device) for k, v in enc.items()}

    @torch.no_grad()
    def verbalize(self, prompt: str, max_new_tokens: Optional[int] = None) -> str:
        max_new = max_new_tokens or self.cfg.verbalize_max_new_tokens
        enc = self._build_verbalize_inputs(prompt)
        eos_ids: List[int] = []
        if self.tokenizer.eos_token_id is not None:
            eos_ids.append(int(self.tokenizer.eos_token_id))
        for tok in ["<|im_end|>", "<|endoftext|>"]:
            try:
                tid = self.tokenizer.convert_tokens_to_ids(tok)
                if isinstance(tid, int) and tid >= 0 and tid not in eos_ids:
                    eos_ids.append(tid)
            except Exception:
                pass
        eos_token_id = eos_ids[0] if len(eos_ids) == 1 else eos_ids
        out = self.model.generate(
            **enc,
            do_sample=False,
            max_new_tokens=max_new,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=eos_token_id,
            use_cache=True,
        )
        gen_tokens = out[0][enc["input_ids"].shape[1]:]
        text = self.tokenizer.decode(gen_tokens, skip_special_tokens=True)
        return text.strip()


class MLP(nn.Module):
    def __init__(self, dims: List[int], dropout: float = 0.0):
        super().__init__()
        layers: List[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LLMFourHeadDistiller(nn.Module):
    def __init__(self, llm_hidden_size: int, action_dim: int, cfg: KnowledgeConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.action_dim = action_dim

        self.text_proj = nn.Linear(llm_hidden_size, cfg.d_text_proj)
        self.phi_g = MLP([6, cfg.hidden_g1, cfg.hidden_g2, cfg.d_g], dropout=0.1)
        self.phi_s = MLP([6, cfg.hidden_s1, cfg.hidden_s2, cfg.d_s], dropout=0.1)
        self.phi_a = MLP([6, cfg.hidden_a1, cfg.hidden_a2, cfg.d_a], dropout=0.1)
        self.phi_o = MLP([4, cfg.hidden_o1, cfg.d_o], dropout=0.1)

        self.query_proj = nn.ModuleDict({
            "topo": nn.Linear(cfg.d_text_proj, cfg.d_k),
            "wire": nn.Linear(cfg.d_text_proj, cfg.d_k),
            "tool": nn.Linear(cfg.d_text_proj, cfg.d_k),
            "out": nn.Linear(cfg.d_text_proj, cfg.d_k),
        })
        self.block_gate = nn.ModuleDict({
            "topo": nn.Linear(cfg.d_text_proj, 4),
            "wire": nn.Linear(cfg.d_text_proj, 4),
            "tool": nn.Linear(cfg.d_text_proj, 4),
            "out": nn.Linear(cfg.d_text_proj, 4),
        })
        self.block_value = nn.ModuleDict({
            "g": nn.Linear(cfg.d_g, cfg.d_v),
            "s": nn.Linear(cfg.d_s, cfg.d_v),
            "a": nn.Linear(cfg.d_a, cfg.d_v),
            "o": nn.Linear(cfg.d_o, cfg.d_v),
        })
        self.key_proj = nn.ModuleDict({
            "topo": nn.Linear(cfg.d_v, cfg.d_k),
            "wire": nn.Linear(cfg.d_v, cfg.d_k),
            "tool": nn.Linear(cfg.d_v, cfg.d_k),
            "out": nn.Linear(cfg.d_v, cfg.d_k),
        })

        self.topo_depth_head = nn.Linear(cfg.d_v, 3)
        self.topo_role_head = nn.Linear(cfg.d_v, 3)
        self.topo_critical_head = nn.Linear(cfg.d_v, 1)

        self.wire_queue_head = nn.Linear(cfg.d_v, 3)
        self.wire_slack_head = nn.Linear(cfg.d_v, 3)
        self.wire_violation_head = nn.Linear(cfg.d_v, 1)

        self.tool_gain_head = nn.Linear(cfg.d_v, action_dim)
        self.outcome_head = nn.Linear(cfg.d_v, 3)

        self.gate_fuse = nn.Linear(4 * cfg.d_v, 4)
        self.fuse = MLP([4 * cfg.d_v, cfg.d_atom, cfg.d_atom], dropout=0.1)
        self.need_head = nn.Linear(cfg.d_atom, 1)
        self.risk_head = nn.Linear(cfg.d_atom, 1)
        self.action_gain_atom_head = nn.Linear(cfg.d_atom, action_dim)
        self.conf_head = nn.Linear(cfg.d_atom, 1)

        self.block_priors = {
            "topo": torch.tensor([0.55, 0.15, 0.10, 0.20], dtype=torch.float32),
            "wire": torch.tensor([0.10, 0.60, 0.10, 0.20], dtype=torch.float32),
            "tool": torch.tensor([0.10, 0.15, 0.55, 0.20], dtype=torch.float32),
            "out": torch.tensor([0.05, 0.20, 0.25, 0.50], dtype=torch.float32),
        }

    def _compose_steps(self, g: torch.Tensor, s: torch.Tensor, a: torch.Tensor, o: torch.Tensor, q_text: torch.Tensor, head_name: str) -> Tuple[torch.Tensor, torch.Tensor]:
        beta = torch.softmax(self.block_gate[head_name](q_text), dim=-1)
        vg = self.block_value["g"](g)
        vs = self.block_value["s"](s)
        va = self.block_value["a"](a)
        vo = self.block_value["o"](o)
        stacked = torch.stack([vg, vs, va, vo], dim=2)
        mixed = (stacked * beta[:, None, :, None]).sum(dim=2)
        return mixed, beta

    def _attend(self, q_text: torch.Tensor, step_mix: torch.Tensor, mask: torch.Tensor, head_name: str) -> Tuple[torch.Tensor, torch.Tensor]:
        q = self.query_proj[head_name](q_text).unsqueeze(1)
        k = self.key_proj[head_name](step_mix)
        score = torch.matmul(q, k.transpose(-1, -2)).squeeze(1) / math.sqrt(self.cfg.d_k)
        score = score.masked_fill(~mask, -1e9)
        alpha = torch.softmax(score, dim=-1)
        z = torch.bmm(alpha.unsqueeze(1), step_mix).squeeze(1)
        return z, alpha

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        q_topo = self.text_proj(batch["q_topo"])
        q_wire = self.text_proj(batch["q_wire"])
        q_tool = self.text_proj(batch["q_tool"])
        q_out = self.text_proj(batch["q_out"])

        g = self.phi_g(batch["g_feat"])
        s = self.phi_s(batch["s_feat"])
        a = self.phi_a(batch["a_feat"])
        o = self.phi_o(batch["o_feat"])
        mask = batch["mask"]

        mix_topo, beta_topo = self._compose_steps(g, s, a, o, q_topo, "topo")
        mix_wire, beta_wire = self._compose_steps(g, s, a, o, q_wire, "wire")
        mix_tool, beta_tool = self._compose_steps(g, s, a, o, q_tool, "tool")
        mix_out, beta_out = self._compose_steps(g, s, a, o, q_out, "out")

        z_topo, alpha_topo = self._attend(q_topo, mix_topo, mask, "topo")
        z_wire, alpha_wire = self._attend(q_wire, mix_wire, mask, "wire")
        z_tool, alpha_tool = self._attend(q_tool, mix_tool, mask, "tool")
        z_out, alpha_out = self._attend(q_out, mix_out, batch.get("out_mask", mask), "out")

        z_all = torch.cat([z_topo, z_wire, z_tool, z_out], dim=-1)
        head_gate = torch.softmax(self.gate_fuse(z_all), dim=-1)
        gated = torch.cat([
            head_gate[:, 0:1] * z_topo,
            head_gate[:, 1:2] * z_wire,
            head_gate[:, 2:3] * z_tool,
            head_gate[:, 3:4] * z_out,
        ], dim=-1)
        h_atom = self.fuse(gated)
        return {
            "z_topo": z_topo,
            "z_wire": z_wire,
            "z_tool": z_tool,
            "z_out": z_out,
            "alpha_topo": alpha_topo,
            "alpha_wire": alpha_wire,
            "alpha_tool": alpha_tool,
            "alpha_out": alpha_out,
            "beta_topo": beta_topo,
            "beta_wire": beta_wire,
            "beta_tool": beta_tool,
            "beta_out": beta_out,
            "head_gate": head_gate,
            "h_atom": h_atom,
        }

    def loss_on_batch(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float], Dict[str, torch.Tensor]]:
        out = self.forward(batch)

        l_topo_depth = F.cross_entropy(self.topo_depth_head(out["z_topo"]), batch["depth_t"])
        l_topo_role = F.cross_entropy(self.topo_role_head(out["z_topo"]), batch["role_t"])
        l_topo_critical = F.binary_cross_entropy_with_logits(self.topo_critical_head(out["z_topo"]).squeeze(-1), batch["critical_t"])
        l_topo_raw = l_topo_depth + l_topo_role + l_topo_critical
        l_topo = l_topo_raw / 3.0 if bool(getattr(self.cfg, "normalize_topo_wire_losses", True)) else l_topo_raw

        l_wire_queue = F.cross_entropy(self.wire_queue_head(out["z_wire"]), batch["queue_t"])
        l_wire_slack = F.cross_entropy(self.wire_slack_head(out["z_wire"]), batch["slack_t"])
        l_wire_violation = F.binary_cross_entropy_with_logits(self.wire_violation_head(out["z_wire"]).squeeze(-1), batch["violation_t"])
        l_wire_raw = l_wire_queue + l_wire_slack + l_wire_violation
        l_wire = l_wire_raw / 3.0 if bool(getattr(self.cfg, "normalize_topo_wire_losses", True)) else l_wire_raw

        tool_pred = self.tool_gain_head(out["z_tool"])
        tool_mask = batch.get("tool_gain_mask_t")
        if bool(getattr(self.cfg, "use_masked_tool_loss", True)) and tool_mask is not None and float(tool_mask.sum().detach().item()) > 0.0:
            l_tool = ((tool_pred - batch["tool_gain_t"]).pow(2) * tool_mask).sum() / tool_mask.sum().clamp_min(1.0)
        else:
            l_tool = F.mse_loss(tool_pred, batch["tool_gain_t"])

        l_out = F.binary_cross_entropy_with_logits(self.outcome_head(out["z_out"]), batch["out_t"])
        l_need = F.binary_cross_entropy_with_logits(self.need_head(out["h_atom"]).squeeze(-1), batch["need_t"])

        atom_gain_pred = self.action_gain_atom_head(out["h_atom"])
        if bool(getattr(self.cfg, "use_masked_tool_loss", True)) and tool_mask is not None and float(tool_mask.sum().detach().item()) > 0.0:
            l_atom_gain = ((atom_gain_pred - batch["tool_gain_t"]).pow(2) * tool_mask).sum() / tool_mask.sum().clamp_min(1.0)
        else:
            l_atom_gain = F.mse_loss(atom_gain_pred, batch["tool_gain_t"])
        l_atom_risk = F.mse_loss(torch.sigmoid(self.risk_head(out["h_atom"]).squeeze(-1)), batch["risk_t"])
        l_atom_conf = F.binary_cross_entropy_with_logits(self.conf_head(out["h_atom"]).squeeze(-1), batch["confidence_t"])
        l_atom = l_atom_risk + l_atom_gain + l_atom_conf

        alphas = {
            "topo": out["alpha_topo"],
            "wire": out["alpha_wire"],
            "tool": out["alpha_tool"],
            "out": out["alpha_out"],
        }
        betas = {
            "topo": out["beta_topo"],
            "wire": out["beta_wire"],
            "tool": out["beta_tool"],
            "out": out["beta_out"],
        }
        y = batch["y"]
        eps = 1e-8
        l_bind_pos = torch.tensor(0.0, device=batch["mask"].device)
        l_sep = torch.tensor(0.0, device=batch["mask"].device)
        for i, name in enumerate(["topo", "wire", "tool", "out"]):
            yi = y[:, :, i]
            yi = yi / yi.sum(dim=-1, keepdim=True).clamp_min(1e-6)
            alpha = alphas[name]
            l_bind_pos = l_bind_pos - torch.mean(torch.sum(yi * torch.log(alpha + eps), dim=-1))
            for j, _other in enumerate(["topo", "wire", "tool", "out"]):
                if i == j:
                    continue
                l_sep = l_sep + torch.mean(torch.sum(y[:, :, j] * alphas[name], dim=-1))

        l_block = torch.tensor(0.0, device=batch["mask"].device)
        for name in ["topo", "wire", "tool", "out"]:
            prior = self.block_priors[name].to(batch["mask"].device).unsqueeze(0).expand_as(betas[name])
            l_block = l_block + F.kl_div(torch.log(betas[name] + 1e-8), prior, reduction="batchmean")

        z_list = [out["z_topo"], out["z_wire"], out["z_tool"], out["z_out"]]
        l_div = torch.tensor(0.0, device=batch["mask"].device)
        for i in range(len(z_list)):
            for j in range(i + 1, len(z_list)):
                zi = F.normalize(z_list[i], dim=-1)
                zj = F.normalize(z_list[j], dim=-1)
                l_div = l_div + torch.mean(torch.sum(zi * zj, dim=-1).pow(2))

        total = (
            float(getattr(self.cfg, "loss_topo_weight", 1.0)) * l_topo
            + float(getattr(self.cfg, "loss_wire_weight", 1.0)) * l_wire
            + float(getattr(self.cfg, "loss_tool_weight", 3.0)) * l_tool
            + float(getattr(self.cfg, "loss_out_weight", 1.0)) * l_out
            + float(getattr(self.cfg, "loss_need_weight", 1.0)) * l_need
            + self.cfg.lambda_atom * l_atom
        )
        if self.cfg.enable_disentangle_losses:
            total = total + self.cfg.lambda_bind_pos * l_bind_pos + self.cfg.lambda_sep * l_sep + self.cfg.lambda_block * l_block + self.cfg.lambda_div * l_div

        stats = {
            "loss_total": float(total.detach().item()),
            "loss_topo": float(l_topo.detach().item()),
            "loss_wire": float(l_wire.detach().item()),
            "loss_tool": float(l_tool.detach().item()),
            "loss_out": float(l_out.detach().item()),
            "loss_need": float(l_need.detach().item()),
            "loss_atom": float(l_atom.detach().item()),
            "loss_topo_depth": float(l_topo_depth.detach().item()),
            "loss_topo_role": float(l_topo_role.detach().item()),
            "loss_topo_critical": float(l_topo_critical.detach().item()),
            "loss_wire_queue": float(l_wire_queue.detach().item()),
            "loss_wire_slack": float(l_wire_slack.detach().item()),
            "loss_wire_violation": float(l_wire_violation.detach().item()),
            "loss_atom_risk": float(l_atom_risk.detach().item()),
            "loss_atom_gain": float(l_atom_gain.detach().item()),
            "loss_atom_conf": float(l_atom_conf.detach().item()),
            "tool_loss_observed_actions": float((batch.get("tool_gain_mask_t", torch.zeros_like(batch["tool_gain_t"]))).sum().detach().item()),
            "loss_bind_pos": float(l_bind_pos.detach().item()),
            "loss_sep": float(l_sep.detach().item()),
            "loss_block": float(l_block.detach().item()),
            "loss_div": float(l_div.detach().item()),
        }
        return total, stats, out


class PrototypeMemory:
    def __init__(self, action_dim: int, cfg: KnowledgeConfig) -> None:
        self.action_dim = action_dim
        self.cfg = cfg
        self.buffer: List[KnowledgeAtom] = []
        self.prototypes: List[Prototype] = []
        self._last_dim_warning: Optional[Tuple[int, int]] = None

    def embedding_dim(self) -> int:
        if self.prototypes and self.prototypes[0].centroid:
            return len(self.prototypes[0].centroid)
        if self.buffer and self.buffer[0].embedding:
            return len(self.buffer[0].embedding)
        return int(self.cfg.d_atom)

    def _align_embedding_dim(self, x: Sequence[float], target_dim: int) -> List[float]:
        vals = list(x)
        if len(vals) == target_dim:
            return vals
        if len(vals) > target_dim:
            return vals[:target_dim]
        return vals + [0.0] * (target_dim - len(vals))

    def _prepare_pair(self, a: Sequence[float], b: Sequence[float]) -> Tuple[torch.Tensor, torch.Tensor]:
        va = list(a)
        vb = list(b)
        la = len(va)
        lb = len(vb)
        target_dim = max(la, lb)
        aa = self._align_embedding_dim(va, target_dim)
        bb = self._align_embedding_dim(vb, target_dim)
        if self.cfg.verbose and la != lb and self._last_dim_warning != (la, lb):
            print(
                f"{self.cfg.print_prefix} memory dim-align | query_dim={la} | proto_dim={lb} | using_dim={target_dim}",
                flush=True,
            )
            self._last_dim_warning = (la, lb)
        ta = torch.tensor(aa, dtype=torch.float32)
        tb = torch.tensor(bb, dtype=torch.float32)
        return ta, tb

    def _cosine(self, a: Sequence[float], b: Sequence[float]) -> float:
        ta, tb = self._prepare_pair(a, b)
        denom = float(ta.norm().item() * tb.norm().item())
        if denom <= 1e-12:
            return 0.0
        return float(torch.dot(ta, tb).item() / denom)

    def _sim(self, atom: KnowledgeAtom, proto: Prototype) -> float:
        type_term = 1.0 if atom.node_type_id == proto.node_type_id else 0.0
        cos = self._cosine(atom.embedding, proto.centroid)
        regime_diff = abs(atom.queue_class - proto.queue_class) + abs(atom.slack_class - proto.slack_class)
        regime = math.exp(-regime_diff / 2.0)
        return 0.4 * type_term + 0.4 * cos + 0.2 * regime

    def _format_action_label(self, idx: int) -> str:
        local_idx = self.action_dim - 2
        pause_idx = self.action_dim - 1
        if idx == local_idx:
            return "local"
        if idx == pause_idx:
            return "pause"
        return f"tool {idx}"

    def _top_actions_text(self, probs: Sequence[float], gains: Sequence[float], top_k: int = 3) -> str:
        if not probs:
            return ""
        top_k = max(1, min(int(top_k), len(probs)))
        order = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)[:top_k]
        items = []
        for idx in order:
            items.append(f"{self._format_action_label(idx)} {probs[idx]:.2f} (gain {gains[idx]:.3f})")
        return ", ".join(items)

    def _fmt_seconds_short(self, value: float) -> str:
        try:
            v = float(value)
        except Exception:
            v = 0.0
        return f"{max(0.0, v):.2f}s"

    def _action_detail_text(
        self,
        stats: Sequence[Dict[str, float]],
        top_actions: Sequence[Dict[str, object]],
        top_k: int = 5,
    ) -> str:
        if not top_actions:
            return "no ranked action details available"
        pieces: List[str] = []
        for rank, row in enumerate(list(top_actions)[: max(1, int(top_k))], start=1):
            idx = int(row.get("index", 0))
            label = str(row.get("label", self._format_action_label(idx)))
            st = stats[idx] if 0 <= idx < len(stats) and isinstance(stats[idx], dict) else {}
            decisions = int(round(float(st.get("decisions", 0.0) or 0.0)))
            if decisions > 0:
                succ = float(st.get("success_rate", 0.0) or 0.0)
                total_s = float(st.get("mean_total_s", 0.0) or 0.0)
                exec_s = float(st.get("mean_exec_s", 0.0) or 0.0)
                comm_s = float(st.get("mean_comm_s", 0.0) or 0.0)
                queue_s = float(st.get("mean_queue_s", 0.0) or 0.0)
                risk = float(st.get("mean_risk", 0.0) or 0.0)
                loss = float(st.get("mean_packet_loss", 0.0) or 0.0)
                rejects = int(round(float(st.get("queue_full_rejects", 0.0) or 0.0)))
                reason_bits = []
                if succ >= 0.80:
                    reason_bits.append("high observed node success")
                elif succ <= 0.20:
                    reason_bits.append("low observed node success")
                if queue_s >= 1.0:
                    reason_bits.append("queue pressure")
                if risk >= 0.50:
                    reason_bits.append("high execution risk")
                if loss >= 0.10:
                    reason_bits.append("packet-loss risk")
                if total_s > 0.0 and exec_s / max(total_s, 1e-9) >= 0.65:
                    reason_bits.append("compute-dominated latency")
                if total_s > 0.0 and comm_s / max(total_s, 1e-9) >= 0.35:
                    reason_bits.append("communication-dominated latency")
                if rejects > 0:
                    reason_bits.append(f"{rejects} queue-full rejects")
                reason = "; ".join(reason_bits) if reason_bits else "balanced observed cost"
                pieces.append(
                    f"#{rank} {label}: model_prob={float(row.get('prob', 0.0)):.2f}, gain={float(row.get('gain', 0.0)):.3f}, "
                    f"used={decisions}, success={succ:.2f}, total={self._fmt_seconds_short(total_s)} "
                    f"(exec {self._fmt_seconds_short(exec_s)}, comm {self._fmt_seconds_short(comm_s)}, queue {self._fmt_seconds_short(queue_s)}), "
                    f"risk={risk:.2f}, pkt_loss={loss:.2f}; evidence={reason}"
                )
            else:
                pieces.append(
                    f"#{rank} {label}: model_prob={float(row.get('prob', 0.0)):.2f}, gain={float(row.get('gain', 0.0)):.3f}, "
                    "no direct execution observations in this prototype"
                )
        return " | ".join(pieces)

    def _prototype_prompt(self, proto: Prototype) -> str:
        state = copy.deepcopy(proto.state_context or {})
        env = copy.deepcopy(proto.environment_context or {})
        candidates = list(proto.candidate_actions or [])
        rec = copy.deepcopy(proto.recommended_action or {})
        observed = copy.deepcopy(proto.observed_outcome or {})
        aspect = list(proto.aspect_weights or [0.0, 0.0, 0.0, 0.0])
        if len(aspect) < 4:
            aspect = aspect + [0.0] * (4 - len(aspect))
        wire = list(proto.wire_factor_scores or [0.0, 0.0, 0.0, 0.0])
        if len(wire) < 4:
            wire = wire + [0.0] * (4 - len(wire))
        outcome = list(proto.outcome_probs or [0.0, 0.0, 0.0])
        if len(outcome) < 3:
            outcome = outcome + [0.0] * (3 - len(outcome))
        gains = list(proto.action_gain_means or [0.0] * self.action_dim)
        if len(gains) < self.action_dim:
            gains = gains + [0.0] * (self.action_dim - len(gains))

        def fmt_float(value: object, digits: int = 3, default: float = 0.0) -> str:
            try:
                return f"{float(value):.{digits}f}"
            except Exception:
                return f"{default:.{digits}f}"

        def action_idx(row: Dict[str, object]) -> int:
            at = str(row.get("action_type", ""))
            if at == "local":
                return self.action_dim - 2
            if at == "pause":
                return self.action_dim - 1
            try:
                return int(row.get("slot", row.get("tool_type_id", row.get("action_index", -1))))
            except Exception:
                return -1

        def action_label(row: Dict[str, object], fallback_idx: int = -1) -> str:
            at = str(row.get("action_type", ""))
            if at == "local":
                return "LOCAL execution"
            if at == "pause":
                return "PAUSE"
            idx = row.get("slot", row.get("tool_type_id", fallback_idx))
            name = row.get("tool_name") or row.get("name") or f"tool {idx}"
            cat = row.get("category", "unknown")
            return f"TOOL slot {idx}: {name} (category={cat})"

        if not rec:
            observed_indices = [i for i, c in enumerate(proto.action_counts or []) if int(c) > 0]
            best_idx = max(observed_indices, key=lambda i: gains[i]) if observed_indices else -1
            rec = {"action_index": best_idx, "action_type": "tool" if best_idx >= 0 and best_idx < self.action_dim - 2 else ("local" if best_idx == self.action_dim - 2 else "pause")}
        rec_idx = int(rec.get("action_index", action_idx(rec)))
        rec_stats = proto.action_detail_stats[rec_idx] if 0 <= rec_idx < len(proto.action_detail_stats) and isinstance(proto.action_detail_stats[rec_idx], dict) else {}
        rec_label = rec.get("action_label") or action_label(rec, rec_idx)
        if str(rec.get("action_type")) == "tool":
            slot = rec.get("slot", rec.get("tool_type_id", rec_idx))
            tool_name = rec.get("tool_name") or rec.get("name") or f"tool {slot}"
            command = rec.get("decision_command") or f"use_tool(slot={slot}, tool_name='{tool_name}')"
        elif str(rec.get("action_type")) == "local":
            command = rec.get("decision_command") or "use_local()"
        else:
            command = rec.get("decision_command") or f"pause(duration_s={fmt_float(rec.get('pause_duration_s', 0.2), 2)})"

        def compact_candidate(row: Dict[str, object]) -> str:
            at = str(row.get("action_type", ""))
            label = action_label(row, action_idx(row))
            if at == "tool":
                return (
                    f"{label}; predicted_total={fmt_float(row.get('predicted_total_s'), 3)}s; "
                    f"queue={fmt_float(row.get('base_queue_delay_s'), 3)}s; risk={fmt_float(row.get('risk_estimate'), 3)}; "
                    f"pkt_loss={fmt_float(row.get('packet_loss_rate'), 3)}; reliability={fmt_float(row.get('server_reliability'), 2)}; "
                    f"completion_gain={fmt_float(row.get('completion_gain'), 2)}"
                )
            if at == "local":
                return f"{label}; predicted_total={fmt_float(row.get('predicted_total_s'), 3)}s; fail_probability={fmt_float(row.get('fail_probability'), 3)}; queue_blocked={bool(row.get('queue_blocked', False))}"
            return f"{label}; duration={fmt_float(row.get('pause_duration_s', 0.2), 2)}s"

        candidate_text = " | ".join(compact_candidate(c) for c in candidates[:6]) if candidates else self._action_detail_text(
            proto.action_detail_stats,
            [{"index": i, "label": self._format_action_label(i), "prob": 0.0, "gain": gains[i]} for i in sorted(range(self.action_dim), key=lambda j: gains[j], reverse=True)[: min(5, self.action_dim)]],
            top_k=5,
        )

        if str(rec.get("action_type")) == "tool":
            tool_detail = (
                f"Description={rec.get('description', '')}. compute={fmt_float(rec.get('compute_frequency_hz'), 1)}Hz, "
                f"uplink={fmt_float(rec.get('uplink_bandwidth_hz'), 1)}Hz, downlink={fmt_float(rec.get('downlink_bandwidth_hz'), 1)}Hz, "
                f"coverage={fmt_float(rec.get('coverage_distance_m'), 1)}m, reliability={fmt_float(rec.get('server_reliability'), 2)}, "
                f"success_probability={fmt_float(rec.get('success_probability'), 2)}, semantic_gain={fmt_float(rec.get('semantic_gain'), 2)}, "
                f"completion_gain={fmt_float(rec.get('completion_gain'), 2)}, energy={fmt_float(rec.get('energy_j'), 3)}J, "
                f"requires_tool_to_complete={rec.get('requires_tool_to_complete', state.get('requires_tool_to_complete', 'unknown'))}."
            )
        elif str(rec.get("action_type")) == "local":
            tool_detail = (
                f"Local details: compute={fmt_float(env.get('local_compute_s'), 3)}s, "
                f"fail_probability={fmt_float(env.get('local_fail_probability'), 3)}, queue_blocked={bool(env.get('local_queue_blocked', False))}."
            )
        else:
            tool_detail = "Pause details: use only when all tool/local actions are infeasible or unsafe under queue/deadline constraints."

        avoid_reason = "Avoid pause if a non-blocked tool/local action can finish before the deadline; avoid local execution on tool-required nodes unless all tools are infeasible."
        if str(rec.get("action_type")) == "local":
            avoid_reason = "Avoid local only when its failure probability or queue block dominates; otherwise it can be a valid fallback."
        elif str(rec.get("action_type")) == "pause":
            avoid_reason = "Avoid forcing execution when queue/deadline conditions make every candidate unsafe."

        return (
            "[Situation]\n"
            f"Current node: {state.get('node_step') or state.get('intent_text') or 'prototype-level node type ' + str(proto.node_type_id)}. "
            f"Node type={int(state.get('node_type_id', proto.node_type_id))}, critical={int(state.get('critical_flag', 0))}, "
            f"queue_class={int(state.get('queue_class', proto.queue_class))}, slack_class={int(state.get('slack_class', proto.slack_class))}, "
            f"deadline remaining={fmt_float(state.get('node_remaining_deadline_s'), 3)}s. Allowed actions: tools={state.get('allowed_slots', [])}, local, pause. "
            f"requires_tool_to_complete={state.get('requires_tool_to_complete', False)}, local_success_probability={state.get('local_success_probability', 'unknown')}.\n"
            "[Environment]\n"
            f"Current pressure={env.get('dominant_pressure', 'balanced')}; local_queue={env.get('local_queue_length', 0)}, local_running={env.get('local_running_jobs', 0)}, "
            f"max_tool_queue={fmt_float(env.get('max_tool_queue_delay_s'), 3)}s, max_tool_risk={fmt_float(env.get('max_tool_risk_estimate'), 3)}, "
            f"max_packet_loss={fmt_float(env.get('max_packet_loss_rate'), 3)}. Local execution: predicted_total={fmt_float(env.get('local_predicted_total_s'), 3)}s, "
            f"fail_probability={fmt_float(env.get('local_fail_probability'), 3)}, queue_blocked={bool(env.get('local_queue_blocked', False))}. Candidate actions: {candidate_text}.\n"
            "[Best action]\n"
            f"Choose {rec_label}. Decision command: {command}.\n"
            "[Tool details]\n"
            f"{tool_detail}\n"
            "[Reason]\n"
            f"This action is preferred because {rec.get('why_best', 'it has the strongest aggregated action evidence in this prototype')}. {avoid_reason}\n"
            "[Evidence]\n"
            f"In similar cases: support={int(proto.support)}, node_success_rate={fmt_float(rec_stats.get('success_rate', observed.get('success_rate', outcome[0])), 3)}, "
            f"task_success_rate={fmt_float(outcome[1], 3)}, mean_total_s={fmt_float(rec_stats.get('mean_total_s', observed.get('mean_total_s', 0.0)), 3)}, "
            f"mean_queue_s={fmt_float(rec_stats.get('mean_queue_s', observed.get('mean_queue_s', 0.0)), 3)}, mean_risk={fmt_float(rec_stats.get('mean_risk', observed.get('mean_risk', 0.0)), 3)}, "
            f"action_gain={fmt_float(gains[rec_idx] if 0 <= rec_idx < len(gains) else observed.get('recommended_action_gain', 0.0), 3)}, "
            f"predicted node/task/violation={outcome[0]:.3f}/{outcome[1]:.3f}/{outcome[2]:.3f}; "
            f"aspect topo/wire/tool/out={aspect[0]:.2f}/{aspect[1]:.2f}/{aspect[2]:.2f}/{aspect[3]:.2f}; "
            f"wire pressure queue/loss/deadline/exec-risk={wire[0]:.2f}/{wire[1]:.2f}/{wire[2]:.2f}/{wire[3]:.2f}."
        )

    def _spawn(self, atom: KnowledgeAtom) -> Prototype:
        proto = Prototype(
            prototype_id=str(uuid.uuid4()),
            node_type_id=atom.node_type_id,
            depth_tier=atom.depth_tier,
            queue_class=atom.queue_class,
            slack_class=atom.slack_class,
            centroid=list(atom.embedding),
            support=1,
            no_tool_risk_mean=float(atom.no_tool_risk_prior),
            action_gain_means=list(atom.action_gain_priors),
            action_counts=list(atom.action_counts),
            confidence_mean=float(atom.confidence),
            prompt_text=atom.prompt_text,
            action_probs=list(atom.action_probs),
            action_detail_stats=copy.deepcopy(atom.action_detail_stats),
            aspect_weights=list(atom.aspect_weights),
            wire_factor_scores=list(atom.wire_factor_scores),
            outcome_probs=list(atom.outcome_probs),
            state_context=copy.deepcopy(atom.state_context),
            environment_context=copy.deepcopy(atom.environment_context),
            candidate_actions=copy.deepcopy(atom.candidate_actions),
            recommended_action=copy.deepcopy(atom.recommended_action),
            observed_outcome=copy.deepcopy(atom.observed_outcome),
        )
        proto.prompt_text = self._prototype_prompt(proto)
        if len(self.prototypes) >= self.cfg.max_prototypes:
            min_idx = min(range(len(self.prototypes)), key=lambda i: self.prototypes[i].support)
            self.prototypes[min_idx] = proto
        else:
            self.prototypes.append(proto)
        return proto

    def _merge_action_detail_stats(
        self,
        old_stats: Sequence[Dict[str, float]],
        new_stats: Sequence[Dict[str, float]],
    ) -> List[Dict[str, float]]:
        keys_mean = [
            "mean_total_s",
            "mean_exec_s",
            "mean_uplink_s",
            "mean_downlink_s",
            "mean_comm_s",
            "mean_queue_s",
            "mean_risk",
            "mean_packet_loss",
        ]
        out: List[Dict[str, float]] = []
        for idx in range(self.action_dim):
            old = old_stats[idx] if idx < len(old_stats) and isinstance(old_stats[idx], dict) else {}
            new = new_stats[idx] if idx < len(new_stats) and isinstance(new_stats[idx], dict) else {}
            n_old = max(0.0, float(old.get("decisions", 0.0) or 0.0))
            n_new = max(0.0, float(new.get("decisions", 0.0) or 0.0))
            n_total = n_old + n_new
            merged = {
                "decisions": n_total,
                "successes": max(0.0, float(old.get("successes", 0.0) or 0.0)) + max(0.0, float(new.get("successes", 0.0) or 0.0)),
                "failures": max(0.0, float(old.get("failures", 0.0) or 0.0)) + max(0.0, float(new.get("failures", 0.0) or 0.0)),
                "queue_full_rejects": max(0.0, float(old.get("queue_full_rejects", 0.0) or 0.0)) + max(0.0, float(new.get("queue_full_rejects", 0.0) or 0.0)),
            }
            for key in keys_mean:
                if n_total > 0.0:
                    merged[key] = (float(old.get(key, 0.0) or 0.0) * n_old + float(new.get(key, 0.0) or 0.0) * n_new) / n_total
                else:
                    merged[key] = 0.0
            merged["success_rate"] = merged["successes"] / max(1.0, n_total)
            merged["failure_rate"] = merged["failures"] / max(1.0, n_total)
            out.append(merged)
        return out

    def _merge(self, atom: KnowledgeAtom, proto: Prototype) -> None:
        n = float(proto.support)
        proto.centroid = [
            (proto.centroid[i] * n + atom.embedding[i]) / (n + 1.0)
            for i in range(len(proto.centroid))
        ]
        proto.no_tool_risk_mean = (proto.no_tool_risk_mean * n + atom.no_tool_risk_prior) / (n + 1.0)
        proto.action_gain_means = [
            (proto.action_gain_means[i] * n + atom.action_gain_priors[i]) / (n + 1.0)
            for i in range(self.action_dim)
        ]
        proto.action_counts = [proto.action_counts[i] + atom.action_counts[i] for i in range(self.action_dim)]
        proto.confidence_mean = (proto.confidence_mean * n + atom.confidence) / (n + 1.0)
        if atom.action_probs:
            if not proto.action_probs:
                proto.action_probs = [0.0] * len(atom.action_probs)
            proto.action_probs = [
                (proto.action_probs[i] * n + atom.action_probs[i]) / (n + 1.0)
                for i in range(len(atom.action_probs))
            ]
        if atom.action_detail_stats:
            proto.action_detail_stats = self._merge_action_detail_stats(proto.action_detail_stats, atom.action_detail_stats)
        if atom.aspect_weights:
            if not proto.aspect_weights:
                proto.aspect_weights = [0.0] * len(atom.aspect_weights)
            proto.aspect_weights = [
                (proto.aspect_weights[i] * n + atom.aspect_weights[i]) / (n + 1.0)
                for i in range(len(atom.aspect_weights))
            ]
        if atom.wire_factor_scores:
            if not proto.wire_factor_scores:
                proto.wire_factor_scores = [0.0] * len(atom.wire_factor_scores)
            proto.wire_factor_scores = [
                (proto.wire_factor_scores[i] * n + atom.wire_factor_scores[i]) / (n + 1.0)
                for i in range(len(atom.wire_factor_scores))
            ]
        if atom.outcome_probs:
            if not proto.outcome_probs:
                proto.outcome_probs = [0.0] * len(atom.outcome_probs)
            proto.outcome_probs = [
                (proto.outcome_probs[i] * n + atom.outcome_probs[i]) / (n + 1.0)
                for i in range(len(atom.outcome_probs))
            ]
        # Keep the richest concrete state/tool context as the representative prototype text.
        # Aggregated numeric evidence remains above, while names/parameters must stay exact.
        if atom.candidate_actions and (not proto.candidate_actions or atom.confidence >= proto.confidence_mean):
            proto.state_context = copy.deepcopy(atom.state_context)
            proto.environment_context = copy.deepcopy(atom.environment_context)
            proto.candidate_actions = copy.deepcopy(atom.candidate_actions)
            proto.recommended_action = copy.deepcopy(atom.recommended_action)
        if atom.observed_outcome:
            proto.observed_outcome = copy.deepcopy(atom.observed_outcome)
        proto.support += 1
        proto.prompt_text = self._prototype_prompt(proto)

    def add_atom(self, atom: KnowledgeAtom) -> Prototype:
        self.buffer.append(atom)
        if len(self.buffer) > self.cfg.max_buffer_atoms:
            self.buffer = self.buffer[-self.cfg.max_buffer_atoms:]
        if not self.prototypes:
            return self._spawn(atom)
        best_idx = max(range(len(self.prototypes)), key=lambda i: self._sim(atom, self.prototypes[i]))
        best_sim = self._sim(atom, self.prototypes[best_idx])
        if best_sim >= self.cfg.tau_assign:
            self._merge(atom, self.prototypes[best_idx])
            return self.prototypes[best_idx]
        return self._spawn(atom)

    def query(self, query_embedding: Sequence[float], node_type_id: int, queue_class: int, slack_class: int, k: int) -> List[Tuple[Prototype, float]]:
        if not self.prototypes:
            return []
        q = KnowledgeAtom(
            atom_id="query",
            node_type_id=node_type_id,
            depth_tier=0,
            role_id=0,
            critical_flag=0,
            queue_class=queue_class,
            slack_class=slack_class,
            no_tool_risk_prior=0.0,
            action_gain_priors=[0.0] * self.action_dim,
            confidence=0.0,
            embedding=self._align_embedding_dim(query_embedding, self.embedding_dim()),
            action_counts=[0] * self.action_dim,
        )
        scored = [(p, self._sim(q, p)) for p in self.prototypes]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]

    def to_json(self) -> Dict[str, object]:
        return {
            "buffer": [asdict(x) for x in self.buffer],
            "prototypes": [asdict(x) for x in self.prototypes],
        }

    def load_json(self, data: Dict[str, object]) -> None:
        atom_fields = set(KnowledgeAtom.__dataclass_fields__.keys())
        proto_fields = set(Prototype.__dataclass_fields__.keys())
        self.buffer = [KnowledgeAtom(**{k: v for k, v in dict(x).items() if k in atom_fields}) for x in data.get("buffer", [])]
        self.prototypes = [Prototype(**{k: v for k, v in dict(x).items() if k in proto_fields}) for x in data.get("prototypes", [])]


class KnowledgeTrainer:
    def __init__(self, dataset: Dict[str, object], action_dim: int, cfg: Optional[KnowledgeConfig] = None) -> None:
        self.dataset = _normalize_dataset_structure(dataset)
        dataset = self.dataset
        self.action_dim = action_dim
        self.cfg = cfg or KnowledgeConfig()
        self.cfg.device = _auto_device(self.cfg.device)
        self.device = torch.device(self.cfg.device)
        self.rng = random.Random(self.cfg.seed)

        self.dataset_index = DatasetIndex(dataset)
        self.extractor = SegmentExtractor(self.dataset_index, action_dim)
        self.llm = LLMBackbone(self.cfg)
        self.distiller = LLMFourHeadDistiller(self.llm.hidden_size, action_dim, self.cfg).to(self.device)
        opt_params = list(self.distiller.parameters()) + list(self.llm.trainable_parameters())
        self.optimizer = torch.optim.AdamW(opt_params, lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)
        self.memory = PrototypeMemory(action_dim, self.cfg)
        self.last_fit_stats: Dict[str, float] = {}
        self.fit_step = 0
        self.last_snapshot: Dict[str, float] = {"prototype_risk_mean": 0.0, "prototype_count": 0.0}
        if self.cfg.verbose:
            trainable = sum(p.numel() for p in opt_params if p.requires_grad)
            print(
                f"{self.cfg.print_prefix} trainer initialized | action_dim={self.action_dim} | batch_size={self.cfg.batch_size} | "
                f"epochs_per_fit={self.cfg.epochs_per_fit} | trainable_params={trainable}",
                flush=True,
            )

    def set_dataset(self, dataset: Dict[str, object]) -> None:
        self.dataset = dataset
        self.dataset_index = DatasetIndex(dataset)
        self.extractor = SegmentExtractor(self.dataset_index, self.action_dim)

    def _segment_fast_vector(self, seg: Dict[str, object], aspect: str) -> List[float]:
        """Build a deterministic segment embedding without calling the Qwen backbone.

        This is used only when cfg.fast_update_text_encoder=True. It preserves the
        same tensor shape expected by the distiller text projections, but derives
        the content from already extracted numeric trajectory features.
        """
        vals: List[float] = []
        static = seg.get("static", {}) or {}
        steps = list(seg.get("steps", []) or [])
        vals.extend([
            float(static.get("node_type_id", 0)) / 16.0,
            float(static.get("depth_ratio", 0.0)),
            float(static.get("indeg", 0.0)) / 8.0,
            float(static.get("outdeg", 0.0)) / 8.0,
            float(static.get("critical_flag", 0.0)),
            float(static.get("role_id", 0.0)) / 2.0,
            math.log1p(float(static.get("comp_cycles", 0.0))) / 30.0,
            math.log1p(float(static.get("uplink_bits", 0.0))) / 30.0,
            math.log1p(float(static.get("downlink_bits", 0.0))) / 30.0,
            float(static.get("node_deadline", 0.0)) / 20.0,
            float(seg.get("depth_tier", 0)) / 8.0,
            float(seg.get("role_id", 0)) / 2.0,
            float(seg.get("critical_flag", 0)),
            float(seg.get("queue_class", 0)) / 2.0,
            float(seg.get("slack_class", 0)) / 2.0,
            float(seg.get("node_success", 0)),
            float(seg.get("task_success", 0)),
            float(seg.get("violation", 0)),
            float(seg.get("tool_invoked", 0)),
            float(seg.get("no_tool_risk_prior", 0.0)),
            float(seg.get("confidence", 0.0)),
        ])
        for block_name, dim in (("g_feat", 6), ("s_feat", 6), ("a_feat", 6), ("o_feat", 4), ("y", 4)):
            if steps:
                for j in range(dim):
                    vals.append(sum(float(step.get(block_name, [0.0] * dim)[j]) for step in steps) / max(1, len(steps)))
            else:
                vals.extend([0.0] * dim)
        action_counts = [float(x) for x in list(seg.get("action_counts", []))[: self.action_dim]]
        if len(action_counts) < self.action_dim:
            action_counts += [0.0] * (self.action_dim - len(action_counts))
        count_sum = max(1.0, sum(action_counts))
        vals.extend([x / count_sum for x in action_counts])
        action_gain = [float(x) for x in list(seg.get("action_gain", []))[: self.action_dim]]
        if len(action_gain) < self.action_dim:
            action_gain += [0.0] * (self.action_dim - len(action_gain))
        vals.extend(action_gain)
        aspect_id = {"topo": 0, "wire": 1, "tool": 2, "out": 3}.get(str(aspect), 0)
        vals.extend([1.0 if i == aspect_id else 0.0 for i in range(4)])

        if not vals:
            vals = [0.0]
        hidden = int(self.llm.hidden_size)
        reps = int(math.ceil(hidden / float(len(vals))))
        out = (vals * max(1, reps))[:hidden]
        return [max(-10.0, min(10.0, float(x))) for x in out]

    def _fast_text_embeddings(self, segments: List[Dict[str, object]], aspect: str) -> torch.Tensor:
        if not segments:
            return torch.zeros((0, int(self.llm.hidden_size)), dtype=torch.float32, device=self.device)
        rows = [self._segment_fast_vector(seg, aspect=aspect) for seg in segments]
        return torch.tensor(rows, dtype=torch.float32, device=self.device)

    def _build_batch(self, segments: List[Dict[str, object]], detach_text: bool = False) -> Dict[str, torch.Tensor]:
        max_len = max(len(seg["steps"]) for seg in segments)
        batch_size = len(segments)

        def _pad(block_name: str, dim: int) -> torch.Tensor:
            out = torch.zeros((batch_size, max_len, dim), dtype=torch.float32)
            for i, seg in enumerate(segments):
                for t, step in enumerate(seg["steps"]):
                    out[i, t] = torch.tensor(step[block_name], dtype=torch.float32)
            return out

        g_feat = _pad("g_feat", 6).to(self.device)
        s_feat = _pad("s_feat", 6).to(self.device)
        a_feat = _pad("a_feat", 6).to(self.device)
        o_feat = _pad("o_feat", 4).to(self.device)
        y = torch.zeros((batch_size, max_len, 4), dtype=torch.float32, device=self.device)
        mask = torch.zeros((batch_size, max_len), dtype=torch.bool, device=self.device)
        out_mask = torch.zeros((batch_size, max_len), dtype=torch.bool, device=self.device)
        for i, seg in enumerate(segments):
            for t, step in enumerate(seg["steps"]):
                y[i, t] = torch.tensor(step["y"], dtype=torch.float32, device=self.device)
                mask[i, t] = True
                out_mask[i, t] = bool(step.get("is_decision_event", False))
            if not bool(out_mask[i].any().item()):
                out_mask[i] = mask[i]

        if bool(getattr(self.cfg, "fast_update_text_encoder", False)):
            topo_emb = self._fast_text_embeddings(segments, aspect="topo")
            wire_emb = self._fast_text_embeddings(segments, aspect="wire")
            tool_emb = self._fast_text_embeddings(segments, aspect="tool")
            out_emb = self._fast_text_embeddings(segments, aspect="out")
        else:
            topo_emb = self.llm.encode_texts([seg["texts"]["topo"] for seg in segments], aspect="topo", detach=detach_text).to(self.device)
            wire_emb = self.llm.encode_texts([seg["texts"]["wire"] for seg in segments], aspect="wire", detach=detach_text).to(self.device)
            tool_emb = self.llm.encode_texts([seg["texts"]["tool"] for seg in segments], aspect="tool", detach=detach_text).to(self.device)
            out_emb = self.llm.encode_texts([seg["texts"]["out"] for seg in segments], aspect="out", detach=detach_text).to(self.device)

        batch = {
            "g_feat": g_feat,
            "s_feat": s_feat,
            "a_feat": a_feat,
            "o_feat": o_feat,
            "mask": mask,
            "out_mask": out_mask,
            "y": y,
            "q_topo": topo_emb,
            "q_wire": wire_emb,
            "q_tool": tool_emb,
            "q_out": out_emb,
            "depth_t": torch.tensor([seg["depth_tier"] for seg in segments], dtype=torch.long, device=self.device),
            "role_t": torch.tensor([seg["role_id"] for seg in segments], dtype=torch.long, device=self.device),
            "critical_t": torch.tensor([seg["critical_flag"] for seg in segments], dtype=torch.float32, device=self.device),
            "queue_t": torch.tensor([seg["queue_class"] for seg in segments], dtype=torch.long, device=self.device),
            "slack_t": torch.tensor([seg["slack_class"] for seg in segments], dtype=torch.long, device=self.device),
            "violation_t": torch.tensor([seg["violation"] for seg in segments], dtype=torch.float32, device=self.device),
            "tool_gain_t": torch.tensor([seg["action_gain"] for seg in segments], dtype=torch.float32, device=self.device),
            "tool_gain_mask_t": torch.tensor([seg.get("action_observed_mask", [1.0] * self.action_dim) for seg in segments], dtype=torch.float32, device=self.device),
            "out_t": torch.tensor(
                [[float(seg["node_success"]), float(seg["task_success"]), float(seg["violation"])] for seg in segments],
                dtype=torch.float32,
                device=self.device,
            ),
            "need_t": torch.tensor([float(seg["tool_invoked"] > 0) for seg in segments], dtype=torch.float32, device=self.device),
            "risk_t": torch.tensor([float(seg["no_tool_risk_prior"]) for seg in segments], dtype=torch.float32, device=self.device),
            "confidence_t": torch.tensor([float(seg["confidence"]) for seg in segments], dtype=torch.float32, device=self.device),
        }
        return batch

    def _risk_bucket_text(self, risk: float) -> str:
        if risk < 0.15:
            return "low"
        if risk < 0.35:
            return "moderate"
        return "high"

    def _action_summary(self, best_idx: int) -> str:
        local_idx = self.action_dim - 2
        pause_idx = self.action_dim - 1
        if best_idx == local_idx:
            return "prefer local execution"
        if best_idx == pause_idx:
            return "prefer a brief pause"
        return f"prefer tool {best_idx}"

    def _softmax_list(self, values: Sequence[float], temperature: float = 1.0) -> List[float]:
        vals = list(values)
        if not vals:
            return []
        t = torch.tensor(vals, dtype=torch.float32)
        probs = torch.softmax(t / max(1e-4, float(temperature)), dim=-1)
        return probs.detach().cpu().tolist()

    def _wire_factor_scores(self, seg: Dict[str, object]) -> List[float]:
        steps = seg.get("steps", [])
        if not steps:
            return [0.25, 0.25, 0.25, 0.25]
        queue_pressure = max(step["s_feat"][2] + step["s_feat"][3] for step in steps)
        loss_pressure = sum(step["s_feat"][1] for step in steps) / max(1, len(steps))
        min_slack = min(step["s_feat"][0] for step in steps)
        slack_pressure = max(0.0, 1.0 - min_slack)
        exec_risk = sum(step["s_feat"][4] for step in steps) / max(1, len(steps))
        vals = [
            max(0.0, min(1.0, queue_pressure / 2.0)),
            max(0.0, min(1.0, loss_pressure)),
            max(0.0, min(1.0, slack_pressure)),
            max(0.0, min(1.0, exec_risk)),
        ]
        total = sum(vals)
        if total <= 1e-8:
            return [0.25, 0.25, 0.25, 0.25]
        return [v / total for v in vals]

    def _top_actions(self, gains: Sequence[float], probs: Sequence[float], top_k: Optional[int] = None) -> List[Dict[str, object]]:
        vals = list(gains)
        pr = list(probs)
        if not vals:
            return []
        k = max(1, min(int(top_k or self.cfg.verbalize_top_k_actions), len(vals)))
        order = sorted(range(len(vals)), key=lambda i: vals[i], reverse=True)[:k]
        rows: List[Dict[str, object]] = []
        for idx in order:
            rows.append({
                "index": int(idx),
                "label": self._action_summary(int(idx)).replace("prefer ", ""),
                "gain": float(vals[idx]),
                "prob": float(pr[idx] if idx < len(pr) else 0.0),
            })
        return rows

    def _extract_prompt_facts(self, seg: Dict[str, object], out: Dict[str, torch.Tensor], batch_index: int) -> Dict[str, object]:
        h = out["h_atom"][batch_index]
        risk = float(torch.sigmoid(self.distiller.risk_head(h)).item())
        gains = self.distiller.action_gain_atom_head(h).detach().cpu().tolist()
        action_probs = self._softmax_list(gains, temperature=self.cfg.verbalize_action_temperature)
        top_actions = self._top_actions(gains, action_probs)
        best_idx = int(top_actions[0]["index"]) if top_actions else 0
        head_gate = out.get("head_gate")
        aspect_weights = head_gate[batch_index].detach().cpu().tolist() if head_gate is not None else [0.25, 0.25, 0.25, 0.25]
        wire_factor_scores = self._wire_factor_scores(seg)
        outcome_probs = torch.sigmoid(self.distiller.outcome_head(out["z_out"][batch_index])).detach().cpu().tolist()
        topo_depth_prob = torch.softmax(self.distiller.topo_depth_head(out["z_topo"][batch_index]), dim=-1).detach().cpu().tolist()
        topo_role_prob = torch.softmax(self.distiller.topo_role_head(out["z_topo"][batch_index]), dim=-1).detach().cpu().tolist()
        topo_critical_prob = float(torch.sigmoid(self.distiller.topo_critical_head(out["z_topo"][batch_index])).item())
        wire_queue_prob = torch.softmax(self.distiller.wire_queue_head(out["z_wire"][batch_index]), dim=-1).detach().cpu().tolist()
        wire_slack_prob = torch.softmax(self.distiller.wire_slack_head(out["z_wire"][batch_index]), dim=-1).detach().cpu().tolist()
        wire_violation_prob = float(torch.sigmoid(self.distiller.wire_violation_head(out["z_wire"][batch_index])).item())
        return {
            "risk": risk,
            "gains": gains,
            "action_probs": action_probs,
            "best_idx": best_idx,
            "top_actions": top_actions,
            "aspect_weights": aspect_weights,
            "wire_factor_scores": wire_factor_scores,
            "outcome_probs": outcome_probs,
            "topo_depth_prob": topo_depth_prob,
            "topo_role_prob": topo_role_prob,
            "topo_critical_prob": topo_critical_prob,
            "wire_queue_prob": wire_queue_prob,
            "wire_slack_prob": wire_slack_prob,
            "wire_violation_prob": wire_violation_prob,
            "action_detail_stats": list(seg.get("action_detail_stats", [])),
        }

    def _fmt_seconds_short(self, value: float) -> str:
        try:
            v = float(value)
        except Exception:
            v = 0.0
        return f"{max(0.0, v):.2f}s"

    def _action_detail_text(
        self,
        stats: Sequence[Dict[str, float]],
        top_actions: Sequence[Dict[str, object]],
        top_k: int = 5,
    ) -> str:
        if not top_actions:
            return "no ranked action details available"
        pieces: List[str] = []
        for rank, row in enumerate(list(top_actions)[: max(1, int(top_k))], start=1):
            idx = int(row.get("index", 0))
            label = str(row.get("label", self._action_summary(idx).replace("prefer ", "")))
            st = stats[idx] if 0 <= idx < len(stats) and isinstance(stats[idx], dict) else {}
            decisions = int(round(float(st.get("decisions", 0.0) or 0.0)))
            if decisions > 0:
                succ = float(st.get("success_rate", 0.0) or 0.0)
                total_s = float(st.get("mean_total_s", 0.0) or 0.0)
                exec_s = float(st.get("mean_exec_s", 0.0) or 0.0)
                comm_s = float(st.get("mean_comm_s", 0.0) or 0.0)
                queue_s = float(st.get("mean_queue_s", 0.0) or 0.0)
                risk = float(st.get("mean_risk", 0.0) or 0.0)
                loss = float(st.get("mean_packet_loss", 0.0) or 0.0)
                rejects = int(round(float(st.get("queue_full_rejects", 0.0) or 0.0)))
                reason_bits = []
                if succ >= 0.80:
                    reason_bits.append("high observed node success")
                elif succ <= 0.20:
                    reason_bits.append("low observed node success")
                if queue_s >= 1.0:
                    reason_bits.append("queue pressure")
                if risk >= 0.50:
                    reason_bits.append("high execution risk")
                if loss >= 0.10:
                    reason_bits.append("packet-loss risk")
                if total_s > 0.0 and exec_s / max(total_s, 1e-9) >= 0.65:
                    reason_bits.append("compute-dominated latency")
                if total_s > 0.0 and comm_s / max(total_s, 1e-9) >= 0.35:
                    reason_bits.append("communication-dominated latency")
                if rejects > 0:
                    reason_bits.append(f"{rejects} queue-full rejects")
                reason = "; ".join(reason_bits) if reason_bits else "balanced observed cost"
                pieces.append(
                    f"#{rank} {label}: model_prob={float(row.get('prob', 0.0)):.2f}, gain={float(row.get('gain', 0.0)):.3f}, "
                    f"used={decisions}, success={succ:.2f}, total={self._fmt_seconds_short(total_s)} "
                    f"(exec {self._fmt_seconds_short(exec_s)}, comm {self._fmt_seconds_short(comm_s)}, queue {self._fmt_seconds_short(queue_s)}), "
                    f"risk={risk:.2f}, pkt_loss={loss:.2f}; evidence={reason}"
                )
            else:
                pieces.append(
                    f"#{rank} {label}: model_prob={float(row.get('prob', 0.0)):.2f}, gain={float(row.get('gain', 0.0)):.3f}, "
                    "no direct execution observations in this memory atom"
                )
        return " | ".join(pieces)

    def _fallback_prompt_text(self, seg: Dict[str, object], facts: Dict[str, object]) -> str:
        state = copy.deepcopy(seg.get("state_context", {}) or {})
        env = copy.deepcopy(seg.get("environment_context", {}) or {})
        candidates = list(seg.get("candidate_actions", []) or [])
        rec = copy.deepcopy(seg.get("recommended_action", {}) or {})
        observed = copy.deepcopy(seg.get("observed_outcome", {}) or {})
        static = seg.get("static", {}) or {}
        action_stats = list(facts.get("action_detail_stats", seg.get("action_detail_stats", [])) or [])
        outcome = list(facts.get("outcome_probs", [float(seg.get("node_success", 0)), float(seg.get("task_success", 0)), float(seg.get("violation", 0))]))
        if len(outcome) < 3:
            outcome = outcome + [0.0] * (3 - len(outcome))

        def fmt_float(value: object, digits: int = 3, default: float = 0.0) -> str:
            try:
                return f"{float(value):.{digits}f}"
            except Exception:
                return f"{default:.{digits}f}"

        def action_label(row: Dict[str, object]) -> str:
            at = str(row.get("action_type", ""))
            if at == "local":
                return "LOCAL execution"
            if at == "pause":
                return "PAUSE"
            slot = row.get("slot", row.get("tool_type_id", row.get("action_index", "?")))
            name = row.get("tool_name") or row.get("name") or f"tool {slot}"
            cat = row.get("category", "unknown")
            return f"TOOL slot {slot}: {name} (category={cat})"

        def compact_candidate(row: Dict[str, object]) -> str:
            at = str(row.get("action_type", ""))
            label = action_label(row)
            if at == "tool":
                return (
                    f"{label}; predicted_total={fmt_float(row.get('predicted_total_s'), 3)}s; "
                    f"queue={fmt_float(row.get('base_queue_delay_s'), 3)}s; risk={fmt_float(row.get('risk_estimate'), 3)}; "
                    f"pkt_loss={fmt_float(row.get('packet_loss_rate'), 3)}; reliability={fmt_float(row.get('server_reliability'), 2)}; "
                    f"completion_gain={fmt_float(row.get('completion_gain'), 2)}; queue_blocked={bool(row.get('queue_blocked', False))}"
                )
            if at == "local":
                return (
                    f"{label}; predicted_total={fmt_float(row.get('predicted_total_s'), 3)}s; "
                    f"compute={fmt_float(row.get('compute_s'), 3)}s; fail_probability={fmt_float(row.get('fail_probability'), 3)}; "
                    f"queue_blocked={bool(row.get('queue_blocked', False))}"
                )
            return f"{label}; duration={fmt_float(row.get('pause_duration_s', 0.2), 2)}s"

        rec_idx = int(rec.get("action_index", -1)) if str(rec.get("action_index", "")).lstrip("-").isdigit() else -1
        rec_stats = action_stats[rec_idx] if 0 <= rec_idx < len(action_stats) and isinstance(action_stats[rec_idx], dict) else {}
        rec_label = rec.get("action_label") or action_label(rec)
        command = rec.get("decision_command", "")
        if not command:
            if str(rec.get("action_type")) == "tool":
                slot = rec.get("slot", rec.get("tool_type_id", rec_idx))
                tool_name = rec.get("tool_name") or rec.get("name") or f"tool {slot}"
                command = f"use_tool(slot={slot}, tool_name='{tool_name}')"
            elif str(rec.get("action_type")) == "local":
                command = "use_local()"
            else:
                command = f"pause(duration_s={fmt_float(rec.get('pause_duration_s', 0.2), 2)})"

        allowed = state.get("allowed_slots", [])
        candidate_text = " | ".join(compact_candidate(c) for c in candidates[:6]) if candidates else "no candidate-action table was recorded"
        tool_detail = ""
        if str(rec.get("action_type")) == "tool":
            tool_detail = (
                f"Description={rec.get('description', '')}. compute={fmt_float(rec.get('compute_frequency_hz'), 1)}Hz, "
                f"uplink={fmt_float(rec.get('uplink_bandwidth_hz'), 1)}Hz, downlink={fmt_float(rec.get('downlink_bandwidth_hz'), 1)}Hz, "
                f"coverage={fmt_float(rec.get('coverage_distance_m'), 1)}m, reliability={fmt_float(rec.get('server_reliability'), 2)}, "
                f"success_probability={fmt_float(rec.get('success_probability'), 2)}, semantic_gain={fmt_float(rec.get('semantic_gain'), 2)}, "
                f"completion_gain={fmt_float(rec.get('completion_gain'), 2)}, energy={fmt_float(rec.get('energy_j'), 3)}J, "
                f"requires_tool_to_complete={rec.get('requires_tool_to_complete', state.get('requires_tool_to_complete', 'unknown'))}."
            )
        elif str(rec.get("action_type")) == "local":
            tool_detail = (
                f"Local details: compute={fmt_float(env.get('local_compute_s'), 3)}s, "
                f"fail_probability={fmt_float(env.get('local_fail_probability'), 3)}, "
                f"queue_blocked={bool(env.get('local_queue_blocked', False))}."
            )
        else:
            tool_detail = "Pause details: use only when all tool/local actions are infeasible or unsafe under the current queue/deadline constraints."

        avoid_reason = "Avoid pause if any non-blocked tool/local action can finish before the node deadline. Avoid local execution on tool-required nodes unless all tools are infeasible."
        if str(rec.get("action_type")) == "local":
            avoid_reason = "Avoid tool calls only when their queues or predicted deadline violations dominate; otherwise prefer concrete tools for tool-required semantic work."
        elif str(rec.get("action_type")) == "pause":
            avoid_reason = "Avoid forcing tool/local execution if all candidates are queue-blocked or predicted to violate the node deadline."

        return (
            "[Situation]\n"
            f"Current node: {state.get('node_step') or state.get('intent_text') or 'unknown'}. "
            f"Node type={int(state.get('node_type_id', static.get('node_type_id', 0)))}, critical={int(state.get('critical_flag', seg.get('critical_flag', 0)))}, "
            f"queue_class={int(state.get('queue_class', seg.get('queue_class', 0)))}, slack_class={int(state.get('slack_class', seg.get('slack_class', 0)))}, "
            f"deadline remaining={fmt_float(state.get('node_remaining_deadline_s'), 3)}s. Allowed actions: tools={allowed}, local, pause. "
            f"requires_tool_to_complete={state.get('requires_tool_to_complete', False)}, local_success_probability={state.get('local_success_probability', 'unknown')}.\n"
            "[Environment]\n"
            f"Current pressure={env.get('dominant_pressure', 'balanced')}; local_queue={env.get('local_queue_length', 0)}, "
            f"local_running={env.get('local_running_jobs', 0)}, max_tool_queue={fmt_float(env.get('max_tool_queue_delay_s'), 3)}s, "
            f"max_tool_risk={fmt_float(env.get('max_tool_risk_estimate'), 3)}, max_packet_loss={fmt_float(env.get('max_packet_loss_rate'), 3)}. "
            f"Local execution: predicted_total={fmt_float(env.get('local_predicted_total_s'), 3)}s, fail_probability={fmt_float(env.get('local_fail_probability'), 3)}, "
            f"queue_blocked={bool(env.get('local_queue_blocked', False))}. Candidate actions: {candidate_text}.\n"
            "[Best action]\n"
            f"Choose {rec_label}. Decision command: {command}.\n"
            "[Tool details]\n"
            f"{tool_detail}\n"
            "[Reason]\n"
            f"This action is preferred because {rec.get('why_best', 'it had the best observed action gain under the current deadline/queue/risk context')}. {avoid_reason}\n"
            "[Evidence]\n"
            f"In similar cases: support=1, node_success_rate={fmt_float(rec_stats.get('success_rate', observed.get('success_rate', seg.get('node_success', 0))), 3)}, "
            f"task_success={int(seg.get('task_success', 0))}, mean_total_s={fmt_float(rec_stats.get('mean_total_s', observed.get('mean_total_s', 0.0)), 3)}, "
            f"mean_queue_s={fmt_float(rec_stats.get('mean_queue_s', observed.get('mean_queue_s', 0.0)), 3)}, "
            f"mean_risk={fmt_float(rec_stats.get('mean_risk', observed.get('mean_risk', 0.0)), 3)}, "
            f"action_gain={fmt_float(rec.get('gain', observed.get('recommended_action_gain', 0.0)), 3)}, "
            f"predicted node/task/violation={outcome[0]:.3f}/{outcome[1]:.3f}/{outcome[2]:.3f}, return={float(seg.get('task_return', 0.0)):.3f}."
        )

    def _clean_verbalized_text(self, text: str, fallback: str) -> str:
        cleaned = (text or "").strip()
        if not cleaned:
            return fallback
        cleaned = cleaned.replace("\r", " ").replace("\n", " ")
        cleaned = re.sub(r"<\|[^>]+\|>", " ", cleaned)
        if self.cfg.verbalize_strip_role_markers:
            split_markers = ["Human:", "Assistant:", "System:", "User:", "Structured facts:", "Facts:", "Recommendation:"]
            cut = len(cleaned)
            lower_cleaned = cleaned.lower()
            for marker in split_markers:
                pos = lower_cleaned.find(marker.lower())
                if pos > 0:
                    cut = min(cut, pos)
            cleaned = cleaned[:cut]
            cleaned = re.sub(r"\b(?:Human|Assistant|System|User)\s*:\s*", " ", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s+", " ", cleaned).strip(" \t\"' :-")
        pieces = [p.strip() for p in re.split(r"(?<=[.!?])\s+", cleaned) if p.strip()]
        max_sent = max(1, int(self.cfg.verbalize_clean_max_sentences))
        if pieces:
            cleaned = " ".join(pieces[:max_sent]).strip()
        cleaned = cleaned.strip(" \t\"'")
        suspicious_patterns = ["you are consolidating", "write at most", "structured facts", "human:", "assistant:", "system:", "user:"]
        lowered = cleaned.lower()
        suspicious = any(pat in lowered for pat in suspicious_patterns)
        if cleaned.count("(") > cleaned.count(")"):
            suspicious = True
        if len(cleaned) < 24:
            suspicious = True
        if suspicious and self.cfg.verbalize_fallback_to_template:
            return fallback
        if cleaned and cleaned[-1] not in ".!?":
            cleaned = cleaned + "."
        return cleaned or fallback

    def _tensor_to_text(self, seg: Dict[str, object], facts: Dict[str, object]) -> str:
        fallback = self._fallback_prompt_text(seg, facts)
        if not self.cfg.use_llm_verbalizer:
            return fallback
        top_actions = facts.get("top_actions", [])
        aspect = list(facts.get("aspect_weights", [0.25, 0.25, 0.25, 0.25]))
        wire = list(facts.get("wire_factor_scores", [0.25, 0.25, 0.25, 0.25]))
        outcome = list(facts.get("outcome_probs", [0.5, 0.5, 0.5]))
        top_action_text = self._action_detail_text(
            facts.get("action_detail_stats", seg.get("action_detail_stats", [])),
            top_actions,
            top_k=max(3, int(self.cfg.verbalize_top_k_actions)),
        )
        gen_prompt = (
            "Convert the following execution facts into a detailed but compact skill note for a downstream policy model. "
            "Mention topology, queue/slack/wire conditions, concrete tool recommendation evidence, and expected outcome. "
            "Include the best tools' total/exec/comm/queue latency, risk, packet loss, success evidence, and when to avoid them. "
            "Do not output labels such as Human, Assistant, System, User, Facts, or Recommendation. "
            f"Facts: node_type={int(seg['static']['node_type_id'])}; depth_tier={int(seg['depth_tier'])}; role_id={int(seg['role_id'])}; critical={int(seg['critical_flag'])}; "
            f"queue_class={int(seg['queue_class'])}; slack_class={int(seg['slack_class'])}; "
            f"aspect_weights_topo_wire_tool_out={aspect[0]:.3f},{aspect[1]:.3f},{aspect[2]:.3f},{aspect[3]:.3f}; "
            f"wire_factors_queue_loss_deadline_exec={wire[0]:.3f},{wire[1]:.3f},{wire[2]:.3f},{wire[3]:.3f}; "
            f"no_tool_risk={float(facts['risk']):.3f}; top_actions={top_action_text}; "
            f"outcome_probs_node_task_violation={outcome[0]:.3f},{outcome[1]:.3f},{outcome[2]:.3f}."
        )
        try:
            out = self.llm.verbalize(gen_prompt)
            return self._clean_verbalized_text(out, fallback)
        except Exception:
            return fallback

    def _segments_to_atoms(
        self,
        segments: List[Dict[str, object]],
        progress_prefix: Optional[str] = None,
        progress_every_batches: int = 10,
    ) -> List[KnowledgeAtom]:
        atoms: List[KnowledgeAtom] = []
        self.distiller.eval()
        self.llm.set_adapter_train(False)
        bs = max(1, int(self.cfg.batch_size))
        total_batches = int(math.ceil(len(segments) / float(bs))) if segments else 0
        progress_t0 = time.perf_counter()
        last_progress_t = progress_t0
        if progress_prefix:
            print(f"{progress_prefix} atom_build_start | segments={len(segments)} batch={bs} batches={total_batches}", flush=True)
        with torch.no_grad():
            for batch_idx, start in enumerate(range(0, len(segments), bs), start=1):
                batch_t0 = time.perf_counter()
                batch_segments = segments[start: start + bs]
                batch = self._build_batch(batch_segments, detach_text=False)
                out = self.distiller.forward(batch)
                risks = torch.sigmoid(self.distiller.risk_head(out["h_atom"]).squeeze(-1))
                gains = self.distiller.action_gain_atom_head(out["h_atom"])
                confs = torch.sigmoid(self.distiller.conf_head(out["h_atom"]).squeeze(-1))
                action_probs = torch.softmax(gains / max(1e-4, float(self.cfg.verbalize_action_temperature)), dim=-1)
                outcome_probs = torch.sigmoid(self.distiller.outcome_head(out["z_out"]))
                head_gate = out.get("head_gate")
                for i, seg in enumerate(batch_segments):
                    h = out["h_atom"][i]
                    facts = self._extract_prompt_facts(seg, out, i)
                    prompt_text = self._tensor_to_text(seg, facts)
                    atoms.append(KnowledgeAtom(
                        atom_id=str(uuid.uuid4()),
                        node_type_id=int(seg["static"]["node_type_id"]),
                        depth_tier=int(seg["depth_tier"]),
                        role_id=int(seg["role_id"]),
                        critical_flag=int(seg["critical_flag"]),
                        queue_class=int(seg["queue_class"]),
                        slack_class=int(seg["slack_class"]),
                        no_tool_risk_prior=float(risks[i].item()),
                        action_gain_priors=gains[i].detach().cpu().tolist(),
                        confidence=float(confs[i].item()),
                        embedding=h.detach().cpu().tolist(),
                        prompt_text=prompt_text,
                        action_counts=list(seg["action_counts"]),
                        action_probs=action_probs[i].detach().cpu().tolist(),
                        action_detail_stats=list(seg.get("action_detail_stats", [])),
                        aspect_weights=(head_gate[i].detach().cpu().tolist() if head_gate is not None else []),
                        wire_factor_scores=list(facts.get("wire_factor_scores", [])),
                        outcome_probs=outcome_probs[i].detach().cpu().tolist(),
                        state_context=copy.deepcopy(seg.get("state_context", {})),
                        environment_context=copy.deepcopy(seg.get("environment_context", {})),
                        candidate_actions=copy.deepcopy(seg.get("candidate_actions", [])),
                        recommended_action=copy.deepcopy(seg.get("recommended_action", {})),
                        observed_outcome=copy.deepcopy(seg.get("observed_outcome", {})),
                    ))
                if progress_prefix:
                    now = time.perf_counter()
                    should_print = (
                        batch_idx == 1
                        or batch_idx == total_batches
                        or batch_idx % max(1, int(progress_every_batches)) == 0
                        or now - last_progress_t >= 15.0
                    )
                    if should_print:
                        last_progress_t = now
                        elapsed = now - progress_t0
                        avg_batch = elapsed / max(1, batch_idx)
                        eta = avg_batch * max(0, total_batches - batch_idx)
                        width = 28
                        done = int(round(width * batch_idx / max(1, total_batches)))
                        bar = "[" + "#" * done + "." * (width - done) + "]"
                        print(
                            f"{progress_prefix} {bar} {batch_idx}/{total_batches} atoms={len(atoms)} "
                            f"last_batch={now - batch_t0:.2f}s avg_batch={avg_batch:.2f}s eta~{eta:.1f}s",
                            flush=True,
                        )
        if progress_prefix:
            print(f"{progress_prefix} atom_build_done | atoms={len(atoms)} time={time.perf_counter() - progress_t0:.2f}s", flush=True)
        return atoms

    def fit_on_task_records(self, task_records: List[Dict[str, object]], batch_size: Optional[int] = None, epochs: Optional[int] = None) -> Dict[str, object]:
        t0 = time.perf_counter()
        if self.cfg.verbose:
            print(f"{self.cfg.print_prefix} fit start | task_records={len(task_records)}", flush=True)
        segments = self.extractor.extract(task_records)
        if self.cfg.verbose:
            print(f"{self.cfg.print_prefix} extracted segments={len(segments)}", flush=True)
        max_segments = int(getattr(self.cfg, "fit_max_segments", 0) or 0)
        if max_segments > 0 and len(segments) > max_segments:
            segments = segments[:max_segments]
            if self.cfg.verbose:
                print(f"{self.cfg.print_prefix} conservative cap | train_segments={len(segments)}", flush=True)
        if not segments:
            self.last_fit_stats = {
                "loss_total": 0.0,
                "loss_topo": 0.0,
                "loss_wire": 0.0,
                "loss_tool": 0.0,
                "loss_out": 0.0,
                "loss_need": 0.0,
                "loss_atom": 0.0,
                "loss_bind_pos": 0.0,
                "loss_sep": 0.0,
                "loss_block": 0.0,
                "loss_div": 0.0,
                "num_segments": 0,
                "num_new_atoms": 0,
                "num_new_prototypes": 0,
                "prototype_risk_drift": 0.0,
                "epoch_history": [],
            }
            return self.last_fit_stats

        bs = batch_size or self.cfg.batch_size
        ep = epochs or self.cfg.epochs_per_fit
        global_stats_accum: Dict[str, float] = {}
        total_updates = 0
        epoch_history: List[Dict[str, object]] = []
        step_history: List[Dict[str, object]] = []

        self.distiller.train()
        train_llm_adapters = not bool(getattr(self.cfg, "freeze_llm_adapters_during_fit", False)) and not bool(getattr(self.cfg, "fast_update_text_encoder", False))
        self.llm.set_adapter_train(train_llm_adapters)
        if self.cfg.verbose:
            print(
                f"{self.cfg.print_prefix} conservative mode | fast_text={bool(getattr(self.cfg, 'fast_update_text_encoder', False))} | "
                f"train_llm_adapters={train_llm_adapters}",
                flush=True,
            )
        order = list(range(len(segments)))
        for epoch_idx in range(ep):
            if self.cfg.shuffle_segments_each_epoch:
                self.rng.shuffle(order)
            epoch_stats_accum: Dict[str, float] = {}
            epoch_updates = 0
            if self.cfg.verbose:
                print(f"{self.cfg.print_prefix} epoch {epoch_idx + 1}/{ep} start | batch_size={bs} | samples={len(order)}", flush=True)
            for start in range(0, len(order), bs):
                idx = order[start: start + bs]
                batch_segments = [segments[i] for i in idx]
                batch = self._build_batch(batch_segments, detach_text=False)
                loss, stats, _ = self.distiller.loss_on_batch(batch)
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                params_to_clip = [
                    p for p in (list(self.distiller.parameters()) + list(self.llm.trainable_parameters()))
                    if p.grad is not None
                ]
                if params_to_clip:
                    torch.nn.utils.clip_grad_norm_(params_to_clip, 2.0)
                self.optimizer.step()
                total_updates += 1
                epoch_updates += 1
                self.fit_step += 1
                for k, v in stats.items():
                    global_stats_accum[k] = global_stats_accum.get(k, 0.0) + float(v)
                    epoch_stats_accum[k] = epoch_stats_accum.get(k, 0.0) + float(v)
                step_rec = {"update": total_updates, "epoch": epoch_idx + 1, **stats}
                step_history.append(step_rec)
                if self.cfg.verbose and self.cfg.train_log_mode in {"step", "both"}:
                    if total_updates <= 2 or total_updates % max(1, self.cfg.log_every_fit_step) == 0:
                        total_expected_updates = max(1, int(ep) * int(math.ceil(len(order) / float(max(1, bs)))))
                        elapsed = time.perf_counter() - t0
                        avg_update = elapsed / max(1, total_updates)
                        eta = avg_update * max(0, total_expected_updates - total_updates)
                        width = 28
                        done = int(round(width * total_updates / max(1, total_expected_updates)))
                        bar = "[" + "#" * done + "." * (width - done) + "]"
                        print(
                            f"{self.cfg.print_prefix} {bar} update={total_updates}/{total_expected_updates} | epoch={epoch_idx + 1}/{ep} | "
                            f"loss_total={float(stats.get('loss_total', 0.0)):.4f} | loss_topo={float(stats.get('loss_topo', 0.0)):.4f} | "
                            f"loss_wire={float(stats.get('loss_wire', 0.0)):.4f} | loss_tool={float(stats.get('loss_tool', 0.0)):.4f} | "
                            f"loss_out={float(stats.get('loss_out', 0.0)):.4f} | avg_update={avg_update:.2f}s eta~{eta:.1f}s",
                            flush=True,
                        )
            epoch_avg = {k: v / max(1, epoch_updates) for k, v in epoch_stats_accum.items()}
            epoch_avg["epoch"] = epoch_idx + 1
            epoch_avg["num_updates"] = epoch_updates
            epoch_avg["num_samples"] = len(order)
            epoch_history.append(epoch_avg)
            if self.cfg.verbose and self.cfg.train_log_mode in {"epoch", "both"}:
                print(
                    f"{self.cfg.print_prefix} epoch={epoch_idx + 1}/{ep} done | loss_total={float(epoch_avg.get('loss_total', 0.0)):.4f} | "
                    f"loss_topo={float(epoch_avg.get('loss_topo', 0.0)):.4f} | loss_wire={float(epoch_avg.get('loss_wire', 0.0)):.4f} | "
                    f"loss_tool={float(epoch_avg.get('loss_tool', 0.0)):.4f} | loss_out={float(epoch_avg.get('loss_out', 0.0)):.4f}",
                    flush=True,
                )

        self.llm.set_adapter_train(False)
        atom_segments = segments
        max_new_atoms = int(getattr(self.cfg, "fit_max_new_atoms", 0) or 0)
        if max_new_atoms > 0 and len(atom_segments) > max_new_atoms:
            atom_segments = atom_segments[:max_new_atoms]
        if self.cfg.verbose:
            print(f"{self.cfg.print_prefix} converting segments to atoms ... | atom_segments={len(atom_segments)}", flush=True)
        atoms = self._segments_to_atoms(atom_segments, progress_prefix=(f"{self.cfg.print_prefix} atom-build" if self.cfg.verbose else None), progress_every_batches=max(1, int(self.cfg.log_every_fit_step)))
        proto_count_before = len(self.memory.prototypes)
        for atom in atoms:
            self.memory.add_atom(atom)
        proto_count_after = len(self.memory.prototypes)

        proto_risk_mean = float(sum(p.no_tool_risk_mean for p in self.memory.prototypes) / max(1, len(self.memory.prototypes)))
        proto_risk_drift = abs(proto_risk_mean - self.last_snapshot.get("prototype_risk_mean", 0.0))
        self.last_snapshot = {
            "prototype_risk_mean": proto_risk_mean,
            "prototype_count": float(len(self.memory.prototypes)),
        }

        out = {k: v / max(1, total_updates) for k, v in global_stats_accum.items()}
        out.update({
            "num_segments": len(segments),
            "num_new_atoms": len(atoms),
            "num_new_prototypes": proto_count_after - proto_count_before,
            "prototype_risk_mean": proto_risk_mean,
            "prototype_risk_drift": proto_risk_drift,
            "fit_seconds": time.perf_counter() - t0,
            "epoch_history": epoch_history,
            "step_history": step_history,
            "total_updates": total_updates,
        })
        self.last_fit_stats = out
        if self.cfg.verbose:
            print(
                f"{self.cfg.print_prefix} fit done | segments={len(segments)} | new_atoms={len(atoms)} | "
                f"new_prototypes={proto_count_after - proto_count_before} | prototype_risk_mean={proto_risk_mean:.4f} | seconds={out['fit_seconds']:.2f}",
                flush=True,
            )
        return out

    def query_memory(self, state_query_embedding: List[float], node_type_id: int, queue_class: int, slack_class: int) -> Dict[str, object]:
        retrieved = self.memory.query(
            query_embedding=state_query_embedding,
            node_type_id=node_type_id,
            queue_class=queue_class,
            slack_class=slack_class,
            k=self.cfg.k_retrieve,
        )
        if not retrieved:
            return {
                "memory_vector": [0.0] * (self.action_dim + 4),
                "action_bias": [0.0] * self.action_dim,
                "selected": [],
                "knowledge_prompt": "",
            }

        total_weight = sum(max(1e-6, score) for _, score in retrieved)
        risk = 0.0
        conf = 0.0
        support = 0.0
        action_bias = [0.0] * self.action_dim
        selected = []
        prompt_lines = []
        for proto, score in retrieved:
            w = max(1e-6, score) / total_weight
            risk += w * proto.no_tool_risk_mean
            conf += w * proto.confidence_mean
            support += w * float(proto.support)
            for i in range(self.action_dim):
                action_bias[i] += w * proto.action_gain_means[i]
            selected.append({
                "prototype_id": proto.prototype_id,
                "score": score,
                "support": proto.support,
                "node_type_id": proto.node_type_id,
                "queue_class": proto.queue_class,
                "slack_class": proto.slack_class,
            })
            if proto.prompt_text:
                prompt_lines.append(proto.prompt_text)

        memory_vector = [risk, conf, support / 10.0, float(len(retrieved)) / max(1.0, float(self.cfg.k_retrieve))] + list(action_bias)
        return {
            "memory_vector": memory_vector,
            "action_bias": action_bias,
            "selected": selected,
            "knowledge_prompt": "\n".join(prompt_lines[: self.cfg.k_retrieve]),
            "query_embedding_dim": len(state_query_embedding),
            "proto_embedding_dim": self.memory.embedding_dim(),
        }

    def knowledge_digest(self) -> Dict[str, object]:
        top_prompts = [p.prompt_text for p in self.memory.prototypes[:5] if p.prompt_text]
        return {
            "num_atoms": len(self.memory.buffer),
            "num_prototypes": len(self.memory.prototypes),
            "prototype_risk_mean": self.last_snapshot.get("prototype_risk_mean", 0.0),
            "sample_prompts": top_prompts,
            "embedding_dim": self.memory.embedding_dim(),
            "adapter_layers": list(self.llm.selected_layer_ids),
        }

    def save(self, model_path: str, memory_path: str) -> Dict[str, str]:
        mp = Path(model_path).expanduser().resolve()
        kp = Path(memory_path).expanduser().resolve()
        mp.parent.mkdir(parents=True, exist_ok=True)
        kp.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "distiller_state_dict": self.distiller.state_dict(),
                "llm_adapter_state_dict": self.llm.adapter_state_dict(),
                "cfg": asdict(self.cfg),
                "action_dim": self.action_dim,
            },
            mp,
        )
        kp.write_text(json.dumps(self.memory.to_json(), ensure_ascii=False, indent=2), encoding="utf-8")
        return {"model_path": str(mp), "memory_path": str(kp)}

    @classmethod
    def load(cls, dataset: Dict[str, object], model_path: str, memory_path: Optional[str] = None, action_dim: Optional[int] = None) -> "KnowledgeTrainer":
        ckpt = torch.load(model_path, map_location="cpu")
        cfg = KnowledgeConfig(**ckpt["cfg"])
        trainer = cls(dataset=dataset, action_dim=int(action_dim or ckpt["action_dim"]), cfg=cfg)
        trainer.distiller.load_state_dict(ckpt["distiller_state_dict"])
        trainer.llm.load_adapter_state_dict(ckpt.get("llm_adapter_state_dict", {}), strict=False)
        if memory_path is not None and Path(memory_path).exists():
            data = json.loads(Path(memory_path).read_text(encoding="utf-8"))
            trainer.memory.load_json(data)
            trainer.last_snapshot["prototype_risk_mean"] = float(
                sum(p.no_tool_risk_mean for p in trainer.memory.prototypes) / max(1, len(trainer.memory.prototypes))
            )
            trainer.last_snapshot["prototype_count"] = float(len(trainer.memory.prototypes))
        return trainer
