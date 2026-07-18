from __future__ import annotations

import copy
import heapq
import json
import math
import random
import statistics
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Deque, Dict, List, Optional, Tuple




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

# ============================================================
# Data models
# ============================================================


def _median(values: List[float], default: float = 1.0) -> float:
    vals = [float(v) for v in values if v is not None]
    if not vals:
        return float(default)
    try:
        return float(statistics.median(vals))
    except Exception:
        vals = sorted(vals)
        return float(vals[len(vals)//2])


def normalize_dataset_for_observation(
    dataset: Dict[str, object],
    tool_profiles: List["ToolTypeProfile"],
    local_profile: "LocalExecutionProfile",
    env_cfg: "EnvironmentConfig",
) -> Dict[str, object]:
    ds = _normalize_dataset_structure(copy.deepcopy(dataset))
    cfg = ds.setdefault("config", {})
    if bool(cfg.get("observation_normalized", False)):
        return ds

    if not bool(getattr(env_cfg, "normalize_dataset_for_observation", True)):
        return ds

    graphs = list(ds.get("graphs", []))
    if not graphs:
        cfg["observation_normalized"] = True
        return ds

    best_tool_freq = max([float(p.compute_frequency_hz) for p in tool_profiles] + [1.0])
    best_ul_bw = max([float(p.uplink_bandwidth_hz) for p in tool_profiles] + [1.0])
    best_dl_bw = max([float(p.downlink_bandwidth_hz) for p in tool_profiles] + [1.0])

    raw_best_compute = []
    raw_best_ul = []
    raw_best_dl = []
    for graph in graphs:
        for node in graph.get("nodes", []):
            raw_best_compute.append(float(node.get("computation_cycles", 0.0)) / max(best_tool_freq, 1e-9))
            raw_best_ul.append(float(node.get("uplink_size_bits", 0.0)) / max(best_ul_bw, 1e-9))
            raw_best_dl.append(float(node.get("downlink_size_bits", 0.0)) / max(best_dl_bw, 1e-9))

    median_best_compute = _median(raw_best_compute, 1.0)
    median_best_ul = _median(raw_best_ul, 0.25)
    median_best_dl = _median(raw_best_dl, 0.25)

    for graph in graphs:
        node_nominal_totals: List[float] = []
        for node in graph.get("nodes", []):
            raw_compute = float(node.get("computation_cycles", 0.0)) / max(best_tool_freq, 1e-9)
            raw_ul = float(node.get("uplink_size_bits", 0.0)) / max(best_ul_bw, 1e-9)
            raw_dl = float(node.get("downlink_size_bits", 0.0)) / max(best_dl_bw, 1e-9)

            compute_ratio = max(0.35, min(2.0, raw_compute / max(median_best_compute, 1e-9)))
            ul_ratio = max(0.35, min(2.0, raw_ul / max(median_best_ul, 1e-9)))
            dl_ratio = max(0.35, min(2.0, raw_dl / max(median_best_dl, 1e-9)))

            best_tool_compute_s = float(env_cfg.target_tool_compute_s) * compute_ratio
            target_uplink_s = float(env_cfg.target_tool_uplink_s) * ul_ratio
            target_downlink_s = float(env_cfg.target_tool_downlink_s) * dl_ratio

            node["computation_cycles"] = float(best_tool_compute_s * best_tool_freq)
            node["uplink_size_bits"] = max(1, int(round(target_uplink_s * best_ul_bw)))
            node["downlink_size_bits"] = max(1, int(round(target_downlink_s * best_dl_bw)))

            nominal_total = best_tool_compute_s + target_uplink_s + target_downlink_s
            node_nominal_totals.append(float(nominal_total))
            node["node_deadline"] = float(max(env_cfg.min_node_deadline_s, env_cfg.node_deadline_multiplier * nominal_total))

        task_nominal_total = float(sum(node_nominal_totals))
        graph["task_deadline"] = float(max(env_cfg.min_task_deadline_s, env_cfg.task_deadline_multiplier * task_nominal_total))

    cfg["observation_normalized"] = True
    cfg["normalization_summary"] = {
        "target_tool_compute_s": float(env_cfg.target_tool_compute_s),
        "target_tool_uplink_s": float(env_cfg.target_tool_uplink_s),
        "target_tool_downlink_s": float(env_cfg.target_tool_downlink_s),
        "node_deadline_multiplier": float(env_cfg.node_deadline_multiplier),
        "task_deadline_multiplier": float(env_cfg.task_deadline_multiplier),
    }
    return ds



@dataclass
class ToolTypeProfile:
    tool_type_id: int
    name: str
    instances: int
    compute_frequency_hz: float
    uplink_bandwidth_hz: float
    downlink_bandwidth_hz: float
    queue_capacity: int
    coverage_distance_m: float
    server_reliability: float
    info_richness: float
    validity_horizon_s: float


@dataclass
class LocalExecutionProfile:
    local_cpu_frequency_hz: float = 3.2e9
    local_failure_base: float = 0.05
    local_failure_compute_scale: float = 8.0e9
    local_jitter_ratio: float = 0.0


@dataclass
class EnvironmentConfig:
    dt: float = 0.10
    seed: int = 7
    q_max: int = 10
    task_arrival_default_gap_range: Tuple[float, float] = (1.0, 1.0)
    use_integer_arrival_slots: bool = True
    arrival_slot_step_s: float = 1.0

    # Compact, observation-friendly timing normalization
    normalize_dataset_for_observation: bool = True
    target_tool_compute_s: float = 1.0
    target_tool_uplink_s: float = 0.25
    target_tool_downlink_s: float = 0.25
    node_deadline_multiplier: float = 3.2
    task_deadline_multiplier: float = 2.6
    min_node_deadline_s: float = 2.0
    min_task_deadline_s: float = 6.0
    use_compact_deterministic_channel: bool = True
    compact_channel_quality_floor: float = 0.55
    compact_channel_quality_span: float = 0.30

    # Wireless: 2.4 GHz IoT-like channel
    carrier_frequency_hz: float = 2.4e9
    tx_power_w: float = 0.1                 # 20 dBm
    noise_power_density_w_hz: float = 3.98e-21   # ~ -174 dBm/Hz
    path_loss_exponent: float = 2.7
    reference_distance_m: float = 1.0
    reference_path_gain: float = 1e-3
    shadowing_sigma_db: float = 4.0
    rician_k_factor: float = 4.0
    packet_loss_snr_mid_db: float = 5.0
    packet_loss_snr_scale_db: float = 2.5

    # Queue and resource semantics
    risk_weights: Tuple[float, float, float, float] = (0.32, 0.28, 0.22, 0.18)
    retry_restart_penalty_s: float = 0.10
    local_max_concurrency: int = 4
    local_queue_capacity: int = 64

    # Prompt-only action mode. When the policy layer no longer masks invalid
    # actions, the environment must convert invalid/rejected decisions into a
    # clear terminal negative signal instead of raising or silently returning.
    # Supported modes: "fail_task" and "keep_ready".
    invalid_action_mode: str = "fail_task"
    invalid_action_penalty_s: float = 0.0
    fail_on_queue_rejection: bool = True
    max_invalid_attempts_per_node: int = 2

    # Metrics sampling
    metrics_record_period_steps: int = 1


@dataclass
class DecisionAction:
    action_type: str  # "tool", "local", "pause"
    tool_type_id: Optional[int] = None
    pause_duration_s: float = 0.0


@dataclass
class NodeRuntime:
    node_id: int
    node_type_id: int
    computation_cycles: float
    uplink_size_bits: int
    downlink_size_bits: int
    allowed_tools_mask: List[int]
    node_deadline: float
    predecessors: List[int] = field(default_factory=list)
    successors: List[int] = field(default_factory=list)
    status: str = "idle"  # idle, ready, paused, queued, queued_local, running_tool, running_local, done
    activation_time: Optional[float] = None
    completion_time: Optional[float] = None
    last_start_time: Optional[float] = None
    attempt_count: int = 0


@dataclass
class TaskRuntime:
    graph_id: str
    num_nodes: int
    task_deadline: float
    arrival_time: float
    nodes: Dict[int, NodeRuntime]
    edges: List[Tuple[int, int]]
    status: str = "waiting"  # waiting, active, completed, failed
    started: bool = False
    completed_nodes: List[int] = field(default_factory=list)
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    restart_count: int = 0
    failure_reason: Optional[str] = None
    log: List[Dict[str, object]] = field(default_factory=list)


@dataclass
class QueueItem:
    task_id: str
    node_id: int
    enqueue_time: float
    predicted_service_s: float
    uplink_s: float
    downlink_s: float
    risk_snapshot: float


@dataclass
class RunningToolJob:
    task_id: str
    node_id: int
    tool_type_id: int
    start_time: float
    finish_time: float
    uplink_s: float
    exec_s: float
    downlink_s: float
    queue_delay_s: float
    packet_loss_rate: float
    risk_estimate: float


@dataclass
class LocalQueueItem:
    task_id: str
    node_id: int
    enqueue_time: float
    predicted_compute_s: float
    fail_probability: float


@dataclass
class RunningLocalJob:
    task_id: str
    node_id: int
    start_time: float
    finish_time: float
    compute_s: float
    fail_probability: float
    queue_delay_s: float = 0.0


@dataclass(order=True)
class TimedEvent:
    trigger_time: float
    event_kind: str
    task_id: str
    node_id: int
    payload: Dict[str, object] = field(compare=False, default_factory=dict)


# ============================================================
# Modular estimators
# ============================================================


def default_queue_delay_estimator(
    queue_items: List[QueueItem],
    candidate_compute_s: float,
    candidate_comm_s: float,
) -> float:
    """Base queue delay = sum of (compute + communication) of all waiting items."""
    total = 0.0
    for item in queue_items:
        total += item.predicted_service_s + item.uplink_s + item.downlink_s
    return total


def default_queue_risk_estimator(
    compute_s: float,
    queue_delay_s: float,
    packet_loss_rate: float,
    congestion_ratio: float,
    weights: Tuple[float, float, float, float],
) -> float:
    wc, wq, wl, wg = weights
    comp_term = 1.0 - math.exp(-compute_s / 2.0)
    queue_term = 1.0 - math.exp(-queue_delay_s / 3.0)
    loss_term = max(0.0, min(1.0, packet_loss_rate))
    cong_term = max(0.0, min(1.0, congestion_ratio))
    risk = wc * comp_term + wq * queue_term + wl * loss_term + wg * cong_term
    return max(0.0, min(1.0, risk))


# ============================================================
# Environment
# ============================================================


class MissionEnvironment:
    def __init__(
        self,
        dataset: Dict[str, object],
        env_cfg: Optional[EnvironmentConfig] = None,
        tool_profiles: Optional[List[ToolTypeProfile]] = None,
        local_profile: Optional[LocalExecutionProfile] = None,
        queue_delay_estimator: Optional[Callable[[List[QueueItem], float, float], float]] = None,
        queue_risk_estimator: Optional[Callable[[float, float, float, float, Tuple[float, float, float, float]], float]] = None,
        tool_library: Optional[Dict[str, object]] = None,
    ) -> None:
        self.env_cfg = env_cfg or EnvironmentConfig()
        self.local_profile = local_profile or LocalExecutionProfile()
        self.rng = random.Random(self.env_cfg.seed)

        base_dataset = _normalize_dataset_structure(copy.deepcopy(dataset))
        base_tool_profiles = tool_profiles or self._build_default_tool_profiles(len(base_dataset["tool_catalog"]))
        self.tool_library = copy.deepcopy(tool_library) if isinstance(tool_library, dict) else None
        normalization_tool_profiles = self._build_library_scale_tool_profiles(base_tool_profiles, self.tool_library)
        self.dataset = normalize_dataset_for_observation(
            dataset=base_dataset,
            tool_profiles=normalization_tool_profiles,
            local_profile=self.local_profile,
            env_cfg=self.env_cfg,
        )
        dataset = self.dataset

        self.tool_profiles = base_tool_profiles
        self.tool_profile_map = {p.tool_type_id: p for p in self.tool_profiles}
        self.normalization_tool_profiles = normalization_tool_profiles
        self.dynamic_tool_profile_map = self._build_dynamic_tool_profile_map(self.tool_library)
        self.dynamic_tool_info_map = self._build_dynamic_tool_info_map(self.tool_library)
        self.queue_delay_estimator = queue_delay_estimator or default_queue_delay_estimator
        self.queue_risk_estimator = queue_risk_estimator or default_queue_risk_estimator

        self.time: float = 0.0
        self.step_count: int = 0
        self.arrival_schedule: Dict[str, float] = {}
        self.tasks: Dict[str, TaskRuntime] = {}

        self.waiting_arrivals: List[Tuple[float, str]] = []
        self.pause_events: List[TimedEvent] = []
        heapq.heapify(self.pause_events)

        self.tool_queues: Dict[int, Deque[QueueItem]] = {p.tool_type_id: deque() for p in self.tool_profiles}
        self.tool_running: Dict[int, List[RunningToolJob]] = {p.tool_type_id: [] for p in self.tool_profiles}
        self.local_queue: Deque[LocalQueueItem] = deque()
        self.local_running: List[RunningLocalJob] = []

        self.metrics_history: List[Dict[str, float]] = []
        self.completed_task_records: List[Dict[str, object]] = []
        self._decision_cache: Dict[Tuple[str, int], Dict[str, object]] = {}

    # --------------------------------------------------------
    # Public setup
    # --------------------------------------------------------

    def _build_default_tool_profiles(self, num_tools: int) -> List[ToolTypeProfile]:
        profiles: List[ToolTypeProfile] = []
        for i in range(num_tools):
            profiles.append(
                ToolTypeProfile(
                    tool_type_id=i,
                    name=f"tool_type_{i}",
                    instances=1 + (i % 3),
                    compute_frequency_hz=2.5e9 + 0.4e9 * i,
                    uplink_bandwidth_hz=1.2e6 + 0.15e6 * i,
                    downlink_bandwidth_hz=1.0e6 + 0.12e6 * i,
                    queue_capacity=self.env_cfg.q_max,
                    coverage_distance_m=60.0 + 8.0 * i,
                    server_reliability=max(0.85, min(0.995, 0.90 + 0.01 * i)),
                    info_richness=max(0.4, min(0.98, 0.55 + 0.04 * i)),
                    validity_horizon_s=4.0 + 1.0 * i,
                )
            )
        return profiles

    @staticmethod
    def _safe_profile_float(value: object, default: float) -> float:
        try:
            if value is None:
                return float(default)
            return float(value)
        except Exception:
            return float(default)

    @staticmethod
    def _safe_profile_int(value: object, default: int) -> int:
        try:
            if value is None:
                return int(default)
            return int(value)
        except Exception:
            return int(default)

    @staticmethod
    def _tool_library_entries(tool_library: Optional[Dict[str, object]]) -> List[Dict[str, object]]:
        if not isinstance(tool_library, dict):
            return []
        entries = tool_library.get("entries", [])
        if not isinstance(entries, list):
            return []
        return [e for e in entries if isinstance(e, dict)]

    @staticmethod
    def _median_positive(values: List[float], default: float) -> float:
        vals = sorted(float(v) for v in values if v is not None and float(v) > 0.0)
        if not vals:
            return float(default)
        mid = len(vals) // 2
        if len(vals) % 2 == 1:
            return float(vals[mid])
        return float(0.5 * (vals[mid - 1] + vals[mid]))

    def _profile_from_library_tool(self, base_profile: ToolTypeProfile, tool: Dict[str, object], slot_id: int) -> ToolTypeProfile:
        return ToolTypeProfile(
            tool_type_id=int(slot_id),
            name=str(tool.get("tool_name") or tool.get("name") or base_profile.name),
            instances=int(base_profile.instances),
            compute_frequency_hz=self._safe_profile_float(tool.get("compute_frequency_hz"), base_profile.compute_frequency_hz),
            uplink_bandwidth_hz=self._safe_profile_float(tool.get("uplink_bandwidth_hz"), base_profile.uplink_bandwidth_hz),
            downlink_bandwidth_hz=self._safe_profile_float(tool.get("downlink_bandwidth_hz"), base_profile.downlink_bandwidth_hz),
            queue_capacity=int(base_profile.queue_capacity),
            coverage_distance_m=self._safe_profile_float(tool.get("coverage_distance_m"), base_profile.coverage_distance_m),
            server_reliability=self._safe_profile_float(
                tool.get("server_reliability", tool.get("success_probability")),
                base_profile.server_reliability,
            ),
            info_richness=self._safe_profile_float(tool.get("info_richness"), base_profile.info_richness),
            validity_horizon_s=self._safe_profile_float(tool.get("validity_horizon_s"), base_profile.validity_horizon_s),
        )

    def _build_library_scale_tool_profiles(
        self,
        base_profiles: List[ToolTypeProfile],
        tool_library: Optional[Dict[str, object]],
    ) -> List[ToolTypeProfile]:
        """
        Build representative per-slot profiles for dataset normalization only.
        Queue capacity and instance count remain the original slot-level resources;
        timing-related scales are aggregated from the dynamic tool library.
        """
        base_map = {int(p.tool_type_id): p for p in base_profiles}
        values: Dict[int, Dict[str, List[float]]] = {}
        names: Dict[int, str] = {}
        for entry in self._tool_library_entries(tool_library):
            for tool in entry.get("tools", []) or []:
                if not isinstance(tool, dict):
                    continue
                slot_id = self._safe_profile_int(tool.get("slot", tool.get("tool_type_id")), -1)
                if slot_id < 0 or slot_id not in base_map:
                    continue
                bucket = values.setdefault(
                    slot_id,
                    {
                        "compute_frequency_hz": [],
                        "uplink_bandwidth_hz": [],
                        "downlink_bandwidth_hz": [],
                        "coverage_distance_m": [],
                        "server_reliability": [],
                        "info_richness": [],
                        "validity_horizon_s": [],
                    },
                )
                names.setdefault(slot_id, str(tool.get("tool_name") or tool.get("name") or base_map[slot_id].name))
                for key in bucket:
                    raw = tool.get(key, tool.get("success_probability")) if key == "server_reliability" else tool.get(key)
                    try:
                        val = float(raw)
                    except Exception:
                        continue
                    if val > 0.0:
                        bucket[key].append(val)

        profiles: List[ToolTypeProfile] = []
        for base in base_profiles:
            slot_id = int(base.tool_type_id)
            bucket = values.get(slot_id)
            if not bucket:
                profiles.append(copy.deepcopy(base))
                continue
            profiles.append(
                ToolTypeProfile(
                    tool_type_id=slot_id,
                    name=names.get(slot_id, base.name),
                    instances=int(base.instances),
                    compute_frequency_hz=self._median_positive(bucket["compute_frequency_hz"], base.compute_frequency_hz),
                    uplink_bandwidth_hz=self._median_positive(bucket["uplink_bandwidth_hz"], base.uplink_bandwidth_hz),
                    downlink_bandwidth_hz=self._median_positive(bucket["downlink_bandwidth_hz"], base.downlink_bandwidth_hz),
                    queue_capacity=int(base.queue_capacity),
                    coverage_distance_m=self._median_positive(bucket["coverage_distance_m"], base.coverage_distance_m),
                    server_reliability=max(0.0, min(1.0, self._median_positive(bucket["server_reliability"], base.server_reliability))),
                    info_richness=max(0.0, min(1.0, self._median_positive(bucket["info_richness"], base.info_richness))),
                    validity_horizon_s=self._median_positive(bucket["validity_horizon_s"], base.validity_horizon_s),
                )
            )
        return profiles

    def _build_dynamic_tool_profile_map(
        self,
        tool_library: Optional[Dict[str, object]],
    ) -> Dict[Tuple[str, int, int], ToolTypeProfile]:
        dynamic_map: Dict[Tuple[str, int, int], ToolTypeProfile] = {}
        for entry in self._tool_library_entries(tool_library):
            graph_id = str(entry.get("graph_id", ""))
            node_id = self._safe_profile_int(entry.get("node_id"), -1)
            if not graph_id or node_id < 0:
                continue
            entry_mask = entry.get("allowed_tools_mask", [])
            for tool in entry.get("tools", []) or []:
                if not isinstance(tool, dict):
                    continue
                slot_id = self._safe_profile_int(tool.get("slot", tool.get("tool_type_id")), -1)
                if slot_id < 0 or slot_id not in self.tool_profile_map:
                    continue
                if isinstance(entry_mask, list) and slot_id < len(entry_mask) and int(entry_mask[slot_id]) != 1:
                    continue
                base_profile = self.tool_profile_map[slot_id]
                dynamic_map[(graph_id, int(node_id), int(slot_id))] = self._profile_from_library_tool(
                    base_profile=base_profile,
                    tool=tool,
                    slot_id=slot_id,
                )
        return dynamic_map

    def _build_dynamic_tool_info_map(
        self,
        tool_library: Optional[Dict[str, object]],
    ) -> Dict[Tuple[str, int, int], Dict[str, object]]:
        """Keep original API tool metadata for decision-context logging.

        The dynamic ToolTypeProfile only preserves numeric execution parameters.
        For knowledge memory, the downstream LLM needs the original tool name,
        category, description, and API-provided attributes.  The key is kept at
        graph-node-slot granularity because the same slot can represent different
        semantic tools in different nodes.
        """
        info_map: Dict[Tuple[str, int, int], Dict[str, object]] = {}
        for entry in self._tool_library_entries(tool_library):
            graph_id = str(entry.get("graph_id", ""))
            node_id = self._safe_profile_int(entry.get("node_id"), -1)
            if not graph_id or node_id < 0:
                continue
            entry_mask = entry.get("allowed_tools_mask", [])
            entry_base = {
                "entry_id": entry.get("entry_id"),
                "graph_id": graph_id,
                "node_id": int(node_id),
                "node_type_id": self._safe_profile_int(entry.get("node_type_id"), -1),
                "intent_text": entry.get("intent_text"),
                "mask_key": entry.get("mask_key"),
                "allowed_tools_mask": copy.deepcopy(entry.get("allowed_tools_mask", [])),
                "requires_tool_to_complete": bool(entry.get("requires_tool_to_complete", False)),
                "local_success_probability": entry.get("local_success_probability"),
            }
            for tool in entry.get("tools", []) or []:
                if not isinstance(tool, dict):
                    continue
                slot_id = self._safe_profile_int(tool.get("slot", tool.get("tool_type_id")), -1)
                if slot_id < 0 or slot_id not in self.tool_profile_map:
                    continue
                if isinstance(entry_mask, list) and slot_id < len(entry_mask) and int(entry_mask[slot_id]) != 1:
                    continue
                row = copy.deepcopy(entry_base)
                row["slot"] = int(slot_id)
                row["tool_type_id"] = int(self._safe_profile_int(tool.get("tool_type_id", slot_id), slot_id))
                row["tool"] = copy.deepcopy(tool)
                for key in (
                    "name",
                    "tool_name",
                    "category",
                    "description",
                    "compute_frequency_hz",
                    "uplink_bandwidth_hz",
                    "downlink_bandwidth_hz",
                    "coverage_distance_m",
                    "server_reliability",
                    "info_richness",
                    "validity_horizon_s",
                    "semantic_gain",
                    "completion_gain",
                    "success_probability",
                    "energy_j",
                    "requires_tool_to_complete",
                    "source",
                ):
                    if key in tool:
                        row[key] = copy.deepcopy(tool.get(key))
                info_map[(graph_id, int(node_id), int(slot_id))] = row
        return info_map

    def _resolve_tool_info(self, task: TaskRuntime, node: NodeRuntime, tool_type_id: int) -> Dict[str, object]:
        key = (str(task.graph_id), int(node.node_id), int(tool_type_id))
        info = copy.deepcopy(self.dynamic_tool_info_map.get(key, {}))
        profile = self._resolve_tool_profile(task, node, tool_type_id)
        if not info:
            info = {
                "graph_id": str(task.graph_id),
                "node_id": int(node.node_id),
                "node_type_id": int(node.node_type_id),
                "slot": int(tool_type_id),
                "tool_type_id": int(tool_type_id),
                "name": profile.name,
                "tool_name": profile.name,
                "category": "unknown",
                "description": "",
            }
        info.setdefault("slot", int(tool_type_id))
        info.setdefault("tool_type_id", int(tool_type_id))
        info.setdefault("name", profile.name)
        info.setdefault("tool_name", info.get("name", profile.name))
        info.setdefault("compute_frequency_hz", float(profile.compute_frequency_hz))
        info.setdefault("uplink_bandwidth_hz", float(profile.uplink_bandwidth_hz))
        info.setdefault("downlink_bandwidth_hz", float(profile.downlink_bandwidth_hz))
        info.setdefault("coverage_distance_m", float(profile.coverage_distance_m))
        info.setdefault("server_reliability", float(profile.server_reliability))
        info.setdefault("info_richness", float(profile.info_richness))
        info.setdefault("validity_horizon_s", float(profile.validity_horizon_s))
        info.setdefault("queue_capacity", int(profile.queue_capacity))
        info.setdefault("instances", int(profile.instances))
        return info

    def _tool_action_context(self, task: TaskRuntime, node: NodeRuntime, tool_type_id: int, timing: Dict[str, object]) -> Dict[str, object]:
        profile = self._resolve_tool_profile(task, node, tool_type_id)
        info = self._resolve_tool_info(task, node, tool_type_id)
        row = {
            "action_type": "tool",
            "slot": int(tool_type_id),
            "tool_type_id": int(tool_type_id),
            "queue_length": int(len(self.tool_queues[int(tool_type_id)])),
            "running_jobs": int(len(self.tool_running[int(tool_type_id)])),
            "queue_capacity": int(profile.queue_capacity),
            "instances": int(profile.instances),
            "queue_blocked": int(len(self.tool_queues[int(tool_type_id)])) >= int(profile.queue_capacity),
        }
        for key in (
            "entry_id",
            "intent_text",
            "mask_key",
            "allowed_tools_mask",
            "requires_tool_to_complete",
            "local_success_probability",
            "name",
            "tool_name",
            "category",
            "description",
            "compute_frequency_hz",
            "uplink_bandwidth_hz",
            "downlink_bandwidth_hz",
            "coverage_distance_m",
            "server_reliability",
            "info_richness",
            "validity_horizon_s",
            "semantic_gain",
            "completion_gain",
            "success_probability",
            "energy_j",
            "source",
        ):
            if key in info:
                row[key] = copy.deepcopy(info.get(key))
        for key, value in dict(timing or {}).items():
            if isinstance(value, (int, float, bool, str)) or value is None:
                row[key] = value
        return row

    def _decision_context_payload(self, task: TaskRuntime, node: NodeRuntime, cached: Optional[Dict[str, object]] = None) -> Dict[str, object]:
        cached = cached or self._get_cached_decision_timing(task.graph_id, node.node_id) or {}
        cached_tools = cached.get("tools") if isinstance(cached.get("tools"), dict) else {}
        tool_options: List[Dict[str, object]] = []
        for tid, allowed in enumerate(node.allowed_tools_mask):
            if int(allowed) != 1:
                continue
            timing = copy.deepcopy(cached_tools.get(int(tid)) or self._predict_tool_timing(task, node, int(tid)))
            tool_options.append(self._tool_action_context(task, node, int(tid), timing))
        local_timing = copy.deepcopy(cached.get("local") or self._predict_local_timing(node))
        local_option = {
            "action_type": "local",
            "compute_s": float(local_timing.get("compute_s", 0.0)),
            "fail_probability": float(local_timing.get("fail_probability", 0.0)),
            "queue_blocked": bool(local_timing.get("queue_blocked", False)),
            "estimated_wait_s": float(local_timing.get("estimated_wait_s", 0.0)),
            "predicted_total_s": float(local_timing.get("predicted_total_s", 0.0)),
            "queue_length": int(local_timing.get("queue_length", len(self.local_queue))),
            "local_running_jobs": int(len(self.local_running)),
            "local_max_concurrency": int(self.env_cfg.local_max_concurrency),
            "local_queue_capacity": int(self.env_cfg.local_queue_capacity),
        }
        allowed_slots = [int(i) for i, allowed in enumerate(node.allowed_tools_mask) if int(allowed) == 1]
        first_tool_info = self._resolve_tool_info(task, node, allowed_slots[0]) if allowed_slots else {}
        return {
            "time": float(self.time),
            "graph_id": str(task.graph_id),
            "node_id": int(node.node_id),
            "node_type_id": int(node.node_type_id),
            "intent_text": first_tool_info.get("intent_text", ""),
            "allowed_slots": allowed_slots,
            "requires_tool_to_complete": bool(first_tool_info.get("requires_tool_to_complete", False)),
            "local_success_probability": first_tool_info.get("local_success_probability"),
            "task_remaining_deadline_s": float(self._task_remaining(task)),
            "node_remaining_deadline_s": float(max(0.0, node.node_deadline - (self.time - (node.activation_time or self.time)))),
            "node_deadline_s": float(node.node_deadline),
            "task_deadline_s": float(task.task_deadline),
            "local_queue_length": int(len(self.local_queue)),
            "local_running_jobs": int(len(self.local_running)),
            "tool_queue_lengths": {str(p.tool_type_id): int(len(self.tool_queues[p.tool_type_id])) for p in self.tool_profiles},
            "tool_running_counts": {str(p.tool_type_id): int(len(self.tool_running[p.tool_type_id])) for p in self.tool_profiles},
            "local_option": local_option,
            "tool_options": tool_options,
            "candidate_actions": tool_options + [local_option, {"action_type": "pause", "pause_duration_s": float(self.env_cfg.dt), "allowed_reason": "Use only when no tool/local action is feasible under queue or deadline constraints."}],
        }

    def _resolve_tool_profile(self, task: TaskRuntime, node: NodeRuntime, tool_type_id: int) -> ToolTypeProfile:
        key = (str(task.graph_id), int(node.node_id), int(tool_type_id))
        return self.dynamic_tool_profile_map.get(key, self.tool_profile_map[int(tool_type_id)])

    def assign_arrival_times(self, arrival_times: Optional[Dict[str, float]] = None) -> None:
        graphs = self.dataset["graphs"]
        self.arrival_schedule = {}
        if arrival_times:
            for graph in graphs:
                gid = graph["graph_id"]
                self.arrival_schedule[gid] = float(arrival_times[gid])
        else:
            if bool(self.env_cfg.use_integer_arrival_slots):
                cursor_slot = 0
                min_gap = max(1, int(round(self.env_cfg.task_arrival_default_gap_range[0])))
                max_gap = max(min_gap, int(round(self.env_cfg.task_arrival_default_gap_range[1])))
                for graph in graphs:
                    gap_slot = self.rng.randint(min_gap, max_gap)
                    cursor_slot += gap_slot
                    self.arrival_schedule[graph["graph_id"]] = round(cursor_slot * float(self.env_cfg.arrival_slot_step_s), 4)
            else:
                cursor = 0.0
                for graph in graphs:
                    gap = self.rng.uniform(*self.env_cfg.task_arrival_default_gap_range)
                    cursor += gap
                    self.arrival_schedule[graph["graph_id"]] = round(cursor, 4)
        self.waiting_arrivals = sorted((t, gid) for gid, t in self.arrival_schedule.items())

    def load_tasks(self) -> None:
        self.tasks = {}
        for graph in self.dataset["graphs"]:
            gid = graph["graph_id"]
            nodes = {}
            for node in graph["nodes"]:
                nodes[node["node_id"]] = NodeRuntime(
                    node_id=node["node_id"],
                    node_type_id=node["node_type_id"],
                    computation_cycles=float(node["computation_cycles"]),
                    uplink_size_bits=int(node["uplink_size_bits"]),
                    downlink_size_bits=int(node["downlink_size_bits"]),
                    allowed_tools_mask=list(node["allowed_tools_mask"]),
                    node_deadline=float(node["node_deadline"]),
                )
            edges = []
            for edge in graph["edges"]:
                u, v = int(edge["src"]), int(edge["dst"])
                edges.append((u, v))
                nodes[u].successors.append(v)
                nodes[v].predecessors.append(u)

            tr = TaskRuntime(
                graph_id=gid,
                num_nodes=int(graph["num_nodes"]),
                task_deadline=float(graph["task_deadline"]),
                arrival_time=float(self.arrival_schedule.get(gid, 0.0)),
                nodes=nodes,
                edges=edges,
            )
            self.tasks[gid] = tr

    def reset(self, arrival_times: Optional[Dict[str, float]] = None) -> None:
        self.time = 0.0
        self.step_count = 0
        self.rng = random.Random(self.env_cfg.seed)
        self.metrics_history = []
        self.completed_task_records = []
        self._decision_cache = {}
        self.invalid_decision_counts = {}
        self.tool_queues = {p.tool_type_id: deque() for p in self.tool_profiles}
        self.tool_running = {p.tool_type_id: [] for p in self.tool_profiles}
        self.local_queue = deque()
        self.local_running = []
        self.pause_events = []
        heapq.heapify(self.pause_events)

        self.assign_arrival_times(arrival_times)
        self.load_tasks()

    # --------------------------------------------------------
    # State helpers
    # --------------------------------------------------------


    def _cache_key(self, task_id: str, node_id: int) -> Tuple[str, int]:
        return (str(task_id), int(node_id))

    def _clear_node_cache(self, task_id: str, node_id: int) -> None:
        self._decision_cache.pop(self._cache_key(task_id, node_id), None)

    def _get_cached_decision_timing(self, task_id: str, node_id: int) -> Optional[Dict[str, object]]:
        return self._decision_cache.get(self._cache_key(task_id, node_id))

    def _task_elapsed(self, task: TaskRuntime) -> float:
        return max(0.0, self.time - task.arrival_time)

    def _task_remaining(self, task: TaskRuntime) -> float:
        return max(0.0, task.task_deadline - self._task_elapsed(task))

    def _frontier_nodes(self, task: TaskRuntime) -> List[int]:
        frontier = []
        for node_id, node in task.nodes.items():
            if node.status in {"done", "queued", "queued_local", "running_tool", "running_local", "paused"}:
                continue
            if all(task.nodes[p].status == "done" for p in node.predecessors):
                frontier.append(node_id)
        return sorted(frontier)

    def _channel_snapshot(self, tool_type_id: int, node: NodeRuntime, task: Optional[TaskRuntime] = None) -> Dict[str, float]:
        profile = self._resolve_tool_profile(task, node, tool_type_id) if task is not None else self.tool_profile_map[tool_type_id]

        if bool(self.env_cfg.use_compact_deterministic_channel):
            denom = max(1, len(self.tool_profiles) - 1)
            tool_rank = float(tool_type_id) / float(denom)
            topology_offset = float((node.node_id % 4)) / 4.0
            quality = float(self.env_cfg.compact_channel_quality_floor) + float(self.env_cfg.compact_channel_quality_span) * (0.65 * tool_rank + 0.35 * (1.0 - topology_offset))
            quality = max(0.25, min(0.98, quality))
            distance = max(5.0, profile.coverage_distance_m * (1.05 - 0.45 * quality))
            rate_ul = profile.uplink_bandwidth_hz * (1.20 + 0.90 * quality)
            rate_dl = profile.downlink_bandwidth_hz * (1.20 + 0.90 * quality)
            snr_ul = max(1e-9, 4.0 * quality)
            snr_dl = max(1e-9, 4.5 * quality)
            snr_db = 10.0 * math.log10(max(snr_ul, 1e-12))
            return {
                "distance_m": float(distance),
                "snr_ul": float(snr_ul),
                "snr_dl": float(snr_dl),
                "snr_db": float(snr_db),
                "rate_ul_bps": float(rate_ul),
                "rate_dl_bps": float(rate_dl),
                # Tool execution success is deterministic in this environment:
                # if the tool finishes before deadline, it succeeds.  Do not use
                # packet-loss probability as a tool failure probability.
                "packet_loss_rate": 0.0,
            }

        distance = 8.0 + (node.node_id % 5) * 7.5 + self.rng.uniform(0.0, profile.coverage_distance_m * 0.35)
        shadow_db = self.rng.gauss(0.0, self.env_cfg.shadowing_sigma_db)
        shadow_lin = 10 ** (shadow_db / 10.0)

        beta = self.env_cfg.reference_path_gain * (
            (max(distance, 1.0) / self.env_cfg.reference_distance_m) ** (-self.env_cfg.path_loss_exponent)
        ) * shadow_lin

        K = self.env_cfg.rician_k_factor
        los_mag = math.sqrt(K / (K + 1.0))
        nlos_real = self.rng.gauss(0.0, math.sqrt(1.0 / (2.0 * (K + 1.0))))
        nlos_imag = self.rng.gauss(0.0, math.sqrt(1.0 / (2.0 * (K + 1.0))))
        g_abs2 = (los_mag + nlos_real) ** 2 + nlos_imag ** 2
        h_abs2 = beta * g_abs2

        noise_ul = self.env_cfg.noise_power_density_w_hz * profile.uplink_bandwidth_hz
        noise_dl = self.env_cfg.noise_power_density_w_hz * profile.downlink_bandwidth_hz

        snr_ul = max(1e-12, self.env_cfg.tx_power_w * h_abs2 / noise_ul)
        snr_dl = max(1e-12, self.env_cfg.tx_power_w * h_abs2 / noise_dl)

        rate_ul = profile.uplink_bandwidth_hz * math.log2(1.0 + snr_ul)
        rate_dl = profile.downlink_bandwidth_hz * math.log2(1.0 + snr_dl)

        snr_db = 10.0 * math.log10(max(snr_ul, 1e-12))
        return {
            "distance_m": distance,
            "snr_ul": snr_ul,
            "snr_dl": snr_dl,
            "snr_db": snr_db,
            "rate_ul_bps": rate_ul,
            "rate_dl_bps": rate_dl,
            # Tool execution success is deterministic in this environment:
            # if the tool finishes before deadline, it succeeds.  Do not use
            # packet-loss probability as a tool failure probability.
            "packet_loss_rate": 0.0,
        }

    def _estimate_tool_queue_wait(self, tool_type_id: int) -> float:
        profile = self.tool_profile_map[tool_type_id]
        capacity = max(1, int(profile.instances))
        server_available = [0.0] * capacity
        running_finish = sorted(max(0.0, job.finish_time - self.time) for job in self.tool_running[tool_type_id])
        for i, v in enumerate(running_finish[:capacity]):
            server_available[i] = v
        for item in list(self.tool_queues[tool_type_id]):
            idx = min(range(capacity), key=lambda j: server_available[j])
            server_available[idx] += float(item.uplink_s + item.predicted_service_s + item.downlink_s)
        return float(min(server_available)) if server_available else 0.0

    def _predict_tool_timing(self, task: TaskRuntime, node: NodeRuntime, tool_type_id: int) -> Dict[str, float]:
        profile = self._resolve_tool_profile(task, node, tool_type_id)
        channel = self._channel_snapshot(tool_type_id, node, task)
        uplink_s = node.uplink_size_bits / max(channel["rate_ul_bps"], 1e-9)
        downlink_s = node.downlink_size_bits / max(channel["rate_dl_bps"], 1e-9)
        exec_s = node.computation_cycles / max(profile.compute_frequency_hz, 1e-9)

        queue_items = list(self.tool_queues[tool_type_id])
        outstanding = len(queue_items) + len(self.tool_running[tool_type_id])
        congestion_ratio = outstanding / max(1, profile.queue_capacity + profile.instances)
        base_queue_delay_s = self._estimate_tool_queue_wait(tool_type_id)

        stale_ratio = max(0.0, (uplink_s + exec_s + downlink_s - profile.validity_horizon_s) / max(1e-6, profile.validity_horizon_s))
        base_risk = self.queue_risk_estimator(
            compute_s=exec_s,
            queue_delay_s=base_queue_delay_s,
            packet_loss_rate=channel["packet_loss_rate"],
            congestion_ratio=congestion_ratio,
            weights=self.env_cfg.risk_weights,
        )
        risk = (
            0.70 * base_risk
            + 0.15 * max(0.0, 1.0 - float(profile.info_richness))
            + 0.15 * min(1.0, stale_ratio)
        )
        risk = max(0.0, min(1.0, risk))
        total_s = uplink_s + base_queue_delay_s + exec_s + downlink_s
        return {
            **channel,
            "uplink_s": float(uplink_s),
            "downlink_s": float(downlink_s),
            "exec_s": float(exec_s),
            "base_queue_delay_s": float(base_queue_delay_s),
            "risk_estimate": float(risk),
            "congestion_ratio": float(congestion_ratio),
            "predicted_total_s": float(total_s),
        }

    def _estimate_local_queue_wait(self, candidate_compute_s: float) -> float:
        capacity = max(1, int(self.env_cfg.local_max_concurrency))
        server_available = [0.0] * capacity
        running_finish = sorted(max(0.0, job.finish_time - self.time) for job in self.local_running)
        for i, v in enumerate(running_finish[:capacity]):
            server_available[i] = v
        for item in list(self.local_queue):
            idx = min(range(capacity), key=lambda j: server_available[j])
            server_available[idx] += float(item.predicted_compute_s)
        return float(min(server_available)) if server_available else 0.0

    def _predict_local_timing(self, node: NodeRuntime) -> Dict[str, float]:
        compute_s = node.computation_cycles / max(self.local_profile.local_cpu_frequency_hz, 1e-9)
        fail_probability = (
            self.local_profile.local_failure_base
            + 0.22 * min(1.0, node.computation_cycles / self.local_profile.local_failure_compute_scale)
        )
        fail_probability = max(0.0, min(0.95, fail_probability))
        compute_s = max(0.001, float(compute_s))
        queue_capacity = max(0, int(getattr(self.env_cfg, "local_queue_capacity", 0)))
        queue_blocked = len(self.local_queue) >= queue_capacity if queue_capacity > 0 else False
        estimated_wait_s = self._estimate_local_queue_wait(compute_s)
        return {
            "compute_s": float(compute_s),
            "fail_probability": float(fail_probability),
            "queue_blocked": bool(queue_blocked),
            "estimated_wait_s": float(estimated_wait_s),
            "predicted_total_s": float(compute_s + estimated_wait_s),
            "queue_length": int(len(self.local_queue)),
        }

    def _purge_task_artifacts(self, task_id: str) -> None:
        """
        清理某个任务在环境中的所有残留执行痕迹：
        - 工具队列
        - 工具运行
        - 本地运行
        - 暂停事件
        """
        for tool_type_id, queue in self.tool_queues.items():
            self.tool_queues[tool_type_id] = deque(
                [item for item in queue if item.task_id != task_id]
            )

        for tool_type_id, running in self.tool_running.items():
            self.tool_running[tool_type_id] = [
                job for job in running if job.task_id != task_id
            ]

        self.local_queue = deque([item for item in self.local_queue if item.task_id != task_id])

        self.local_running = [
            job for job in self.local_running if job.task_id != task_id
        ]

        self.pause_events = [
            ev for ev in self.pause_events if ev.task_id != task_id
        ]
        heapq.heapify(self.pause_events)
        stale_keys = [key for key in self._decision_cache if key[0] == str(task_id)]
        for key in stale_keys:
            self._decision_cache.pop(key, None)

    def _has_runtime_artifacts(self) -> bool:
        """
        判断环境里是否仍然存在运行/排队/暂停/待到达痕迹。
        """
        if self.waiting_arrivals:
            return True

        if self.pause_events:
            return True

        if self.local_queue:
            return True

        if self.local_running:
            return True

        for tool_type_id in self.tool_queues:
            if self.tool_queues[tool_type_id]:
                return True
            if self.tool_running[tool_type_id]:
                return True

        return False

    def get_state(self) -> Dict[str, object]:
        active_tasks = []
        for task_id, task in self.tasks.items():
            if task.status in {"active", "waiting"}:
                active_tasks.append(
                    {
                        "task_id": task_id,
                        "status": task.status,
                        "remaining_deadline_s": self._task_remaining(task),
                        "frontier_nodes": self._frontier_nodes(task) if task.status == "active" else [],
                        "completed_nodes": sorted(task.completed_nodes),
                    }
                )
        tool_states = []
        for profile in self.tool_profiles:
            q = self.tool_queues[profile.tool_type_id]
            tool_states.append(
                {
                    "tool_type_id": profile.tool_type_id,
                    "queue_length": len(q),
                    "running_jobs": len(self.tool_running[profile.tool_type_id]),
                    "instances": profile.instances,
                }
            )
        return {
            "time": round(self.time, 4),
            "step_count": self.step_count,
            "tasks": active_tasks,
            "tool_states": tool_states,
            "local_queue_length": len(self.local_queue),
            "local_running_jobs": len(self.local_running),
            "active_pause_events": len(self.pause_events),
        }

    # --------------------------------------------------------
    # Core execution
    # --------------------------------------------------------

    def _activate_arrivals(self) -> None:
        while self.waiting_arrivals and self.waiting_arrivals[0][0] <= self.time + 1e-12:
            _, task_id = self.waiting_arrivals.pop(0)
            task = self.tasks[task_id]
            task.status = "active"
            task.started = True
            task.start_time = self.time
            self._log(task, None, "task_arrival", {"arrival_time": self.time})

    def _release_pause_events(self) -> List[Tuple[str, int]]:
        released = []
        while self.pause_events and self.pause_events[0].trigger_time <= self.time + 1e-12:
            event = heapq.heappop(self.pause_events)
            task = self.tasks.get(event.task_id)
            if task is None or task.status != "active":
                continue
            node = task.nodes[event.node_id]
            if node.status == "paused":
                node.status = "idle"
                released.append((event.task_id, event.node_id))
                self._log(task, node, "pause_released", {"time": self.time})
        return released

    def _admit_tool_jobs(self) -> None:
        for profile in self.tool_profiles:
            tool_type_id = profile.tool_type_id
            running = self.tool_running[tool_type_id]
            queue = self.tool_queues[tool_type_id]
            available = profile.instances - len(running)
            while available > 0 and queue:
                item = queue.popleft()

                task = self.tasks.get(item.task_id)
                if task is None or task.status != "active":
                    continue

                node = task.nodes[item.node_id]
                if node.status != "queued":
                    continue

                finish_time = self.time + item.uplink_s + item.predicted_service_s + item.downlink_s
                job = RunningToolJob(
                    task_id=item.task_id,
                    node_id=item.node_id,
                    tool_type_id=tool_type_id,
                    start_time=self.time,
                    finish_time=finish_time,
                    uplink_s=item.uplink_s,
                    exec_s=item.predicted_service_s,
                    downlink_s=item.downlink_s,
                    queue_delay_s=max(0.0, self.time - item.enqueue_time),
                    packet_loss_rate=0.0,
                    risk_estimate=item.risk_snapshot,
                )
                running.append(job)
                node.status = "running_tool"
                node.last_start_time = self.time
                self._log(
                    task,
                    node,
                    "tool_execution_started",
                    {
                        "tool_type_id": tool_type_id,
                        "queue_delay_s": job.queue_delay_s,
                        "predicted_exec_s": item.predicted_service_s,
                    },
                )
                available -= 1

    def _admit_local_jobs(self) -> None:
        available = max(1, int(self.env_cfg.local_max_concurrency)) - len(self.local_running)
        while available > 0 and self.local_queue:
            item = self.local_queue.popleft()
            task = self.tasks.get(item.task_id)
            if task is None or task.status != "active":
                continue
            node = task.nodes[item.node_id]
            if node.status != "queued_local":
                continue
            queue_delay_s = max(0.0, self.time - item.enqueue_time)
            job = RunningLocalJob(
                task_id=item.task_id,
                node_id=item.node_id,
                start_time=self.time,
                finish_time=self.time + item.predicted_compute_s,
                compute_s=item.predicted_compute_s,
                fail_probability=item.fail_probability,
                queue_delay_s=queue_delay_s,
            )
            self.local_running.append(job)
            node.status = "running_local"
            node.last_start_time = self.time
            self._log(
                task,
                node,
                "local_execution_started",
                {
                    "queue_delay_s": queue_delay_s,
                    "predicted_compute_s": item.predicted_compute_s,
                    "fail_probability": item.fail_probability,
                },
            )
            available -= 1

    def _complete_tool_jobs(self) -> None:
        for tool_type_id, running_jobs in self.tool_running.items():
            remain = []
            for job in running_jobs:
                task = self.tasks.get(job.task_id)
                if task is None or task.status != "active":
                    continue

                node = task.nodes[job.node_id]
                if node.status != "running_tool":
                    continue

                if job.finish_time <= self.time + 1e-12:
                    self._complete_node_success(
                        task=task,
                        node=node,
                        outcome_kind="tool_success",
                        detail={
                            "tool_type_id": tool_type_id,
                            "finish_time": self.time,
                            "queue_delay_s": job.queue_delay_s,
                            "exec_s": job.exec_s,
                            "uplink_s": job.uplink_s,
                            "downlink_s": job.downlink_s,
                            "risk_estimate": job.risk_estimate,
                        },
                    )
                else:
                    remain.append(job)
            self.tool_running[tool_type_id] = remain

    def _complete_local_jobs(self) -> None:
        remain = []
        for job in self.local_running:
            task = self.tasks.get(job.task_id)
            if task is None or task.status != "active":
                continue

            node = task.nodes[job.node_id]
            if node.status != "running_local":
                continue

            if job.finish_time <= self.time + 1e-12:
                if self.rng.random() < job.fail_probability:
                    self._log(
                        task,
                        node,
                        "local_failed",
                        {
                            "compute_s": job.compute_s,
                            "fail_probability": job.fail_probability,
                            "queue_delay_s": job.queue_delay_s,
                        },
                    )
                    self._restart_or_fail_task(task, node, "local_failure")
                else:
                    self._complete_node_success(
                        task=task,
                        node=node,
                        outcome_kind="local_success",
                        detail={
                            "finish_time": self.time,
                            "compute_s": job.compute_s,
                            "fail_probability": job.fail_probability,
                            "queue_delay_s": job.queue_delay_s,
                        },
                    )
            else:
                remain.append(job)
        self.local_running = remain

    def _check_deadlines(self) -> None:
        for task in self.tasks.values():
            if task.status not in {"active", "waiting"}:
                continue

            if self._task_elapsed(task) > task.task_deadline + 1e-12:
                task.status = "failed"
                task.end_time = self.time
                task.failure_reason = "task_deadline_exceeded"
                self._log(task, None, "task_failed", {"reason": task.failure_reason})
                self._purge_task_artifacts(task.graph_id)
                self._finalize_task_record(task)
                continue

            if task.status == "active":
                for node in task.nodes.values():
                    if node.status in {"done"}:
                        continue
                    if node.activation_time is not None:
                        elapsed = self.time - node.activation_time
                        if elapsed > node.node_deadline + 1e-12:
                            self._log(task, node, "node_timeout", {"elapsed_s": elapsed})
                            self._restart_or_fail_task(task, node, "node_timeout")
                            break

    def _complete_node_success(
        self,
        task: TaskRuntime,
        node: NodeRuntime,
        outcome_kind: str,
        detail: Dict[str, object],
    ) -> None:
        if task.status != "active":
            return
        if node.status not in {"running_tool", "running_local"}:
            return

        self._clear_node_cache(task.graph_id, node.node_id)
        node.status = "done"
        node.completion_time = self.time
        if node.node_id not in task.completed_nodes:
            task.completed_nodes.append(node.node_id)
        self._log(task, node, outcome_kind, detail)

        if len(task.completed_nodes) == task.num_nodes:
            task.status = "completed"
            task.end_time = self.time
            self._log(task, None, "task_completed", {"latency_s": self.time - task.arrival_time})
            self._purge_task_artifacts(task.graph_id)
            self._finalize_task_record(task)

    def _restart_or_fail_task(self, task: TaskRuntime, node: Optional[NodeRuntime], reason: str) -> None:
        if task.status != "active":
            return

        self._purge_task_artifacts(task.graph_id)

        if self._task_elapsed(task) >= task.task_deadline:
            task.status = "failed"
            task.end_time = self.time
            task.failure_reason = reason
            self._log(task, node, "task_failed", {"reason": reason})
            self._finalize_task_record(task)
            return

        task.restart_count += 1
        task.completed_nodes = []

        for n in task.nodes.values():
            self._clear_node_cache(task.graph_id, n.node_id)
            n.status = "idle"
            n.activation_time = None
            n.completion_time = None
            n.last_start_time = None

        self.time += self.env_cfg.retry_restart_penalty_s
        self._log(task, node, "task_restarted", {"reason": reason, "restart_count": task.restart_count})

    def _log(self, task: TaskRuntime, node: Optional[NodeRuntime], event_type: str, payload: Dict[str, object]) -> None:
        record = {
            "time": round(self.time, 4),
            "event_type": event_type,
            "task_id": task.graph_id,
            "node_id": None if node is None else node.node_id,
            "task_status": task.status,
            "payload": copy.deepcopy(payload),
        }
        task.log.append(record)

    def _finalize_task_record(self, task: TaskRuntime) -> None:
        existing_ids = {rec["task_id"] for rec in self.completed_task_records}
        if task.graph_id in existing_ids:
            return

        self.completed_task_records.append(
            {
                "task_id": task.graph_id,
                "status": task.status,
                "arrival_time": task.arrival_time,
                "end_time": task.end_time,
                "latency_s": None if task.end_time is None else round(task.end_time - task.arrival_time, 4),
                "task_deadline_s": float(task.task_deadline),
                "restart_count": task.restart_count,
                "failure_reason": task.failure_reason,
                "log": copy.deepcopy(task.log),
            }
        )

    # --------------------------------------------------------
    # Decision interface
    # --------------------------------------------------------

    def collect_decision_points(self) -> List[Dict[str, object]]:
        decision_points = []
        self._decision_cache = {}
        for task in self.tasks.values():
            if task.status != "active":
                continue
            frontier = self._frontier_nodes(task)
            for node_id in frontier:
                node = task.nodes[node_id]
                if node.activation_time is None:
                    node.activation_time = self.time
                node.status = "ready"
                tool_options = []
                cached_tools: Dict[int, Dict[str, float]] = {}
                for tid, allowed in enumerate(node.allowed_tools_mask):
                    if allowed != 1:
                        continue
                    profile = self._resolve_tool_profile(task, node, tid)
                    queue_len = len(self.tool_queues[tid])
                    queue_blocked = queue_len >= profile.queue_capacity
                    timing = self._predict_tool_timing(task, node, tid)
                    cached_tools[int(tid)] = copy.deepcopy(timing)
                    tool_options.append(
                        {
                            "tool_type_id": tid,
                            "queue_blocked": queue_blocked,
                            **timing,
                        }
                    )
                local_timing = self._predict_local_timing(node)
                self._decision_cache[self._cache_key(task.graph_id, node.node_id)] = {
                    "local": copy.deepcopy(local_timing),
                    "tools": cached_tools,
                    "cache_time": float(self.time),
                }
                decision_points.append(
                    {
                        "task_id": task.graph_id,
                        "node_id": node.node_id,
                        "node_type_id": node.node_type_id,
                        "task_deadline_s": float(task.task_deadline),
                        "task_remaining_deadline_s": self._task_remaining(task),
                        "task_elapsed_time_s": self._task_elapsed(task),
                        "ready_frontier_count": len(frontier),
                        "completed_node_count": len(task.completed_nodes),
                        "num_nodes": int(task.num_nodes),
                        "restart_count": int(task.restart_count),
                        "computation_cycles": float(node.computation_cycles),
                        "uplink_size_bits": int(node.uplink_size_bits),
                        "downlink_size_bits": int(node.downlink_size_bits),
                        "node_deadline_s": float(node.node_deadline),
                        "node_remaining_deadline_s": max(
                            0.0,
                            node.node_deadline - (self.time - (node.activation_time or self.time)),
                        ),
                        "allowed_tools_mask": list(node.allowed_tools_mask),
                        "num_predecessors": len(node.predecessors),
                        "num_successors": len(node.successors),
                        "running_tool_jobs_total": sum(len(v) for v in self.tool_running.values()),
                        "queued_tool_jobs_total": sum(len(v) for v in self.tool_queues.values()),
                        "running_local_jobs": len(self.local_running),
                        "queued_local_jobs": len(self.local_queue),
                        "tool_options": tool_options,
                        "local_option": local_timing,
                        "decision_context": self._decision_context_payload(task, node),
                    }
                )
        decision_points.sort(key=lambda x: (x["task_id"], x["node_id"]))
        return decision_points

    def _handle_invalid_decision(
        self,
        task: TaskRuntime,
        node: NodeRuntime,
        event_type: str,
        reason: str,
        payload: Dict[str, object],
    ) -> None:
        detail = copy.deepcopy(payload)
        detail["reason"] = str(reason)
        detail["invalid_action_mode"] = str(getattr(self.env_cfg, "invalid_action_mode", "fail_task"))
        count_key = (str(task.graph_id), int(node.node_id), str(reason), int(task.restart_count))
        prev_count = int(getattr(self, "invalid_decision_counts", {}).get(count_key, 0))
        attempt_count = prev_count + 1
        self.invalid_decision_counts[count_key] = attempt_count
        detail["invalid_attempt_count_for_node"] = int(attempt_count)
        detail["max_invalid_attempts_per_node"] = int(getattr(self.env_cfg, "max_invalid_attempts_per_node", 0) or 0)
        self._log(task, node, event_type, detail)

        mode = str(getattr(self.env_cfg, "invalid_action_mode", "fail_task")).lower()
        if mode in {"fail_task", "task_failed", "fail", "failed"}:
            self._purge_task_artifacts(task.graph_id)
            task.status = "failed"
            task.end_time = self.time
            task.failure_reason = str(reason)
            self._log(task, node, "task_failed", {"reason": str(reason), "source_event": str(event_type)})
            self._finalize_task_record(task)
            return

        max_invalid = int(getattr(self.env_cfg, "max_invalid_attempts_per_node", 0) or 0)
        if max_invalid > 0 and int(attempt_count) > max_invalid:
            self._log(
                task, node, "invalid_action_attempt_limit",
                {
                    "reason": str(reason),
                    "invalid_attempt_count_for_node": int(attempt_count),
                    "max_invalid_attempts_per_node": int(max_invalid),
                    "source_event": str(event_type),
                },
            )
            self._restart_or_fail_task(task, node, f"{reason}_repeated")
            return

        node.status = "ready"
        node.last_start_time = None
        penalty_s = max(0.0, float(getattr(self.env_cfg, "invalid_action_penalty_s", 0.0) or 0.0))
        if penalty_s > 0.0:
            self.time += penalty_s
            self._log(task, node, "invalid_action_penalty", {"penalty_s": penalty_s, "reason": str(reason)})

    def apply_decision(self, task_id: str, node_id: int, action: DecisionAction) -> None:
        task = self.tasks[task_id]
        if task.status != "active":
            return
        node = task.nodes[node_id]
        if node.status not in {"ready", "idle"}:
            return

        node.attempt_count += 1

        if action.action_type == "invalid_generated":
            reason = "llm_generation_parse_failed_or_out_of_range"
            self._log(
                task,
                node,
                "decision_invalid_generated_action",
                {
                    "reason": reason,
                    "invalid_action_mode": "direct_fail",
                    "decision_context": self._decision_context_payload(task, node),
                },
            )
            self._purge_task_artifacts(task.graph_id)
            task.status = "failed"
            task.end_time = self.time
            task.failure_reason = reason
            self._log(task, node, "task_failed", {"reason": reason, "source_event": "decision_invalid_generated_action"})
            self._finalize_task_record(task)
            return

        if action.action_type == "pause":
            node.status = "paused"
            trigger_time = self.time + max(self.env_cfg.dt, action.pause_duration_s)
            heapq.heappush(
                self.pause_events,
                TimedEvent(
                    trigger_time=trigger_time,
                    event_kind="pause_release",
                    task_id=task_id,
                    node_id=node_id,
                ),
            )
            cached = self._get_cached_decision_timing(task_id, node_id) or {}
            self._log(
                task,
                node,
                "decision_pause",
                {
                    "pause_duration_s": action.pause_duration_s,
                    "decision_context": self._decision_context_payload(task, node, cached),
                },
            )
            return

        if action.action_type == "local":
            cached = self._get_cached_decision_timing(task_id, node_id) or {}
            timing = copy.deepcopy(cached.get("local") or self._predict_local_timing(node))
            timing["decision_context"] = self._decision_context_payload(task, node, cached)
            if bool(timing.get("queue_blocked", False)):
                if bool(getattr(self.env_cfg, "fail_on_queue_rejection", True)):
                    self._handle_invalid_decision(task, node, "decision_local_rejected_queue_full", "local_queue_full", timing)
                else:
                    self._log(task, node, "decision_local_rejected_queue_full", timing)
                return
            self._clear_node_cache(task_id, node_id)
            self._log(task, node, "decision_local", timing)
            if len(self.local_running) < max(1, int(self.env_cfg.local_max_concurrency)) and not self.local_queue:
                node.status = "running_local"
                node.last_start_time = self.time
                self.local_running.append(
                    RunningLocalJob(
                        task_id=task_id,
                        node_id=node_id,
                        start_time=self.time,
                        finish_time=self.time + timing["compute_s"],
                        compute_s=timing["compute_s"],
                        fail_probability=timing["fail_probability"],
                        queue_delay_s=0.0,
                    )
                )
                self._log(
                    task,
                    node,
                    "local_execution_started",
                    {
                        "queue_delay_s": 0.0,
                        "predicted_compute_s": timing["compute_s"],
                        "fail_probability": timing["fail_probability"],
                    },
                )
            else:
                node.status = "queued_local"
                node.last_start_time = self.time
                self.local_queue.append(
                    LocalQueueItem(
                        task_id=task_id,
                        node_id=node_id,
                        enqueue_time=self.time,
                        predicted_compute_s=timing["compute_s"],
                        fail_probability=timing["fail_probability"],
                    )
                )
                self._log(
                    task,
                    node,
                    "local_queued",
                    {
                        "queue_length": len(self.local_queue),
                        "estimated_wait_s": timing.get("estimated_wait_s", 0.0),
                        "predicted_total_s": timing.get("predicted_total_s", timing["compute_s"]),
                    },
                )
            return

        if action.action_type == "tool":
            if action.tool_type_id is None:
                self._handle_invalid_decision(
                    task,
                    node,
                    "decision_tool_invalid_missing_id",
                    "tool_type_id_missing",
                    {"decision_context": self._decision_context_payload(task, node)},
                )
                return
            tid = int(action.tool_type_id)
            if tid < 0 or tid >= len(node.allowed_tools_mask) or tid not in self.tool_profile_map:
                self._handle_invalid_decision(
                    task,
                    node,
                    "decision_tool_invalid_unknown_tool",
                    "unknown_tool_type",
                    {"tool_type_id": tid, "decision_context": self._decision_context_payload(task, node)},
                )
                return
            if node.allowed_tools_mask[tid] != 1:
                self._handle_invalid_decision(
                    task,
                    node,
                    "decision_tool_invalid_not_allowed",
                    "tool_not_allowed_by_node",
                    {"tool_type_id": tid, "decision_context": self._decision_context_payload(task, node)},
                )
                return
            profile = self._resolve_tool_profile(task, node, tid)
            queue = self.tool_queues[tid]
            if len(queue) >= profile.queue_capacity:
                reject_payload = {"tool_type_id": tid, "decision_context": self._decision_context_payload(task, node)}
                if bool(getattr(self.env_cfg, "fail_on_queue_rejection", True)):
                    self._handle_invalid_decision(task, node, "decision_tool_rejected_queue_full", "tool_queue_full", reject_payload)
                else:
                    self._log(task, node, "decision_tool_rejected_queue_full", reject_payload)
                return
            cached = self._get_cached_decision_timing(task_id, node_id) or {}
            timing = copy.deepcopy((cached.get("tools") or {}).get(int(tid)) or self._predict_tool_timing(task, node, tid))
            queue.append(
                QueueItem(
                    task_id=task_id,
                    node_id=node_id,
                    enqueue_time=self.time,
                    predicted_service_s=timing["exec_s"],
                    uplink_s=timing["uplink_s"],
                    downlink_s=timing["downlink_s"],
                    risk_snapshot=timing["risk_estimate"],
                )
            )
            node.status = "queued"
            node.last_start_time = self.time
            selected_tool_info = self._tool_action_context(task, node, int(tid), timing)
            decision_context = self._decision_context_payload(task, node, cached)
            self._clear_node_cache(task_id, node_id)
            self._log(
                task,
                node,
                "decision_tool",
                {
                    "tool_type_id": tid,
                    "base_queue_delay_s": timing["base_queue_delay_s"],
                    "risk_estimate": timing["risk_estimate"],
                    "packet_loss_rate": timing["packet_loss_rate"],
                    "uplink_s": timing["uplink_s"],
                    "downlink_s": timing["downlink_s"],
                    "exec_s": timing["exec_s"],
                    "predicted_total_s": timing.get(
                        "predicted_total_s",
                        timing["base_queue_delay_s"] + timing["uplink_s"] + timing["exec_s"] + timing["downlink_s"],
                    ),
                    "congestion_ratio": timing.get("congestion_ratio", 0.0),
                    "queue_length_after_enqueue": len(queue),
                    "selected_tool_info": selected_tool_info,
                    "decision_context": decision_context,
                },
            )
            return

        raise ValueError(f"Unknown action type: {action.action_type}")

    # --------------------------------------------------------
    # Step / run
    # --------------------------------------------------------

    def step(self) -> Dict[str, object]:
        self._activate_arrivals()
        self._release_pause_events()
        self._admit_tool_jobs()
        self._admit_local_jobs()
        self._complete_tool_jobs()
        self._complete_local_jobs()
        self._check_deadlines()

        state = self.get_state()
        if self.step_count % self.env_cfg.metrics_record_period_steps == 0:
            self._record_metrics()

        self.step_count += 1
        self.time = round(self.time + self.env_cfg.dt, 10)
        return state

    def done(self) -> bool:
        all_tasks_terminal = all(
            task.status in {"completed", "failed"} for task in self.tasks.values()
        )
        if not all_tasks_terminal:
            return False

        if self._has_runtime_artifacts():
            return False

        return True

    def _record_metrics(self) -> None:
        finished = [t for t in self.tasks.values() if t.status in {"completed", "failed"}]
        completed = [t for t in self.tasks.values() if t.status == "completed"]
        latency_values = []
        for t in finished:
            if t.status == "completed" and t.end_time is not None:
                latency_values.append(max(0.0, t.end_time - t.arrival_time))
            else:
                latency_values.append(max(0.0, float(t.task_deadline)))
        avg_latency = (sum(latency_values) / len(latency_values)) if latency_values else 0.0

        tool_decisions = 0
        local_decisions = 0
        for task in self.tasks.values():
            for item in task.log:
                if item["event_type"] == "decision_tool":
                    tool_decisions += 1
                elif item["event_type"] == "decision_local":
                    local_decisions += 1

        total_exec_decisions = tool_decisions + local_decisions
        tool_ratio = (tool_decisions / total_exec_decisions) if total_exec_decisions > 0 else 0.0
        tool_queue_lengths = [len(self.tool_queues[p.tool_type_id]) for p in self.tool_profiles]
        tool_running_counts = [len(self.tool_running[p.tool_type_id]) for p in self.tool_profiles]
        active_tasks = sum(1 for t in self.tasks.values() if t.status == "active")

        self.metrics_history.append(
            {
                "time": round(self.time, 4),
                "completion_rate": (len(completed) / len(self.tasks)) if self.tasks else 0.0,
                "average_latency_s": round(avg_latency, 4),
                "tool_usage_ratio": round(tool_ratio, 4),
                "num_finished": len(finished),
                "num_active_tasks": int(active_tasks),
                "tool_queue_lengths": list(tool_queue_lengths),
                "tool_running_counts": list(tool_running_counts),
                "local_queue_length": int(len(self.local_queue)),
                "local_running_jobs": int(len(self.local_running)),
            }
        )

    def export_metrics(self, path: str) -> str:
        p = Path(path).expanduser().resolve()
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as f:
            json.dump(self.metrics_history, f, ensure_ascii=False, indent=2)
        return str(p)

    def export_task_records(self, path: str) -> str:
        p = Path(path).expanduser().resolve()
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as f:
            json.dump(self.completed_task_records, f, ensure_ascii=False, indent=2)
        return str(p)


def load_dataset(dataset_path: str) -> Dict[str, object]:
    p = Path(dataset_path).expanduser().resolve()
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)