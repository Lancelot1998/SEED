# -*- coding: utf-8 -*-
"""
Run an independent knowledge + Qwen encoder + D3QN baseline.

This module reuses the shared baseline runtime while providing an independent
D3QN policy and training loop. Run it through the preserved root entry point:

    python train_d3qn_baseline.py

Design:
  - Reuses the uploaded baseline's dataset loading, dynamic tool library,
    environment, prompt construction, knowledge retrieval, rollout logging,
    metrics, CSV writers, and checkpoint directories.
  - Uses a frozen Qwen backbone as the state encoder, matching the lightweight
    baseline mode in the shared runtime.
  - Replaces the policy action head with a dueling Q-network.
  - Trains with Double-DQN targets and a separate target dueling head.
  - Builds approximate node-level transitions from rollout traces. For each
    task, the next decision trace in the same task is treated as next_state;
    if no later trace exists for that task, the transition is terminal.
"""

from __future__ import annotations

import copy
import json
import math
import random
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from baseline_suite.config.paths import RunArtifacts
from baseline_suite.training import baseline_runtime as bc


# ============================================================
# User-editable D3QN globals
# ============================================================

D3QN_REPLAY_CAPACITY = 4096
D3QN_MIN_REPLAY_SIZE = 64
D3QN_BATCH_SIZE = 64
D3QN_UPDATES_PER_EPOCH = 4
D3QN_GAMMA = 0.92
D3QN_REWARD_SCALE = 0.01
D3QN_TARGET_UPDATE_EVERY_EPOCHS = 1
D3QN_TARGET_TAU = 0.20
D3QN_HIDDEN_SIZE = 512
D3QN_DROPOUT = 0.05
D3QN_GRAD_CLIP = 1.0
D3QN_HUBER_BETA = 1.0

D3QN_EPSILON_START = 0.60
D3QN_EPSILON_END = 0.05
D3QN_EPSILON_DECAY_EPOCHS = 60
D3QN_EVAL_EPSILON = 0.0

D3QN_MAX_NEW_TRANSITIONS_PER_EPOCH = 0  # 0 means keep all rollout transitions
D3QN_SAVE_REPLAY_JSONL = True
D3QN_POLICY_NAME = "D3QN"


# ============================================================
# D3QN data structures
# ============================================================


@dataclass
class D3QNTransition:
    prompt: str
    action: int
    reward: float
    next_prompt: str
    done: bool
    action_names: List[str] = field(default_factory=list)
    next_action_names: List[str] = field(default_factory=list)
    trace_id: str = ""
    task_id: str = ""
    node_id: int = -1
    next_trace_id: str = ""
    raw_reward: float = 0.0
    outcome_event_type: str = ""


class D3QNReplayBuffer:
    def __init__(self, capacity: int, seed: int = 0) -> None:
        self.capacity = int(max(1, capacity))
        self.rows: List[D3QNTransition] = []
        self.rng = random.Random(int(seed))

    def __len__(self) -> int:
        return len(self.rows)

    def extend(self, transitions: Sequence[D3QNTransition]) -> None:
        for tr in transitions:
            self.rows.append(tr)
        if len(self.rows) > self.capacity:
            self.rows = self.rows[-self.capacity:]

    def sample(self, batch_size: int) -> List[D3QNTransition]:
        n = min(int(batch_size), len(self.rows))
        if n <= 0:
            return []
        return self.rng.sample(self.rows, k=n)


# ============================================================
# Dueling Q-network policy
# ============================================================


class DuelingQHead(nn.Module):
    def __init__(self, input_dim: int, action_dim: int, hidden_dim: int = D3QN_HIDDEN_SIZE, dropout: float = D3QN_DROPOUT) -> None:
        super().__init__()
        hidden_dim = int(max(64, hidden_dim))
        self.feature = nn.Sequential(
            nn.Linear(int(input_dim), hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Dropout(float(dropout)),
        )
        self.value = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.advantage = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, int(action_dim)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.feature(x.float())
        value = self.value(z)
        advantage = self.advantage(z)
        return value + advantage - advantage.mean(dim=-1, keepdim=True)


class D3QNPolicy(bc.QwenActionPolicy):
    def __init__(self, action_dim: int, cfg: bc.PolicyConfig) -> None:
        super().__init__(action_dim=action_dim, cfg=cfg)
        hidden_size = int(getattr(self.model.config, "hidden_size", getattr(self.model.config, "n_embd", 4096)))
        device = bc.last_parameter_device(self.model)

        self.action_head = DuelingQHead(
            input_dim=hidden_size,
            action_dim=int(action_dim),
            hidden_dim=int(D3QN_HIDDEN_SIZE),
            dropout=float(D3QN_DROPOUT),
        ).to(device)
        self.target_action_head = copy.deepcopy(self.action_head).to(device)
        for p in self.target_action_head.parameters():
            p.requires_grad = False
        if hasattr(self, "value_head"):
            for p in self.value_head.parameters():
                p.requires_grad = False

        self.epsilon = float(D3QN_EPSILON_START)
        self._d3qn_rng = random.Random(int(cfg.seed) + 7919)
        self.sync_target(tau=1.0)

        trainable = sum(p.numel() for p in self.trainable_parameters() if p.requires_grad)
        print(
            f"[d3qn-policy] initialized dueling q-head | action_dim={self.action_dim} | "
            f"hidden={hidden_size} | trainable_params={trainable}",
            flush=True,
        )

    def trainable_parameters(self) -> List[nn.Parameter]:
        return [p for p in self.action_head.parameters() if p.requires_grad]

    def action_logp_matrix(
        self,
        prompts: Sequence[str],
        action_names: Optional[Sequence[Sequence[str]]] = None,
        require_grad: bool = True,
    ) -> torch.Tensor:
        prompts = [str(x) for x in prompts]
        if not prompts:
            return torch.zeros((0, self.action_dim), dtype=torch.float32, device=next(self.action_head.parameters()).device)
        reps = self.encode_prompts(prompts)
        ctx = torch.enable_grad() if bool(require_grad) else torch.no_grad()
        with ctx:
            return self.action_head(reps.float())

    def target_q_matrix(
        self,
        prompts: Sequence[str],
        action_names: Optional[Sequence[Sequence[str]]] = None,
    ) -> torch.Tensor:
        prompts = [str(x) for x in prompts]
        if not prompts:
            return torch.zeros((0, self.action_dim), dtype=torch.float32, device=next(self.target_action_head.parameters()).device)
        with torch.no_grad():
            reps = self.encode_prompts(prompts)
            return self.target_action_head(reps.float())

    def sync_target(self, tau: float = 1.0) -> None:
        tau = float(max(0.0, min(1.0, tau)))
        with torch.no_grad():
            for target_p, online_p in zip(self.target_action_head.parameters(), self.action_head.parameters()):
                target_p.data.mul_(1.0 - tau).add_(online_p.data, alpha=tau)

    def values_for_prompts(self, prompts: Sequence[str], require_grad: bool = True) -> torch.Tensor:
        prompts = list(prompts)
        device = next(self.action_head.parameters()).device
        if not prompts:
            return torch.zeros((0,), dtype=torch.float32, device=device)
        q = self.action_logp_matrix(prompts, require_grad=bool(require_grad))
        return q.max(dim=-1).values

    @torch.no_grad()
    def select_actions(
        self,
        dps: Sequence[Dict[str, object]],
        num_tools: int,
        use_memory: bool,
        knowledge_trainer=None,
        sample: bool = True,
        temperature: float = bc.ACTION_TEMPERATURE,
        graph_metas: Optional[Sequence[Dict[str, object]]] = None,
    ) -> List[Tuple[object, bc.ActionInfo]]:
        dps = list(dps)
        graph_metas = list(graph_metas or [{} for _ in dps])
        prompts: List[str] = []
        knowledge_prompts: List[str] = []
        action_names: List[List[str]] = []
        masks: List[List[int]] = []

        for dp, gm in zip(dps, graph_metas):
            ko = bc.query_knowledge_for_dp(dp, num_tools, knowledge_trainer, use_memory)
            prompt, names = bc.build_prompt_from_dp(dp, num_tools, str(ko.get("knowledge_prompt", "") or ""), gm)
            prompts.append(prompt)
            knowledge_prompts.append(str(ko.get("knowledge_prompt", "") or ""))
            action_names.append(list(names))
            masks.append(bc.valid_action_mask_from_dp(dp, num_tools))

        q_values = self.action_logp_matrix(prompts, action_names=action_names, require_grad=False)
        results: List[Tuple[object, bc.ActionInfo]] = []
        for i, dp in enumerate(dps):
            row = q_values[i].float()
            valid_indices = bc.action_mask_to_indices(masks[i]) or list(range(self.action_dim))
            if bool(sample) and self._d3qn_rng.random() < float(self.epsilon):
                idx = int(self._d3qn_rng.choice(valid_indices))
            else:
                masked = row.clone()
                invalid = [j for j in range(self.action_dim) if j not in set(valid_indices)]
                if invalid:
                    masked[torch.tensor(invalid, dtype=torch.long, device=masked.device)] = -1.0e9
                idx = int(torch.argmax(masked).item())

            probs = torch.softmax(row / max(1e-6, float(temperature)), dim=-1)
            env_action = bc.action_index_to_env_action(idx, num_tools)
            info = bc.ActionInfo(
                action_index=int(idx),
                action_type=str(env_action.action_type),
                tool_type_id=None if env_action.tool_type_id is None else int(env_action.tool_type_id),
                log_prob=float(torch.log(torch.clamp(probs[idx], min=1e-12)).detach().cpu().item()),
                probability=float(probs[idx].detach().cpu().item()),
                value=float(row[idx].detach().cpu().item()),
                valid_action_indices=list(valid_indices),
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
            "algorithm": "d3qn",
            "online_q_state_dict": {k: v.detach().cpu() for k, v in self.action_head.state_dict().items()},
            "target_q_state_dict": {k: v.detach().cpu() for k, v in self.target_action_head.state_dict().items()},
            "epsilon": float(self.epsilon),
            "extra": extra or {},
        }
        torch.save(state, p)
        return str(p)


# ============================================================
# Transition building and D3QN update
# ============================================================


def epsilon_by_epoch(epoch: int) -> float:
    if int(D3QN_EPSILON_DECAY_EPOCHS) <= 0:
        return float(D3QN_EPSILON_END)
    ratio = min(1.0, max(0.0, (int(epoch) - 1) / max(1, int(D3QN_EPSILON_DECAY_EPOCHS))))
    return float(D3QN_EPSILON_START + ratio * (D3QN_EPSILON_END - D3QN_EPSILON_START))


def transition_to_json(tr: D3QNTransition) -> Dict[str, object]:
    return dict(tr.__dict__)


def build_d3qn_transitions_from_traces(traces: Sequence[bc.RolloutTrace]) -> List[D3QNTransition]:
    by_task: Dict[str, List[bc.RolloutTrace]] = {}
    for tr in traces:
        if not str(tr.prompt or ""):
            continue
        by_task.setdefault(str(tr.task_id), []).append(tr)

    transitions: List[D3QNTransition] = []
    terminal_failure_events = {
        "task_failed",
        "node_timeout",
        "decision_tool_invalid_missing_id",
        "decision_tool_invalid_unknown_tool",
        "decision_tool_invalid_not_allowed",
        "decision_tool_rejected_queue_full",
        "decision_local_rejected_queue_full",
        "decision_invalid_generated_action",
    }

    for task_id, rows in by_task.items():
        rows = sorted(rows, key=lambda x: (float(x.env_time), int(x.step_count), int(x.node_id), str(x.trace_id)))
        for i, tr in enumerate(rows):
            next_tr = rows[i + 1] if i + 1 < len(rows) else None
            hard_terminal = str(tr.outcome_event_type) in terminal_failure_events and next_tr is None
            done = bool(next_tr is None or hard_terminal)
            next_prompt = "" if done else str(next_tr.prompt)
            next_action_names = [] if done else list(next_tr.action_names)
            transitions.append(
                D3QNTransition(
                    prompt=str(tr.prompt),
                    action=int(tr.exec_action),
                    reward=float(tr.reward) * float(D3QN_REWARD_SCALE),
                    next_prompt=next_prompt,
                    done=done,
                    action_names=list(tr.action_names),
                    next_action_names=next_action_names,
                    trace_id=str(tr.trace_id),
                    task_id=str(task_id),
                    node_id=int(tr.node_id),
                    next_trace_id="" if next_tr is None else str(next_tr.trace_id),
                    raw_reward=float(tr.reward),
                    outcome_event_type=str(tr.outcome_event_type),
                )
            )
            if int(D3QN_MAX_NEW_TRANSITIONS_PER_EPOCH) > 0 and len(transitions) >= int(D3QN_MAX_NEW_TRANSITIONS_PER_EPOCH):
                return transitions
    return transitions


def _safe_tensor(values: Sequence[float], device: torch.device) -> torch.Tensor:
    return torch.tensor([float(v) for v in values], dtype=torch.float32, device=device)


def train_d3qn(policy: D3QNPolicy, optimizer: torch.optim.Optimizer, replay: D3QNReplayBuffer) -> Dict[str, float]:
    if len(replay) < int(D3QN_MIN_REPLAY_SIZE):
        return {
            "loss": 0.0,
            "pref_acc": 0.0,
            "logp_margin": 0.0,
            "confidence_mean": 0.0,
            "grad_norm": 0.0,
            "grad_norm_clipped": 0.0,
            "update_seconds": 0.0,
            "tdpo_seconds": 0.0,
            "pair_gap_mean": 0.0,
            "q_selected_mean": 0.0,
            "q_target_mean": 0.0,
            "td_error_abs_mean": 0.0,
        }

    t0 = time.time()
    stats: List[Dict[str, float]] = []
    n_updates = max(1, int(D3QN_UPDATES_PER_EPOCH))
    for _ in range(n_updates):
        batch = replay.sample(int(D3QN_BATCH_SIZE))
        if not batch:
            continue

        prompts = [tr.prompt for tr in batch]
        next_prompts = [tr.next_prompt if not tr.done else tr.prompt for tr in batch]
        action_names = [tr.action_names for tr in batch]
        next_action_names = [tr.next_action_names if not tr.done else tr.action_names for tr in batch]

        q_values = policy.action_logp_matrix(prompts, action_names=action_names, require_grad=True).float()
        device = q_values.device
        actions = torch.tensor([int(tr.action) for tr in batch], dtype=torch.long, device=device)
        rewards = _safe_tensor([tr.reward for tr in batch], device)
        dones = _safe_tensor([1.0 if tr.done else 0.0 for tr in batch], device)
        q_selected = q_values.gather(1, actions.view(-1, 1)).squeeze(1)

        with torch.no_grad():
            # Double-DQN: online head selects the next action; target head evaluates it.
            next_online_q = policy.action_logp_matrix(next_prompts, action_names=next_action_names, require_grad=False).float()
            next_actions = torch.argmax(next_online_q, dim=-1)
            next_target_q = policy.target_q_matrix(next_prompts, action_names=next_action_names).float().to(device)
            next_selected = next_target_q.gather(1, next_actions.view(-1, 1)).squeeze(1)
            target = rewards + float(D3QN_GAMMA) * (1.0 - dones) * next_selected

        loss = F.smooth_l1_loss(q_selected, target, beta=float(D3QN_HUBER_BETA))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(policy.trainable_parameters(), float(D3QN_GRAD_CLIP))
        optimizer.step()

        with torch.no_grad():
            td_error = target - q_selected.detach()
            greedy_now = torch.argmax(q_values.detach(), dim=-1)
            selected_is_greedy = (greedy_now == actions).float().mean().item()
            q_margin = (q_selected.detach() - q_values.detach().mean(dim=-1)).mean().item()
            stats.append(
                {
                    "loss": float(loss.detach().cpu().item()),
                    "grad": float(grad),
                    "selected_is_greedy": float(selected_is_greedy),
                    "q_margin": float(q_margin),
                    "q_selected": float(q_selected.detach().mean().cpu().item()),
                    "q_target": float(target.detach().mean().cpu().item()),
                    "td_abs": float(td_error.abs().mean().cpu().item()),
                    "max_q": float(q_values.detach().max(dim=-1).values.mean().cpu().item()),
                }
            )

    return {
        "loss": bc.safe_mean([s["loss"] for s in stats]),
        "pref_acc": bc.safe_mean([s["selected_is_greedy"] for s in stats]),
        "logp_margin": bc.safe_mean([s["q_margin"] for s in stats]),
        "confidence_mean": bc.safe_mean([s["max_q"] for s in stats]),
        "grad_norm": bc.safe_mean([s["grad"] for s in stats]),
        "grad_norm_clipped": min(float(D3QN_GRAD_CLIP), bc.safe_mean([s["grad"] for s in stats])),
        "update_seconds": time.time() - t0,
        "tdpo_seconds": time.time() - t0,
        "pair_gap_mean": bc.safe_mean([s["q_target"] for s in stats]),
        "q_selected_mean": bc.safe_mean([s["q_selected"] for s in stats]),
        "q_target_mean": bc.safe_mean([s["q_target"] for s in stats]),
        "td_error_abs_mean": bc.safe_mean([s["td_abs"] for s in stats]),
    }


# ============================================================
# Main D3QN training loop
# ============================================================


def run_training_d3qn() -> None:
    random.seed(bc.SEED)
    torch.manual_seed(bc.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(bc.SEED)

    bc.print_cuda_layout()
    algorithm = "d3qn"
    policy_name = D3QN_POLICY_NAME
    run_name = "qwen7b_knowledge_d3qn_baseline"
    run_dir = bc.OUTPUT_ROOT / f"{run_name}_{bc.now_tag()}"
    artifacts = RunArtifacts(run_dir)
    for subdirectory in artifacts.d3qn_subdirectories:
        bc.ensure_dir(subdirectory)

    dataset, scan_info = bc.load_dataset_from_dir(bc.DATA_DIR, seed=bc.SEED)
    tool_library = bc.load_dynamic_tool_library()
    coverage = bc.validate_dynamic_tool_library_coverage(dataset, tool_library)
    if bc.SAVE_AUXILIARY_JSON:
        bc.save_json(artifacts.dataset_scan, scan_info)

    num_tools = len(dataset.get("tool_catalog", []))
    action_dim = bc.action_dim_from_num_tools(num_tools)
    graph_meta = bc.build_graph_meta_map(dataset)

    bc.save_json(
        artifacts.run_config,
        {
            "algorithm": algorithm,
            "DATA_DIR": str(bc.DATA_DIR),
            "SAMPLE_NUM_GRAPHS": int(bc.SAMPLE_NUM_GRAPHS),
            "ROLLOUT_CHUNK_GRAPHS": int(bc.ROLLOUT_CHUNK_GRAPHS),
            "SAMPLE_IN_ORDER": bool(bc.SAMPLE_IN_ORDER),
            "SAMPLE_RANDOM_POOL_LIMIT": int(bc.SAMPLE_RANDOM_POOL_LIMIT),
            "dataset_scan": scan_info,
            "USE_DYNAMIC_TOOL_LIBRARY": bool(bc.USE_DYNAMIC_TOOL_LIBRARY),
            "USE_MULTIPLE_TOOL_LIBRARIES": bool(bc.USE_MULTIPLE_TOOL_LIBRARIES),
            "TOOL_LIBRARY_PATHS": [str(p) for p in bc._selected_tool_library_paths()],
            "ALLOW_TOOL_LIBRARY_FALLBACK": bool(bc.ALLOW_TOOL_LIBRARY_FALLBACK),
            "tool_library_coverage": coverage,
            "ENV_RELAXED_RESOURCE_MODE": bool(bc.ENV_RELAXED_RESOURCE_MODE),
            "ENV_Q_MAX": int(bc.ENV_Q_MAX),
            "ENV_TOOL_INSTANCE_MULTIPLIER": float(bc.ENV_TOOL_INSTANCE_MULTIPLIER),
            "ENV_TOOL_INSTANCE_ADDITIVE": int(bc.ENV_TOOL_INSTANCE_ADDITIVE),
            "ENV_MIN_TOOL_INSTANCES": int(bc.ENV_MIN_TOOL_INSTANCES),
            "ENV_ARRIVAL_GAP_RANGE": list(bc.ENV_ARRIVAL_GAP_RANGE),
            "ENV_NODE_DEADLINE_MULTIPLIER": float(bc.ENV_NODE_DEADLINE_MULTIPLIER),
            "ENV_TASK_DEADLINE_MULTIPLIER": float(bc.ENV_TASK_DEADLINE_MULTIPLIER),
            "ENV_LOCAL_MAX_CONCURRENCY": int(bc.ENV_LOCAL_MAX_CONCURRENCY),
            "ENV_LOCAL_QUEUE_CAPACITY": int(bc.ENV_LOCAL_QUEUE_CAPACITY),
            "QWEN_MODEL_PATH": bc.QWEN_MODEL_PATH,
            "PRETRAINED_KNOWLEDGE_MODEL_PATH": str(bc.PRETRAINED_KNOWLEDGE_MODEL_PATH),
            "USE_KNOWLEDGE_MEMORY": bool(bc.USE_KNOWLEDGE_MEMORY),
            "GPU_PROFILE": bc.GPU_PROFILE,
            "CUDA_VISIBLE_DEVICES_DEFAULT": bc.CUDA_VISIBLE_DEVICES_DEFAULT,
            "LLM_ENABLE_MULTI_GPU": bool(bc.LLM_ENABLE_MULTI_GPU),
            "LLM_DEVICE_MAP": bc.LLM_DEVICE_MAP,
            "LLM_MAX_MEMORY_PER_GPU": bc.LLM_MAX_MEMORY_PER_GPU,
            "NUM_EPOCHS": int(bc.NUM_EPOCHS),
            "LEARNING_RATE": float(bc.LEARNING_RATE),
            "WEIGHT_DECAY": float(bc.WEIGHT_DECAY),
            "ACTION_TEMPERATURE": float(bc.ACTION_TEMPERATURE),
            "TRAIN_WITH_SAMPLING": bool(bc.TRAIN_WITH_SAMPLING),
            "FAST_COMPACT_PROMPT": bool(bc.FAST_COMPACT_PROMPT),
            "KNOWLEDGE_PROMPT_MAX_CHARS": int(bc.KNOWLEDGE_PROMPT_MAX_CHARS),
            "LIGHTWEIGHT_HEAD_ONLY_POLICY": bool(bc.LIGHTWEIGHT_HEAD_ONLY_POLICY),
            "LLM_USE_LORA": bool(bc.LLM_USE_LORA),
            "LLM_MAX_LENGTH": int(bc.LLM_MAX_LENGTH),
            "D3QN_REPLAY_CAPACITY": int(D3QN_REPLAY_CAPACITY),
            "D3QN_MIN_REPLAY_SIZE": int(D3QN_MIN_REPLAY_SIZE),
            "D3QN_BATCH_SIZE": int(D3QN_BATCH_SIZE),
            "D3QN_UPDATES_PER_EPOCH": int(D3QN_UPDATES_PER_EPOCH),
            "D3QN_GAMMA": float(D3QN_GAMMA),
            "D3QN_REWARD_SCALE": float(D3QN_REWARD_SCALE),
            "D3QN_TARGET_UPDATE_EVERY_EPOCHS": int(D3QN_TARGET_UPDATE_EVERY_EPOCHS),
            "D3QN_TARGET_TAU": float(D3QN_TARGET_TAU),
            "D3QN_EPSILON_START": float(D3QN_EPSILON_START),
            "D3QN_EPSILON_END": float(D3QN_EPSILON_END),
            "D3QN_EPSILON_DECAY_EPOCHS": int(D3QN_EPSILON_DECAY_EPOCHS),
            "KNOWLEDGE_UPDATE_STRATEGY": int(bc.KNOWLEDGE_UPDATE_STRATEGY),
            "KNOWLEDGE_UPDATE_EVERY_EPOCHS": int(bc.KNOWLEDGE_UPDATE_EVERY_EPOCHS),
            "KNOWLEDGE_FAST_UPDATE_TEXT_ENCODER": bool(bc.KNOWLEDGE_FAST_UPDATE_TEXT_ENCODER),
        },
    )

    knowledge_rows: List[Dict[str, object]] = []
    knowledge_trainer = (
        bc.load_knowledge_trainer(dataset, action_dim, run_dir)
        if bool(bc.USE_KNOWLEDGE_MEMORY) or int(bc.KNOWLEDGE_UPDATE_STRATEGY) in {1, 2, 4}
        else None
    )
    if knowledge_trainer is not None and bool(bc.RELEASE_KNOWLEDGE_BACKBONE_BEFORE_POLICY):
        release_info = bc.release_knowledge_backbone_for_policy(knowledge_trainer)
        if bc.SAVE_AUXILIARY_JSON:
            bc.save_json(artifacts.knowledge_release, release_info)
        print(f"[knowledge-release] {release_info}", flush=True)
    bc.safe_cuda_empty_cache()

    policy_cfg = bc.PolicyConfig(
        model_path=bc.resolve_llm_path(bc.QWEN_MODEL_PATH, bc.QWEN_MODEL_PATH_FALLBACK),
        tokenizer_path=None,
        dtype=bc.LLM_DTYPE,
        max_length=int(bc.LLM_MAX_LENGTH),
        use_lora=bool(bc.LLM_USE_LORA),
        lora_layer_ids=tuple(bc.LLM_LORA_LAYER_IDS),
        lora_target_modules=tuple(bc.LLM_LORA_TARGET_MODULES),
        lora_r=int(bc.LLM_LORA_R),
        lora_alpha=float(bc.LLM_LORA_ALPHA),
        lora_dropout=float(bc.LLM_LORA_DROPOUT),
        tune_last_n_blocks=int(bc.LLM_TUNE_LAST_N_BLOCKS),
        train_final_norm=bool(bc.LLM_TRAIN_FINAL_NORM),
        gradient_checkpointing=bool(bc.LLM_GRADIENT_CHECKPOINTING),
        enable_multi_gpu=bool(bc.LLM_ENABLE_MULTI_GPU),
        device_map=str(bc.LLM_DEVICE_MAP),
        max_memory_per_gpu=str(bc.LLM_MAX_MEMORY_PER_GPU),
        low_cpu_mem_usage=bool(bc.LLM_LOW_CPU_MEM_USAGE),
        offload_folder=str(artifacts.policy_offload),
        seed=int(bc.SEED),
        verbose=True,
    )
    policy = D3QNPolicy(action_dim=action_dim, cfg=policy_cfg)
    optimizer = torch.optim.AdamW(policy.trainable_parameters(), lr=float(bc.LEARNING_RATE), weight_decay=float(bc.WEIGHT_DECAY))
    replay = D3QNReplayBuffer(capacity=int(D3QN_REPLAY_CAPACITY), seed=int(bc.SEED))

    epoch_rows: List[Dict[str, object]] = []
    eval_rows: List[Dict[str, object]] = []
    energy_rows: List[Dict[str, object]] = []
    delay_rows: List[Dict[str, object]] = []
    best_score = -1.0e18
    best_ckpt = ""

    for epoch in range(1, int(bc.NUM_EPOCHS) + 1):
        epoch_t0 = time.time()
        policy.epsilon = epsilon_by_epoch(epoch)
        print(f"[d3qn] epoch {epoch}/{bc.NUM_EPOCHS} start | epsilon={policy.epsilon:.4f}", flush=True)

        rollout_summary, task_records, traces = bc.run_policy_rollout_chunked(
            dataset,
            graph_meta,
            policy,
            knowledge_trainer,
            epoch,
            int(bc.SEED + epoch),
            bool(bc.USE_KNOWLEDGE_MEMORY),
            bool(bc.TRAIN_WITH_SAMPLING),
            policy_name,
        )
        print(
            f"[d3qn-stage] epoch={epoch} rollout_done time={bc.fmt_seconds(float(rollout_summary.get('rollout_seconds', 0.0)))} traces={len(traces)}",
            flush=True,
        )

        pair_t0 = time.time()
        new_transitions = build_d3qn_transitions_from_traces(traces)
        replay.extend(new_transitions)
        pair_seconds = time.time() - pair_t0

        train_stats = train_d3qn(policy, optimizer, replay)
        if int(D3QN_TARGET_UPDATE_EVERY_EPOCHS) > 0 and epoch % int(D3QN_TARGET_UPDATE_EVERY_EPOCHS) == 0:
            policy.sync_target(tau=float(D3QN_TARGET_TAU))
        bc.safe_cuda_empty_cache()

        # Greedy evaluation uses sample=False in the shared runtime. Set epsilon to 0
        # around evaluation for clarity, then restore it for the next epoch.
        old_eps = float(policy.epsilon)
        policy.epsilon = float(D3QN_EVAL_EPSILON)
        eval_row = bc._eval_after_epoch(dataset, graph_meta, policy, knowledge_trainer, epoch, policy_name, eval_rows, energy_rows, delay_rows, run_dir)
        policy.epsilon = old_eps

        knowledge_stats: Dict[str, object] = {}
        if knowledge_trainer is not None and int(bc.KNOWLEDGE_UPDATE_STRATEGY) in {1, 2, 4} and epoch % max(1, int(bc.KNOWLEDGE_UPDATE_EVERY_EPOCHS)) == 0:
            print(f"[knowledge-update] epoch={epoch} strategy={bc.KNOWLEDGE_UPDATE_STRATEGY} records={len(task_records)}", flush=True)
            knowledge_stats = bc.update_knowledge_from_records(knowledge_trainer, task_records, epoch=epoch)
            if bc.SAVE_AUXILIARY_JSON:
                bc.save_json(artifacts.knowledge_update(epoch), knowledge_stats)
            bc.save_knowledge_artifacts(knowledge_trainer, run_dir, tag="best")
            knowledge_rows.append(bc.build_knowledge_row("epoch_update", epoch, knowledge_stats, knowledge_trainer.knowledge_digest()))
            bc.write_knowledge_csv(artifacts.knowledge_metrics, knowledge_rows)

        digest = bc.knowledge_digest_or_empty(knowledge_trainer)
        score_source = eval_row if eval_row is not None else rollout_summary
        score = float(score_source.get("completion_rate", 0.0)) * 1000.0 - float(score_source.get("avg_latency_s", 0.0))
        is_best = False
        if bool(bc.SAVE_EVERY_EPOCH):
            policy.save_checkpoint(
                str(artifacts.policy_checkpoint("d3qn", epoch)),
                extra={"epoch": epoch, "summary": rollout_summary, "train_stats": train_stats, "knowledge_digest": digest},
            )
        if score > best_score:
            is_best = True
            best_score = score
            best_ckpt = policy.save_checkpoint(
                str(artifacts.best_checkpoint),
                extra={"epoch": epoch, "score": score, "summary": rollout_summary, "train_stats": train_stats, "knowledge_digest": digest},
            )

        if bc.SAVE_TASK_RECORDS_EVERY_EPOCH:
            bc.save_json(artifacts.task_records(epoch), task_records)
        if bc.SAVE_TRACE_JSONL:
            bc.append_jsonl(artifacts.traces, [bc.trace_to_json(t) for t in traces])
            if bool(D3QN_SAVE_REPLAY_JSONL):
                bc.append_jsonl(artifacts.d3qn_transitions, [transition_to_json(t) for t in new_transitions])
        if bc.WRITE_PROMPT_SAMPLES and epoch == 1:
            bc.save_json(
                artifacts.prompt_samples,
                [
                    {
                        "trace_id": t.trace_id,
                        "task_id": t.task_id,
                        "node_id": t.node_id,
                        "exec_action": t.exec_action,
                        "prompt": t.prompt,
                        "knowledge_prompt": t.knowledge_prompt,
                        "action_names": t.action_names,
                    }
                    for t in traces[: int(bc.MAX_PROMPT_SAMPLES)]
                ],
            )

        counts = {
            "num_pairs_added": len(new_transitions),
            "num_pairs_buffer": len(replay),
            "num_success_buffer": sum(1 for tr in replay.rows if tr.raw_reward > 0.0),
            "num_explore_buffer": len(replay),
            "num_pairs_train": min(len(replay), int(D3QN_BATCH_SIZE) * max(1, int(D3QN_UPDATES_PER_EPOCH))),
            "num_tier_a_pairs": len(new_transitions),
            "num_tier_a_raw": len(new_transitions),
        }
        times = {
            "pair_seconds": pair_seconds,
            "new_ref_seconds": 0.0,
            "ref_seconds": 0.0,
            "knowledge_update_seconds": float(knowledge_stats.get("fit_seconds", knowledge_stats.get("update_seconds", 0.0)) if knowledge_stats else 0.0),
            "epoch_seconds": time.time() - epoch_t0,
        }
        row = bc._row_from_epoch(epoch, rollout_summary, traces, train_stats, counts, times, digest, is_best)
        epoch_rows.append(row)
        bc.write_epoch_csv(artifacts.epoch_metrics, epoch_rows)

        e_rows, d_rows = bc.build_energy_delay_detail_rows(epoch, "rollout", task_records)
        energy_rows.extend(e_rows)
        delay_rows.extend(d_rows)
        bc.write_energy_csv(artifacts.energy_metrics, energy_rows)
        bc.write_delay_csv(artifacts.delay_metrics, delay_rows)

        if bc.SAVE_LATEST_SUMMARY_JSON:
            bc.save_json(
                artifacts.latest_summary,
                {
                    "epoch": epoch,
                    "row": row,
                    "rollout_summary": rollout_summary,
                    "train_stats": train_stats,
                    "knowledge_stats": knowledge_stats,
                    "knowledge_digest": digest,
                    "best_ckpt": best_ckpt,
                    "epsilon": float(policy.epsilon),
                    "replay_size": len(replay),
                },
            )

        print(
            f"[d3qn] epoch={epoch} comp={row['completion_rate']:.4f} lat={row['avg_latency_s']:.4f} "
            f"transitions+={row['num_pairs_added']} replay={row['num_pairs_buffer']} "
            f"loss={row['tdpo_loss']:.4f} greedy_match={row['tdpo_pref_acc']:.4f} "
            f"q_margin={row['tdpo_margin']:.4f} grad={row['tdpo_grad_norm']:.3e}/{row['tdpo_grad_norm_clipped']:.3e} "
            f"rollout={bc.fmt_seconds(row['rollout_seconds'])} update={bc.fmt_seconds(row['tdpo_update_seconds'])} "
            f"knowledge_proto={row['knowledge_num_prototypes']}",
            flush=True,
        )

    final_knowledge = bc.save_knowledge_artifacts(knowledge_trainer, run_dir, tag="final") if knowledge_trainer is not None else {"disabled": "true"}
    if bc.SAVE_FINAL_RESULT_JSON:
        bc.save_json(
            artifacts.final_result,
            {
                "algorithm": algorithm,
                "run_dir": str(run_dir.resolve()),
                "best_ckpt": best_ckpt,
                "final_knowledge": final_knowledge,
                "num_epochs": int(bc.NUM_EPOCHS),
                "best_score": float(best_score),
                "eval_csv": str(artifacts.eval_metrics.resolve()) if bool(bc.RUN_GREEDY_EVAL_EACH_EPOCH) else "",
            },
        )
    print(f"[d3qn] done | run_dir={run_dir.resolve()}", flush=True)
    print(f"[d3qn] best_ckpt={best_ckpt}", flush=True)


if __name__ == "__main__":
    run_training_d3qn()
