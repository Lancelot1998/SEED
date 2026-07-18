# -*- coding: utf-8 -*-
"""
decision_TDPO.py

LLM + TDPO decision module for the existing task_env.py and knowledge_aspect_adapter.py
interfaces.  This file intentionally keeps all tunable parameters near the top so that
experiments can be reproduced by editing globals only.

Action convention used by this file:
    0 ... NUM_TOOL_TYPES-1        -> tool action with tool_type_id=index
    NUM_TOOL_TYPES                -> local/no-tool action
    NUM_TOOL_TYPES + 1            -> pause action
This follows the existing knowledge_aspect_adapter convention where the final two
actions are local and pause.
"""

from __future__ import annotations
import re
import copy
import json
import math
import os
import random
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# Keep allocator behavior deterministic and fragmentation-tolerant.  Newer PyTorch
# versions prefer PYTORCH_ALLOC_CONF; the legacy variable is deliberately not set
# here to avoid the deprecation warning shown in the user's run log.
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except Exception as exc:  # pragma: no cover
    raise RuntimeError("transformers is required. Please install transformers before running decision_TDPO.py") from exc

try:
    from llm_tdpo_ablation.environment import task_env as task_env_mod
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "The llm_tdpo_ablation environment package must be importable."
    ) from exc

try:
    from llm_tdpo_ablation.knowledge import adapter as knowledge_mod
except Exception:
    knowledge_mod = None

from llm_tdpo_ablation.config.paths import (
    POLICY_OFFLOAD_DIR,
    QWEN_MODEL_PATH,
    QWEN_MODEL_PATH_FALLBACK,
)


# ============================================================
# User-editable globals: LLM policy
# ============================================================

LLM_MODEL_PATH = QWEN_MODEL_PATH
LLM_MODEL_PATH_FALLBACK = QWEN_MODEL_PATH_FALLBACK
LLM_TOKENIZER_PATH: Optional[str] = None
LLM_DEVICE = "auto"
LLM_DTYPE = "bfloat16"  # one of: float32, float16, bfloat16
LLM_MAX_LENGTH = 512

# LoRA fine-tuning.  The base Qwen weights remain frozen; only LoRA adapters
# inserted into the selected transformer blocks plus the decision heads are trained.
# Use negative ids for indexing from the end, e.g., (-2, -1) means the last two blocks.
LLM_USE_LORA = True
LLM_LORA_NUM_LAYERS = 4
LLM_LORA_LAYER_IDS: Tuple[int, ...] = tuple(range(-int(LLM_LORA_NUM_LAYERS), 0)) if int(LLM_LORA_NUM_LAYERS) > 0 else ()
LLM_LORA_TARGET_MODULES: Tuple[str, ...] = (
    "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"
)
LLM_LORA_R = 8
LLM_LORA_ALPHA = 16.0
LLM_LORA_DROPOUT = 0.05

# Fallback mode only: if LLM_USE_LORA=False, this controls full-block fine-tuning.
LLM_TUNE_LAST_N_BLOCKS = 0
LLM_TRAIN_FINAL_NORM = False
LLM_GRADIENT_CHECKPOINTING = True
LLM_USE_CACHE = False

# Multi-GPU model parallel loading.  This is the key change for 4 x RTX 4090:
# the Qwen backbone is sharded across all visible GPUs instead of being moved
# wholesale to cuda:0.  Keep CUDA_VISIBLE_DEVICES="0,1,2,3" when launching.
LLM_ENABLE_MULTI_GPU = True
LLM_DEVICE_MAP = "balanced"         # Recommended: "balanced" for 4 GPUs.  Use "none" for single GPU.
LLM_MAX_MEMORY_PER_GPU = "18GiB"    # Leaves headroom for gradients, AdamW states, and activations on 24GB cards.
LLM_LOW_CPU_MEM_USAGE = True
LLM_OFFLOAD_FOLDER = str(POLICY_OFFLOAD_DIR)

# Action head and memory fusion.
ACTION_HEAD_DROPOUT = 0.05
MEMORY_VECTOR_DIM_FALLBACK = 64
MEMORY_FUSION_HIDDEN = 256
# MEMORY_SCALE = 0.35
# MEMORY_ACTION_BIAS_SCALE = 0.25
PAUSE_LOGIT_BIAS = -3.0
INVALID_ACTION_LOGIT = -1.0e9

########
ACTION_TEMPERATURE = 0.60
TDPO_BETA = 0.05
TDPO_CONFIDENCE_MIN = 0.60
TDPO_GRAD_CLIP_NORM = 1.0
# Prompt-only knowledge injection.
# The compact state vector is used only as a retrieval key for matched memory.
# Retrieved memory is appended to the LLM prompt as text; numeric memory vectors
# and action-bias priors are not fused into hidden states or logits.
MEMORY_TEXT_ONLY_IN_PROMPT = True
MEMORY_SCALE = 0.0
MEMORY_ACTION_BIAS_SCALE = 0.0
# Prompt memory text control.  Set KNOWLEDGE_PROMPT_COMPRESS=True to restore
# compact top-k truncation; the default keeps the retrieved memory text intact.
KNOWLEDGE_PROMPT_COMPRESS = False
KNOWLEDGE_PROMPT_TOP_K = 2
KNOWLEDGE_PROMPT_MAX_CHARS_PER_ITEM = 220
KNOWLEDGE_PROMPT_TOTAL_MAX_CHARS = 700
KNOWLEDGE_ACTION_BIAS_CLIP = 1.2
KNOWLEDGE_RECOMMEND_TOP_K = 2
#########

# Action selection.
# ACTION_TEMPERATURE = 0.85
GREEDY_DURING_EVAL = True
SAMPLE_DURING_TRAIN = False
MIN_ACTION_PROB = 1.0e-8

# Prompt construction.
# No-skill baseline: do not expose concrete tool-state parameters in the LLM prompt.
# The environment still uses the same dynamic tool library and action mask internally.
PROMPT_INCLUDE_FULL_TOOL_TABLE = False
PROMPT_INCLUDE_LOCAL_OPTION_DETAILS = False
PROMPT_INCLUDE_INTENT_IF_AVAILABLE = False
PROMPT_DECIMAL_DIGITS = 4
# Rich prompt + prompt-only legality mode.  The prompt exposes task/node/DAG
# summaries and an allowed-tool indicator as text, but it does not expose a tool
# table, per-tool queue state, predicted latency, risk, or packet-loss fields.
PROMPT_RICH_TASK_STATE = True
PROMPT_INCLUDE_ALLOWED_TOOL_MASK_TEXT = True
PROMPT_INTENT_MAX_CHARS = 240
PROMPT_ALLOWED_TOOL_RULE_EXAMPLE = True
# Disable policy-layer action masking so the LLM generator must learn legality
# from prompt text and environment feedback.  Environment-level validity checks
# still execute and invalid/rejected actions become negative TDPO outcomes.
DISABLE_POLICY_ACTION_MASK = True
# Use a PDF-state-like prompt by default: keep feasible actions and twin-state
# latency components, but hide direct oracle-like fields such as predicted_total_s,
# risk_estimate, and packet_loss_rate from the LLM prompt. Set to "full" to restore
# the previous prompt.
PROMPT_TOOL_TABLE_MODE = "full"  # full, pdf_state

# Keep the original lightweight classifier scoring for speed and memory safety.
# The stronger overfitting source is reduced by random counterfactual sampling and
# by hiding oracle-like prompt fields, rather than by using the much heavier verbalizer.
ACTION_SCORING_MODE = "generative"

# Direct LLM action generation mode.  No external action head is trained or used.
# The frozen Qwen lm_head scores/generates short completions of the form
# ACTION_INDEX=<integer>, while only LoRA adapter parameters are trainable.
LLM_DIRECT_ACTION_GENERATION = True
LLM_ACTION_OUTPUT_PREFIX = "ACTION_INDEX="
LLM_GENERATE_MAX_NEW_TOKENS = 8
LLM_GENERATION_FALLBACK_TO_SCORING = True
# Rollout speed switch.  When True, select_actions() does not score all
# candidate ACTION_INDEX completions before generation.  It only calls
# model.generate() during rollout; candidate completion scoring remains used
# for TDPO/DPO loss, reference log-probs, and optional parse-failure fallback.
LLM_ROLLOUT_GENERATE_ONLY = True
# Controls what happens when Qwen does not generate a valid in-range ACTION_INDEX.
# True: fall back to candidate completion scoring, so rollout continues.
# False: send a special invalid_generated action to task_env.py; the environment
#        records the parse failure and directly marks the current task failed.
LLM_PARSE_FAILURE_DIRECT_FAIL_REASON = "llm_generation_parse_failed_or_out_of_range"
LLM_COMPLETION_SCORE_BATCH_SIZE = 24
TRAIN_LM_HEAD = False
TRAIN_INPUT_EMBEDDINGS = False
MERGE_LORA_INTO_BASE = False

# TDPO loss.
# TDPO_BETA = 0.10
# TDPO_CONFIDENCE_MIN = 0.05
# TDPO_GRAD_CLIP_NORM = 1.0
TDPO_EPS = 1.0e-8

# DT-sandbox utility defaults.
UTILITY_W_NODE = 1.0
UTILITY_W_MISSION = 1.5
UTILITY_W_VIOL = 2.0
UTILITY_W_LATENCY = 5.0
PAUSE_SANDBOX_DURATION_S = 0.20
PAUSE_SANDBOX_VIOLATION_BONUS = 0.10

# Completion-first preference shaping.
# When enabled, TDPO pair construction and counterfactual selection use a hierarchical
# utility scale: mission completion dominates node success, deadline/violation, and latency.
# This changes only the decision preference signal in this file; environment dynamics,
# masks, queues, knowledge memory, and train_TDPO.py remain unchanged.
COMPLETION_FIRST_OBJECTIVE = False
COMPLETION_FIRST_MISSION_WEIGHT = 200.0
COMPLETION_FIRST_NODE_WEIGHT = 20.0
COMPLETION_FIRST_VIOLATION_WEIGHT = 80.0
COMPLETION_FIRST_LATENCY_WEIGHT = 0.05
COMPLETION_FIRST_LATENCY_CAP = 2.0

# Wireless-regime confidence for Tier-B pairs.
WIRELESS_TAU = 1.0
TIER_B_MAX_HISTORY_PER_TYPE = 256

# Counterfactual selection for Tier-A pairs. "random" avoids using the DT sandbox
# as a best-action teacher. "best" restores the previous strongest-counterfactual behavior.
SANDBOX_COUNTERFACTUAL_SELECTION = "random"  # random, best


# ============================================================
# Basic helpers
# ============================================================


def resolve_llm_path(path: str = LLM_MODEL_PATH, fallback: str = LLM_MODEL_PATH_FALLBACK) -> str:
    p = Path(path).expanduser()
    if p.exists():
        return str(p)
    pf = Path(fallback).expanduser()
    if pf.exists():
        return str(pf)
    return path


def dtype_from_name(name: str) -> torch.dtype:
    name = str(name).lower()
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if name in {"fp16", "float16", "half"}:
        return torch.float16
    return torch.float32


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


def safe_empty_cuda_cache() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def safe_float(x: object, default: float = 0.0) -> float:
    try:
        if x is None:
            return float(default)
        return float(x)
    except Exception:
        return float(default)


def sigmoid_float(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def minmax_unit(x: float, lo: float, hi: float) -> float:
    if hi <= lo:
        return 0.0
    return max(0.0, min(1.0, (float(x) - float(lo)) / (float(hi) - float(lo))))


def pad_or_trim(values: Sequence[float], dim: int) -> List[float]:
    vals = [float(v) for v in values]
    if len(vals) >= dim:
        return vals[:dim]
    return vals + [0.0] * (dim - len(vals))


def mean_or_zero(values: Sequence[float]) -> float:
    vals = [float(v) for v in values if v is not None]
    return float(sum(vals) / len(vals)) if vals else 0.0


def normalize_layer_ids(layer_ids: Sequence[int], num_layers: int) -> List[int]:
    """Normalize positive/negative block ids and remove duplicates while preserving order."""
    out: List[int] = []
    for raw in layer_ids:
        idx = int(raw)
        if idx < 0:
            idx = int(num_layers) + idx
        if 0 <= idx < int(num_layers) and idx not in out:
            out.append(idx)
    return out


class LoRALinear(nn.Module):
    """Minimal LoRA wrapper for nn.Linear that leaves the original weight frozen."""

    def __init__(self, base_layer: nn.Linear, r: int, alpha: float, dropout: float) -> None:
        super().__init__()
        if not isinstance(base_layer, nn.Linear):
            raise TypeError(f"LoRALinear expects nn.Linear, got {type(base_layer)}")
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
        # Keep LoRA in fp32 for optimizer stability.  Forward casts only the small
        # LoRA branch input/output, so bf16 sharded Qwen activations still work.
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


def inject_lora_into_block(
    block: nn.Module,
    target_module_names: Sequence[str],
    r: int,
    alpha: float,
    dropout: float,
) -> List[str]:
    """Replace selected child Linear modules in one transformer block with LoRA wrappers."""
    target_set = {str(x) for x in target_module_names}
    replaced: List[str] = []
    for module_path, module in list(block.named_modules()):
        for child_name, child in list(module.named_children()):
            if child_name in target_set and isinstance(child, nn.Linear) and not isinstance(child, LoRALinear):
                setattr(module, child_name, LoRALinear(child, r=int(r), alpha=float(alpha), dropout=float(dropout)))
                full_name = f"{module_path}.{child_name}" if module_path else child_name
                replaced.append(full_name)
    return replaced


# ============================================================
# Dataclasses
# ============================================================


@dataclass
class TDPOPolicyConfig:
    model_path: str = LLM_MODEL_PATH
    tokenizer_path: Optional[str] = LLM_TOKENIZER_PATH
    device: str = LLM_DEVICE
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
    use_cache: bool = LLM_USE_CACHE
    enable_multi_gpu: bool = LLM_ENABLE_MULTI_GPU
    device_map: str = LLM_DEVICE_MAP
    max_memory_per_gpu: str = LLM_MAX_MEMORY_PER_GPU
    low_cpu_mem_usage: bool = LLM_LOW_CPU_MEM_USAGE
    offload_folder: str = LLM_OFFLOAD_FOLDER
    action_head_dropout: float = ACTION_HEAD_DROPOUT
    memory_vector_dim: int = MEMORY_VECTOR_DIM_FALLBACK
    memory_fusion_hidden: int = MEMORY_FUSION_HIDDEN
    memory_scale: float = MEMORY_SCALE
    memory_action_bias_scale: float = MEMORY_ACTION_BIAS_SCALE
    pause_logit_bias: float = PAUSE_LOGIT_BIAS
    beta: float = TDPO_BETA
    seed: int = 20260425
    verbose: bool = True
    direct_action_generation: bool = LLM_DIRECT_ACTION_GENERATION
    train_lm_head: bool = TRAIN_LM_HEAD
    train_input_embeddings: bool = TRAIN_INPUT_EMBEDDINGS
    merge_lora_into_base: bool = MERGE_LORA_INTO_BASE


@dataclass
class TDPOActionInfo:
    action_index: int
    action_type: str
    tool_type_id: Optional[int]
    log_prob: float
    probability: float
    valid_action_indices: List[int]
    prompt: str
    memory_vector: List[float]
    memory_action_bias: List[float]
    knowledge_prompt: str
    action_mask: List[int]
    prompt_allowed_tool_mask: List[int] = field(default_factory=list)
    generated_action_text: str = ""


@dataclass
class DecisionTrace:
    trace_id: str
    epoch: int
    step_count: int
    env_time: float
    task_id: str
    node_id: int
    node_type_id: int
    prompt: str
    memory_vector: List[float]
    memory_action_bias: List[float]
    knowledge_prompt: str
    action_mask: List[int]
    prompt_allowed_tool_mask: List[int]
    exec_action: int
    cf_action: int
    valid_actions: List[int]
    node_remaining_deadline_s: float
    task_remaining_deadline_s: float
    sandbox_utilities: Dict[int, float] = field(default_factory=dict)
    wireless_regime: Tuple[int, int] = (1, 1)
    realized_utility: Optional[float] = None
    mission_success: Optional[int] = None
    node_success: Optional[int] = None
    violation: Optional[int] = None
    actual_latency_s: Optional[float] = None


@dataclass
class PreferencePair:
    pair_id: str
    tier: str
    prompt: str
    memory_vector: List[float]
    action_mask: List[int]
    positive_action: int
    negative_action: int
    confidence: float
    ref_logp_positive: Optional[float] = None
    ref_logp_negative: Optional[float] = None
    meta: Dict[str, object] = field(default_factory=dict)
    memory_action_bias: List[float] = field(default_factory=list)


# ============================================================
# Action-space helpers
# ============================================================


def action_dim_from_num_tools(num_tools: int) -> int:
    return int(num_tools) + 2


def local_action_index(num_tools: int) -> int:
    return int(num_tools)


def pause_action_index(num_tools: int) -> int:
    return int(num_tools) + 1


def action_index_to_label(action_index: int, num_tools: int) -> str:
    if int(action_index) < int(num_tools):
        return f"tool_{int(action_index)}"
    if int(action_index) == local_action_index(num_tools):
        return "local"
    if int(action_index) == pause_action_index(num_tools):
        return "pause"
    return f"invalid_{int(action_index)}"


def _clip_action_bias(values: Sequence[float], action_dim: int) -> List[float]:
    vals = pad_or_trim(values, int(action_dim))
    clip = float(KNOWLEDGE_ACTION_BIAS_CLIP)
    vals = [max(-clip, min(clip, float(v))) for v in vals]
    mean_v = sum(vals) / max(1, len(vals))
    return [float(v - mean_v) for v in vals]


def _short_text(text: object, limit: int) -> str:
    t = " ".join(str(text or "").replace("\n", " ").split())
    limit = max(16, int(limit))
    return t if len(t) <= limit else t[: limit - 3] + "..."


def _top_actions_from_bias(action_bias: Sequence[float], num_tools: int, top_k: int = KNOWLEDGE_RECOMMEND_TOP_K) -> List[int]:
    vals = [float(v) for v in list(action_bias or [])]
    if not vals:
        return []
    if max(vals) - min(vals) < 1.0e-8:
        return []
    order = sorted(range(len(vals)), key=lambda i: float(vals[i]), reverse=True)
    return [int(i) for i in order[: max(1, int(top_k))]]


def _compress_knowledge_result(out: Dict[str, object], num_tools: int, action_dim: int) -> Dict[str, object]:
    """Return matched memory as plain prompt text only.

    The retrieval score/vector/action-bias fields are kept only for debugging in
    retrieval_* keys.  They are deliberately not converted into action priors and
    are not injected into the policy logits.
    """
    selected = list(out.get("selected", []) or [])

    raw_prompt = str(out.get("knowledge_prompt", "") or "")
    out = dict(out)
    out["action_bias"] = [0.0] * int(action_dim)
    out["skill_top_actions"] = []
    out["skill_top_action"] = -1
    out["knowledge_prompt_raw"] = raw_prompt

    if not bool(KNOWLEDGE_PROMPT_COMPRESS):
        out["knowledge_prompt"] = raw_prompt
        return out

    raw_lines = [ln.strip() for ln in raw_prompt.splitlines() if ln.strip()]
    prompt_lines: List[str] = []

    for i, ln in enumerate(raw_lines[: max(1, int(KNOWLEDGE_PROMPT_TOP_K))], start=1):
        prompt_lines.append(f"Matched memory {i}: " + _short_text(ln, int(KNOWLEDGE_PROMPT_MAX_CHARS_PER_ITEM)))

    # If a memory entry has no prompt_text, keep only minimal provenance so the
    # prompt still records that a memory item was matched.  Do not expose action
    # scores or action recommendations here.
    if not prompt_lines:
        def _sel_key(x: object) -> Tuple[float, float]:
            if not isinstance(x, dict):
                return (0.0, 0.0)
            return (safe_float(x.get("score"), 0.0), safe_float(x.get("support"), 0.0))
        selected_sorted = sorted(selected, key=_sel_key, reverse=True)[: max(1, int(KNOWLEDGE_PROMPT_TOP_K))]
        for i, sel in enumerate(selected_sorted, start=1):
            if isinstance(sel, dict):
                prompt_lines.append(
                    f"Matched memory {i}: source={sel.get('source', 'run_memory')}, "
                    f"support={int(safe_float(sel.get('support'), 0.0))}, score={safe_float(sel.get('score'), 0.0):.4f}."
                )

    compressed = "\n".join(prompt_lines)
    compressed = _short_text(compressed, int(KNOWLEDGE_PROMPT_TOTAL_MAX_CHARS))
    out["knowledge_prompt"] = compressed
    return out


def action_index_to_env_action(action_index: int, num_tools: int, pause_duration_s: float = PAUSE_SANDBOX_DURATION_S):
    idx = int(action_index)
    if idx < int(num_tools):
        return task_env_mod.DecisionAction(action_type="tool", tool_type_id=idx)
    if idx == local_action_index(num_tools):
        return task_env_mod.DecisionAction(action_type="local")
    return task_env_mod.DecisionAction(action_type="pause", pause_duration_s=float(pause_duration_s))


def env_action_to_index(action, num_tools: int) -> int:
    if action.action_type == "tool":
        return int(action.tool_type_id)
    if action.action_type == "local":
        return local_action_index(num_tools)
    return pause_action_index(num_tools)


def valid_action_mask_from_dp(dp: Dict[str, object], num_tools: int, allow_pause: bool = True) -> List[int]:
    action_dim = action_dim_from_num_tools(num_tools)
    if bool(DISABLE_POLICY_ACTION_MASK):
        return [1] * action_dim
    mask = [0] * action_dim
    has_executable_action = False
    for opt in list(dp.get("tool_options", []) or []):
        tid = int(opt.get("tool_type_id", -1))
        if 0 <= tid < num_tools and not bool(opt.get("queue_blocked", False)):
            mask[tid] = 1
            has_executable_action = True
    local_opt = dp.get("local_option", {}) or {}
    if not bool(local_opt.get("queue_blocked", False)):
        mask[local_action_index(num_tools)] = 1
        has_executable_action = True
    # Pause is only legal when no tool/local action is executable.
    # This prevents the policy from wasting deadline while real execution is available.
    if allow_pause and not has_executable_action:
        mask[pause_action_index(num_tools)] = 1
    if sum(mask) <= 0:
        mask[pause_action_index(num_tools)] = 1
    return mask


def action_mask_to_indices(mask: Sequence[int]) -> List[int]:
    return [i for i, v in enumerate(mask) if int(v) == 1]


def is_action_legal_by_indicator(action_index: int, allowed_tool_mask: Sequence[int], num_tools: Optional[int] = None) -> bool:
    """Check legality under prompt-visible allowed_tool_indicator_text_only."""
    if num_tools is None:
        num_tools = len(list(allowed_tool_mask or []))
    idx = int(action_index)
    n = int(num_tools)
    if idx < 0:
        return False
    if idx < n:
        vals = list(allowed_tool_mask or [])
        return idx < len(vals) and int(vals[idx]) == 1
    if idx == local_action_index(n) or idx == pause_action_index(n):
        return True
    return False


def first_legal_action_from_indicator(allowed_tool_mask: Sequence[int], num_tools: Optional[int] = None, prefer_local: bool = False) -> int:
    if num_tools is None:
        num_tools = len(list(allowed_tool_mask or []))
    n = int(num_tools)
    if bool(prefer_local):
        return local_action_index(n)
    for i, v in enumerate(list(allowed_tool_mask or [])[:n]):
        if int(v) == 1:
            return int(i)
    return local_action_index(n)


# ============================================================
# Prompt and memory helpers
# ============================================================


def _fmt(x: object, digits: int = PROMPT_DECIMAL_DIGITS) -> str:
    try:
        return f"{float(x):.{digits}f}"
    except Exception:
        return str(x)


def build_state_query_embedding(dp: Dict[str, object], num_tools: int, target_dim: int = MEMORY_VECTOR_DIM_FALLBACK) -> List[float]:
    values: List[float] = []
    values.append(safe_float(dp.get("node_type_id"), 0.0) / 16.0)
    values.append(safe_float(dp.get("task_remaining_deadline_s"), 0.0) / 100.0)
    values.append(safe_float(dp.get("node_remaining_deadline_s"), 0.0) / 50.0)

    tool_by_id = {int(opt.get("tool_type_id", -1)): opt for opt in list(dp.get("tool_options", []) or [])}
    for tid in range(num_tools):
        opt = tool_by_id.get(tid, {})
        values.extend([
            0.0 if not opt else 1.0,
            1.0 if bool(opt.get("queue_blocked", False)) else 0.0,
            safe_float(opt.get("predicted_total_s"), 0.0) / 20.0,
            safe_float(opt.get("risk_estimate"), 0.0),
            safe_float(opt.get("packet_loss_rate"), 0.0),
            safe_float(opt.get("base_queue_delay_s"), 0.0) / 10.0,
        ])

    local = dp.get("local_option", {}) or {}
    values.extend([
        0.0 if bool(local.get("queue_blocked", False)) else 1.0,
        safe_float(local.get("predicted_total_s", local.get("compute_s", 0.0)), 0.0) / 20.0,
        safe_float(local.get("fail_probability"), 0.0),
        safe_float(local.get("estimated_wait_s"), 0.0) / 10.0,
    ])
    return pad_or_trim(values, target_dim)


def infer_queue_class(dp: Dict[str, object]) -> int:
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


def infer_slack_class(dp: Dict[str, object]) -> int:
    remain = safe_float(dp.get("node_remaining_deadline_s"), 0.0)
    fastest = fastest_predicted_latency(dp, default=1.0)
    ratio = remain / max(fastest, 1.0e-6)
    if ratio < 1.2:
        return 0
    if ratio < 2.5:
        return 1
    return 2


def infer_wireless_regime(dp: Dict[str, object]) -> Tuple[int, int]:
    losses = [safe_float(opt.get("packet_loss_rate"), 0.0) for opt in list(dp.get("tool_options", []) or [])]
    queues = [safe_float(opt.get("base_queue_delay_s"), 0.0) for opt in list(dp.get("tool_options", []) or [])]
    loss = mean_or_zero(losses)
    queue = max(queues or [0.0])
    # Here higher class means harsher regime.  It is a compact substitute for (SNR, queue) bins.
    if loss < 0.05:
        snr_regime = 0
    elif loss < 0.20:
        snr_regime = 1
    else:
        snr_regime = 2
    if queue < 0.25:
        q_regime = 0
    elif queue < 1.0:
        q_regime = 1
    else:
        q_regime = 2
    return int(snr_regime), int(q_regime)


def retrieve_knowledge_for_dp(
    knowledge_trainer,
    dp: Dict[str, object],
    num_tools: int,
    use_memory: bool,
    target_dim: int = MEMORY_VECTOR_DIM_FALLBACK,
) -> Dict[str, object]:
    action_dim = action_dim_from_num_tools(num_tools)
    if (not use_memory) or knowledge_trainer is None:
        return {
            "memory_vector": [0.0] * (action_dim + 4),
            "action_bias": [0.0] * action_dim,
            "selected": [],
            "knowledge_prompt": "",
        }
    query_embedding = build_state_query_embedding(dp, num_tools=num_tools, target_dim=target_dim)
    out = knowledge_trainer.query_memory(
        state_query_embedding=query_embedding,
        node_type_id=int(dp.get("node_type_id", 0)),
        queue_class=infer_queue_class(dp),
        slack_class=infer_slack_class(dp),
    )
    mv = list(out.get("memory_vector", []))
    ab = list(out.get("action_bias", []))
    out["retrieval_memory_vector"] = pad_or_trim(mv, action_dim + 4)
    out["retrieval_action_bias"] = _clip_action_bias(ab, action_dim)

    # Prompt-only injection: use the numeric state embedding only to retrieve
    # memory.  After retrieval, force numeric memory features and action priors
    # to zero so the policy can only consume matched memory through prompt text.
    out["memory_vector"] = [0.0] * (action_dim + 4)
    out["action_bias"] = [0.0] * action_dim
    out = _compress_knowledge_result(out, num_tools=num_tools, action_dim=action_dim)
    return out


def _prompt_text_value(value: object, max_chars: int = PROMPT_INTENT_MAX_CHARS) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list, tuple)):
        try:
            text = json.dumps(value, ensure_ascii=False, sort_keys=True)
        except Exception:
            text = str(value)
    else:
        text = str(value)
    text = " ".join(text.replace("\n", " ").split())
    if len(text) > int(max_chars):
        text = text[: max(0, int(max_chars) - 3)] + "..."
    return text


def _graph_node_meta(graph_meta: Optional[Dict[str, object]], node_id: int) -> Dict[str, object]:
    if not isinstance(graph_meta, dict):
        return {}
    for node in list(graph_meta.get("nodes", []) or []):
        if isinstance(node, dict) and int(node.get("node_id", -1)) == int(node_id):
            return node
    return {}


def _graph_degree_pair(graph_meta: Optional[Dict[str, object]], node_id: int) -> Tuple[int, int]:
    if not isinstance(graph_meta, dict):
        return 0, 0
    indeg = 0
    outdeg = 0
    for edge in list(graph_meta.get("edges", []) or []):
        if not isinstance(edge, dict):
            continue
        try:
            src = int(edge.get("src", -1))
            dst = int(edge.get("dst", -1))
        except Exception:
            continue
        if dst == int(node_id):
            indeg += 1
        if src == int(node_id):
            outdeg += 1
    return indeg, outdeg


def _allowed_tool_mask_values(dp: Dict[str, object], graph_meta: Optional[Dict[str, object]], node_id: int, num_tools: int) -> List[int]:
    mask = dp.get("allowed_tools_mask")
    if mask is None:
        node_meta = _graph_node_meta(graph_meta, node_id)
        mask = node_meta.get("allowed_tools_mask") if isinstance(node_meta, dict) else None
    vals = [0] * int(num_tools)
    if isinstance(mask, (list, tuple)):
        raw = [int(x) for x in list(mask)]
        for i in range(min(len(raw), int(num_tools))):
            vals[i] = 1 if int(raw[i]) == 1 else 0
    else:
        for opt in list(dp.get("tool_options", []) or []):
            tid = int(opt.get("tool_type_id", -1))
            if 0 <= tid < int(num_tools):
                vals[tid] = 1
    return vals


def _allowed_tool_mask_text(dp: Dict[str, object], graph_meta: Optional[Dict[str, object]], node_id: int, num_tools: int) -> str:
    vals = _allowed_tool_mask_values(dp, graph_meta, node_id, num_tools)
    return "[" + ",".join(str(v) for v in vals) + "]"


def build_prompt_from_dp(
    dp: Dict[str, object],
    num_tools: int,
    knowledge_prompt: str = "",
    graph_meta: Optional[Dict[str, object]] = None,
) -> str:
    """Build a rich prompt-only decision input without policy-layer masking.

    The prompt gives task/node/DAG/load summaries and action semantics, but hides
    per-tool queue state, predicted latency, risk, packet loss, tool profiles,
    retrieved memory, and all execution logs.
    """
    node_id = int(dp.get("node_id", 0))
    graph_meta = graph_meta or {}
    node_meta = _graph_node_meta(graph_meta, node_id)
    indeg = int(dp.get("num_predecessors", _graph_degree_pair(graph_meta, node_id)[0]) or 0)
    outdeg = int(dp.get("num_successors", _graph_degree_pair(graph_meta, node_id)[1]) or 0)

    intent_text = _prompt_text_value(
        graph_meta.get("intent", graph_meta.get("segmented_intent", graph_meta.get("intent_meta", "")))
    )
    node_text = _prompt_text_value(node_meta.get("text", node_meta.get("intent", "")), max_chars=180)

    allowed_mask_vals = _allowed_tool_mask_values(dp, graph_meta, node_id, num_tools)
    allowed_mask = "[" + ",".join(str(v) for v in allowed_mask_vals) + "]"
    allowed_tool_indices = [i for i, v in enumerate(allowed_mask_vals) if int(v) == 1]
    unsuitable_tool_indices = [i for i, v in enumerate(allowed_mask_vals) if int(v) != 1]
    local_idx = local_action_index(num_tools)
    pause_idx = pause_action_index(num_tools)

    if not bool(PROMPT_RICH_TASK_STATE):
        lines: List[str] = []
        lines.append("You are a stateless decision policy.")
        lines.append("Use only the current node summary below. Do not use prior logs, prior actions, or dialogue history.")
        lines.append("Current node summary:")
        lines.append(f"- node_id: {node_id}")
        lines.append(f"- task_remaining_deadline_s: {_fmt(dp.get('task_remaining_deadline_s'))}")
        lines.append(f"- node_remaining_deadline_s: {_fmt(dp.get('node_remaining_deadline_s'))}")
        lines.append("Choose one action for this single step.")
        return "\n".join(lines)

    lines = []
    lines.append("You are a prompt-only TDPO decision policy for DAG task execution.")
    lines.append("Goal: maximize task completion first, then reduce latency and energy.")
    lines.append("No policy action mask is applied. Generate the action only from this prompt.")
    lines.append("Invalid or rejected actions are handled by the environment and become negative outcomes during TDPO training.")
    lines.append("Do not assume hidden tool tables, per-tool queue states, predicted tool latency, risk, packet loss, memory, or prior logs.")
    lines.append("Action space:")
    lines.append(f"- tool actions: indices 0 to {max(0, int(num_tools) - 1)} select the corresponding tool queue.")
    lines.append(f"- local action: index {local_idx} executes the node locally.")
    lines.append(f"- pause action: index {pause_idx} waits briefly and decides later.")
    if bool(PROMPT_INCLUDE_ALLOWED_TOOL_MASK_TEXT):
        lines.append("Allowed tool rule:")
        lines.append(f"- allowed_tool_indicator_text_only = {allowed_mask}")
        lines.append("- A tool action i is suitable only if allowed_tool_indicator_text_only[i] = 1.")
        lines.append("- A tool action i is unsuitable if allowed_tool_indicator_text_only[i] = 0.")
        lines.append("- Unsuitable tool actions may be rejected by the environment and become negative TDPO outcomes.")
        lines.append("- This indicator is text only; no policy action mask is applied.")
        lines.append(f"- suitable_tool_actions_from_indicator: {allowed_tool_indices}")
        if bool(PROMPT_ALLOWED_TOOL_RULE_EXAMPLE):
            lines.append(f"- unsuitable_tool_actions_from_indicator: {unsuitable_tool_indices}")
            lines.append(f"- Example rule: with {allowed_mask}, choose only tool indices in {allowed_tool_indices}; local={local_idx} and pause={pause_idx} remain explicit non-tool actions.")

    lines.append("Task state:")
    if intent_text:
        lines.append(f"- mission_intent: {intent_text}")
    lines.append(f"- task_deadline_s: {_fmt(dp.get('task_deadline_s', graph_meta.get('task_deadline')))}")
    lines.append(f"- task_remaining_deadline_s: {_fmt(dp.get('task_remaining_deadline_s'))}")
    lines.append(f"- task_elapsed_time_s: {_fmt(dp.get('task_elapsed_time_s'))}")
    lines.append(f"- ready_frontier_nodes: {int(dp.get('ready_frontier_count', 0) or 0)}")
    lines.append(f"- completed_nodes: {int(dp.get('completed_node_count', 0) or 0)} / {int(dp.get('num_nodes', graph_meta.get('num_nodes', 0)) or 0)}")

    lines.append("Current node:")
    if node_text:
        lines.append(f"- node_text: {node_text}")
    lines.append(f"- node_id: {node_id}")
    lines.append(f"- node_type_id: {int(dp.get('node_type_id', 0) or 0)}")
    lines.append(f"- computation_cycles: {_fmt(dp.get('computation_cycles', node_meta.get('computation_cycles')))}")
    lines.append(f"- uplink_size_bits: {int(safe_float(dp.get('uplink_size_bits', node_meta.get('uplink_size_bits', 0)), 0.0))}")
    lines.append(f"- downlink_size_bits: {int(safe_float(dp.get('downlink_size_bits', node_meta.get('downlink_size_bits', 0)), 0.0))}")
    lines.append(f"- node_deadline_s: {_fmt(dp.get('node_deadline_s', node_meta.get('node_deadline')))}")
    lines.append(f"- node_remaining_deadline_s: {_fmt(dp.get('node_remaining_deadline_s'))}")

    lines.append("DAG topology:")
    lines.append(f"- current_node_predecessors: {indeg}")
    lines.append(f"- current_node_successors: {outdeg}")

    lines.append("System aggregate load:")
    lines.append(f"- running_tool_jobs_total: {int(dp.get('running_tool_jobs_total', 0) or 0)}")
    lines.append(f"- queued_tool_jobs_total: {int(dp.get('queued_tool_jobs_total', 0) or 0)}")
    lines.append(f"- running_local_jobs: {int(dp.get('running_local_jobs', 0) or 0)}")
    lines.append(f"- queued_local_jobs: {int(dp.get('queued_local_jobs', 0) or 0)}")
    lines.append("Output format:")
    lines.append(f"- Output exactly one line: {LLM_ACTION_OUTPUT_PREFIX}<integer>")
    lines.append("- Do not output explanations, JSON, code blocks, or multiple numbers.")
    lines.append("Choose exactly one action index for this node.")
    return "\n".join(lines)


# ============================================================
# Sandbox utility helpers
# ============================================================


def fastest_predicted_latency(dp: Dict[str, object], default: float = 1.0) -> float:
    vals: List[float] = []
    for opt in list(dp.get("tool_options", []) or []):
        if not bool(opt.get("queue_blocked", False)):
            vals.append(safe_float(opt.get("predicted_total_s"), default))
    local = dp.get("local_option", {}) or {}
    if not bool(local.get("queue_blocked", False)):
        vals.append(safe_float(local.get("predicted_total_s", local.get("compute_s", default)), default))
    return min(vals) if vals else float(default)


def completion_first_utility(
    node_success: float,
    mission_success: float,
    violation: float,
    latency_s: float,
    deadline_s: float,
) -> float:
    deadline = max(1.0e-6, float(deadline_s))
    latency_ratio = min(float(COMPLETION_FIRST_LATENCY_CAP), max(0.0, float(latency_s) / deadline))
    node_s = max(0.0, min(1.0, float(node_success)))
    mission_s = max(0.0, min(1.0, float(mission_success)))
    viol = max(0.0, float(violation))

    # A predicted action that misses the remaining node deadline should not look
    # completion-safe in the sandbox, even when its raw reliability is high.
    if float(latency_s) > deadline:
        node_s = 0.0
        mission_s = 0.0
        viol = max(viol, 1.0)

    if not bool(COMPLETION_FIRST_OBJECTIVE):
        return (
            UTILITY_W_NODE * node_s
            + UTILITY_W_MISSION * mission_s
            - UTILITY_W_VIOL * viol
            - UTILITY_W_LATENCY * latency_ratio
        )

    return (
        float(COMPLETION_FIRST_MISSION_WEIGHT) * mission_s
        + float(COMPLETION_FIRST_NODE_WEIGHT) * node_s
        - float(COMPLETION_FIRST_VIOLATION_WEIGHT) * viol
        - float(COMPLETION_FIRST_LATENCY_WEIGHT) * latency_ratio
    )


def sandbox_utility_for_action(dp: Dict[str, object], action_index: int, num_tools: int) -> float:
    deadline = max(1.0e-6, safe_float(dp.get("node_remaining_deadline_s"), 1.0))
    idx = int(action_index)
    node_success = 0.0
    mission_proxy = 0.0
    violation = 0.0
    latency = deadline

    if idx < int(num_tools):
        opt = None
        for item in list(dp.get("tool_options", []) or []):
            if int(item.get("tool_type_id", -1)) == idx:
                opt = item
                break
        if opt is None or bool(opt.get("queue_blocked", False)):
            return completion_first_utility(0.0, 0.0, 1.5, deadline, deadline)
        latency = safe_float(opt.get("predicted_total_s"), deadline)
        risk = max(0.0, min(1.0, safe_float(opt.get("risk_estimate"), 0.0)))
        loss = max(0.0, min(1.0, safe_float(opt.get("packet_loss_rate"), 0.0)))
        # Align sandbox semantics with task_env.py: a tool job that finishes before
        # the node deadline is treated as successful.  Risk and packet loss are kept
        # only as mild preference penalties instead of directly lowering success.
        if latency > deadline:
            node_success = 0.0
            mission_proxy = 0.0
            violation = 1.0
        else:
            node_success = 1.0
            mission_proxy = 1.0
            violation = min(0.5, 0.25 * risk + 0.25 * loss)
    elif idx == local_action_index(num_tools):
        local = dp.get("local_option", {}) or {}
        if bool(local.get("queue_blocked", False)):
            return completion_first_utility(0.0, 0.0, 1.5, deadline, deadline)
        latency = safe_float(local.get("predicted_total_s", local.get("compute_s", deadline)), deadline)
        fail_prob = max(0.0, min(1.0, safe_float(local.get("fail_probability"), 0.0)))
        success_prob = max(0.0, min(1.0, 1.0 - fail_prob))
        node_success = success_prob
        mission_proxy = success_prob
        violation = 1.0 if latency > deadline else fail_prob
    else:
        latency = max(PAUSE_SANDBOX_DURATION_S, safe_float(dp.get("env_dt", PAUSE_SANDBOX_DURATION_S), PAUSE_SANDBOX_DURATION_S))
        slack_after_pause = max(0.0, deadline - latency)
        node_success = 0.0
        mission_proxy = 0.0
        violation = PAUSE_SANDBOX_VIOLATION_BONUS + (1.0 if slack_after_pause <= 0.0 else 0.0)

    return completion_first_utility(
        node_success=node_success,
        mission_success=mission_proxy,
        violation=violation,
        latency_s=latency,
        deadline_s=deadline,
    )


def sandbox_utilities(dp: Dict[str, object], num_tools: int, mask: Optional[Sequence[int]] = None) -> Dict[int, float]:
    action_dim = action_dim_from_num_tools(num_tools)
    if mask is None:
        mask = valid_action_mask_from_dp(dp, num_tools)
    out: Dict[int, float] = {}
    for idx in range(action_dim):
        if int(mask[idx]) == 1:
            out[idx] = sandbox_utility_for_action(dp, idx, num_tools)
    return out


def choose_counterfactual_action(dp: Dict[str, object], exec_action: int, num_tools: int, rng: Optional[random.Random] = None) -> int:
    mask = valid_action_mask_from_dp(dp, num_tools)
    candidates = [idx for idx in action_mask_to_indices(mask) if idx != int(exec_action)]
    if not candidates:
        return int(exec_action)
    if str(SANDBOX_COUNTERFACTUAL_SELECTION).lower() == "best":
        utils = sandbox_utilities(dp, num_tools, mask)
        best_u = max(utils.get(idx, -1.0e9) for idx in candidates)
        best = [idx for idx in candidates if abs(utils.get(idx, -1.0e9) - best_u) <= 1.0e-9]
        return int((rng or random).choice(best))
    return int((rng or random).choice(candidates))


def realized_utility(
    node_success: int,
    mission_success: int,
    violation: int,
    actual_latency_s: float,
    node_remaining_deadline_s: float,
) -> float:
    deadline = max(1.0e-6, float(node_remaining_deadline_s))
    return completion_first_utility(
        node_success=float(node_success),
        mission_success=float(mission_success),
        violation=float(violation),
        latency_s=float(actual_latency_s),
        deadline_s=deadline,
    )


def build_tier_a_pair(trace: DecisionTrace, confidence_min: float = TDPO_CONFIDENCE_MIN) -> Optional[PreferencePair]:
    if trace.exec_action == trace.cf_action:
        return None
    u_exec = trace.realized_utility
    if u_exec is None:
        u_exec = trace.sandbox_utilities.get(int(trace.exec_action))
    u_cf = trace.sandbox_utilities.get(int(trace.cf_action))
    if u_exec is None or u_cf is None:
        return None
    if float(u_exec) >= float(u_cf):
        pos, neg = int(trace.exec_action), int(trace.cf_action)
        gap = float(u_exec) - float(u_cf)
    else:
        pos, neg = int(trace.cf_action), int(trace.exec_action)
        gap = float(u_cf) - float(u_exec)
    allowed_mask = list(getattr(trace, "prompt_allowed_tool_mask", []) or [])
    num_tools = len(allowed_mask)
    if num_tools > 0:
        pos_legal = is_action_legal_by_indicator(pos, allowed_mask, num_tools)
        neg_legal = is_action_legal_by_indicator(neg, allowed_mask, num_tools)
        if not bool(pos_legal):
            return None
    else:
        pos_legal = True
        neg_legal = True
    conf = sigmoid_float(gap)
    if conf < confidence_min:
        return None
    return PreferencePair(
        pair_id=f"A_{trace.trace_id}", tier="A", prompt=trace.prompt,
        memory_vector=list(trace.memory_vector), memory_action_bias=list(trace.memory_action_bias),
        action_mask=list(trace.action_mask), positive_action=pos, negative_action=neg, confidence=conf,
        meta={
            "trace_id": trace.trace_id, "task_id": trace.task_id, "node_id": trace.node_id,
            "node_type_id": trace.node_type_id, "u_exec": float(u_exec), "u_cf": float(u_cf),
            "gap": float(gap), "exec_action": int(trace.exec_action), "cf_action": int(trace.cf_action),
            "positive_from_exec": bool(int(pos) == int(trace.exec_action)),
            "positive_realized": bool(int(pos) == int(trace.exec_action)),
            "positive_task_completed": int(trace.mission_success or 0) if int(pos) == int(trace.exec_action) else 0,
            "positive_node_success": int(trace.node_success or 0) if int(pos) == int(trace.exec_action) else 0,
            "negative_from_exec": bool(int(neg) == int(trace.exec_action)),
            "negative_task_completed": int(trace.mission_success or 0) if int(neg) == int(trace.exec_action) else 0,
            "negative_node_success": int(trace.node_success or 0) if int(neg) == int(trace.exec_action) else 0,
            "prompt_allowed_tool_mask": list(allowed_mask),
            "positive_action_legal": int(bool(pos_legal)), "negative_action_legal": int(bool(neg_legal)),
        },
    )


def build_legality_pair(trace: DecisionTrace, confidence: float = 0.98) -> Optional[PreferencePair]:
    allowed_mask = list(getattr(trace, "prompt_allowed_tool_mask", []) or [])
    num_tools = len(allowed_mask)
    if num_tools <= 0:
        return None
    neg = int(trace.exec_action)
    if is_action_legal_by_indicator(neg, allowed_mask, num_tools):
        return None
    pos = first_legal_action_from_indicator(allowed_mask, num_tools, prefer_local=False)
    if int(pos) == int(neg) or not is_action_legal_by_indicator(pos, allowed_mask, num_tools):
        pos = local_action_index(num_tools)
    if int(pos) == int(neg):
        return None
    return PreferencePair(
        pair_id=f"L_{trace.trace_id}", tier="A", prompt=trace.prompt,
        memory_vector=list(trace.memory_vector), memory_action_bias=list(trace.memory_action_bias),
        action_mask=list(trace.action_mask), positive_action=int(pos), negative_action=int(neg),
        confidence=float(confidence),
        meta={
            "trace_id": trace.trace_id, "task_id": trace.task_id, "node_id": trace.node_id,
            "node_type_id": trace.node_type_id, "pair_source": "legality", "legality_pair": 1,
            "gap": 20.0, "utility_gap": 20.0, "positive_realized": False,
            "positive_task_completed": 0, "positive_node_success": 0, "negative_from_exec": True,
            "negative_task_completed": int(trace.mission_success or 0),
            "negative_node_success": int(trace.node_success or 0), "exec_action": int(trace.exec_action),
            "prompt_allowed_tool_mask": list(allowed_mask), "positive_action_legal": 1, "negative_action_legal": 0,
        },
    )


def wireless_similarity_confidence(regime_a: Tuple[int, int], regime_b: Tuple[int, int], tau: float = WIRELESS_TAU) -> float:
    diff_sq = float((int(regime_a[0]) - int(regime_b[0])) ** 2 + (int(regime_a[1]) - int(regime_b[1])) ** 2)
    return math.exp(-diff_sq / max(1.0e-6, float(tau)))


def build_tier_b_pairs(
    traces: Sequence[DecisionTrace],
    history_by_type: Dict[int, List[DecisionTrace]],
    confidence_min: float = TDPO_CONFIDENCE_MIN,
    max_pairs: int = 512,
) -> List[PreferencePair]:
    pairs: List[PreferencePair] = []
    for tr in traces:
        if tr.realized_utility is None:
            continue
        hist = history_by_type.get(int(tr.node_type_id), [])
        for old in reversed(hist[-TIER_B_MAX_HISTORY_PER_TYPE:]):
            if old.realized_utility is None:
                continue
            if int(old.exec_action) == int(tr.exec_action):
                continue
            if float(tr.realized_utility) >= float(old.realized_utility):
                pos, neg = int(tr.exec_action), int(old.exec_action)
                pos_trace, neg_trace = tr, old
                prompt = tr.prompt
                memory_vector = tr.memory_vector
                memory_action_bias = tr.memory_action_bias
                action_mask = tr.action_mask
                gap = float(tr.realized_utility) - float(old.realized_utility)
            else:
                pos, neg = int(old.exec_action), int(tr.exec_action)
                pos_trace, neg_trace = old, tr
                prompt = tr.prompt
                memory_vector = tr.memory_vector
                memory_action_bias = tr.memory_action_bias
                action_mask = tr.action_mask
                gap = float(old.realized_utility) - float(tr.realized_utility)
            sim = wireless_similarity_confidence(tr.wireless_regime, old.wireless_regime, WIRELESS_TAU)
            conf = sigmoid_float(gap) * sim
            if conf < confidence_min:
                continue
            allowed_mask = list(getattr(tr, "prompt_allowed_tool_mask", []) or [])
            if allowed_mask and not is_action_legal_by_indicator(pos, allowed_mask, len(allowed_mask)):
                continue
            if pos >= len(action_mask) or neg >= len(action_mask):
                continue
            if int(action_mask[pos]) != 1 or int(action_mask[neg]) != 1:
                # Cross-episode action may be invalid in the current prompt; skip to keep DPO mask legal.
                continue
            pairs.append(
                PreferencePair(
                    pair_id=f"B_{tr.trace_id}_{old.trace_id}",
                    tier="B",
                    prompt=prompt,
                    memory_vector=list(memory_vector),
                    memory_action_bias=list(memory_action_bias),
                    action_mask=list(action_mask),
                    positive_action=pos,
                    negative_action=neg,
                    confidence=float(conf),
                    meta={
                        "new_trace_id": tr.trace_id,
                        "old_trace_id": old.trace_id,
                        "node_type_id": int(tr.node_type_id),
                        "gap": float(gap),
                        "wireless_similarity": float(sim),
                        "positive_trace_id": pos_trace.trace_id,
                        "negative_trace_id": neg_trace.trace_id,
                        "positive_realized": True,
                        "positive_task_completed": int(pos_trace.mission_success or 0),
                        "positive_node_success": int(pos_trace.node_success or 0),
                        "negative_task_completed": int(neg_trace.mission_success or 0),
                        "negative_node_success": int(neg_trace.node_success or 0),
                        "prompt_allowed_tool_mask": list(allowed_mask),
                        "positive_action_legal": 1,
                        "negative_action_legal": int(is_action_legal_by_indicator(neg, allowed_mask, len(allowed_mask))) if allowed_mask else 1,
                    },
                )
            )
            if len(pairs) >= int(max_pairs):
                return pairs
    return pairs


# ============================================================
# LLM policy
# ============================================================


class TDPOPolicy(nn.Module):
    def __init__(self, action_dim: int, cfg: Optional[TDPOPolicyConfig] = None) -> None:
        super().__init__()
        self.action_dim = int(action_dim)
        self.cfg = cfg or TDPOPolicyConfig()
        self.cfg.model_path = resolve_llm_path(self.cfg.model_path)
        requested_device = self.cfg.device if self.cfg.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        if str(requested_device) == "cuda":
            requested_device = "cuda:0"
        self.device_name = str(requested_device)
        self.device = torch.device(self.device_name)
        self.torch_dtype = dtype_from_name(self.cfg.dtype)
        self.rng = random.Random(self.cfg.seed)

        tokenizer_path = self.cfg.tokenizer_path or self.cfg.model_path
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        requested_map = str(getattr(self.cfg, "device_map", "none") or "none").lower()
        self.use_device_map = (
            bool(getattr(self.cfg, "enable_multi_gpu", True))
            and torch.cuda.is_available()
            and requested_map not in {"", "none", "false", "single", "0"}
            and torch.cuda.device_count() > 1
        )
        model_kwargs = {
            "torch_dtype": self.torch_dtype,
            "trust_remote_code": True,
            "low_cpu_mem_usage": bool(getattr(self.cfg, "low_cpu_mem_usage", True)),
        }
        if self.use_device_map:
            model_kwargs["device_map"] = getattr(self.cfg, "device_map", "auto")
            max_memory = build_max_memory_map(getattr(self.cfg, "max_memory_per_gpu", "18GiB"))
            if max_memory:
                model_kwargs["max_memory"] = max_memory
            offload_folder = str(getattr(self.cfg, "offload_folder", "") or "")
            if offload_folder:
                Path(offload_folder).mkdir(parents=True, exist_ok=True)
                model_kwargs["offload_folder"] = offload_folder
        self.model = AutoModelForCausalLM.from_pretrained(self.cfg.model_path, **model_kwargs)
        self.model.config.use_cache = bool(self.cfg.use_cache)
        if not self.use_device_map:
            self.model.to(self.device)

        self.input_device = self._infer_input_device()
        self.output_device = self._infer_output_device()
        if bool(self.cfg.gradient_checkpointing) and hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable()
            if hasattr(self.model, "enable_input_require_grads"):
                self.model.enable_input_require_grads()

        hidden_size = int(getattr(self.model.config, "hidden_size", 0) or getattr(self.model.config, "n_embd", 0))
        if hidden_size <= 0:
            raise RuntimeError("Could not infer hidden size from Qwen config.")
        self.hidden_size = hidden_size

        self._freeze_and_select_blocks()
        self.direct_action_generation = bool(getattr(self.cfg, "direct_action_generation", LLM_DIRECT_ACTION_GENERATION))
        if self.direct_action_generation and not bool(getattr(self.cfg, "use_lora", True)):
            raise RuntimeError("Direct LLM action generation mode requires LLM_USE_LORA=True; full Qwen fine-tuning is disabled.")
        if self.direct_action_generation:
            # No external action head and no memory-fusion head in direct generation mode.
            # Qwen base weights, input embeddings, lm_head, final norm, and all non-LoRA
            # parameters remain frozen; only LoRA adapter parameters are trainable.
            self.memory_proj = None
            self.action_head = None
        else:
            self.memory_proj = nn.Sequential(
                nn.Linear(int(self.cfg.memory_vector_dim), int(self.cfg.memory_fusion_hidden)),
                nn.SiLU(),
                nn.Linear(int(self.cfg.memory_fusion_hidden), hidden_size),
            )
            self.action_head = nn.Sequential(
                nn.Dropout(float(self.cfg.action_head_dropout)),
                nn.Linear(hidden_size, self.action_dim),
            )
            self.memory_proj.to(self.output_device)
            self.action_head.to(self.output_device)

        if self.cfg.verbose:
            trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
            llm_trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            print(
                f"[decision-tdpo] initialized | action_dim={self.action_dim} | hidden={hidden_size} | "
                f"device_map={getattr(self.model, 'hf_device_map', None)} | input_device={self.input_device} | "
                f"output_device={self.output_device} | use_lora={getattr(self.cfg, 'use_lora', False)} | "
                f"selected_layers={getattr(self, 'selected_layer_ids', [])} | "
                f"lora_modules={len(getattr(self, 'lora_replaced_modules', []))} | "
                f"direct_generation={self.direct_action_generation} | "
                f"trainable_params={trainable} | llm_trainable_params={llm_trainable}",
                flush=True,
            )

    def _infer_input_device(self) -> torch.device:
        try:
            emb = self.model.get_input_embeddings()
            if emb is not None and getattr(emb, "weight", None) is not None:
                return emb.weight.device
        except Exception:
            pass
        return first_parameter_device(self.model, self.device)

    def _infer_output_device(self) -> torch.device:
        for attr_path in ("model.norm", "transformer.ln_f", "lm_head"):
            obj = self.model
            ok = True
            for name in attr_path.split("."):
                obj = getattr(obj, name, None)
                if obj is None:
                    ok = False
                    break
            if ok:
                return first_parameter_device(obj, last_parameter_device(self.model, self.device))
        return last_parameter_device(self.model, self.device)

    def _ensure_heads_on_device(self, device: torch.device) -> None:
        if self.action_head is None or self.memory_proj is None:
            self.output_device = device
            return
        current = first_parameter_device(self.action_head, self.output_device)
        if current != device:
            self.memory_proj.to(device)
            self.action_head.to(device)
            self.output_device = device

    def _forward_backbone_last_hidden(self, enc: Dict[str, torch.Tensor]) -> torch.Tensor:
        # For Qwen-like AutoModelForCausalLM, call the backbone directly.  This avoids
        # materializing full vocabulary logits and avoids output_hidden_states=True, both
        # of which create unnecessary CUDA memory pressure for a classifier-style head.
        backbone = getattr(self.model, "model", None)
        if backbone is None:
            backbone = getattr(self.model, "transformer", None)
        if backbone is not None:
            out = backbone(**enc, use_cache=False, return_dict=True)
            if hasattr(out, "last_hidden_state") and out.last_hidden_state is not None:
                return out.last_hidden_state
            if isinstance(out, (tuple, list)) and len(out) > 0:
                return out[0]
        out = self.model(
            **enc,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
        return out.hidden_states[-1]

    def _resolve_layers(self) -> nn.ModuleList:
        candidates = [
            ("model.layers", lambda m: getattr(getattr(m, "model", None), "layers", None)),
            ("transformer.h", lambda m: getattr(getattr(m, "transformer", None), "h", None)),
            ("gpt_neox.layers", lambda m: getattr(getattr(m, "gpt_neox", None), "layers", None)),
        ]
        for _, getter in candidates:
            layers = getter(self.model)
            if layers is not None:
                return layers
        raise RuntimeError("Could not locate transformer block list. Expected model.model.layers for Qwen-like models.")

    def _freeze_and_select_blocks(self) -> None:
        """Configure trainable LLM parameters.

        Default path uses LoRA in explicitly selected blocks, so the original Qwen
        weights are not fully fine-tuned.  Set cfg.use_lora=False only when you
        intentionally want the old full-block fine-tuning behavior.
        """
        for p in self.model.parameters():
            p.requires_grad = False
        layers = self._resolve_layers()
        n = len(layers)
        self.selected_layer_ids: List[int] = []
        self.lora_replaced_modules: List[str] = []

        if bool(getattr(self.cfg, "use_lora", True)):
            raw_ids = tuple(getattr(self.cfg, "lora_layer_ids", (-2, -1)) or ())
            if not raw_ids:
                k = max(0, min(int(getattr(self.cfg, "tune_last_n_blocks", 0)), n))
                raw_ids = tuple(range(n - k, n)) if k > 0 else ()
            self.selected_layer_ids = normalize_layer_ids(raw_ids, n)
            for idx in self.selected_layer_ids:
                names = inject_lora_into_block(
                    layers[idx],
                    target_module_names=tuple(getattr(self.cfg, "lora_target_modules", LLM_LORA_TARGET_MODULES)),
                    r=int(getattr(self.cfg, "lora_r", LLM_LORA_R)),
                    alpha=float(getattr(self.cfg, "lora_alpha", LLM_LORA_ALPHA)),
                    dropout=float(getattr(self.cfg, "lora_dropout", LLM_LORA_DROPOUT)),
                )
                self.lora_replaced_modules.extend([f"layers.{idx}.{name}" for name in names])
            if not self.lora_replaced_modules:
                raise RuntimeError(
                    "LLM_USE_LORA=True but no target Linear modules were replaced. "
                    f"Check lora_target_modules={getattr(self.cfg, 'lora_target_modules', None)}."
                )
            # Safety guard for generative TDPO: never train Qwen base weights, input
            # embeddings, lm_head, or final norm in LoRA mode.  The adapted policy is
            # saved as separate LoRA parameters and is not merged into the base model.
            for attr_path in ("lm_head", "model.embed_tokens", "model.norm", "transformer.wte", "transformer.ln_f"):
                obj = self.model
                ok = True
                for name in attr_path.split("."):
                    obj = getattr(obj, name, None)
                    if obj is None:
                        ok = False
                        break
                if ok:
                    for p in obj.parameters():
                        p.requires_grad = False
            return

        # Fallback: old full-block tuning, disabled by default.
        k = max(0, min(int(self.cfg.tune_last_n_blocks), n))
        self.selected_layer_ids = list(range(n - k, n)) if k > 0 else []
        for idx in self.selected_layer_ids:
            for p in layers[idx].parameters():
                p.requires_grad = True
        if bool(self.cfg.train_final_norm):
            for attr_path in ("model.norm", "transformer.ln_f"):
                obj = self.model
                ok = True
                for name in attr_path.split("."):
                    obj = getattr(obj, name, None)
                    if obj is None:
                        ok = False
                        break
                if ok:
                    for p in obj.parameters():
                        p.requires_grad = True

    def _tokenize(self, prompts: Sequence[str]) -> Dict[str, torch.Tensor]:
        enc = self.tokenizer(
            list(prompts),
            padding=True,
            truncation=True,
            max_length=int(self.cfg.max_length),
            return_tensors="pt",
        )
        return {k: v.to(self.input_device) for k, v in enc.items()}

    def _action_completion_text(self, action_index: int) -> str:
        return "\n" + str(LLM_ACTION_OUTPUT_PREFIX) + str(int(action_index))

    def _build_completion_batch(
        self,
        prompts: Sequence[str],
        actions: Sequence[int],
    ) -> Dict[str, torch.Tensor]:
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id if self.tokenizer.eos_token_id is not None else 0
        max_total = max(8, int(self.cfg.max_length))
        input_rows: List[List[int]] = []
        label_rows: List[List[int]] = []
        for prompt, action in zip(list(prompts), list(actions)):
            comp_ids = self.tokenizer.encode(
                self._action_completion_text(int(action)),
                add_special_tokens=False,
            )
            if not comp_ids:
                comp_ids = self.tokenizer.encode(str(int(action)), add_special_tokens=False)
            max_prompt_len = max(1, max_total - max(1, len(comp_ids)))
            prompt_ids = self.tokenizer.encode(
                str(prompt),
                add_special_tokens=True,
                truncation=True,
                max_length=max_prompt_len,
            )
            ids = list(prompt_ids) + list(comp_ids)
            labels = [-100] * len(prompt_ids) + list(comp_ids)
            if len(ids) > max_total:
                overflow = len(ids) - max_total
                ids = ids[overflow:]
                labels = labels[overflow:]
            input_rows.append(ids)
            label_rows.append(labels)
        max_len = max(len(x) for x in input_rows) if input_rows else 1
        input_ids = []
        attention = []
        labels = []
        for ids, labs in zip(input_rows, label_rows):
            pad_n = max_len - len(ids)
            input_ids.append(ids + [pad_id] * pad_n)
            attention.append([1] * len(ids) + [0] * pad_n)
            labels.append(labs + [-100] * pad_n)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long, device=self.input_device),
            "attention_mask": torch.tensor(attention, dtype=torch.long, device=self.input_device),
            "labels": torch.tensor(labels, dtype=torch.long, device=self.input_device),
        }

    def completion_log_probs_for_actions(
        self,
        prompts: Sequence[str],
        actions: Sequence[int],
        batch_size: Optional[int] = None,
    ) -> torch.Tensor:
        """Return autoregressive log p(ACTION_INDEX=i | prompt) for selected actions.

        This is the training/scoring path for direct LLM action generation.  It uses
        the frozen Qwen lm_head to evaluate short action completions while gradients
        flow only through LoRA adapter parameters.
        """
        prompts = list(prompts)
        actions = [int(a) for a in list(actions)]
        if len(prompts) != len(actions):
            raise ValueError("prompts and actions must have the same length")
        if not prompts:
            return torch.empty(0, dtype=torch.float32, device=self.output_device)
        bs = max(1, int(batch_size or LLM_COMPLETION_SCORE_BATCH_SIZE))
        out_scores: List[torch.Tensor] = []
        for start in range(0, len(prompts), bs):
            p_batch = prompts[start:start + bs]
            a_batch = actions[start:start + bs]
            batch = self._build_completion_batch(p_batch, a_batch)
            labels = batch.pop("labels")
            model_out = self.model(**batch, use_cache=False, return_dict=True)
            logits = model_out.logits
            labels = labels.to(logits.device)
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            valid = shift_labels.ne(-100)
            safe_labels = shift_labels.masked_fill(~valid, 0)
            token_logp = F.log_softmax(shift_logits.float(), dim=-1).gather(
                -1,
                safe_labels.unsqueeze(-1),
            ).squeeze(-1)
            token_logp = token_logp.masked_fill(~valid, 0.0)
            denom = valid.float().sum(dim=-1).clamp(min=1.0)
            # Average by completion length so actions with one- vs two-digit ids are comparable.
            out_scores.append(token_logp.sum(dim=-1) / denom)
        return torch.cat(out_scores, dim=0)

    def candidate_action_log_probs(
        self,
        prompts: Sequence[str],
        action_masks: Optional[Sequence[Sequence[int]]] = None,
    ) -> torch.Tensor:
        prompts = list(prompts)
        if not prompts:
            return torch.empty((0, self.action_dim), dtype=torch.float32, device=self.output_device)
        rep_prompts: List[str] = []
        rep_actions: List[int] = []
        for p in prompts:
            for a in range(self.action_dim):
                rep_prompts.append(p)
                rep_actions.append(a)
        scores = self.completion_log_probs_for_actions(rep_prompts, rep_actions)
        scores = scores.view(len(prompts), self.action_dim)
        if action_masks is not None:
            masks = torch.tensor([[int(v) for v in m] for m in action_masks], dtype=torch.bool, device=scores.device)
            scores = scores.masked_fill(~masks, INVALID_ACTION_LOGIT)
        return F.log_softmax(scores.float(), dim=-1)

    @staticmethod
    def _parse_generated_action(text: str, action_dim: int) -> Optional[int]:
        t = str(text or "")
        m = re.search(r"ACTION_INDEX\s*=\s*(-?\d+)", t)
        if m is None:
            m = re.search(r"(-?\d+)", t)
        if m is None:
            return None
        try:
            idx = int(m.group(1))
        except Exception:
            return None
        if 0 <= idx < int(action_dim):
            return idx
        return None

    @torch.no_grad()
    def _generate_action_texts(
        self,
        prompts: Sequence[str],
        sample: bool,
        temperature: float,
    ) -> List[str]:
        enc = self.tokenizer(
            list(prompts),
            padding=True,
            truncation=True,
            max_length=int(self.cfg.max_length),
            return_tensors="pt",
        )
        enc = {k: v.to(self.input_device) for k, v in enc.items()}
        input_len = int(enc["input_ids"].shape[1])
        gen_kwargs = {
            "max_new_tokens": int(LLM_GENERATE_MAX_NEW_TOKENS),
            "do_sample": bool(sample),
            "temperature": max(1.0e-6, float(temperature)) if bool(sample) else None,
            "pad_token_id": self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "use_cache": True,
        }
        gen_kwargs = {k: v for k, v in gen_kwargs.items() if v is not None}
        generated = self.model.generate(**enc, **gen_kwargs)
        new_tokens = generated[:, input_len:]
        return self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)

    def forward_logits(self, prompts: Sequence[str], memory_vectors: Optional[Sequence[Sequence[float]]] = None) -> torch.Tensor:
        if self.direct_action_generation:
            raise RuntimeError("forward_logits/action_head is disabled in direct LLM generation mode.")
        enc = self._tokenize(prompts)
        hidden = self._forward_backbone_last_hidden(enc)
        self._ensure_heads_on_device(hidden.device)
        attn = enc.get("attention_mask")
        if attn is None:
            pooled = hidden[:, -1, :]
        else:
            attn = attn.to(hidden.device)
            last_idx = attn.long().sum(dim=1).clamp(min=1) - 1
            pooled = hidden[torch.arange(hidden.size(0), device=hidden.device), last_idx]

        # The Qwen backbone may run in bfloat16/float16 under device_map, while the
        # lightweight decision heads are intentionally kept in float32 for optimizer
        # stability. Align only the pooled representation and memory tensor to the
        # head dtype/device before Linear layers; this fixes BFloat16-vs-Float matmul
        # errors without casting the full hidden sequence or the sharded LLM.
        head_param = next(self.memory_proj.parameters())
        head_device = head_param.device
        head_dtype = head_param.dtype
        pooled_for_head = pooled.to(device=head_device, dtype=head_dtype)

        if memory_vectors is None:
            mem = torch.zeros(
                (len(prompts), int(self.cfg.memory_vector_dim)),
                dtype=head_dtype,
                device=head_device,
            )
        else:
            mem_list = [pad_or_trim(v, int(self.cfg.memory_vector_dim)) for v in memory_vectors]
            mem = torch.tensor(mem_list, dtype=head_dtype, device=head_device)

        if (not bool(MEMORY_TEXT_ONLY_IN_PROMPT)) and abs(float(self.cfg.memory_scale)) > 0.0:
            fused = pooled_for_head + float(self.cfg.memory_scale) * self.memory_proj(mem)
        else:
            fused = pooled_for_head
        logits = self.action_head(fused)
        if self.action_dim >= 2:
            logits[:, -1] = logits[:, -1] + float(self.cfg.pause_logit_bias)
        return logits.float()

    def masked_log_probs(
        self,
        prompts: Sequence[str],
        memory_vectors: Sequence[Sequence[float]],
        action_masks: Sequence[Sequence[int]],
        memory_action_biases: Optional[Sequence[Sequence[float]]] = None,
    ) -> torch.Tensor:
        if self.direct_action_generation:
            return self.candidate_action_log_probs(prompts, action_masks)
        logits = self.forward_logits(prompts, memory_vectors)
        masks = torch.tensor([[int(v) for v in m] for m in action_masks], dtype=torch.bool, device=logits.device)
        if (not bool(MEMORY_TEXT_ONLY_IN_PROMPT)) and memory_action_biases is not None and float(self.cfg.memory_action_bias_scale) != 0.0:
            bias = torch.tensor(
                [pad_or_trim(b, self.action_dim) for b in memory_action_biases],
                dtype=logits.dtype,
                device=logits.device,
            )
            # Keep knowledge as a weak action prior rather than a hard selector.
            # Centering removes global offsets; clipping prevents a few prototypes from
            # collapsing the policy to one action and killing Tier-B diversity.
            bias = bias - bias.mean(dim=-1, keepdim=True)
            bias = torch.clamp(bias, -float(KNOWLEDGE_ACTION_BIAS_CLIP), float(KNOWLEDGE_ACTION_BIAS_CLIP))
            logits = logits + float(self.cfg.memory_action_bias_scale) * bias
        logits = logits.masked_fill(~masks, INVALID_ACTION_LOGIT)
        return F.log_softmax(logits, dim=-1)

    @torch.no_grad()
    def select_actions(
        self,
        dps: Sequence[Dict[str, object]],
        num_tools: int,
        use_memory: bool = False,
        knowledge_trainer=None,
        sample: bool = True,
        temperature: float = ACTION_TEMPERATURE,
        graph_metas: Optional[Sequence[Optional[Dict[str, object]]]] = None,
    ) -> List[Tuple[object, TDPOActionInfo]]:
        """Select actions for multiple decision points.

        In direct LLM generation mode, Qwen generates a short action completion
        ACTION_INDEX=<integer>; candidate completion scores are used only to obtain
        log probabilities for traces and DPO.  No external action head is used.
        """
        self.eval()
        dps = list(dps)
        if not dps:
            return []
        if graph_metas is None:
            graph_metas = [None] * len(dps)
        else:
            graph_metas = list(graph_metas)
            if len(graph_metas) < len(dps):
                graph_metas = graph_metas + [None] * (len(dps) - len(graph_metas))

        prompts: List[str] = []
        memory_vectors: List[List[float]] = []
        memory_action_biases: List[List[float]] = []
        knowledge_prompts: List[str] = []
        masks: List[List[int]] = []

        for dp, gmeta in zip(dps, graph_metas):
            mem = retrieve_knowledge_for_dp(
                knowledge_trainer=knowledge_trainer,
                dp=dp,
                num_tools=num_tools,
                use_memory=use_memory,
                target_dim=int(self.cfg.memory_vector_dim),
            )
            memory_vector = pad_or_trim(mem.get("memory_vector", []), int(self.cfg.memory_vector_dim))
            memory_action_bias = pad_or_trim(mem.get("action_bias", []), self.action_dim)
            knowledge_prompt = str(mem.get("knowledge_prompt", "") or "")
            prompt = build_prompt_from_dp(
                dp,
                num_tools=num_tools,
                knowledge_prompt=knowledge_prompt,
                graph_meta=gmeta or {},
            )
            mask = valid_action_mask_from_dp(dp, num_tools)
            prompts.append(prompt)
            memory_vectors.append(memory_vector)
            memory_action_biases.append(memory_action_bias)
            knowledge_prompts.append(knowledge_prompt)
            masks.append(mask)

        # Direct generation rollout path: do not pre-score every candidate action.
        # Scoring all ACTION_INDEX completions here is expensive and duplicates the
        # later generation call.  TDPO/DPO training still uses completion scoring
        # in dpo_loss() and fill_reference_logps(); rollout uses generate()+parse.
        log_probs = None
        if (not self.direct_action_generation) or (not bool(LLM_ROLLOUT_GENERATE_ONLY)):
            log_probs = self.masked_log_probs(
                prompts,
                memory_vectors,
                masks,
                memory_action_biases,
            )
        generated_texts = [""] * len(prompts)
        generated_indices: List[Optional[int]] = [None] * len(prompts)
        if self.direct_action_generation:
            generated_texts = self._generate_action_texts(prompts, sample=sample, temperature=temperature)
            generated_indices = [self._parse_generated_action(t, self.action_dim) for t in generated_texts]

        results: List[Tuple[object, TDPOActionInfo]] = []
        for i, dp in enumerate(dps):
            row = None if log_probs is None else log_probs[i]
            idx = generated_indices[i] if self.direct_action_generation else None
            parse_failed = bool(self.direct_action_generation and idx is None)
            generated_text = str(generated_texts[i] if i < len(generated_texts) else "")

            if parse_failed and not bool(LLM_GENERATION_FALLBACK_TO_SCORING):
                # Strict generation mode: do not rescue malformed/out-of-range LLM output.
                # Send a special environment action so task_env.py can record the parse
                # failure and directly fail the current task.  The synthetic action index
                # is outside the normal action space and is used only for diagnostics.
                idx = int(self.action_dim)
                action = task_env_mod.DecisionAction(action_type="invalid_generated")
                log_prob_value = float(math.log(MIN_ACTION_PROB))
                prob_value = float(MIN_ACTION_PROB)
                generated_text = generated_text + ("" if generated_text else "<EMPTY>") + " | parse_failed_no_fallback"
            else:
                if idx is None:
                    # Optional fallback after generation failure only.  This is not a
                    # pre-scoring step: it is called only for malformed/out-of-range
                    # generation, and only for the failed prompt.
                    single_log_probs = self.masked_log_probs(
                        [prompts[i]],
                        [memory_vectors[i]],
                        [masks[i]],
                        [memory_action_biases[i]],
                    )
                    row = single_log_probs[0]
                    if sample:
                        probs = torch.softmax(row / max(1.0e-6, float(temperature)), dim=-1)
                        idx = int(torch.multinomial(probs, num_samples=1).item())
                    else:
                        idx = int(torch.argmax(row).item())
                    generated_text = generated_text + ("" if generated_text else "<EMPTY>") + " | parse_failed_fallback_scoring"
                if row is None:
                    # Generate-only rollout deliberately avoids computing exact action
                    # log-probability for speed.  The real TDPO loss/reference logps
                    # are computed later by completion_log_probs_for_actions().
                    log_prob_value = 0.0
                    prob_value = 1.0
                    generated_text = generated_text + ("" if generated_text else "<EMPTY>") + " | generate_only_no_prescore"
                else:
                    prob_value = float(torch.exp(row[int(idx)]).detach().cpu().item())
                    log_prob_value = float(row[int(idx)].detach().cpu().item())
                action = action_index_to_env_action(int(idx), num_tools)

            info = TDPOActionInfo(
                action_index=int(idx),
                action_type=str(action.action_type),
                tool_type_id=None if action.tool_type_id is None else int(action.tool_type_id),
                log_prob=float(log_prob_value),
                probability=max(MIN_ACTION_PROB, float(prob_value)),
                valid_action_indices=action_mask_to_indices(masks[i]),
                prompt=prompts[i],
                memory_vector=memory_vectors[i],
                memory_action_bias=memory_action_biases[i],
                knowledge_prompt=knowledge_prompts[i],
                action_mask=masks[i],
                prompt_allowed_tool_mask=list(dp.get("allowed_tools_mask", []) or []),
                generated_action_text=generated_text,
            )
            results.append((action, info))
        return results

    @torch.no_grad()
    def select_action(
        self,
        dp: Dict[str, object],
        num_tools: int,
        use_memory: bool = False,
        knowledge_trainer=None,
        sample: bool = True,
        temperature: float = ACTION_TEMPERATURE,
        graph_meta: Optional[Dict[str, object]] = None,
    ) -> Tuple[object, TDPOActionInfo]:
        # Backward-compatible single-decision wrapper.  Existing callers outside
        # train_TDPO.py can keep using select_action() unchanged.
        return self.select_actions(
            [dp],
            num_tools=num_tools,
            use_memory=use_memory,
            knowledge_trainer=knowledge_trainer,
            sample=sample,
            temperature=temperature,
            graph_metas=[graph_meta],
        )[0]

    def dpo_loss(self, batch_pairs: Sequence[PreferencePair]) -> Tuple[torch.Tensor, Dict[str, float]]:
        prompts = [p.prompt for p in batch_pairs]
        if self.direct_action_generation:
            pos_actions = [int(p.positive_action) for p in batch_pairs]
            neg_actions = [int(p.negative_action) for p in batch_pairs]
            logp_pos = self.completion_log_probs_for_actions(prompts, pos_actions)
            logp_neg = self.completion_log_probs_for_actions(prompts, neg_actions)
            conf = torch.tensor([float(p.confidence) for p in batch_pairs], dtype=logp_pos.dtype, device=logp_pos.device)
        else:
            mem = [pad_or_trim(p.memory_vector, int(self.cfg.memory_vector_dim)) for p in batch_pairs]
            masks = [p.action_mask for p in batch_pairs]
            # The action bias is already fused into memory during decision.  For preference replay, keep only memory_vector
            # to avoid requiring an extra field in every pair.
            biases = [pad_or_trim(getattr(p, "memory_action_bias", []), self.action_dim) for p in batch_pairs]
            logp = self.masked_log_probs(prompts, mem, masks, memory_action_biases=biases)
            pos = torch.tensor([int(p.positive_action) for p in batch_pairs], dtype=torch.long, device=logp.device)
            neg = torch.tensor([int(p.negative_action) for p in batch_pairs], dtype=torch.long, device=logp.device)
            conf = torch.tensor([float(p.confidence) for p in batch_pairs], dtype=logp.dtype, device=logp.device)
            logp_pos = logp.gather(1, pos.view(-1, 1)).squeeze(1)
            logp_neg = logp.gather(1, neg.view(-1, 1)).squeeze(1)
        ref_pos = torch.tensor(
            [float(p.ref_logp_positive if p.ref_logp_positive is not None else 0.0) for p in batch_pairs],
            dtype=logp_pos.dtype,
            device=logp_pos.device,
        )
        ref_neg = torch.tensor(
            [float(p.ref_logp_negative if p.ref_logp_negative is not None else 0.0) for p in batch_pairs],
            dtype=logp_pos.dtype,
            device=logp_pos.device,
        )
        beta = float(self.cfg.beta)
        logits = beta * ((logp_pos - logp_neg) - (ref_pos - ref_neg))
        loss_vec = -conf * F.logsigmoid(logits)
        loss = loss_vec.mean()
        with torch.no_grad():
            acc = (logp_pos > logp_neg).float().mean().item()
            margin = (logp_pos - logp_neg).mean().item()
        return loss, {
            "loss": float(loss.detach().cpu().item()),
            "pref_acc": float(acc),
            "logp_margin": float(margin),
            "confidence_mean": float(conf.detach().mean().cpu().item()),
        }

    @torch.no_grad()
    def fill_reference_logps(
        self,
        pairs: Sequence[PreferencePair],
        batch_size: int = 4,
        show_progress: bool = False,
        stage: str = "[tdpo-ref]",
        print_every_batches: int = 10,
    ) -> None:
        self.eval()
        pairs = list(pairs)
        bs = max(1, int(batch_size))
        total_batches = int(math.ceil(len(pairs) / float(bs))) if pairs else 0
        t0 = math.nan
        last_print_t = math.nan
        if show_progress:
            import time as _time
            t0 = _time.time()
            last_print_t = t0
            print(f"{stage} start | pairs={len(pairs)} batch={bs} batches={total_batches}", flush=True)
        for batch_idx, start in enumerate(range(0, len(pairs), bs), start=1):
            if show_progress:
                import time as _time
                batch_t0 = _time.time()
            batch = list(pairs[start:start + bs])
            if not batch:
                continue
            prompts = [p.prompt for p in batch]
            if self.direct_action_generation:
                pos_actions = [int(p.positive_action) for p in batch]
                neg_actions = [int(p.negative_action) for p in batch]
                ref_pos = self.completion_log_probs_for_actions(prompts, pos_actions)
                ref_neg = self.completion_log_probs_for_actions(prompts, neg_actions)
                for i, p in enumerate(batch):
                    p.ref_logp_positive = float(ref_pos[i].detach().cpu().item())
                    p.ref_logp_negative = float(ref_neg[i].detach().cpu().item())
            else:
                mem = [pad_or_trim(p.memory_vector, int(self.cfg.memory_vector_dim)) for p in batch]
                masks = [p.action_mask for p in batch]
                biases = [pad_or_trim(getattr(p, "memory_action_bias", []), self.action_dim) for p in batch]
                logp = self.masked_log_probs(prompts, mem, masks, memory_action_biases=biases)
                for i, p in enumerate(batch):
                    p.ref_logp_positive = float(logp[i, int(p.positive_action)].detach().cpu().item())
                    p.ref_logp_negative = float(logp[i, int(p.negative_action)].detach().cpu().item())
            if show_progress:
                import time as _time
                now = _time.time()
                should_print = (
                    batch_idx == 1
                    or batch_idx == total_batches
                    or batch_idx % max(1, int(print_every_batches)) == 0
                    or now - last_print_t >= 15.0
                )
                if should_print:
                    last_print_t = now
                    elapsed = now - float(t0)
                    avg_batch = elapsed / max(1, batch_idx)
                    eta = avg_batch * max(0, total_batches - batch_idx)
                    width = 28
                    done = int(round(width * batch_idx / max(1, total_batches)))
                    bar = "[" + "#" * done + "." * (width - done) + "]"
                    print(
                        f"{stage} {bar} {batch_idx}/{total_batches} pairs={min(start + bs, len(pairs))}/{len(pairs)} "
                        f"last_batch={now - batch_t0:.2f}s avg_batch={avg_batch:.2f}s eta~{eta:.1f}s",
                        flush=True,
                    )
        if show_progress:
            import time as _time
            print(f"{stage} done | pairs={len(pairs)} time={_time.time() - float(t0):.2f}s", flush=True)

    def trainable_parameters(self) -> List[nn.Parameter]:
        return [p for p in self.parameters() if p.requires_grad]

    def trainable_state_dict(self) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {}
        for name, param in self.named_parameters():
            if param.requires_grad:
                out[name] = param.detach().cpu()
        return out

    def save_checkpoint(self, path: str, extra: Optional[Dict[str, object]] = None) -> str:
        p = Path(path).expanduser().resolve()
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "cfg": asdict(self.cfg),
                "action_dim": self.action_dim,
                "hidden_size": self.hidden_size,
                "selected_layer_ids": list(getattr(self, "selected_layer_ids", [])),
                "lora_replaced_modules": list(getattr(self, "lora_replaced_modules", [])),
                "direct_action_generation": bool(getattr(self, "direct_action_generation", False)),
                "base_weights_frozen": True,
                "lm_head_trained": False,
                "embedding_trained": False,
                "merged_lora_into_base": False,
                "trainable_state_dict": self.trainable_state_dict(),
                "extra": extra or {},
            },
            p,
        )
        return str(p)

    def load_trainable_checkpoint(self, path: str, strict: bool = False) -> Dict[str, object]:
        ckpt = torch.load(path, map_location="cpu")
        state = ckpt.get("trainable_state_dict", {})
        own = dict(self.named_parameters())
        missing = []
        loaded = []
        with torch.no_grad():
            for name, tensor in state.items():
                if name in own and own[name].shape == tensor.shape:
                    own[name].copy_(tensor.to(device=own[name].device, dtype=own[name].dtype))
                    loaded.append(name)
                else:
                    missing.append(name)
        if strict and missing:
            raise RuntimeError(f"Missing or shape-mismatched checkpoint parameters: {missing[:10]}")
        return {"loaded": loaded, "missing": missing, "extra": ckpt.get("extra", {})}
