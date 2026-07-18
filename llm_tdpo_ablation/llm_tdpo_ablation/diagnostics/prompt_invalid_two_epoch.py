# -*- coding: utf-8 -*-
"""
test_prompt_invalid_two_epoch.py

Two-epoch diagnostic runner for train_blank_llm_skill_memory.py.

Purpose:
  1) Run exactly two epochs with the original LLM+skill-memory route.
  2) Save prompts after tokenizer max_length truncation, so you can see the exact
     text that enters Qwen after the length constraint.
  3) Count invalid actions and save each invalid action's generated output,
     selected ACTION_INDEX, original prompt, and truncated prompt.

Run from the project root:
  python test_prompt_invalid_two_epoch.py

Optional environment overrides:
  PROMPT_INVALID_TEST_SAMPLE_GRAPHS=20
  PROMPT_INVALID_TEST_OUTPUT_ROOT=./llm_ablation_runs/skill_memory_prompt_invalid_test
  PROMPT_INVALID_TEST_MAX_PROMPT_SAMPLES=50
  PROMPT_INVALID_TEST_SAVE_ALL_ACTIONS=0
"""

from __future__ import annotations

import json
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from llm_tdpo_ablation.config.paths import (
    PROMPT_INVALID_ALL_EPOCHS_FILE,
    PROMPT_INVALID_TEST_CONFIG_FILE,
    PROMPT_INVALID_TEST_DIR_PREFIX,
    all_actions_epoch_file,
    invalid_actions_epoch_file,
    invalid_summary_epoch_file,
    prompt_invalid_output_root,
    truncated_prompt_samples_epoch_file,
)
from llm_tdpo_ablation.training import blank_llm_skill_memory_runner as train_mod

decision_mod = train_mod.decision_mod

TEST_OUTPUT_ROOT = prompt_invalid_output_root()
TEST_TAG = time.strftime("%Y%m%d_%H%M%S")
TEST_DIR = TEST_OUTPUT_ROOT / f"{PROMPT_INVALID_TEST_DIR_PREFIX}_{TEST_TAG}"
TEST_DIR.mkdir(parents=True, exist_ok=True)

TEST_SAMPLE_GRAPHS = int(os.environ.get("PROMPT_INVALID_TEST_SAMPLE_GRAPHS", str(getattr(train_mod, "SAMPLE_NUM_GRAPHS", 20))))
MAX_PROMPT_SAMPLES = int(os.environ.get("PROMPT_INVALID_TEST_MAX_PROMPT_SAMPLES", "50"))
SAVE_ALL_ACTIONS = os.environ.get("PROMPT_INVALID_TEST_SAVE_ALL_ACTIONS", "0") == "1"

_CURRENT_EPOCH: Optional[int] = None
_epoch_counts: Dict[int, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
_epoch_prompt_samples: Dict[int, List[Dict[str, Any]]] = defaultdict(list)


# -----------------------------------------------------------------------------
# Test-only configuration patching.
# -----------------------------------------------------------------------------
train_mod.NUM_EPOCHS = 2
train_mod.SAMPLE_NUM_GRAPHS = TEST_SAMPLE_GRAPHS
train_mod.MAX_TRAIN_GRAPHS = TEST_SAMPLE_GRAPHS
train_mod.TDPO_ROLLOUT_CHUNK_GRAPHS = int(getattr(train_mod, "TDPO_ROLLOUT_CHUNK_GRAPHS", TEST_SAMPLE_GRAPHS)) or TEST_SAMPLE_GRAPHS
train_mod.OUTPUT_ROOT = TEST_OUTPUT_ROOT
train_mod.RUN_NAME = "prompt_invalid_two_epoch"
train_mod.SAVE_TRACE_JSONL = True
train_mod.SAVE_TASK_RECORDS_EVERY_EPOCH = True
train_mod.WRITE_PROMPT_SAMPLES = True
train_mod.MAX_PROMPT_SAMPLES = max(int(getattr(train_mod, "MAX_PROMPT_SAMPLES", 20)), MAX_PROMPT_SAMPLES)
train_mod.SAVE_AUXILIARY_JSON = True
train_mod.SAVE_FINAL_RESULT_JSON = True
train_mod.SAVE_EVERY_EPOCH = False
train_mod.RUN_GREEDY_EVAL_EACH_EPOCH = False
train_mod.EVAL_WITH_GREEDY = False
# Keep this as the ablation route: no TDPO update, no knowledge update.  The goal
# is prompt/action diagnostics, not additional training.
train_mod.ABLATION_SKIP_TDPO_UPDATE = True
train_mod.KNOWLEDGE_UPDATE_STRATEGY = 3
train_mod.KNOWLEDGE_UPDATE_MEMORY_DURING_TDPO = False
train_mod.KNOWLEDGE_UPDATE_WEIGHTS_DURING_TDPO = False


# -----------------------------------------------------------------------------
# Prompt truncation and legality helpers.
# -----------------------------------------------------------------------------
def _json_append(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _json_write(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _token_ids(tokenizer, text: str, max_length: Optional[int] = None) -> List[int]:
    kwargs = {"add_special_tokens": True}
    if max_length is not None:
        kwargs.update({"truncation": True, "max_length": int(max_length)})
    else:
        kwargs.update({"truncation": False})
    enc = tokenizer(str(text or ""), **kwargs)
    return list(enc.get("input_ids", []) or [])


def _truncated_prompt_record(policy, prompt: str) -> Dict[str, Any]:
    max_len = int(policy.cfg.max_length)
    full_ids = _token_ids(policy.tokenizer, prompt, max_length=None)
    trunc_ids = _token_ids(policy.tokenizer, prompt, max_length=max_len)
    truncated_text = policy.tokenizer.decode(trunc_ids, skip_special_tokens=True)
    return {
        "max_length": int(max_len),
        "original_token_count": int(len(full_ids)),
        "truncated_token_count": int(len(trunc_ids)),
        "was_truncated": bool(len(full_ids) > max_len),
        "original_char_count": int(len(str(prompt or ""))),
        "truncated_char_count": int(len(truncated_text)),
        "truncated_prompt": truncated_text,
    }


def _is_action_invalid_by_prompt_mask(info, num_tools: int) -> Tuple[bool, str]:
    idx = int(getattr(info, "action_index", -1))
    action_type = str(getattr(info, "action_type", ""))
    generated = str(getattr(info, "generated_action_text", "") or "")
    action_dim = int(num_tools) + 2
    allowed_mask = [int(x) for x in list(getattr(info, "prompt_allowed_tool_mask", []) or [])]

    if action_type == "invalid_generated" or "parse_failed_no_fallback" in generated:
        return True, "parse_failed_or_out_of_range_generation"
    if idx < 0 or idx >= action_dim:
        return True, "action_index_out_of_range"
    if idx < int(num_tools):
        if idx >= len(allowed_mask):
            return True, "tool_index_not_in_allowed_mask"
        if int(allowed_mask[idx]) != 1:
            return True, "tool_index_forbidden_by_allowed_mask"
        return False, "legal_tool_by_allowed_mask"
    if idx == int(num_tools):
        return False, "legal_local_action"
    if idx == int(num_tools) + 1:
        return False, "legal_pause_action"
    return True, "unknown_invalid_action"


def _action_debug_row(policy, dp: Dict[str, Any], action, info, num_tools: int) -> Dict[str, Any]:
    invalid, reason = _is_action_invalid_by_prompt_mask(info, num_tools)
    prompt = str(getattr(info, "prompt", "") or "")
    trunc = _truncated_prompt_record(policy, prompt)
    row = {
        "epoch": int(_CURRENT_EPOCH or -1),
        "task_id": str(dp.get("task_id", dp.get("graph_id", ""))),
        "node_id": int(dp.get("node_id", -1) or -1),
        "node_type_id": int(dp.get("node_type_id", 0) or 0),
        "selected_action_index": int(getattr(info, "action_index", -1)),
        "selected_action_type": str(getattr(info, "action_type", "")),
        "selected_tool_type_id": None if getattr(info, "tool_type_id", None) is None else int(getattr(info, "tool_type_id")),
        "generated_action_text": str(getattr(info, "generated_action_text", "") or ""),
        "invalid_by_prompt_mask": bool(invalid),
        "invalid_reason": str(reason),
        "allowed_tool_mask": [int(x) for x in list(getattr(info, "prompt_allowed_tool_mask", []) or [])],
        "valid_action_indices_from_policy_mask": [int(x) for x in list(getattr(info, "valid_action_indices", []) or [])],
        "knowledge_prompt": str(getattr(info, "knowledge_prompt", "") or ""),
        "prompt_original": prompt,
        **trunc,
    }
    return row


# -----------------------------------------------------------------------------
# Monkey patches.
# -----------------------------------------------------------------------------
_original_select_actions = decision_mod.TDPOPolicy.select_actions


def _debug_select_actions(self, dps, num_tools, *args, **kwargs):
    results = _original_select_actions(self, dps, num_tools, *args, **kwargs)
    epoch = int(_CURRENT_EPOCH or -1)
    for dp, (action, info) in zip(list(dps), results):
        row = _action_debug_row(self, dp, action, info, int(num_tools))
        counts = _epoch_counts[epoch]
        counts["total_decisions"] += 1
        if bool(row["invalid_by_prompt_mask"]):
            counts["invalid_actions"] += 1
            counts[f"invalid_reason::{row['invalid_reason']}"] += 1
            _json_append(TEST_DIR / invalid_actions_epoch_file(epoch), row)
        else:
            counts["legal_actions"] += 1
        if SAVE_ALL_ACTIONS:
            _json_append(TEST_DIR / all_actions_epoch_file(epoch), row)
        samples = _epoch_prompt_samples[epoch]
        if len(samples) < MAX_PROMPT_SAMPLES:
            samples.append(row)
    return results


decision_mod.TDPOPolicy.select_actions = _debug_select_actions

_original_rollout = train_mod.run_tdpo_rollout


def _debug_rollout(*args, **kwargs):
    global _CURRENT_EPOCH
    epoch = kwargs.get("epoch", None)
    if epoch is None:
        # run_tdpo_rollout(..., epoch, seed, use_memory, sample) in positional form.
        # The current train file usually calls it with keyword arguments.
        try:
            epoch = int(args[4])
        except Exception:
            epoch = -1
    previous_epoch = _CURRENT_EPOCH
    _CURRENT_EPOCH = int(epoch)
    try:
        return _original_rollout(*args, **kwargs)
    finally:
        epoch_i = int(epoch)
        samples = _epoch_prompt_samples.get(epoch_i, [])
        if samples:
            _json_write(TEST_DIR / truncated_prompt_samples_epoch_file(epoch_i), samples)
        _json_write(TEST_DIR / invalid_summary_epoch_file(epoch_i), dict(_epoch_counts.get(epoch_i, {})))
        _CURRENT_EPOCH = previous_epoch


train_mod.run_tdpo_rollout = _debug_rollout


def main() -> None:
    config = {
        "test_dir": str(TEST_DIR),
        "num_epochs": 2,
        "sample_num_graphs": int(TEST_SAMPLE_GRAPHS),
        "save_all_actions": bool(SAVE_ALL_ACTIONS),
        "max_prompt_samples_per_epoch": int(MAX_PROMPT_SAMPLES),
        "llm_max_length_from_train": int(getattr(train_mod, "LLM_MAX_LENGTH", -1)),
        "llm_max_length_from_decision_default": int(getattr(decision_mod, "LLM_MAX_LENGTH", -1)),
        "fallback_to_scoring": bool(getattr(decision_mod, "LLM_GENERATION_FALLBACK_TO_SCORING", False)),
        "rollout_generate_only": bool(getattr(decision_mod, "LLM_ROLLOUT_GENERATE_ONLY", False)),
        "env_invalid_action_mode": str(getattr(train_mod, "ENV_INVALID_ACTION_MODE", "")),
        "env_fail_on_queue_rejection": bool(getattr(train_mod, "ENV_FAIL_ON_QUEUE_REJECTION", False)),
    }
    _json_write(TEST_DIR / PROMPT_INVALID_TEST_CONFIG_FILE, config)
    print(f"[prompt-invalid-test] output_dir={TEST_DIR}", flush=True)
    print(f"[prompt-invalid-test] running NUM_EPOCHS=2 SAMPLE_NUM_GRAPHS={TEST_SAMPLE_GRAPHS}", flush=True)
    train_mod.main()
    final_summary = {str(k): dict(v) for k, v in sorted(_epoch_counts.items())}
    _json_write(TEST_DIR / PROMPT_INVALID_ALL_EPOCHS_FILE, final_summary)
    print(f"[prompt-invalid-test] done. debug outputs saved under: {TEST_DIR}", flush=True)
    print(json.dumps(final_summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
