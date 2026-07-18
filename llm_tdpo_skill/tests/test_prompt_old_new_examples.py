# -*- coding: utf-8 -*-
"""
test_prompt_old_new_examples.py

Build the same environment decision points and save full, untruncated old/new TDPO
prompts for manual comparison.  This script does not run LLM generation and does
not perform TDPO update.

Example:
  python test_prompt_old_new_examples.py \
    --train-file ./train_fast_mem.py \
    --old-decision-file ./llm_tdpo/policy/legacy_tdpo.py \
    --new-decision-file ./llm_tdpo/policy/tdpo.py \
    --task-env-file ./llm_tdpo/environment/task_env.py \
    --knowledge-file ./llm_tdpo/knowledge/adapter.py \
    --qwen-model-path /models/Qwen \
    --sample-num-graphs 20 \
    --use-memory 1 \
    --num-examples 3 \
    --save-json prompt_old_new_examples.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import random
import statistics
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from llm_tdpo.config.paths import (
    CURRENT_POLICY_ENTRY_FILE,
    KNOWLEDGE_ENTRY_FILE,
    LEGACY_POLICY_ENTRY_FILE,
    PROMPT_COMPARISON_REPORT_FILE,
    QWEN_MODEL_PATH,
    TASK_ENV_ENTRY_FILE,
    TRAIN_ENTRY_FILE,
)


def load_module(module_name: str, path: Path):
    path = Path(path).expanduser().resolve()
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {module_name} from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return sys.modules.get(module_name, mod)


def percentile(vals: Sequence[int], q: float) -> float:
    if not vals:
        return 0.0
    xs = sorted(int(v) for v in vals)
    if len(xs) == 1:
        return float(xs[0])
    pos = (len(xs) - 1) * float(q)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(xs[lo])
    return float(xs[lo] * (hi - pos) + xs[hi] * (pos - lo))


def summarize(lengths: Sequence[int], max_length: int) -> Dict[str, Any]:
    if not lengths:
        return {"count": 0}
    over = [int(x) for x in lengths if int(x) > int(max_length)]
    return {
        "count": len(lengths),
        "max_length_limit": int(max_length),
        "mean_tokens": round(float(statistics.mean(lengths)), 2),
        "median_tokens": round(float(statistics.median(lengths)), 2),
        "p90_tokens": round(percentile(lengths, 0.90), 2),
        "p95_tokens": round(percentile(lengths, 0.95), 2),
        "max_tokens": int(max(lengths)),
        "min_tokens": int(min(lengths)),
        "num_over_limit": len(over),
        "ratio_over_limit": round(len(over) / max(1, len(lengths)), 4),
        "mean_overflow_tokens": round(float(statistics.mean([x - int(max_length) for x in over])), 2) if over else 0.0,
        "max_overflow_tokens": int(max([x - int(max_length) for x in over])) if over else 0,
    }


def choose_lightweight_action(decision_mod, task_env_mod, dp: Dict[str, Any], num_tools: int):
    allowed = list(dp.get("allowed_tools_mask", []) or [])[: int(num_tools)]
    for i, v in enumerate(allowed):
        if int(v) == 1:
            return task_env_mod.DecisionAction(action_type="tool", tool_type_id=int(i))
    return task_env_mod.DecisionAction(action_type="local")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-file", default=TRAIN_ENTRY_FILE)
    parser.add_argument("--old-decision-file", default=LEGACY_POLICY_ENTRY_FILE)
    parser.add_argument("--new-decision-file", default=CURRENT_POLICY_ENTRY_FILE)
    parser.add_argument("--task-env-file", default=TASK_ENV_ENTRY_FILE)
    parser.add_argument("--knowledge-file", default=KNOWLEDGE_ENTRY_FILE)
    parser.add_argument("--qwen-model-path", default=QWEN_MODEL_PATH)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--num-examples", type=int, default=3)
    parser.add_argument("--scan-prompts", type=int, default=100, help="Collect up to this many prompts for length summaries; full text is saved only for num-examples.")
    parser.add_argument("--max-env-steps", type=int, default=2000)
    parser.add_argument("--sample-num-graphs", type=int, default=None)
    parser.add_argument("--sample-random-pool-limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20260425)
    parser.add_argument("--use-memory", type=int, default=1)
    parser.add_argument("--tmp-run-dir", default="")
    parser.add_argument("--save-json", default=PROMPT_COMPARISON_REPORT_FILE)
    args = parser.parse_args()

    task_env_mod = load_module("task_env", Path(args.task_env_file))
    if args.knowledge_file:
        load_module("knowledge_aspect_adapter", Path(args.knowledge_file))

    # Load the new decision module under the canonical name so train.py imports it.
    new_decision = load_module("decision_TDPO_ye", Path(args.new_decision_file))
    train_mod = load_module("train", Path(args.train_file))
    # Load the old decision module under an isolated name after train.py is ready.
    old_decision = load_module("decision_TDPO_ye_old_for_compare", Path(args.old_decision_file))

    if args.sample_num_graphs is not None:
        train_mod.SAMPLE_NUM_GRAPHS = int(args.sample_num_graphs)
        train_mod.MAX_TRAIN_GRAPHS = int(args.sample_num_graphs)
    if args.sample_random_pool_limit is not None:
        train_mod.SAMPLE_RANDOM_POOL_LIMIT = int(args.sample_random_pool_limit)

    dataset, _ = train_mod.load_dataset_from_dir(Path(train_mod.DATA_DIR), seed=int(args.seed))
    if getattr(train_mod, "MAX_TRAIN_GRAPHS", 0):
        dataset = train_mod.subset_dataset(dataset, int(train_mod.MAX_TRAIN_GRAPHS))
    graph_meta = train_mod.build_graph_meta_map(dataset)
    num_tools = len(dataset.get("tool_catalog", []))
    action_dim = new_decision.action_dim_from_num_tools(num_tools)

    knowledge_trainer = None
    if bool(args.use_memory):
        tmp_run_dir = Path(args.tmp_run_dir).expanduser().resolve() if args.tmp_run_dir else Path(tempfile.mkdtemp(prefix="prompt_compare_"))
        tmp_run_dir.mkdir(parents=True, exist_ok=True)
        loaded = train_mod.load_knowledge_trainer(dataset, action_dim, tmp_run_dir)
        if isinstance(loaded, (tuple, list)):
            knowledge_trainer = loaded[0] if loaded else None
        else:
            knowledge_trainer = loaded
        if hasattr(train_mod, "release_knowledge_backbone_for_tdpo") and knowledge_trainer is not None:
            train_mod.release_knowledge_backbone_for_tdpo(knowledge_trainer)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.qwen_model_path, trust_remote_code=True)
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"

    env = train_mod.build_env(dataset, seed=int(args.seed))
    env.reset()

    examples: List[Dict[str, Any]] = []
    old_lengths: List[int] = []
    new_lengths: List[int] = []
    old_knowledge_lengths: List[int] = []
    new_knowledge_lengths: List[int] = []

    for _ in range(int(args.max_env_steps)):
        dps = env.collect_decision_points()
        if dps:
            for dp in dps:
                gm = graph_meta.get(str(dp.get("task_id", "")), {})
                old_kp = ""
                new_kp = ""
                if knowledge_trainer is not None:
                    old_mem = old_decision.retrieve_knowledge_for_dp(
                        knowledge_trainer=knowledge_trainer,
                        dp=dp,
                        num_tools=num_tools,
                        use_memory=True,
                        target_dim=int(action_dim + 4),
                    )
                    new_mem = new_decision.retrieve_knowledge_for_dp(
                        knowledge_trainer=knowledge_trainer,
                        dp=dp,
                        num_tools=num_tools,
                        use_memory=True,
                        target_dim=int(action_dim + 4),
                    )
                    old_kp = str(old_mem.get("knowledge_prompt", "") or "")
                    new_kp = str(new_mem.get("knowledge_prompt", "") or "")

                old_prompt = old_decision.build_prompt_from_dp(dp, num_tools=num_tools, knowledge_prompt=old_kp, graph_meta=gm)
                new_prompt = new_decision.build_prompt_from_dp(dp, num_tools=num_tools, knowledge_prompt=new_kp, graph_meta=gm)

                old_n = len(tokenizer.encode(old_prompt, add_special_tokens=True, truncation=False))
                new_n = len(tokenizer.encode(new_prompt, add_special_tokens=True, truncation=False))
                old_kn = len(tokenizer.encode(old_kp, add_special_tokens=True, truncation=False)) if old_kp.strip() else 0
                new_kn = len(tokenizer.encode(new_kp, add_special_tokens=True, truncation=False)) if new_kp.strip() else 0
                old_lengths.append(old_n)
                new_lengths.append(new_n)
                old_knowledge_lengths.append(old_kn)
                new_knowledge_lengths.append(new_kn)

                if len(examples) < int(args.num_examples):
                    examples.append({
                        "index": len(examples),
                        "task_id": str(dp.get("task_id", "")),
                        "node_id": int(dp.get("node_id", -1)),
                        "old_prompt_tokens": old_n,
                        "new_prompt_tokens": new_n,
                        "old_knowledge_prompt_tokens": old_kn,
                        "new_knowledge_prompt_tokens": new_kn,
                        "token_reduction": old_n - new_n,
                        "old_prompt": old_prompt,
                        "new_prompt": new_prompt,
                        "old_knowledge_prompt": old_kp,
                        "new_knowledge_prompt": new_kp,
                    })

                if len(old_lengths) >= int(args.scan_prompts):
                    break

            for dp in dps:
                action = choose_lightweight_action(new_decision, task_env_mod, dp, num_tools)
                env.apply_decision(str(dp["task_id"]), int(dp["node_id"]), action)

        if len(old_lengths) >= int(args.scan_prompts):
            break
        env.step()
        if env.done():
            break

    report = {
        "old_decision_file": str(Path(args.old_decision_file).expanduser().resolve()),
        "new_decision_file": str(Path(args.new_decision_file).expanduser().resolve()),
        "max_length": int(args.max_length),
        "num_tools": int(num_tools),
        "length_summary": {
            "old_prompt": summarize(old_lengths, int(args.max_length)),
            "new_prompt": summarize(new_lengths, int(args.max_length)),
            "old_knowledge_prompt": summarize(old_knowledge_lengths, int(args.max_length)),
            "new_knowledge_prompt": summarize(new_knowledge_lengths, int(args.max_length)),
        },
        "examples": examples,
    }

    out = Path(args.save_json).expanduser().resolve()
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["length_summary"], ensure_ascii=False, indent=2))
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
