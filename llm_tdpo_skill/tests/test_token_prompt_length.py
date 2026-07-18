# -*- coding: utf-8 -*-
"""
test_token_prompt_length.py

Purpose:
  Measure prompt token lengths for the current TDPO prompt builder without running
  LLM generation or TDPO update.

Usage patterns:
  1) Fastest, using saved prompt samples:
       python test_token_prompt_length.py --prompt-samples ./prompt_samples_epoch_001.json

  2) Build a real environment and call decision_TDPO_ye.build_prompt_from_dp():
       python test_token_prompt_length.py \
         --train-file ./train_fast_mem.py \
         --decision-file ./llm_tdpo/policy/tdpo.py \
         --task-env-file ./llm_tdpo/environment/task_env.py \
         --max-prompts 200

  3) Include retrieved memory text in prompt construction. This may load the
     knowledge trainer and can be slower/heavier:
       python test_token_prompt_length.py --use-memory 1 --max-prompts 100
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
    PROMPT_LENGTH_REPORT_FILE,
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


def summarize_lengths(lengths: List[int], max_length: int) -> Dict[str, Any]:
    if not lengths:
        return {"count": 0}
    over = [x for x in lengths if x > int(max_length)]
    return {
        "count": len(lengths),
        "max_length_limit": int(max_length),
        "mean_tokens": round(float(statistics.mean(lengths)), 2),
        "median_tokens": round(float(statistics.median(lengths)), 2),
        "p90_tokens": round(percentile(lengths, 0.90), 2),
        "p95_tokens": round(percentile(lengths, 0.95), 2),
        "p99_tokens": round(percentile(lengths, 0.99), 2),
        "max_tokens": int(max(lengths)),
        "min_tokens": int(min(lengths)),
        "num_over_limit": len(over),
        "ratio_over_limit": round(len(over) / max(1, len(lengths)), 4),
        "mean_overflow_tokens": round(float(statistics.mean([x - int(max_length) for x in over])), 2) if over else 0.0,
        "max_overflow_tokens": int(max([x - int(max_length) for x in over])) if over else 0,
    }


def load_prompts_from_samples(path: Path) -> List[Dict[str, Any]]:
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = obj if isinstance(obj, list) else obj.get("samples", []) if isinstance(obj, dict) else []
    out = []
    for i, row in enumerate(rows):
        if isinstance(row, dict):
            prompt = str(row.get("prompt", "") or "")
            knowledge_prompt = str(row.get("knowledge_prompt", "") or "")
        else:
            prompt = str(row or "")
            knowledge_prompt = ""
        if prompt.strip():
            out.append({"prompt": prompt, "knowledge_prompt": knowledge_prompt, "source_index": i})
    return out


def choose_lightweight_action(decision_mod, task_env_mod, dp: Dict[str, Any], num_tools: int):
    # This is only to advance the simulator and collect more prompts. It is not
    # used for training, evaluation, or policy decisions.
    allowed = list(dp.get("allowed_tools_mask", []) or [])[: int(num_tools)]
    for i, v in enumerate(allowed):
        if int(v) == 1:
            return task_env_mod.DecisionAction(action_type="tool", tool_type_id=int(i))
    return task_env_mod.DecisionAction(action_type="local")


def build_prompts_from_environment(args) -> List[Dict[str, Any]]:
    decision_path = Path(args.decision_file).expanduser().resolve()
    train_path = Path(args.train_file).expanduser().resolve()
    task_env_path = Path(args.task_env_file).expanduser().resolve()
    knowledge_path = Path(args.knowledge_file).expanduser().resolve() if args.knowledge_file else None

    # Preload dependency modules so train.py imports these exact files.
    task_env_mod = load_module("task_env", task_env_path)
    if knowledge_path and knowledge_path.exists():
        load_module("knowledge_aspect_adapter", knowledge_path)
    decision_mod = load_module("decision_TDPO_ye", decision_path)
    train_mod = load_module("train", train_path)
    decision_mod = getattr(train_mod, "decision_mod", decision_mod)

    if args.sample_num_graphs is not None:
        train_mod.SAMPLE_NUM_GRAPHS = int(args.sample_num_graphs)
        train_mod.MAX_TRAIN_GRAPHS = int(args.sample_num_graphs)
    if args.sample_random_pool_limit is not None:
        train_mod.SAMPLE_RANDOM_POOL_LIMIT = int(args.sample_random_pool_limit)

    dataset, scan_info = train_mod.load_dataset_from_dir(Path(train_mod.DATA_DIR), seed=int(args.seed))
    if getattr(train_mod, "MAX_TRAIN_GRAPHS", 0):
        dataset = train_mod.subset_dataset(dataset, int(train_mod.MAX_TRAIN_GRAPHS))
    graph_meta = train_mod.build_graph_meta_map(dataset)
    num_tools = len(dataset.get("tool_catalog", []))
    action_dim = decision_mod.action_dim_from_num_tools(num_tools)

    knowledge_trainer = None
    if bool(args.use_memory):
        tmp_run_dir = Path(args.tmp_run_dir).expanduser().resolve() if args.tmp_run_dir else Path(tempfile.mkdtemp(prefix="prompt_len_"))
        tmp_run_dir.mkdir(parents=True, exist_ok=True)
        loaded_knowledge = train_mod.load_knowledge_trainer(dataset, action_dim, tmp_run_dir)
        if isinstance(loaded_knowledge, (tuple, list)):
            knowledge_trainer = loaded_knowledge[0] if loaded_knowledge else None
        else:
            knowledge_trainer = loaded_knowledge
        if hasattr(train_mod, "release_knowledge_backbone_for_tdpo") and knowledge_trainer is not None:
            train_mod.release_knowledge_backbone_for_tdpo(knowledge_trainer)

    env = train_mod.build_env(dataset, seed=int(args.seed))
    env.reset()
    prompts: List[Dict[str, Any]] = []

    step_limit = int(args.max_env_steps)
    rng = random.Random(int(args.seed))
    for _ in range(step_limit):
        dps = env.collect_decision_points()
        if dps:
            for dp in dps:
                gm = graph_meta.get(str(dp.get("task_id", "")), {})
                knowledge_prompt = ""
                if knowledge_trainer is not None:
                    mem = decision_mod.retrieve_knowledge_for_dp(
                        knowledge_trainer=knowledge_trainer,
                        dp=dp,
                        num_tools=num_tools,
                        use_memory=True,
                        target_dim=int(action_dim + 4),
                    )
                    knowledge_prompt = str(mem.get("knowledge_prompt", "") or "")
                prompt = decision_mod.build_prompt_from_dp(
                    dp,
                    num_tools=num_tools,
                    knowledge_prompt=knowledge_prompt,
                    graph_meta=gm,
                )
                prompts.append({
                    "prompt": prompt,
                    "knowledge_prompt": knowledge_prompt,
                    "task_id": str(dp.get("task_id", "")),
                    "node_id": int(dp.get("node_id", -1)),
                })
                if len(prompts) >= int(args.max_prompts):
                    return prompts
            # Advance the environment with a deterministic cheap action so new
            # nodes can become ready. This is only for prompt coverage.
            for dp in dps:
                action = choose_lightweight_action(decision_mod, task_env_mod, dp, num_tools)
                env.apply_decision(str(dp["task_id"]), int(dp["node_id"]), action)
        env.step()
        if env.done():
            break
    return prompts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt-samples", default="", help="Existing prompt_samples_epoch_XXX.json; bypasses env construction.")
    parser.add_argument("--train-file", default=TRAIN_ENTRY_FILE)
    parser.add_argument("--decision-file", default=CURRENT_POLICY_ENTRY_FILE)
    parser.add_argument("--task-env-file", default=TASK_ENV_ENTRY_FILE)
    parser.add_argument("--knowledge-file", default=KNOWLEDGE_ENTRY_FILE)
    parser.add_argument("--qwen-model-path", default="", help="Tokenizer path. Default uses train_mod.QWEN_MODEL_PATH or /models/Qwen.")
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--max-prompts", type=int, default=200)
    parser.add_argument("--max-env-steps", type=int, default=2000)
    parser.add_argument("--sample-num-graphs", type=int, default=None)
    parser.add_argument("--sample-random-pool-limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20260425)
    parser.add_argument("--use-memory", type=int, default=0, help="1: include retrieved memory prompt; may load knowledge trainer.")
    parser.add_argument("--tmp-run-dir", default="")
    parser.add_argument("--save-json", default=PROMPT_LENGTH_REPORT_FILE)
    parser.add_argument("--print-examples", type=int, default=3)
    args = parser.parse_args()

    if args.prompt_samples:
        prompts = load_prompts_from_samples(Path(args.prompt_samples))[: int(args.max_prompts)]
        tokenizer_path = args.qwen_model_path or os.environ.get("QWEN_MODEL_PATH", QWEN_MODEL_PATH)
    else:
        prompts = build_prompts_from_environment(args)
        # Import train only after build_prompts so it is available in sys.modules.
        train_mod = sys.modules.get("train")
        tokenizer_path = args.qwen_model_path or str(getattr(train_mod, "QWEN_MODEL_PATH", QWEN_MODEL_PATH))

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"

    lengths = []
    knowledge_lengths = []
    rows = []
    for i, row in enumerate(prompts):
        prompt = str(row.get("prompt", "") or "")
        kp = str(row.get("knowledge_prompt", "") or "")
        ids = tokenizer.encode(prompt, add_special_tokens=True, truncation=False)
        k_ids = tokenizer.encode(kp, add_special_tokens=True, truncation=False) if kp.strip() else []
        n = len(ids)
        lengths.append(n)
        knowledge_lengths.append(len(k_ids))
        rows.append({
            "index": i,
            "task_id": row.get("task_id"),
            "node_id": row.get("node_id"),
            "prompt_tokens": n,
            "knowledge_prompt_tokens": len(k_ids),
            "over_limit": bool(n > int(args.max_length)),
            "overflow_tokens": max(0, n - int(args.max_length)),
            "prompt_head": prompt[:500],
            "prompt_tail": prompt[-500:],
        })

    report = {
        "tokenizer_path": tokenizer_path,
        "max_length": int(args.max_length),
        "prompt_summary": summarize_lengths(lengths, int(args.max_length)),
        "knowledge_prompt_summary": summarize_lengths(knowledge_lengths, int(args.max_length)),
        "examples_over_limit": [r for r in rows if r["over_limit"]][: int(args.print_examples)],
        "examples_under_or_equal_limit": [r for r in rows if not r["over_limit"]][: int(args.print_examples)],
    }

    out_path = Path(args.save_json).expanduser().resolve()
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["prompt_summary"], ensure_ascii=False, indent=2))
    print(f"[saved] {out_path}")
    if report["examples_over_limit"]:
        print("\n[examples_over_limit]")
        for ex in report["examples_over_limit"]:
            print(json.dumps({k: ex[k] for k in ("index", "task_id", "node_id", "prompt_tokens", "overflow_tokens")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
