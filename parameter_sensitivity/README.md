# Knowledge-Environment and TDPO Sweep Suite

This project studies how knowledge trained under different environment settings
affects TDPO policies. It contains three knowledge-training experiments, a
skill-enabled TDPO pipeline, a no-skill TDPO ablation, and six evaluation sweeps.

## Layout

```text
experiment_suite/
  config/paths.py                 # Every input and output path
  shared/knowledge_adapter.py     # One deduplicated knowledge implementation
  knowledge_training/             # Bound, deadline, and arrival-time training
  tdpo_skill/                      # Skill-enabled policy, environment, and trainer
    evaluation/                    # Time/deadline and same-skill controls
  tdpo_no_skill/                   # No-skill policy, environment, and trainer
    evaluation/                    # Time and deadline ablation sweeps
tdpo/                              # Preserved skill-enabled command names
tdpo_no/                           # Preserved no-skill command names
train_knowledge_dynamic_tools_*.py # Preserved knowledge-training commands
```

Notebook checkpoints, bytecode caches, empty output directories, duplicated
knowledge adapters, and the unreferenced `Untitled Folder` draft are excluded.

## Configuration

Edit `experiment_suite/config/paths.py` to relocate datasets, Qwen models,
knowledge checkpoints, memories, tool libraries, policy checkpoints, sweep cases,
or output directories. All current values match the original source.

## Installation

```bash
pip install -r requirements.txt
```

## Knowledge training

```bash
python train_knowledge_dynamic_tools_bind_sep_csv.py
python train_knowledge_dynamic_tools_sweep_dead.py
python train_knowledge_dynamic_tools_sweep_time.py
```

## Skill-enabled TDPO

```bash
python tdpo/train_fast_mem.py
python tdpo/eval_sweep_dead.py
python tdpo/eval_sweep_dead_same.py
python tdpo/eval_sweep_time.py
python tdpo/eval_sweep_time_same.py
```

The `*_same.py` evaluations vary the environment while using one fixed knowledge
skill/memory case, separating environment effects from skill-selection effects.

## No-skill TDPO ablation

```bash
python tdpo_no/train_TDPO_no_skill.py
python tdpo_no/eval_tdpo_no_sweep_dead.py
python tdpo_no/eval_tdpo_no_sweep_nume.py
```

