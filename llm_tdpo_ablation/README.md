# LLM-TDPO Ablation Experiments

## Variants

| Entry point | Experiment |
| --- | --- |
| `train_TDPO_no_skill.py` | TDPO policy without skill-memory guidance |
| `train_blank_llm_direct.py` | Frozen blank LLM with direct action generation |
| `train_blank_llm_skill_memory.py` | Frozen blank LLM with skill-memory prompting |

## Layout

```text
llm_tdpo_ablation/
  config/       Central paths and variant registry
  environment/  Shared task environment
  knowledge/    Shared knowledge and memory implementation
  policies/     Three isolated decision-policy variants
  training/     Three isolated experiment runners
  diagnostics/  Two-epoch prompt and invalid-action diagnostic
train_TDPO_no_skill.py                 No-skill TDPO training entry
train_blank_llm_direct.py              Direct blank-LLM training entry
train_blank_llm_skill_memory.py        Skill-memory training entry
test_prompt_invalid_two_epoch.py       Prompt/invalid-action analysis entry
```

The root contains only active training and analysis entry points plus project
metadata. All implementation modules live exclusively under
`llm_tdpo_ablation/`.

## Configure paths

Edit `llm_tdpo_ablation/config/paths.py`. It controls all default dataset, model,
knowledge, memory, tool-library, offload, training-output, and diagnostic-output
paths. Original values and environment-variable overrides are retained.

## Install and run

```bash
python -m pip install -r requirements.txt
python train_TDPO_no_skill.py
python train_blank_llm_direct.py
python train_blank_llm_skill_memory.py
```

Run the original two-epoch diagnostic with:

```bash
python test_prompt_invalid_two_epoch.py
```
