# LLM-TDPO Skill

## Project layout

```text
llm_tdpo/
  config/       Central input and output path configuration
  environment/  Dataset normalization and task execution environment
  knowledge/    Knowledge extraction, distillation, and memory
  policy/       Current and legacy TDPO policies
  training/     Training and evaluation orchestration
  tools/        Offline skill-rematching utility
tests/          Executable diagnostics and regression utilities
data/reference/ Original JSON reports supplied with the code
train_fast_mem.py                         Training entry point
rematch_epoch75_task1090_skills.py        Offline analysis entry point
test_policy_legality_rollout.py           Policy diagnostic entry point
test_prompt_old_new_examples.py           Prompt comparison entry point
test_token_prompt_length.py               Prompt-length analysis entry point
```

The repository root contains only active training and analysis entry points plus
project metadata. Implementation modules live exclusively under `llm_tdpo/`,
and diagnostic implementations live under `tests/`.

## Configure paths

Edit `llm_tdpo/config/paths.py` to relocate any dataset, model, tool library,
knowledge checkpoint, memory file, or output. All defaults retain the original
values and are documented in that file.

## Install and run

```bash
python -m pip install -r requirements.txt
python train_fast_mem.py
```

Optional diagnostics:

```bash
python test_policy_legality_rollout.py --help
python test_prompt_old_new_examples.py --help
python test_token_prompt_length.py --help
python rematch_epoch75_task1090_skills.py --help
```
