# Knowledge Pretraining and CMASD Ablations

## Layout

```text
knowledge_pretraining/
  config/paths.py              # All input and output paths
  environment/task_env.py      # Task/DAG simulation environment
  knowledge/base_adapter.py    # Full knowledge model and memory
  knowledge/ablation_adapter.py# CMASD ablation variants
  training/base_runner.py      # Standard pretraining workflow
  training/ablation_runner.py  # Ablation workflow and CLI
launchers/                     # Serial and parallel experiment launchers
train_knowledge_dynamic_tools.py
                               # Standard training entry point
train_knowledge_dynamic_tools_ablation.py
                               # Ablation training entry point
```

The root contains only the two active Python training entry points and project
metadata. Package implementations live exclusively under `knowledge_pretraining/`;
shell launchers live exclusively under `launchers/`.

## Configuration

Edit `knowledge_pretraining/config/paths.py` to change dataset, model, tool-library,
checkpoint, or output locations. All values currently match the original source.
Relative output paths remain relative to the directory from which training is run.

Training hyperparameters remain in their respective runner modules because they
are behavioral settings rather than file locations.

## Usage

Install dependencies:

```bash
pip install -r requirements.txt
```

Run standard knowledge pretraining:

```bash
python train_knowledge_dynamic_tools.py
```

Run one ablation:

```bash
bash launchers/run_cmasd_ablation_single.sh no_fusion
```

Run the default ablation suite serially or on four GPUs:

```bash
bash launchers/run_cmasd_ablation_all.sh serial
bash launchers/run_cmasd_ablation_all.sh parallel4
```

Supported modes are `full`, `single_modal_numeric`, `no_fusion`, `no_topo`,
`no_wire`, and `action_out_only`.

Each run writes only `loss.csv`, `best.pt`, and `memory.json` under the configured
output root.
