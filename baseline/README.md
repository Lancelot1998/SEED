# Knowledge-Assisted Baseline Suite

This project trains four comparison baselines (`SFT`, `DPO`, `PPO`, and `D3QN`)
in the same task environment with the same Qwen encoder and optional pretrained
knowledge/memory inputs. It also aggregates five experiment CSV files into
MATLAB/MLX tables and four-panel bar charts.

## Project layout

```text
baseline_suite/
  config/paths.py              # All filesystem inputs and outputs
  config/settings.py           # Behavioral and training settings
  environment/task_env.py      # DAG task simulation
  knowledge/adapter.py         # Knowledge model and prototype memory
  training/baseline_runtime.py # Shared SFT/DPO/PPO runtime
  training/d3qn_runner.py      # Dueling Double-DQN implementation
  analysis/bar_report.py       # Shared CSV aggregation and plotting
train_*_baseline.py            # Preserved training entry points
make_mlx_bar_from_5_csv*.py    # Preserved analysis entry points
```

## Configuration

Edit `baseline_suite/config/paths.py` to change datasets, Qwen locations,
knowledge checkpoints, memory libraries, tool libraries, run outputs, analysis
inputs, or analysis outputs. All current values exactly match the original code.

Edit `baseline_suite/config/settings.py` for sampling, environment, optimizer,
GPU, policy, knowledge-update, memory, and logging behavior.

## Installation

```bash
pip install -r requirements.txt
```

## Training

```bash
python train_sft_baseline.py
python train_dpo_baseline.py
python train_ppo_baseline.py
python train_d3qn_baseline.py
```

Training outputs remain under `./baseline_decision_runs`, relative to the working
directory used to launch the command.

## Analysis

```bash
python make_mlx_bar_from_5_csv.py
python make_mlx_bar_from_5_csv_with_last20.py
```

The first command retains the original last-10 aggregation and `bar_outputs`
directory. The second retains the original last-20 aggregation, truncated CSV
exports, and `bar_outputs_5` directory.

