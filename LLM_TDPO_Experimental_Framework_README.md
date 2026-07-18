# LLM-based Skill Learning and TDPO Experimental Framework

## 1. Overview

This repository contains the complete experimental framework for
LLM-based task decision making with skill learning, knowledge
enhancement, TDPO optimization, baseline comparison, ablation studies,
parameter sensitivity analysis, and skill pretraining.

The framework includes five major components:

1.  Skill pretraining and knowledge learning.
2.  Skill-based TDPO optimization.
3.  No-skill TDPO ablation.
4.  Frozen LLM and traditional baseline comparison.
5.  Parameter sensitivity experiments.

The implementation is organized as independent modules so that each
experiment can be reproduced separately.

------------------------------------------------------------------------

# 2. Repository Structure

    new_floder_up/

    ├── skill_pretraining/
    │   └── Knowledge and skill pretraining experiments
    │
    ├── llm_tdpo_skill/
    │   └── Main Skill-enabled TDPO framework
    │
    ├── llm_tdpo_ablation/
    │   └── Ablation experiments:
    │       - No-skill TDPO
    │       - Frozen LLM direct generation
    │       - Frozen LLM with skill memory
    │
    ├── parameter_sensitivity/
    │   └── Deadline, time, and task-number sensitivity experiments
    │
    └── baseline/
        └── SFT, DPO, PPO, and D3QN baseline implementations

------------------------------------------------------------------------

# 3. Dataset

The dataset is not included in this repository.

Users should download the required dataset separately and configure the
dataset path before running experiments.

## Dataset Download

[DATASET DOWNLOAD URL]

------------------------------------------------------------------------

## Dataset Directory

After downloading, organize the dataset as follows:

    dataset_root/

    ├── task_graphs/
    │
    ├── dag_dataset/
    │
    ├── metadata/
    │
    └── other required files

The exact structure depends on the provided dataset version.

------------------------------------------------------------------------

## Dataset Configuration

Dataset paths are controlled by configuration files.

Modify the corresponding path files:

    skill_pretraining/knowledge_pretraining/config/

    llm_tdpo_skill/llm_tdpo/config/

    llm_tdpo_ablation/llm_tdpo_ablation/config/

    parameter_sensitivity/experiment_suite/config/

Example:

``` python
DATASET_PATH = "YOUR_DATASET_PATH"
```

------------------------------------------------------------------------

# 4. Pretrained Model

The framework uses a pretrained Qwen-based large language model.

Configure the model path in the corresponding configuration files.

Example:

``` python
MODEL_PATH = "YOUR_QWEN_MODEL_PATH"
```

Required models:

-   Qwen backbone model.
-   Knowledge adapter checkpoint.
-   Skill memory checkpoint.

------------------------------------------------------------------------

# 5. Skill Pretraining Module

Directory:

    skill_pretraining/

Purpose:

-   Train knowledge representations.
-   Generate skill-related knowledge.
-   Produce checkpoints used by later TDPO experiments.

Main entry files:

## Knowledge training

``` bash
python train_knowledge_dynamic_tools.py
```

## Knowledge ablation training

``` bash
python train_knowledge_dynamic_tools_ablation.py
```

Outputs include:

    checkpoints/

    knowledge_adapter.pt

    memory files

    training logs

    metrics

------------------------------------------------------------------------

# 6. Skill-enabled TDPO

Directory:

    llm_tdpo_skill/

Purpose:

Evaluate the proposed skill-memory-enhanced TDPO decision framework.

Main training entry:

``` bash
python train_fast_mem.py
```

The framework includes:

-   Task environment.
-   Skill memory.
-   Knowledge adapter.
-   Tool library.
-   TDPO optimization.

Typical outputs:

    outputs/

    ├── checkpoints/
    ├── metrics.csv
    ├── task_records.json
    ├── memory/
    └── logs/

------------------------------------------------------------------------

# 7. TDPO Ablation Experiments

Directory:

    llm_tdpo_ablation/

Contains:

## 7.1 Frozen LLM Direct Generation

Entry:

``` bash
python train_blank_llm_direct.py
```

Purpose:

Evaluate the direct decision ability of a frozen LLM without skill
memory.

------------------------------------------------------------------------

## 7.2 Frozen LLM + Skill Memory

Entry:

``` bash
python train_blank_llm_skill_memory.py
```

Purpose:

Evaluate the contribution of skill memory.

------------------------------------------------------------------------

## 7.3 No-skill TDPO

Entry:

``` bash
python train_TDPO_no_skill.py
```

Purpose:

Remove skill memory and compare TDPO performance.

Outputs:

    checkpoints/

    metrics.csv

    task_records.json

    evaluation results

------------------------------------------------------------------------

# 8. Parameter Sensitivity Experiments

Directory:

    parameter_sensitivity/

This module evaluates the influence of different environment parameters.

Supported experiments:

-   Deadline sensitivity.
-   Time sensitivity.
-   Task number sensitivity.

------------------------------------------------------------------------

## Knowledge sensitivity

``` bash
python train_knowledge_dynamic_tools_bind_sep_csv.py

python train_knowledge_dynamic_tools_sweep_dead.py

python train_knowledge_dynamic_tools_sweep_time.py
```

------------------------------------------------------------------------

## Skill TDPO sensitivity

Examples:

``` bash
python tdpo/eval_sweep_dead.py

python tdpo/eval_sweep_time.py
```

------------------------------------------------------------------------

## No-skill TDPO sensitivity

Examples:

``` bash
python tdpo_no/eval_tdpo_no_sweep_dead.py

python tdpo_no/eval_tdpo_no_sweep_nume.py
```

------------------------------------------------------------------------

# 9. Baseline Experiments

Directory:

    baseline/

Implemented baselines:

-   Supervised Fine-tuning (SFT)
-   Direct Preference Optimization (DPO)
-   Proximal Policy Optimization (PPO)
-   Dueling Double Deep Q Network (D3QN)

Training commands:

``` bash
python train_sft_baseline.py

python train_dpo_baseline.py

python train_ppo_baseline.py

python train_d3qn_baseline.py
```

------------------------------------------------------------------------

# 10. Tool Library

The framework requires external tool libraries.

Configure:

    tool_library_path

in the corresponding configuration files.

Example:

``` python
TOOL_LIBRARY_PATH = "YOUR_TOOL_LIBRARY_PATH"
```

------------------------------------------------------------------------

# 11. Output Management

All experiments generate:

-   Model checkpoints.
-   Training metrics.
-   Task execution records.
-   Evaluation results.

Output paths are controlled through configuration files.

Before running experiments, update:

    paths.py

instead of modifying training scripts directly.

------------------------------------------------------------------------

# 12. Environment Requirements

Recommended environment:

    Python >= 3.10

    PyTorch >= 2.0

    CUDA >= 11.8

Install dependencies:

``` bash
pip install -r requirements.txt
```

Each sub-module provides its own requirements file.

------------------------------------------------------------------------

# 13. Recommended Experiment Workflow

A complete experiment pipeline:

    Step 1
    Dataset preparation

            ↓

    Step 2
    Skill pretraining

            ↓

    Step 3
    Skill-enabled TDPO training

            ↓

    Step 4
    Ablation experiments

            ↓

    Step 5
    Baseline comparison

            ↓

    Step 6
    Parameter sensitivity analysis

------------------------------------------------------------------------

# 14. Reproducibility Notes

For fair comparison:

-   Keep the same dataset.
-   Keep the same environment parameters.
-   Keep the same tool libraries.
-   Keep the same model checkpoint.
-   Record random seeds.
-   Do not modify output directories during comparison.

------------------------------------------------------------------------

# 15. Citation and Usage

If this framework is used in academic research, please cite the
corresponding paper or acknowledge the authors.
