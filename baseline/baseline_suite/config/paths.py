"""Centralized input and output paths.

Edit only this module when datasets, models, memories, tool libraries, or output
locations move. Every value intentionally preserves the corresponding original
path. Relative training outputs remain relative to the process working directory.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Training inputs: generated graph dataset and accepted file names.
DATA_DIR = Path("/models/Qwen/lyq_jsac_2026_03_17/new_code/new_llm/newcode0411/deepseek_intent_dag_dataset_500")
DATASET_FILE_PATTERNS = ("dataset_*.json", "sample_*.json")
DATASET_METADATA_FILES = ("pool_metadata.json", "dataset_pool_metadata.json")

# Training inputs: Qwen model directories, checked in order.
QWEN_MODEL_PATH = "/models/Qwen"
QWEN_MODEL_PATH_FALLBACK = "/model/Qwen"

# Training inputs: pretrained knowledge weights and memory.
PRETRAINED_KNOWLEDGE_MODEL_PATH = Path(
    "/models/Qwen/lyq_jsac_2026_03_17/new_code/new_llm/newcode0411/new_0419/"
    "new_0509_skill/knowledge_dynamic_tool_runs_c/"
    "knowledge_dynamic_tools_heuristic_20260509_171515/best.pt"
)
OPTIONAL_PRETRAINED_KNOWLEDGE_MEMORY_PATH = Path(
    "/models/Qwen/lyq_jsac_2026_03_17/new_code/new_llm/newcode0411/new_0419/"
    "new_0509_skill/knowledge_dynamic_tool_runs_c/"
    "knowledge_dynamic_tools_heuristic_20260509_171515/memory.json"
)
EXTERNAL_MEMORY_LIBRARY_PATH = Path(
    "/models/Qwen/lyq_jsac_2026_03_17/new_code/new_llm/newcode0411/new_0419/"
    "new_0503_yes/knowledge_dynamic_tool_runs_c/"
    "knowledge_dynamic_tools_heuristic_20260504_131050/memory.json"
)

# Training inputs: dynamic tool libraries. Ordering is preserved for merging.
TOOL_LIBRARY_PATHS = [
    Path("/models/Qwen/lyq_jsac_2026_03_17/new_code/new_llm/newcode0411/new_0419/new_0425_api/tool/api_tool_0_500/subset_api_schema_retry_20260427_141944/tool_library.json"),
    Path("/models/Qwen/lyq_jsac_2026_03_17/new_code/new_llm/newcode0411/new_0419/new_0425_api/tool/api_tool_500_999/subset_api_schema_retry_20260427_141947/tool_library.json"),
    Path("/models/Qwen/lyq_jsac_2026_03_17/new_code/new_llm/newcode0411/new_0419/new_0425_api/tool_new/api_tool_1000_1499/subset_api_schema_retry_20260427_164323/tool_library.json"),
    Path("/models/Qwen/lyq_jsac_2026_03_17/new_code/new_llm/newcode0411/new_0419/new_0425_api/tool/api_tool_1500_1999/subset_api_schema_retry_20260427_141953/tool_library.json"),
]
LOCAL_TOOL_LIBRARY_PATH = PROJECT_ROOT / "tool_library.json"

# Training output root. This retains the original relative-path behavior.
OUTPUT_ROOT = Path("./baseline_decision_runs")
DEFAULT_POLICY_OFFLOAD_PATH = Path("./baseline_offload")

# Analysis inputs: five existing epoch-metric files.
BAR_CSV_CONFIGS = [
    {"method": "Ours", "path": "/models/Qwen/lyq_jsac_2026_03_17/new_code/new_llm/newcode0411/new_0419/new_0509_all/llm_tdpo_skill_fast1/tdpo_decision_runs_batch_fast_skill/qwen7b_tdpo_with_skill_memory_20260511_140148/epoch_metrics.csv"},
    {"method": "DPO", "path": "/models/Qwen/lyq_jsac_2026_03_17/new_code/new_llm/newcode0411/new_0419/new_0513/baseline_decision_runs/qwen7b_knowledge_dpo_baseline_20260513_183424/epoch_metrics.csv"},
    {"method": "PPO", "path": "/models/Qwen/lyq_jsac_2026_03_17/new_code/new_llm/newcode0411/new_0419/new_0513/baseline_decision_runs/qwen7b_knowledge_ppo_baseline_20260513_181406/epoch_metrics.csv"},
    {"method": "SFT", "path": "/models/Qwen/lyq_jsac_2026_03_17/new_code/new_llm/newcode0411/new_0419/new_0513/baseline_decision_runs/qwen7b_knowledge_sft_baseline_20260513_185126/epoch_metrics.csv"},
    {"method": "D3QN", "path": "/models/Qwen/lyq_jsac_2026_03_17/new_code/new_llm/newcode0411/new_0419/new_0513/baseline_decision_runs/qwen7b_knowledge_d3qn_baseline_20260515_000438/epoch_metrics.csv"},
]
BAR_OUTPUT_DIR = PROJECT_ROOT / "bar_outputs"
BAR_LAST20_OUTPUT_DIR = PROJECT_ROOT / "bar_outputs_5"
BAR_SUMMARY_WIDE_CSV = "bar_summary_for_mlx.csv"
BAR_SUMMARY_LONG_CSV = "bar_summary_long_for_mlx.csv"
BAR_MATLAB_ARRAYS_FILE = "bar_arrays_for_mlx.m"
BAR_PYTHON_FIG_PNG = "python_four_panel_bar.png"
BAR_PYTHON_FIG_PDF = "python_four_panel_bar.pdf"
BAR_TRUNCATED_DIR = "last20_epoch_csv"
BAR_TRUNCATED_SUFFIX = "epoch_metrics.csv"
BAR_TRUNCATED_MANIFEST = "truncated_last20_manifest.csv"


@dataclass(frozen=True)
class RunArtifacts:
    """Build every file path produced inside one timestamped training run."""

    root: Path

    @property
    def subdirectories(self) -> tuple[Path, ...]:
        return tuple(self.root / name for name in ("ckpts", "records", "pairs", "traces", "knowledge"))

    @property
    def d3qn_subdirectories(self) -> tuple[Path, ...]:
        return self.subdirectories + (self.root / "replay",)

    dataset_scan = property(lambda self: self.root / "dataset_scan.json")
    run_config = property(lambda self: self.root / "run_config.json")
    knowledge_release = property(lambda self: self.root / "knowledge" / "release_before_policy.json")
    knowledge_metrics = property(lambda self: self.root / "knowledge_metrics.csv")
    best_checkpoint = property(lambda self: self.root / "ckpts" / "best.pt")
    traces = property(lambda self: self.root / "traces" / "traces.jsonl")
    dpo_pairs = property(lambda self: self.root / "pairs" / "pairs.jsonl")
    sft_examples = property(lambda self: self.root / "pairs" / "sft_examples.jsonl")
    d3qn_transitions = property(lambda self: self.root / "replay" / "transitions.jsonl")
    prompt_samples = property(lambda self: self.root / "prompt_samples_epoch_001.json")
    epoch_metrics = property(lambda self: self.root / "epoch_metrics.csv")
    eval_metrics = property(lambda self: self.root / "eval_metrics.csv")
    energy_metrics = property(lambda self: self.root / "energy.csv")
    delay_metrics = property(lambda self: self.root / "delay.csv")
    latest_summary = property(lambda self: self.root / "latest_summary.json")
    final_result = property(lambda self: self.root / "final_result.json")
    policy_offload = property(lambda self: self.root / "baseline_offload")
    pretrained_knowledge_copy = property(lambda self: self.root / "knowledge" / "pretrained_input_best.pt")
    loaded_knowledge_info = property(lambda self: self.root / "knowledge" / "loaded_knowledge_info.json")

    def knowledge_update(self, epoch: int) -> Path:
        return self.root / "knowledge" / f"update_stats_epoch_{epoch:03d}.json"

    def policy_checkpoint(self, algorithm: str, epoch: int) -> Path:
        return self.root / "ckpts" / f"{algorithm}_policy_epoch_{epoch:03d}.pt"

    def task_records(self, epoch: int) -> Path:
        return self.root / "records" / f"task_records_epoch_{epoch:03d}.json"

    def knowledge_model(self, tag: str) -> Path:
        return self.root / "knowledge" / f"{tag}.pt"

    def knowledge_memory(self, tag: str) -> Path:
        return self.root / "knowledge" / f"{tag}_memory.json"
