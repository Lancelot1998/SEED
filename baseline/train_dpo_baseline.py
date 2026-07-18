"""Training entry point for the knowledge-assisted DPO baseline."""

from baseline_suite.training.baseline_runtime import run_training


if __name__ == "__main__":
    run_training("dpo")

