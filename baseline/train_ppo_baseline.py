"""Training entry point for the knowledge-assisted PPO baseline."""

from baseline_suite.training.baseline_runtime import run_training


if __name__ == "__main__":
    run_training("ppo")

