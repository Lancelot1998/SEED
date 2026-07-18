"""Preserved entry point for the same-environment time sweep."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiment_suite.tdpo_skill.evaluation.time_same_environment import main


if __name__ == "__main__":
    main()

