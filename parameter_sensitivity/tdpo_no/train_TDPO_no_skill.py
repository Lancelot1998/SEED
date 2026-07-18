"""Preserved entry point for no-skill TDPO training."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiment_suite.tdpo_no_skill.trainer import main


if __name__ == "__main__":
    main()

