"""Preserved entry point for the skill-enabled deadline sweep."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiment_suite.tdpo_skill.evaluation.deadline_sweep import main


if __name__ == "__main__":
    main()

