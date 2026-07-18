"""Preserved import name for the no-skill TDPO policy."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiment_suite.tdpo_no_skill import decision as _implementation

sys.modules[__name__] = _implementation

