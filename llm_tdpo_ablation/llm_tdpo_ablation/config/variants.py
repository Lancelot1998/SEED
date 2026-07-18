"""Human-readable registry of the three preserved ablation variants."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Tuple


@dataclass(frozen=True)
class AblationVariant:
    name: str
    policy_module: str
    training_module: str
    uses_skill_memory: bool
    performs_tdpo_update: bool


VARIANTS: Final[Tuple[AblationVariant, ...]] = (
    AblationVariant(
        name="tdpo_no_skill",
        policy_module="llm_tdpo_ablation.policies.no_skill_tdpo",
        training_module="llm_tdpo_ablation.training.no_skill_runner",
        uses_skill_memory=False,
        performs_tdpo_update=True,
    ),
    AblationVariant(
        name="blank_llm_direct",
        policy_module="llm_tdpo_ablation.policies.blank_llm_direct",
        training_module="llm_tdpo_ablation.training.blank_llm_direct_runner",
        uses_skill_memory=False,
        performs_tdpo_update=False,
    ),
    AblationVariant(
        name="blank_llm_skill_memory",
        policy_module="llm_tdpo_ablation.policies.blank_llm_skill_memory",
        training_module="llm_tdpo_ablation.training.blank_llm_skill_memory_runner",
        uses_skill_memory=True,
        performs_tdpo_update=False,
    ),
)
