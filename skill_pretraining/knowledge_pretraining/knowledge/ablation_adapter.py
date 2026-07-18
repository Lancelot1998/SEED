from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from knowledge_pretraining.knowledge import base_adapter as base
from knowledge_pretraining.knowledge.base_adapter import *  # noqa: F403 - preserve the public interface


ABLATION_MODE_ALIASES = {
    "full": "full",
    "cmasd_full": "full",
    "single_modal": "single_modal_numeric",
    "single_modal_numeric": "single_modal_numeric",
    "numeric_only": "single_modal_numeric",
    "cmasd_single_modal": "single_modal_numeric",
    "no_fusion": "no_fusion",
    "cmasd_no_fusion": "no_fusion",
    "no_topo": "no_topo",
    "no_topology": "no_topo",
    "cmasd_no_topo": "no_topo",
    "no_wire": "no_wire",
    "no_wireless": "no_wire",
    "cmasd_no_wire": "no_wire",
    "action_out_only": "action_out_only",
    "action_result_only": "action_out_only",
    "tool_out_only": "action_out_only",
    "cmasd_action_out_only": "action_out_only",
}

VALID_ABLATION_MODES = tuple(sorted(set(ABLATION_MODE_ALIASES.values())))
ASPECT_ORDER = ("topo", "wire", "tool", "out")


@dataclass
class KnowledgeConfig(base.KnowledgeConfig):
    ablation_mode: str = "full"
    use_text_modality: bool = True
    use_cross_modal_fusion: bool = True
    active_aspects: Tuple[str, ...] = ASPECT_ORDER
    zero_g_feat: bool = False
    zero_s_feat: bool = False
    zero_a_feat: bool = False
    zero_o_feat: bool = False
    strict_ablation: bool = True


def normalize_ablation_mode(mode: str) -> str:
    key = str(mode or "full").strip().lower().replace("-", "_").replace(" ", "_")
    if key not in ABLATION_MODE_ALIASES:
        raise ValueError(f"Unsupported CMASD ablation mode: {mode!r}. Valid modes: {VALID_ABLATION_MODES}")
    return ABLATION_MODE_ALIASES[key]


def apply_ablation_to_cfg(cfg: KnowledgeConfig, mode: Optional[str] = None) -> KnowledgeConfig:
    mode = normalize_ablation_mode(mode if mode is not None else getattr(cfg, "ablation_mode", "full"))
    cfg.ablation_mode = mode
    cfg.use_text_modality = True
    cfg.use_cross_modal_fusion = True
    cfg.active_aspects = ASPECT_ORDER
    cfg.zero_g_feat = False
    cfg.zero_s_feat = False
    cfg.zero_a_feat = False
    cfg.zero_o_feat = False

    if not hasattr(cfg, "loss_topo_weight"):
        cfg.loss_topo_weight = 1.0
    if not hasattr(cfg, "loss_wire_weight"):
        cfg.loss_wire_weight = 1.0
    if not hasattr(cfg, "loss_tool_weight"):
        cfg.loss_tool_weight = 3.0
    if not hasattr(cfg, "loss_out_weight"):
        cfg.loss_out_weight = 1.0
    if not hasattr(cfg, "loss_need_weight"):
        cfg.loss_need_weight = 1.0

    if mode == "full":
        return cfg
    if mode == "single_modal_numeric":
        cfg.use_text_modality = False
        cfg.use_cross_modal_fusion = False
        return cfg
    if mode == "no_fusion":
        cfg.use_text_modality = True
        cfg.use_cross_modal_fusion = False
        return cfg
    if mode == "no_topo":
        cfg.active_aspects = ("wire", "tool", "out")
        cfg.loss_topo_weight = 0.0
        cfg.zero_g_feat = True
        return cfg
    if mode == "no_wire":
        cfg.active_aspects = ("topo", "tool", "out")
        cfg.loss_wire_weight = 0.0
        cfg.zero_s_feat = True
        return cfg
    if mode == "action_out_only":
        cfg.active_aspects = ("tool", "out")
        cfg.loss_topo_weight = 0.0
        cfg.loss_wire_weight = 0.0
        cfg.zero_g_feat = True
        cfg.zero_s_feat = True
        return cfg
    raise ValueError(f"Unsupported CMASD ablation mode after normalization: {mode!r}")


def _zero_or_original(x: torch.Tensor, enabled: bool) -> torch.Tensor:
    return torch.zeros_like(x) if bool(enabled) else x


class AblationLLMFourHeadDistiller(base.LLMFourHeadDistiller):
    def __init__(self, llm_hidden_size: int, action_dim: int, cfg: KnowledgeConfig) -> None:
        apply_ablation_to_cfg(cfg)
        super().__init__(llm_hidden_size=llm_hidden_size, action_dim=action_dim, cfg=cfg)
        self.aspect_query = nn.ParameterDict({
            name: nn.Parameter(torch.zeros(cfg.d_text_proj)) for name in ASPECT_ORDER
        })
        self.no_fusion_text_value = nn.ModuleDict({
            name: nn.Linear(cfg.d_text_proj, cfg.d_v) for name in ASPECT_ORDER
        })
        for p in self.aspect_query.values():
            nn.init.normal_(p, mean=0.0, std=0.02)

    def _project_text_or_query(self, batch: Dict[str, torch.Tensor], aspect: str, batch_size: int, device: torch.device) -> torch.Tensor:
        active = set(getattr(self.cfg, "active_aspects", ASPECT_ORDER))
        if aspect not in active:
            return torch.zeros((batch_size, int(self.cfg.d_text_proj)), dtype=torch.float32, device=device)
        if not bool(getattr(self.cfg, "use_text_modality", True)):
            return self.aspect_query[aspect].unsqueeze(0).expand(batch_size, -1)
        q_raw = batch.get(f"q_{aspect}")
        if q_raw is None:
            raise KeyError(f"Missing q_{aspect} in batch for ablation mode {getattr(self.cfg, 'ablation_mode', 'full')}")
        return self.text_proj(q_raw)

    def _masked_mean(self, x: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mask_f = mask.to(dtype=x.dtype)
        denom = mask_f.sum(dim=-1, keepdim=True).clamp_min(1.0)
        z = (x * mask_f.unsqueeze(-1)).sum(dim=1) / denom
        alpha = mask_f / denom
        return z, alpha

    def _fixed_numeric_mix(self, g: torch.Tensor, s: torch.Tensor, a: torch.Tensor, o: torch.Tensor, head_name: str) -> Tuple[torch.Tensor, torch.Tensor]:
        vg = self.block_value["g"](g)
        vs = self.block_value["s"](s)
        va = self.block_value["a"](a)
        vo = self.block_value["o"](o)
        stacked = torch.stack([vg, vs, va, vo], dim=2)
        prior = self.block_priors[head_name].to(stacked.device).to(stacked.dtype)
        beta = prior.unsqueeze(0).expand(stacked.size(0), -1)
        mixed = (stacked * beta[:, None, :, None]).sum(dim=2)
        return mixed, beta

    def _head_forward(
        self,
        g: torch.Tensor,
        s: torch.Tensor,
        a: torch.Tensor,
        o: torch.Tensor,
        q_text: torch.Tensor,
        mask: torch.Tensor,
        head_name: str,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if bool(getattr(self.cfg, "use_cross_modal_fusion", True)):
            mix, beta = self._compose_steps(g, s, a, o, q_text, head_name)
            z, alpha = self._attend(q_text, mix, mask, head_name)
            return z, alpha, beta

        mix, beta = self._fixed_numeric_mix(g, s, a, o, head_name)
        z_num, alpha = self._masked_mean(mix, mask)
        if bool(getattr(self.cfg, "use_text_modality", True)):
            z_text = self.no_fusion_text_value[head_name](q_text)
            z = 0.5 * z_num + 0.5 * z_text
        else:
            z = z_num
        return z, alpha, beta

    def _apply_aspect_mask(self, z_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        active = set(getattr(self.cfg, "active_aspects", ASPECT_ORDER))
        if not active:
            raise ValueError(f"No active aspects are left in ablation mode {getattr(self.cfg, 'ablation_mode', 'unknown')}")
        out: Dict[str, torch.Tensor] = {}
        for name, z in z_dict.items():
            out[name] = z if name in active else torch.zeros_like(z)
        return out

    def _masked_head_gate(self, z_all: torch.Tensor) -> torch.Tensor:
        active = set(getattr(self.cfg, "active_aspects", ASPECT_ORDER))
        if not active:
            raise ValueError("active_aspects cannot be empty")
        logits = self.gate_fuse(z_all)
        aspect_mask = torch.tensor([name in active for name in ASPECT_ORDER], device=logits.device, dtype=torch.bool).unsqueeze(0)
        logits = logits.masked_fill(~aspect_mask, -1e9)
        return torch.softmax(logits, dim=-1)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        B = int(batch["g_feat"].shape[0])
        device = batch["g_feat"].device
        q_topo = self._project_text_or_query(batch, "topo", B, device)
        q_wire = self._project_text_or_query(batch, "wire", B, device)
        q_tool = self._project_text_or_query(batch, "tool", B, device)
        q_out = self._project_text_or_query(batch, "out", B, device)

        g = self.phi_g(_zero_or_original(batch["g_feat"], getattr(self.cfg, "zero_g_feat", False)))
        s = self.phi_s(_zero_or_original(batch["s_feat"], getattr(self.cfg, "zero_s_feat", False)))
        a = self.phi_a(_zero_or_original(batch["a_feat"], getattr(self.cfg, "zero_a_feat", False)))
        o = self.phi_o(_zero_or_original(batch["o_feat"], getattr(self.cfg, "zero_o_feat", False)))
        mask = batch["mask"]

        z_topo, alpha_topo, beta_topo = self._head_forward(g, s, a, o, q_topo, mask, "topo")
        z_wire, alpha_wire, beta_wire = self._head_forward(g, s, a, o, q_wire, mask, "wire")
        z_tool, alpha_tool, beta_tool = self._head_forward(g, s, a, o, q_tool, mask, "tool")
        z_out, alpha_out, beta_out = self._head_forward(g, s, a, o, q_out, batch.get("out_mask", mask), "out")

        z_masked = self._apply_aspect_mask({"topo": z_topo, "wire": z_wire, "tool": z_tool, "out": z_out})
        z_topo = z_masked["topo"]
        z_wire = z_masked["wire"]
        z_tool = z_masked["tool"]
        z_out = z_masked["out"]

        z_all = torch.cat([z_topo, z_wire, z_tool, z_out], dim=-1)
        head_gate = self._masked_head_gate(z_all)
        gated = torch.cat([
            head_gate[:, 0:1] * z_topo,
            head_gate[:, 1:2] * z_wire,
            head_gate[:, 2:3] * z_tool,
            head_gate[:, 3:4] * z_out,
        ], dim=-1)
        h_atom = self.fuse(gated)
        return {
            "z_topo": z_topo,
            "z_wire": z_wire,
            "z_tool": z_tool,
            "z_out": z_out,
            "alpha_topo": alpha_topo,
            "alpha_wire": alpha_wire,
            "alpha_tool": alpha_tool,
            "alpha_out": alpha_out,
            "beta_topo": beta_topo,
            "beta_wire": beta_wire,
            "beta_tool": beta_tool,
            "beta_out": beta_out,
            "head_gate": head_gate,
            "h_atom": h_atom,
        }


class KnowledgeTrainer(base.KnowledgeTrainer):
    def __init__(self, dataset: Dict[str, object], action_dim: int, cfg: Optional[KnowledgeConfig] = None) -> None:
        if cfg is None:
            cfg = KnowledgeConfig()
        elif not isinstance(cfg, KnowledgeConfig):
            raw = {f.name: getattr(cfg, f.name) for f in fields(base.KnowledgeConfig) if hasattr(cfg, f.name)}
            cfg = KnowledgeConfig(**raw)
        apply_ablation_to_cfg(cfg)
        old_distiller = base.LLMFourHeadDistiller
        base.LLMFourHeadDistiller = AblationLLMFourHeadDistiller
        try:
            super().__init__(dataset=dataset, action_dim=action_dim, cfg=cfg)
        finally:
            base.LLMFourHeadDistiller = old_distiller
        if self.cfg.verbose:
            print(
                f"{self.cfg.print_prefix} ablation mode | mode={self.cfg.ablation_mode} | "
                f"text={bool(self.cfg.use_text_modality)} | fusion={bool(self.cfg.use_cross_modal_fusion)} | "
                f"active_aspects={tuple(self.cfg.active_aspects)} | zero_g={bool(self.cfg.zero_g_feat)} | zero_s={bool(self.cfg.zero_s_feat)}",
                flush=True,
            )

    def _encode_aspect_for_batch(self, segments: List[Dict[str, object]], aspect: str, detach_text: bool) -> torch.Tensor:
        B = len(segments)
        hidden = int(self.llm.hidden_size)
        device = self.device
        active = set(getattr(self.cfg, "active_aspects", ASPECT_ORDER))
        if aspect not in active or not bool(getattr(self.cfg, "use_text_modality", True)):
            return torch.zeros((B, hidden), dtype=torch.float32, device=device)
        if bool(getattr(self.cfg, "fast_update_text_encoder", False)):
            return self._fast_text_embeddings(segments, aspect=aspect)
        return self.llm.encode_texts([seg["texts"][aspect] for seg in segments], aspect=aspect, detach=detach_text).to(device)

    def _build_batch(self, segments: List[Dict[str, object]], detach_text: bool = False) -> Dict[str, torch.Tensor]:
        max_len = max(len(seg["steps"]) for seg in segments)
        batch_size = len(segments)

        def _pad(block_name: str, dim: int) -> torch.Tensor:
            out = torch.zeros((batch_size, max_len, dim), dtype=torch.float32)
            for i, seg in enumerate(segments):
                for t, step in enumerate(seg["steps"]):
                    out[i, t] = torch.tensor(step[block_name], dtype=torch.float32)
            return out

        g_feat = _pad("g_feat", 6).to(self.device)
        s_feat = _pad("s_feat", 6).to(self.device)
        a_feat = _pad("a_feat", 6).to(self.device)
        o_feat = _pad("o_feat", 4).to(self.device)
        y = torch.zeros((batch_size, max_len, 4), dtype=torch.float32, device=self.device)
        mask = torch.zeros((batch_size, max_len), dtype=torch.bool, device=self.device)
        out_mask = torch.zeros((batch_size, max_len), dtype=torch.bool, device=self.device)
        for i, seg in enumerate(segments):
            for t, step in enumerate(seg["steps"]):
                y[i, t] = torch.tensor(step["y"], dtype=torch.float32, device=self.device)
                mask[i, t] = True
                out_mask[i, t] = bool(step.get("is_decision_event", False))
            if not bool(out_mask[i].any().item()):
                out_mask[i] = mask[i]

        topo_emb = self._encode_aspect_for_batch(segments, "topo", detach_text=detach_text)
        wire_emb = self._encode_aspect_for_batch(segments, "wire", detach_text=detach_text)
        tool_emb = self._encode_aspect_for_batch(segments, "tool", detach_text=detach_text)
        out_emb = self._encode_aspect_for_batch(segments, "out", detach_text=detach_text)

        batch = {
            "g_feat": g_feat,
            "s_feat": s_feat,
            "a_feat": a_feat,
            "o_feat": o_feat,
            "mask": mask,
            "out_mask": out_mask,
            "y": y,
            "q_topo": topo_emb,
            "q_wire": wire_emb,
            "q_tool": tool_emb,
            "q_out": out_emb,
            "depth_t": torch.tensor([seg["depth_tier"] for seg in segments], dtype=torch.long, device=self.device),
            "role_t": torch.tensor([seg["role_id"] for seg in segments], dtype=torch.long, device=self.device),
            "critical_t": torch.tensor([seg["critical_flag"] for seg in segments], dtype=torch.float32, device=self.device),
            "queue_t": torch.tensor([seg["queue_class"] for seg in segments], dtype=torch.long, device=self.device),
            "slack_t": torch.tensor([seg["slack_class"] for seg in segments], dtype=torch.long, device=self.device),
            "violation_t": torch.tensor([seg["violation"] for seg in segments], dtype=torch.float32, device=self.device),
            "tool_gain_t": torch.tensor([seg["action_gain"] for seg in segments], dtype=torch.float32, device=self.device),
            "tool_gain_mask_t": torch.tensor([seg.get("action_observed_mask", [1.0] * self.action_dim) for seg in segments], dtype=torch.float32, device=self.device),
            "out_t": torch.tensor(
                [[float(seg["node_success"]), float(seg["task_success"]), float(seg["violation"])] for seg in segments],
                dtype=torch.float32,
                device=self.device,
            ),
            "need_t": torch.tensor([float(seg["tool_invoked"] > 0) for seg in segments], dtype=torch.float32, device=self.device),
            "risk_t": torch.tensor([float(seg["no_tool_risk_prior"]) for seg in segments], dtype=torch.float32, device=self.device),
            "confidence_t": torch.tensor([float(seg["confidence"]) for seg in segments], dtype=torch.float32, device=self.device),
        }
        return batch

    @classmethod
    def load(cls, dataset: Dict[str, object], model_path: str, memory_path: Optional[str] = None, action_dim: Optional[int] = None) -> "KnowledgeTrainer":
        ckpt = torch.load(model_path, map_location="cpu")
        raw_cfg = dict(ckpt.get("cfg", {}))
        valid = {f.name for f in fields(KnowledgeConfig)}
        cfg_kwargs = {k: v for k, v in raw_cfg.items() if k in valid}
        cfg = KnowledgeConfig(**cfg_kwargs)
        apply_ablation_to_cfg(cfg)
        trainer = cls(dataset=dataset, action_dim=int(action_dim or ckpt["action_dim"]), cfg=cfg)
        trainer.distiller.load_state_dict(ckpt["distiller_state_dict"], strict=True)
        trainer.llm.load_adapter_state_dict(ckpt.get("llm_adapter_state_dict", {}), strict=False)
        if memory_path is not None and Path(memory_path).exists():
            data = json.loads(Path(memory_path).read_text(encoding="utf-8"))
            trainer.memory.load_json(data)
            trainer.last_snapshot["prototype_risk_mean"] = float(
                sum(p.no_tool_risk_mean for p in trainer.memory.prototypes) / max(1, len(trainer.memory.prototypes))
            )
            trainer.last_snapshot["prototype_count"] = float(len(trainer.memory.prototypes))
        return trainer
