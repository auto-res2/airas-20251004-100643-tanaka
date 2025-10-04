#!/usr/bin/env python
"""src/model.py
Model architectures and loss functions for DPSM experiments.
"""

from __future__ import annotations

import math
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch import nn

# HuggingFace – heavy deps are acceptable for the full experiment
from transformers import GPT2Config, GPT2LMHeadModel

# --------------------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------------------

def expected_calibration_error(probs: torch.Tensor, labels: torch.Tensor, num_bins: int = 10) -> torch.Tensor:
    """Vectorised Expected Calibration Error implementation."""

    confidences, predictions = probs.max(dim=1)
    accuracies = predictions.eq(labels)

    ece = torch.zeros(1, device=probs.device)
    bin_boundaries = torch.linspace(0, 1, num_bins + 1, device=probs.device)
    for i in range(num_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        mask = (confidences > lo) & (confidences <= hi)
        if mask.any():
            bin_acc = accuracies[mask].float().mean()
            bin_conf = confidences[mask].mean()
            ece += (mask.float().mean()) * torch.abs(bin_conf - bin_acc)
    return ece


# --------------------------------------------------------------------------------------
# Minimal LSTM (kept for smoke-tests)
# --------------------------------------------------------------------------------------


class DummyLanguageModel(nn.Module):
    """A tiny 1-layer LSTM language model suitable for quick tests."""

    def __init__(self, vocab_size: int, embedding_dim: int = 64, hidden_dim: int = 128, num_layers: int = 1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.proj = nn.Linear(hidden_dim, vocab_size)
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)
            if hasattr(m, "bias") and m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, input_ids: torch.Tensor):  # [B, T]
        x = self.embed(input_ids)
        out, _ = self.lstm(x)
        return self.proj(out)


# --------------------------------------------------------------------------------------
# HuggingFace GPT-2 wrapper
# --------------------------------------------------------------------------------------


class GPT2LMWrapper(nn.Module):
    """Thin wrapper that exposes `.forward -> logits` for training loop."""

    def __init__(self, model_name: str = "gpt2", from_pretrained: bool = False):
        super().__init__()
        if from_pretrained:
            self.inner = GPT2LMHeadModel.from_pretrained(model_name)
        else:
            cfg = GPT2Config.from_pretrained(model_name)
            self.inner = GPT2LMHeadModel(cfg)

    def forward(self, input_ids: torch.Tensor):
        return self.inner(input_ids=input_ids).logits


# --------------------------------------------------------------------------------------
# Loss functions
# --------------------------------------------------------------------------------------

class CrossEntropyLoss(nn.Module):
    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))


class BrierLoss(nn.Module):
    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        p = logits.softmax(-1)
        oh = F.one_hot(targets, logits.size(-1)).type_as(p)
        return (p - oh).pow(2).sum(-1).mean()


class DPSMLoss(nn.Module):
    """Dynamic Proper-Score Mixing between CE and Brier."""

    def __init__(
        self,
        vocab_size: int,
        warmup_steps: int = 1000,
        schedule: str = "linear",
        fixed_alpha: Optional[float] = None,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.warmup_steps = max(1, warmup_steps)
        self.schedule = schedule.lower()
        self.fixed_alpha = fixed_alpha
        self.register_buffer("global_step", torch.tensor(0.0))

    def _alpha(self) -> torch.Tensor:
        if self.fixed_alpha is not None:
            return torch.tensor(self.fixed_alpha, device=self.global_step.device)
        x = torch.clamp(self.global_step / self.warmup_steps, 0.0, 1.0)
        if self.schedule == "linear":
            return x
        elif self.schedule == "cosine":
            return 0.5 * (1 - torch.cos(math.pi * x))
        else:
            raise ValueError(f"Unknown DPSM schedule '{self.schedule}'")

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        self.global_step += 1.0
        ce = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), reduction="none")
        p = logits.softmax(-1)
        oh = F.one_hot(targets, logits.size(-1)).type_as(p)
        brier = (p - oh).pow(2).sum(-1)
        alpha = self._alpha()
        loss = (1 - alpha) * ce + alpha * brier
        return loss.mean()


# --------------------------------------------------------------------------------------
# Factories
# --------------------------------------------------------------------------------------

def get_model(cfg: Dict, vocab_size: int) -> nn.Module:
    name = cfg["model"]["name"].lower()

    if name == "dummy":
        mc = cfg["model"]
        return DummyLanguageModel(
            vocab_size=vocab_size,
            embedding_dim=mc.get("embedding_dim", 64),
            hidden_dim=mc.get("hidden_dim", 128),
            num_layers=mc.get("num_layers", 1),
        )

    if name in {"gpt2-small", "gpt2", "gpt2-medium", "gpt2-large"}:
        from_pretrained = cfg["model"].get("from_pretrained", False)
        return GPT2LMWrapper(model_name=name.replace("-small", ""), from_pretrained=from_pretrained)

    raise NotImplementedError(f"Model '{name}' is not implemented.")


def get_loss_fn(cfg: Dict, vocab_size: int, device: torch.device):
    raw = cfg["training"]["loss"].lower()

    if raw == "ce":
        return CrossEntropyLoss().to(device)
    if raw == "brier":
        return BrierLoss().to(device)

    if raw.startswith("dpsm"):
        # Determine schedule / fixed alpha from string or config
        if "cosine" in raw:
            schedule = "cosine"
        elif "linear" in raw:
            schedule = "linear"
        elif "fixed" in raw:
            schedule = "fixed"
        else:
            schedule = cfg["training"].get("schedule", "linear")

        fixed_alpha = None
        if "fixed" in raw:
            # try to parse trailing alpha e.g. dpsm-fixed-alpha0.5
            if "alpha" in raw:
                try:
                    fixed_alpha = float(raw.split("alpha")[-1])
                except ValueError:
                    pass
            fixed_alpha = fixed_alpha or cfg["training"].get("fixed_alpha", 0.5)

        warmup = int(cfg["training"].get("warmup_steps", 1000))
        return DPSMLoss(vocab_size, warmup_steps=warmup, schedule=schedule, fixed_alpha=fixed_alpha).to(device)

    raise NotImplementedError(f"Loss '{raw}' is not supported." )