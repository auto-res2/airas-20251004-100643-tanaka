#!/usr/bin/env python
"""src/model.py
Model architectures & loss functions (fully implemented).
"""

from __future__ import annotations

import math
from typing import Dict, Union

import torch
import torch.nn.functional as F
from torch import nn

try:
    from transformers import AutoModelForSeq2SeqLM
except ImportError:  # pragma: no cover – handled via optional deps
    AutoModelForSeq2SeqLM = None  # type: ignore

# --------------------------------------------------------------------------------------
# Calibration metric
# --------------------------------------------------------------------------------------

def expected_calibration_error(probs: torch.Tensor, labels: torch.Tensor, num_bins: int = 10) -> torch.Tensor:
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
# Dummy language model (used for smoke tests)
# --------------------------------------------------------------------------------------

class DummyLanguageModel(nn.Module):
    """A tiny LSTM-based LM supporting *any* vocab size."""

    def __init__(self, vocab_size: int, embedding_dim: int = 64, hidden_dim: int = 128, num_layers: int = 1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.proj = nn.Linear(hidden_dim, vocab_size)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0, std=0.02)

    def forward(self, input_ids: torch.Tensor):
        x = self.embed(input_ids)
        out, _ = self.lstm(x)
        return self.proj(out)


# --------------------------------------------------------------------------------------
# HuggingFace Seq2Seq wrapper
# --------------------------------------------------------------------------------------

class Seq2SeqModelWrapper(nn.Module):
    """Thin wrapper around HuggingFace AutoModelForSeq2SeqLM that exposes logits."""

    def __init__(self, pretrained_name: str):
        super().__init__()
        if AutoModelForSeq2SeqLM is None:
            raise ImportError("transformers must be installed to use Seq2Seq models")
        self.model = AutoModelForSeq2SeqLM.from_pretrained(pretrained_name)

    def forward(self, inputs):  # type: ignore[override]
        outputs = self.model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            decoder_input_ids=inputs["decoder_input_ids"],
            use_cache=False,
        )
        return outputs.logits


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
    """Dynamic Proper-Score Mixing (CE ↔ Brier)."""

    def __init__(self, vocab_size: int, warmup_steps: int = 1000, schedule: str = "linear"):
        super().__init__()
        self.vocab_size = vocab_size
        self.warmup_steps = warmup_steps
        self.schedule = schedule.lower()
        self.register_buffer("global_step", torch.tensor(0.0))

    def _alpha(self):
        x = torch.clamp(self.global_step / self.warmup_steps, 0.0, 1.0)
        if self.schedule == "linear":
            return x
        elif self.schedule == "cosine":
            return 0.5 * (1 - torch.cos(math.pi * x))
        else:
            raise ValueError(f"Unknown schedule '{self.schedule}'")

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        self.global_step += 1.0
        ce = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), reduction="none")
        p = logits.view(-1, logits.size(-1)).softmax(-1)
        oh = F.one_hot(targets.view(-1), logits.size(-1)).type_as(p)
        brier = (p - oh).pow(2).sum(-1)
        alpha = self._alpha()
        loss = (1 - alpha) * ce + alpha * brier
        return loss.mean()


# --------------------------------------------------------------------------------------
# Factories (public API)
# --------------------------------------------------------------------------------------

def get_model(cfg: Dict, vocab_size: int) -> nn.Module:
    name = cfg["model"]["name"].lower()
    if name == "dummy":
        return DummyLanguageModel(
            vocab_size=vocab_size,
            embedding_dim=cfg["model"].get("embedding_dim", 64),
            hidden_dim=cfg["model"].get("hidden_dim", 128),
            num_layers=cfg["model"].get("num_layers", 1),
        )
    else:
        # Assume HuggingFace identifier
        return Seq2SeqModelWrapper(pretrained_name=cfg["model"]["name"])


def get_loss_fn(cfg: Dict, vocab_size: int, device: torch.device):
    loss_name = cfg["training"]["loss"].lower()
    if loss_name in {"ce", "cross_entropy", "ce+tempscale", "ce_tempscale"}:
        return CrossEntropyLoss().to(device)
    elif loss_name == "brier":
        return BrierLoss().to(device)
    elif loss_name == "dpsm":
        warmup = cfg["training"].get("warmup_steps", 1000)
        schedule = cfg["training"].get("schedule", "linear")
        return DPSMLoss(vocab_size=vocab_size, warmup_steps=warmup, schedule=schedule).to(device)
    else:
        raise NotImplementedError(f"Loss '{loss_name}' not implemented.")
