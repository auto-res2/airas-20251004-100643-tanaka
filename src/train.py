#!/usr/bin/env python
"""src/train.py
Runs a single experiment variation.  
This script is *invoked as a subprocess* by src/main.py so that every run has an
isolated Python interpreter and clean GPU memory.  The CLI is intentionally
minimal – **all run-specific information is provided through an on-disk YAML
file** produced by main.py.

Standard-output protocol (MUST NOT CHANGE – relied upon by evaluate.py & CI)
1. Human-readable experiment description (multi-line, free-form).
2. A single **JSON line** with the structure documented below – this is parsed
   by main.py & evaluate.py.

{
  "run_id": "<unique name from YAML>",
  "epoch_metrics": [
      {"epoch": 1, "train_loss": 4.83, "val_ppl": 125.1, "val_ece": 0.38},
      ...
  ],
  "final":        {"val_ppl": 37.2, "val_ece": 0.09, "wall_clock": 713.4}
}
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .preprocess import load_dataset
from .model import (
    get_model,
    get_loss_fn,
    expected_calibration_error,
)
from .utils import set_seed, save_pdf

# --------------------------------------------------------------------------------------
# Helper functions
# --------------------------------------------------------------------------------------

def _describe_experiment(cfg: Dict[str, Any]) -> str:
    ds = cfg["dataset"]["name"]
    model = cfg["model"]["name"]
    loss = cfg["training"]["loss"]
    epochs = cfg["training"]["epochs"]
    bs = cfg["training"]["batch_size"]
    return (
        f"Running experiment '{cfg['run_id']}'\n"
        f"  • Task       : {cfg['task_type']}\n"
        f"  • Dataset    : {ds}\n"
        f"  • Model      : {model}\n"
        f"  • Loss       : {loss}\n"
        f"  • Epochs     : {epochs}\n"
        f"  • Batch size : {bs}\n"
    )


# --------------------------------------------------------------------------------------
# Training / Evaluation routines (model-agnostic)
# --------------------------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    loss_fn: nn.Module,
    data_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
):
    """Train for exactly one epoch and return average training loss."""

    model.train()
    total_loss = 0.0
    total_tokens = 0

    for batch in data_loader:
        inputs, targets = [x.to(device) for x in batch]
        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = loss_fn(logits, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * targets.numel()
        total_tokens += targets.numel()

    return total_loss / max(total_tokens, 1)


def evaluate(
    model: nn.Module,
    loss_fn: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
):
    """Return perplexity & ECE on the supplied validation / test split."""

    model.eval()
    all_logits: List[torch.Tensor] = []
    all_targets: List[torch.Tensor] = []
    with torch.no_grad():
        for batch in data_loader:
            inputs, targets = [x.to(device) for x in batch]
            logits = model(inputs)
            _ = loss_fn(logits, targets)  # keep internal step counters consistent
            all_logits.append(logits.detach())
            all_targets.append(targets.detach())

    logits = torch.cat(all_logits, dim=0)
    targets = torch.cat(all_targets, dim=0)

    ppl = torch.exp(
        F.cross_entropy(
            logits.view(-1, logits.size(-1)), targets.view(-1), reduction="mean"
        )
    ).item()

    probs = logits.softmax(-1).view(-1, logits.size(-1))
    labels = targets.view(-1)
    ece = expected_calibration_error(probs, labels, num_bins=10).item()

    return ppl, ece


# --------------------------------------------------------------------------------------
# Main entry point
# --------------------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Train a single experimental run")
    parser.add_argument("--config", type=str, required=True, help="Path to run config YAML")
    parser.add_argument("--results-dir", type=str, required=True, help="Root directory to save run-specific outputs")
    args = parser.parse_args()

    import yaml  # local import to keep start-up time minimal

    cfg: Dict[str, Any] = yaml.safe_load(Path(args.config).read_text())
    run_id: str = cfg["run_id"]
    results_dir = Path(args.results_dir)
    run_dir = results_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    images_dir = run_dir / "images"
    images_dir.mkdir(exist_ok=True, parents=True)

    # ------------------------------------------------------------------
    # Reproducibility & device
    # ------------------------------------------------------------------
    seed = int(cfg.get("seed", 42))
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    train_loader, val_loader, vocab_size = load_dataset(cfg)

    # ------------------------------------------------------------------
    # Model & Loss
    # ------------------------------------------------------------------
    model = get_model(cfg, vocab_size=vocab_size).to(device)
    loss_fn = get_loss_fn(cfg, vocab_size=vocab_size, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["training"]["learning_rate"])

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    num_epochs = int(cfg["training"]["epochs"])
    epoch_metrics: List[Dict[str, float]] = []
    start_time = time.time()

    print(_describe_experiment(cfg), flush=True)

    for epoch in range(1, num_epochs + 1):
        train_loss = train_one_epoch(model, loss_fn, train_loader, optimizer, device)
        val_ppl, val_ece = evaluate(model, loss_fn, val_loader, device)

        epoch_metrics.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_ppl": val_ppl,
                "val_ece": val_ece,
            }
        )

    wall_clock = time.time() - start_time

    # ------------------------------------------------------------------
    # Save artifacts
    # ------------------------------------------------------------------
    torch.save(model.state_dict(), run_dir / "model.pt")
    with (run_dir / "epoch_metrics.json").open("w") as f:
        json.dump(epoch_metrics, f, indent=2)

    # ----- Figures -----
    import matplotlib.pyplot as plt

    epochs = [m["epoch"] for m in epoch_metrics]
    losses = [m["train_loss"] for m in epoch_metrics]
    ppls = [m["val_ppl"] for m in epoch_metrics]
    eces = [m["val_ece"] for m in epoch_metrics]

    # Training loss curve
    plt.figure(figsize=(6, 4))
    plt.plot(epochs, losses, label="Train loss")
    plt.scatter(epochs[-1], losses[-1], color="red")
    plt.text(epochs[-1], losses[-1], f"{losses[-1]:.2f}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.legend()
    plt.tight_layout()
    save_pdf(plt, images_dir / "training_loss.pdf")

    # Validation PPL & ECE (twin axes)
    fig, ax1 = plt.subplots(figsize=(6, 4))
    ax2 = ax1.twinx()
    ax1.plot(epochs, ppls, color="blue", label="PPL")
    ax2.plot(epochs, eces, color="orange", label="ECE")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Perplexity", color="blue")
    ax2.set_ylabel("ECE", color="orange")

    # annotate final values
    ax1.scatter(epochs[-1], ppls[-1], color="blue")
    ax1.text(epochs[-1], ppls[-1], f"{ppls[-1]:.2f}")
    ax2.scatter(epochs[-1], eces[-1], color="orange")
    ax2.text(epochs[-1], eces[-1], f"{eces[-1]:.3f}")

    fig.suptitle("Validation Metrics")
    fig.tight_layout()
    save_pdf(plt, images_dir / "validation_metrics.pdf")

    # ------------------------------------------------------------------
    # Print final metrics to STDOUT (machine-readable part)
    # ------------------------------------------------------------------
    final_payload = {
        "run_id": run_id,
        "epoch_metrics": epoch_metrics,
        "final": {
            "val_ppl": ppls[-1],
            "val_ece": eces[-1],
            "wall_clock": wall_clock,
        },
    }

    print(json.dumps(final_payload), flush=True)


if __name__ == "__main__":
    main()
