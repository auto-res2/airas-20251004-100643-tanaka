#!/usr/bin/env python
"""src/train.py
Runs a single experiment variation.
See module-level docstring in the common base for the STDOUT protocol.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

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
# Utility helpers for dict-aware tensor handling
# --------------------------------------------------------------------------------------

def _move_inputs_to_device(inputs: Union[torch.Tensor, Dict[str, torch.Tensor]], device: torch.device):
    if isinstance(inputs, dict):
        return {k: v.to(device) for k, v in inputs.items()}
    return inputs.to(device)


# --------------------------------------------------------------------------------------
# Temperature scaling (for CE+TempScale baseline)
# --------------------------------------------------------------------------------------

class TemperatureScaledModel(nn.Module):
    """Wraps any model and divides its logits by a learned temperature."""

    def __init__(self, model: nn.Module, temperature: torch.Tensor):
        super().__init__()
        self.model = model
        self.temperature = nn.Parameter(temperature)

    def forward(self, inputs):  # type: ignore[override]
        logits = self.model(inputs)
        return logits / self.temperature


def _tune_temperature(model: nn.Module, val_loader: DataLoader, device: torch.device) -> torch.Tensor:
    """Optimise a single scalar temperature on the validation set."""

    model.eval()
    logits_list: List[torch.Tensor] = []
    targets_list: List[torch.Tensor] = []

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = _move_inputs_to_device(inputs, device)
            targets = targets.to(device)
            logits = model(inputs)
            logits_list.append(logits.view(-1, logits.size(-1)))
            targets_list.append(targets.view(-1))

    logits_all = torch.cat(logits_list, dim=0)
    targets_all = torch.cat(targets_list, dim=0)

    temperature = torch.nn.Parameter(torch.ones([], device=device) * 1.5)
    optimizer = torch.optim.LBFGS([temperature], max_iter=50, line_search_fn="strong_wolfe")
    nll_criterion = torch.nn.CrossEntropyLoss()

    def _eval():
        optimizer.zero_grad()
        loss = nll_criterion(logits_all / temperature, targets_all)
        loss.backward()
        return loss

    optimizer.step(_eval)
    return temperature.detach()


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

    for inputs, targets in data_loader:
        inputs = _move_inputs_to_device(inputs, device)
        targets = targets.to(device)

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
) -> Tuple[float, float]:
    """Return perplexity & ECE on the supplied validation / test split."""

    model.eval()
    total_ce_loss = 0.0
    total_tokens = 0
    all_probs: List[torch.Tensor] = []
    all_labels: List[torch.Tensor] = []

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs = _move_inputs_to_device(inputs, device)
            targets = targets.to(device)
            logits = model(inputs)
            _ = loss_fn(logits, targets)  # keep internal step counters consistent

            # Compute CE loss for perplexity
            ce_loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), reduction="sum"
            )
            total_ce_loss += ce_loss.item()
            total_tokens += targets.numel()

            # For ECE, accumulate probs and labels (move to CPU to save GPU memory)
            probs = logits.softmax(-1).view(-1, logits.size(-1)).cpu()
            labels = targets.view(-1).cpu()
            all_probs.append(probs)
            all_labels.append(labels)

            # Clear GPU cache
            del logits, probs, labels
            torch.cuda.empty_cache()

    # Compute perplexity
    ppl = torch.exp(torch.tensor(total_ce_loss / max(total_tokens, 1))).item()

    # Compute ECE on CPU
    probs_all = torch.cat(all_probs, dim=0)
    labels_all = torch.cat(all_labels, dim=0)
    ece = expected_calibration_error(probs_all, labels_all, num_bins=10).item()

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
    learning_rate = float(cfg["training"]["learning_rate"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

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

    # ------------------------------------------------------------------
    # Temperature scaling baseline (post-hoc calibration)
    # ------------------------------------------------------------------
    loss_variant = cfg["training"]["loss"].lower()
    if loss_variant in {"ce+tempscale", "ce_tempscale"}:
        temperature = _tune_temperature(model, val_loader, device)
        model = TemperatureScaledModel(model, temperature).to(device)
        val_ppl, val_ece = evaluate(model, loss_fn, val_loader, device)
        # Overwrite / append final metrics
        epoch_metrics[-1]["val_ppl"] = val_ppl
        epoch_metrics[-1]["val_ece"] = val_ece
        epoch_metrics[-1]["temperature"] = float(temperature.cpu())

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
