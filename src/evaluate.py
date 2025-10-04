#!/usr/bin/env python
"""src/evaluate.py
Aggregates results from multiple experimental runs (i.e. sub-directories of the
`results_dir`) and generates comparison figures + a JSON summary printed to
stdout.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Any

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .utils import save_pdf


# --------------------------------------------------------------------------------------
# CLI & helpers
# --------------------------------------------------------------------------------------

def _collect_results(results_dir: Path) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for sub in results_dir.iterdir():
        if not (sub / "epoch_metrics.json").exists():
            continue  # skip non-run folders
        with (sub / "epoch_metrics.json").open() as f:
            epoch_metrics = json.load(f)
        final = epoch_metrics[-1]
        rows.append(
            {
                "run_id": sub.name,
                "val_ppl": final["val_ppl"],
                "val_ece": final["val_ece"],
            }
        )
    if not rows:
        raise RuntimeError(f"No result folders found in {results_dir}")
    return pd.DataFrame(rows)


def _plot_bar(df: pd.DataFrame, metric: str, images_dir: Path):
    plt.figure(figsize=(max(4, len(df) * 1.5), 4))
    sns.barplot(x="run_id", y=metric, data=df, palette="viridis")
    for i, v in enumerate(df[metric]):
        plt.text(i, v, f"{v:.3f}", ha="center", va="bottom")
    plt.title(f"Final {metric.upper()} Comparison")
    plt.xlabel("Run ID")
    plt.ylabel(metric.upper())
    plt.tight_layout()
    save_pdf(plt, images_dir / f"{metric}_comparison.pdf")


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate all experiment variations")
    parser.add_argument("--results-dir", required=True, type=str, help="Directory with all runs' sub-folders")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    images_dir = results_dir / "images"
    images_dir.mkdir(exist_ok=True)

    df = _collect_results(results_dir)

    # ----- Figures -----
    for metric in ["val_ppl", "val_ece"]:
        _plot_bar(df, metric, images_dir)

    # ----- JSON summary -----
    summary = df.to_dict(orient="list")
    print(json.dumps(summary, indent=None))


if __name__ == "__main__":
    main()
