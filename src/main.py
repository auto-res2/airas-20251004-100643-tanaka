#!/usr/bin/env python
"""src/main.py
Master orchestrator.  
Usage:
  uv run python -m src.main --smoke-test  --results-dir <path>
  uv run python -m src.main --full-experiment --results-dir <path>
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Any

import yaml

# --------------------------------------------------------------------------------------
# IO helpers
# --------------------------------------------------------------------------------------

def _tee_stream(stream, log_file):
    """Forward *stream* (stdout/stderr of subprocess) to both terminal and file."""
    for line in iter(stream.readline, b""):
        sys.stdout.buffer.write(line)
        log_file.write(line)
        sys.stdout.flush()
        log_file.flush()


def _run_subprocess(cmd: List[str], env: Dict[str, str], stdout_path: Path, stderr_path: Path):
    with stdout_path.open("wb") as out_f, stderr_path.open("wb") as err_f:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
        # Real-time tee
        import threading

        t_out = threading.Thread(target=_tee_stream, args=(proc.stdout, out_f))
        t_err = threading.Thread(target=_tee_stream, args=(proc.stderr, err_f))
        t_out.start(); t_err.start()
        proc.wait()
        t_out.join(); t_err.join()
        if proc.returncode != 0:
            raise RuntimeError(f"Subprocess failed with code {proc.returncode}: {' '.join(cmd)}")


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Run all experiment variations sequentially")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--smoke-test", action="store_true")
    group.add_argument("--full-experiment", action="store_true")
    parser.add_argument("--results-dir", required=True, type=str, help="Directory to store all outputs")
    args = parser.parse_args()

    cfg_path = Path("config/smoke_test.yaml" if args.smoke_test else "config/full_experiment.yaml")
    with cfg_path.open() as f:
        master_cfg = yaml.safe_load(f)

    experiments: List[Dict[str, Any]] = master_cfg["experiments"]
    results_root = Path(args.results_dir)
    if results_root.exists():
        # Allow re-runs: remove previous contents
        try:
            shutil.rmtree(results_root, ignore_errors=True)
        except Exception:
            # If that fails, try a more forceful approach
            import subprocess
            subprocess.run(["rm", "-rf", str(results_root)], check=False)
    results_root.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Run sequentially
    # ------------------------------------------------------------------
    for exp in experiments:
        run_id = exp["run_id"]
        run_dir = results_root / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        # Dump per-run config (visible to train.py only)
        run_cfg_path = run_dir / "config.yaml"
        with run_cfg_path.open("w") as f:
            yaml.safe_dump(exp, f)

        # Subprocess call
        cmd = [
            sys.executable,
            "-m",
            "src.train",
            "--config",
            str(run_cfg_path),
            "--results-dir",
            str(results_root),
        ]
        env = os.environ.copy()
        _run_subprocess(cmd, env, run_dir / "stdout.log", run_dir / "stderr.log")

    # ------------------------------------------------------------------
    # After all runs – aggregate & evaluate
    # ------------------------------------------------------------------
    eval_cmd = [
        sys.executable,
        "-m",
        "src.evaluate",
        "--results-dir",
        str(results_root),
    ]
    _run_subprocess(eval_cmd, os.environ.copy(), results_root / "evaluate_stdout.log", results_root / "evaluate_stderr.log")

    print("All experiments completed successfully.")


if __name__ == "__main__":
    main()
