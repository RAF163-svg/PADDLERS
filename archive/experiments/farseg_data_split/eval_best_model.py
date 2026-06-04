#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from pathlib import Path


WORK_DIR = Path(__file__).resolve().parent
if str(WORK_DIR) not in sys.path:
    sys.path.insert(0, str(WORK_DIR))

from train import DEFAULT_CONFIG_PATH, build_runtime, parse_args as _unused_parse_args  # noqa: F401


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluation entry skeleton for the best 4-head FarSeg model."
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--save-dir", default=None)
    parser.add_argument("--resume-checkpoint", default=None)
    parser.add_argument("--train-batch-size", type=int, default=None)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--final-eval-split", choices=["val", "test"], default="test")
    return parser.parse_args()


def main():
    args = parse_args()
    runtime = build_runtime(args)
    save_dir = runtime["save_dir"]

    best_model_dir = save_dir / "best_model"
    metrics_dir = save_dir / "metrics"

    print(f"[eval] model_type=ClusterRoutedFarSeg num_heads={runtime['num_heads']} final_eval_split={args.final_eval_split}")
    print(f"[eval] best_model_dir={best_model_dir}")
    print(f"[eval] metrics_dir={metrics_dir}")
    print("[eval] best-model evaluation is intentionally not executed in this scaffolding turn.")


if __name__ == "__main__":
    main()
