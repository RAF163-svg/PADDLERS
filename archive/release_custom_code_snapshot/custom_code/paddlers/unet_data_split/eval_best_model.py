#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from pathlib import Path


WORK_DIR = Path(__file__).resolve().parent
if str(WORK_DIR) not in sys.path:
    sys.path.insert(0, str(WORK_DIR))

from train import DEFAULT_CONFIG_PATH, build_runtime, create_output_structure, run_best_model_evaluation, write_config_snapshot


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate the best UNet data-split model.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--save-dir", default=None)
    parser.add_argument("--resume-checkpoint", default=None)
    parser.add_argument("--train-batch-size", type=int, default=None)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    parser.add_argument("--max-test-batches", type=int, default=None)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)
    parser.add_argument("--device", choices=["auto", "gpu", "cpu"], default="auto")
    parser.add_argument("--final-eval-split", choices=["val", "test"], default="test")
    return parser.parse_args()


def main():
    args = parse_args()
    runtime = build_runtime(args)
    create_output_structure(runtime["save_dir"])
    write_config_snapshot(runtime)
    run_best_model_evaluation(runtime, split_name=args.final_eval_split, log_file_name="eval")


if __name__ == "__main__":
    main()
