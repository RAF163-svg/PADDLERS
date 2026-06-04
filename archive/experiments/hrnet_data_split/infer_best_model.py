#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
WORK_DIR = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(WORK_DIR) not in sys.path:
    sys.path.insert(0, str(WORK_DIR))

from metrics_utils import StageLogger, write_json
from train import build_runtime, create_output_structure, evaluate_checkpoint_on_split, maybe_import_training_stack


def parse_args():
    parser = argparse.ArgumentParser(description="Export prediction samples from best_model for 4-head routed HRNet.")
    parser.add_argument("--config", default=str(WORK_DIR / "train_config_multihead.json"))
    parser.add_argument("--save-dir", default=None)
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--device", default="gpu")
    parser.add_argument("--prediction-sample-count", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    stack = maybe_import_training_stack()
    runtime = build_runtime(args)
    create_output_structure(runtime["save_dir"])

    logger = StageLogger(runtime["save_dir"] / "logs")
    paddle = stack["paddle"]
    paddle.set_device(args.device)

    checkpoint_dir = runtime["save_dir"] / "best_model"
    prediction_sample_count = (
        args.prediction_sample_count
        if args.prediction_sample_count is not None
        else int(runtime["config"]["evaluation"].get("prediction_sample_count", 12))
    )
    metrics, _ = evaluate_checkpoint_on_split(
        runtime,
        stack,
        checkpoint_dir=checkpoint_dir,
        split_name=args.split,
        logger=None,
        export_predictions=True,
        prediction_sample_count=prediction_sample_count,
    )
    manifest_path = runtime["save_dir"] / "prediction_samples" / f"{args.split}_manifest.json"
    write_json(manifest_path, metrics["prediction_samples"])
    logger.info("EXPORT", f"Prediction samples exported: split={args.split}, count={len(metrics['prediction_samples'])}, manifest={manifest_path}")


if __name__ == "__main__":
    main()
