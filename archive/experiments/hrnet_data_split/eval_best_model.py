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

from metrics_utils import StageLogger, format_eval_log, format_test_log, write_confusion_csv, write_json, write_metrics_text, write_per_class_csv
from train import build_runtime, create_output_structure, evaluate_checkpoint_on_split, maybe_import_training_stack


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate best_model for 4-head routed HRNet.")
    parser.add_argument("--config", default=str(WORK_DIR / "train_config_multihead.json"))
    parser.add_argument("--save-dir", default=None)
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--device", default="gpu")
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
    metrics, confusion = evaluate_checkpoint_on_split(
        runtime,
        stack,
        checkpoint_dir=checkpoint_dir,
        split_name=args.split,
        logger=None,
        export_predictions=bool(runtime["config"]["evaluation"].get("export_prediction_samples", True)),
        prediction_sample_count=int(runtime["config"]["evaluation"].get("prediction_sample_count", 12)),
    )

    if args.split == "test":
        logger.info(
            "TEST",
            format_test_log(
                test_loss=metrics["mean_loss"],
                oa=metrics["overall_accuracy"],
                miou=metrics["mean_iou"],
                macc=metrics["mean_accuracy"],
                kappa=metrics["kappa"],
            ),
        )
    else:
        logger.info(
            "EVAL",
            format_eval_log(
                epoch=int(metrics.get("epoch", 0)),
                total_epochs=int(runtime["config"]["training"]["epochs"]),
                val_loss=metrics["mean_loss"],
                oa=metrics["overall_accuracy"],
                miou=metrics["mean_iou"],
                macc=metrics["mean_accuracy"],
                kappa=metrics["kappa"],
            ),
        )

    write_json(runtime["save_dir"] / "metrics" / f"best_model_{args.split}.json", metrics)
    write_per_class_csv(runtime["save_dir"] / "metrics" / f"best_model_{args.split}_per_class.csv", metrics)
    write_confusion_csv(runtime["save_dir"] / "metrics" / f"best_model_{args.split}_confusion.csv", confusion, runtime["class_names"])
    if args.split == "test":
        write_metrics_text(runtime["save_dir"] / "best_model_metrics.txt", metrics, split_name="test", checkpoint_path=str(checkpoint_dir))


if __name__ == "__main__":
    main()
