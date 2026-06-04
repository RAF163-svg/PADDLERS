#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path


WORK_DIR = Path(__file__).resolve().parent
if str(WORK_DIR) not in sys.path:
    sys.path.insert(0, str(WORK_DIR))

from train import (  # noqa: E402
    DEFAULT_CONFIG_PATH,
    build_seg_datasets,
    evaluate_and_export_best_model,
    load_config,
    resolve_runtime,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate the best HRNet GID-15 model and regenerate exported metrics/artifacts."
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--save-dir", default=None)
    parser.add_argument("--final-eval-split", choices=["auto", "val", "test"], default="auto")
    return parser.parse_args()


def main():
    args = parse_args()
    runtime = resolve_runtime(load_config(args.config), args=args)
    resolved = runtime["_resolved"]

    bundles = build_seg_datasets(runtime)
    eval_dataset = bundles["test_dataset"] if resolved["final_eval_split"] == "test" else bundles["val_dataset"]
    eval_pairs = bundles["test_pairs"] if resolved["final_eval_split"] == "test" else bundles["val_pairs"]
    payload = evaluate_and_export_best_model(runtime, eval_dataset, eval_pairs, bundles["eval_transforms"])

    print("[info] best_model_dir=", payload["best_model_dir"])
    print("[info] evaluated_split=", payload["evaluated_split"])
    print("[info] best_model_metrics=", resolved["save_dir"] / "best_model_metrics.json")
    print("[info] per_class_metrics=", resolved["save_dir"] / "per_class_metrics.csv")
    print("[info] confusion_matrix=", resolved["save_dir"] / "confusion_matrix.npy")


if __name__ == "__main__":
    main()
