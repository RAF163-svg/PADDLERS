#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


WORK_DIR = Path(__file__).resolve().parent
if str(WORK_DIR) not in sys.path:
    sys.path.insert(0, str(WORK_DIR))

from metrics_utils import ensure_dir
from train import (
    DEFAULT_CONFIG_PATH,
    build_datasets,
    build_model,
    build_optimizer,
    build_runtime,
    create_output_structure,
    create_phase_logger,
    export_prediction_samples,
    load_checkpoint,
    log_message,
    maybe_import_training_stack,
    set_seed,
    write_config_snapshot,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Export prediction samples from the best UNet data-split model.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--save-dir", default=None)
    parser.add_argument("--device", choices=["auto", "gpu", "cpu"], default="auto")
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--count", type=int, default=12)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)
    parser.add_argument("--train-batch-size", type=int, default=None)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--max-epochs", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    runtime = build_runtime(args)
    create_output_structure(runtime["save_dir"])
    write_config_snapshot(runtime)

    stack = maybe_import_training_stack()
    paddle = stack["paddle"]
    requested_device = args.device
    if requested_device == "auto":
        if paddle.device.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0:
            requested_device = "gpu"
        else:
            requested_device = "cpu"
    device = paddle.set_device(requested_device)

    logger, _ = create_phase_logger(runtime["save_dir"], "infer")
    log_message(logger, "INIT", f"Exporting prediction samples from best_model on split={args.split}.")
    log_message(logger, "SYSTEM", f"device={device} pid={os.getpid()}")

    set_seed(int(runtime["config"]["training"]["seed"]), paddle=paddle)
    datasets = build_datasets(runtime, stack)
    dataset = datasets[args.split]
    model = build_model(runtime)
    optimizer, _ = build_optimizer(runtime, paddle, model, runtime["planned_total_steps"])
    best_model_dir = runtime["save_dir"] / "best_model"
    load_checkpoint(model, optimizer, best_model_dir, paddle)

    exported_paths = export_prediction_samples(
        model=model,
        dataset=dataset,
        save_dir=runtime["save_dir"],
        split_name=args.split,
        count=args.count,
        stack=stack,
        paddle=paddle,
    )
    ensure_dir(runtime["save_dir"] / "prediction_samples")
    log_message(
        logger,
        "TEST",
        f"split={args.split} exported_prediction_files={len(exported_paths)} best_model_dir={best_model_dir}",
    )


if __name__ == "__main__":
    main()
