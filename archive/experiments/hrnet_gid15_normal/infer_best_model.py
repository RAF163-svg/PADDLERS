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
    load_config,
    load_runtime_model_from_checkpoint,
    resolve_runtime,
)
from utils_metrics import export_prediction_artifacts, write_json  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export prediction labels and visualizations from the HRNet best model."
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--save-dir", default=None)
    parser.add_argument("--split", choices=["auto", "val", "test"], default="auto")
    parser.add_argument("--pred-subdir", default="pred")
    parser.add_argument("--vis-subdir", default="vis")
    return parser.parse_args()


def main():
    args = parse_args()
    runtime = resolve_runtime(
        load_config(args.config),
        args=argparse.Namespace(
            save_dir=args.save_dir,
            resume_checkpoint=None,
            final_eval_split=args.split,
        ),
    )
    resolved = runtime["_resolved"]
    bundles = build_seg_datasets(runtime)
    split_name = resolved["final_eval_split"]
    pairs = bundles["test_pairs"] if split_name == "test" else bundles["val_pairs"]

    best_model_dir = resolved["save_dir"] / "best_model"
    if not best_model_dir.exists():
        raise FileNotFoundError(f"Best model directory not found: {best_model_dir}")

    best_model = load_runtime_model_from_checkpoint(
        runtime,
        best_model_dir,
        eval_transforms=bundles["eval_transforms"],
    )
    manifest = export_prediction_artifacts(
        model=best_model,
        pairs=pairs,
        transforms=bundles["eval_transforms"],
        dataset_root=resolved["dataset_root"],
        pred_dir=resolved["save_dir"] / args.pred_subdir,
        vis_dir=resolved["save_dir"] / args.vis_subdir,
        num_classes=resolved["num_classes"],
        split_name=split_name,
        export_predictions=True,
        export_visualizations=True,
    )
    write_json(resolved["save_dir"] / "pred_manifest.json", manifest)

    print("[info] best_model_dir=", best_model_dir)
    print("[info] split=", split_name)
    print("[info] pred_dir=", resolved["save_dir"] / args.pred_subdir)
    print("[info] vis_dir=", resolved["save_dir"] / args.vis_subdir)
    print("[info] manifest=", resolved["save_dir"] / "pred_manifest.json")


if __name__ == "__main__":
    main()
