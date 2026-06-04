#!/usr/bin/env python3

from __future__ import annotations

import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluation skeleton for FastSCNN data split. No evaluation is executed in this scaffolding round."
    )
    parser.add_argument("--config", default="train_config_multihead.json")
    parser.add_argument("--save-dir", default=None)
    parser.add_argument("--final-eval-split", choices=["val", "test"], default="test")
    return parser.parse_args()


def main():
    args = parse_args()
    print("[status] mode=prepare_only action=evaluate_not_executed")
    print(f"[status] config={args.config}")
    print(f"[status] save_dir={args.save_dir or 'use_config_default'}")
    print(f"[status] final_eval_split={args.final_eval_split}")
    print("[status] note=This script is a skeleton only. No model loading or dataset iteration has been performed.")


if __name__ == "__main__":
    main()

