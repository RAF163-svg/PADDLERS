#!/usr/bin/env python3

import argparse
import copy
import json
import os
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import paddle


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import paddlers as pdrs
from paddlers import transforms as T
from paddlers.tasks.load_model import load_model

from utils_metrics import (
    build_evaluation_payload,
    ensure_dir,
    export_prediction_artifacts,
    save_confusion_matrix,
    to_builtin,
    write_json,
    write_metrics_text,
    write_per_class_csv,
)


WORK_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = WORK_DIR / "train_config.json"
DEFAULT_DATA_CANDIDATES = [
    Path("${GID15_DATASET_ROOT}/gid15_seed20260320_tile640_stride448"),
    Path("${GID15_DATASET_ROOT}/paddlers_gid15_benchmark_20260320/data/gid15_seed20260320_tile640_stride448"),
    Path("${GID15_DATASET_ROOT}/gid-15"),
]
IGNORE_INDEX = 255


def get_world_size():
    env_world_size = int(os.environ.get("PADDLE_TRAINERS_NUM", "1"))
    if env_world_size > 1:
        return env_world_size
    world_size = paddle.distributed.get_world_size()
    if world_size and world_size > 0:
        return world_size
    return 1


def get_rank():
    env_rank = os.environ.get("PADDLE_TRAINER_ID")
    if env_rank is not None:
        return int(env_rank)
    if get_world_size() <= 1:
        return 0
    return paddle.distributed.get_rank()


def init_dist_if_needed():
    if get_world_size() > 1 and not paddle.distributed.parallel.parallel_helper._is_parallel_ctx_initialized():
        paddle.distributed.init_parallel_env()


def dist_barrier():
    if get_world_size() > 1 and paddle.distributed.parallel.parallel_helper._is_parallel_ctx_initialized():
        paddle.distributed.barrier()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    paddle.seed(seed)


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_existing_path(base_dir, path_value, allow_missing=False):
    if path_value is None:
        return None
    path = Path(path_value)
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    if path.exists() or allow_missing:
        return path
    raise FileNotFoundError(f"Path does not exist: {path}")


def resolve_dataset_root(config):
    candidates = []
    if config.get("dataset_root"):
        candidates.append(Path(config["dataset_root"]))
    for item in config.get("dataset_root_candidates", []):
        candidates.append(Path(item))
    candidates.extend(DEFAULT_DATA_CANDIDATES)

    for candidate in candidates:
        resolved = candidate
        if not resolved.is_absolute():
            resolved = (WORK_DIR / resolved).resolve()
        if resolved.exists():
            return resolved

    fallback = candidates[0] if candidates else Path("/path/to/your/gid15_dataset")
    if not fallback.is_absolute():
        fallback = (WORK_DIR / fallback).resolve()
    return fallback


def read_label_names(label_path):
    if label_path is None or not Path(label_path).exists():
        return []
    with open(label_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def resolve_class_names(config, label_path):
    config_class_names = list(config.get("class_names", []))
    label_names = read_label_names(label_path)
    if config_class_names and label_names and config_class_names != label_names:
        raise ValueError(
            "class_names in train_config.json does not match labels.txt. "
            f"config={config_class_names}, labels.txt={label_names}"
        )
    class_names = config_class_names or label_names
    num_classes = int(config.get("num_classes") or len(class_names))
    if num_classes <= 0:
        raise ValueError("num_classes must be positive.")
    if not class_names:
        class_names = [f"class_{idx}" for idx in range(num_classes)]
    if len(class_names) != num_classes:
        raise ValueError(
            f"class_names length ({len(class_names)}) does not equal num_classes ({num_classes})."
        )
    return class_names, num_classes


def read_file_pairs(file_list_path, data_dir):
    pairs = []
    with open(file_list_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            items = line.strip().split()
            if not items:
                continue
            if len(items) != 2:
                raise ValueError(
                    f"{file_list_path} line {line_no} should contain 2 columns, got {len(items)}."
                )
            image_path = Path(items[0])
            mask_path = Path(items[1])
            if not image_path.is_absolute():
                image_path = Path(data_dir) / image_path
            if not mask_path.is_absolute():
                mask_path = Path(data_dir) / mask_path
            pairs.append((image_path, mask_path))
    return pairs


def read_mask(mask_path):
    mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise FileNotFoundError(f"Unable to read mask: {mask_path}")
    if mask.ndim == 3:
        if not np.array_equal(mask[..., 0], mask[..., 1]) or not np.array_equal(
            mask[..., 0], mask[..., 2]
        ):
            print(
                f"[warn] mask {mask_path} is 3-channel with different values per channel. "
                "Only the first channel will be used."
            )
        mask = mask[..., 0]
    return mask.astype(np.int64)


def analyze_split(split_name, pairs, ignore_index):
    pixel_counts = {}
    image_counts = {}
    invalid_file_samples = []
    max_valid_label = -1
    total_pixels = 0
    ignore_pixels = 0

    for image_path, mask_path in pairs:
        if not image_path.exists():
            raise FileNotFoundError(f"{split_name}: image not found: {image_path}")
        if not mask_path.exists():
            raise FileNotFoundError(f"{split_name}: mask not found: {mask_path}")

        mask = read_mask(mask_path)
        total_pixels += int(mask.size)
        values, counts = np.unique(mask, return_counts=True)
        values = values.astype(np.int64)
        counts = counts.astype(np.int64)
        image_classes = set()

        for value, count in zip(values.tolist(), counts.tolist()):
            pixel_counts[value] = pixel_counts.get(value, 0) + int(count)
            if value == ignore_index:
                ignore_pixels += int(count)
                continue
            if value >= 0:
                max_valid_label = max(max_valid_label, int(value))
                image_classes.add(int(value))

        for value in image_classes:
            image_counts[value] = image_counts.get(value, 0) + 1

        invalid_values = [value for value in values.tolist() if value < 0]
        if invalid_values:
            invalid_file_samples.append(
                {"mask": str(mask_path), "invalid_values": invalid_values[:10]}
            )

    return {
        "split": split_name,
        "num_images": len(pairs),
        "total_pixels": total_pixels,
        "ignore_pixels": ignore_pixels,
        "max_valid_label": max_valid_label,
        "raw_pixel_counts": pixel_counts,
        "raw_image_counts": image_counts,
        "invalid_file_samples": invalid_file_samples[:20],
    }


def finalize_split_stats(summary, num_classes, ignore_index):
    raw_pixel_counts = {int(k): int(v) for k, v in summary["raw_pixel_counts"].items()}
    raw_image_counts = {int(k): int(v) for k, v in summary["raw_image_counts"].items()}
    class_pixel_counts = [int(raw_pixel_counts.get(i, 0)) for i in range(num_classes)]
    class_image_counts = [int(raw_image_counts.get(i, 0)) for i in range(num_classes)]
    invalid_label_counts = {}
    for value, count in raw_pixel_counts.items():
        if value == ignore_index:
            continue
        if value < 0 or value >= num_classes:
            invalid_label_counts[str(value)] = int(count)

    valid_pixels = int(sum(class_pixel_counts))
    summary["class_pixel_counts"] = class_pixel_counts
    summary["class_image_counts"] = class_image_counts
    summary["valid_pixels"] = valid_pixels
    summary["class_pixel_ratio"] = [
        float(count / valid_pixels) if valid_pixels else 0.0 for count in class_pixel_counts
    ]
    summary["missing_classes"] = [
        idx for idx, count in enumerate(class_pixel_counts) if count == 0
    ]
    summary["invalid_label_counts"] = invalid_label_counts
    summary["ignore_index"] = ignore_index
    del summary["raw_pixel_counts"]
    del summary["raw_image_counts"]
    return summary


def build_dataset_report(train_stats, val_stats, test_stats, label_names, num_classes):
    report = {
        "num_classes": num_classes,
        "labels": label_names,
        "train": train_stats,
        "val": val_stats,
    }
    if test_stats is not None:
        report["test"] = test_stats

    train_missing = set(train_stats["missing_classes"])
    report["classes_missing_in_train"] = sorted(train_missing)
    report["classes_present_in_val_but_missing_in_train"] = sorted(
        idx
        for idx, count in enumerate(val_stats["class_pixel_counts"])
        if count > 0 and idx in train_missing
    )
    if test_stats is not None:
        report["classes_present_in_test_but_missing_in_train"] = sorted(
            idx
            for idx, count in enumerate(test_stats["class_pixel_counts"])
            if count > 0 and idx in train_missing
        )
    report["very_rare_train_classes"] = sorted(
        idx for idx, ratio in enumerate(train_stats["class_pixel_ratio"]) if ratio < 0.01
    )
    report["val_missing_classes"] = sorted(val_stats["missing_classes"])
    if test_stats is not None:
        report["test_missing_classes"] = sorted(test_stats["missing_classes"])
    return report


def resolve_target_class(label_names, num_classes, target_index, target_label, train_stats):
    if target_index is not None:
        if target_index < 0 or target_index >= num_classes:
            raise ValueError(
                f"oversample target index {target_index} is out of range for num_classes={num_classes}"
            )
        resolved_index = int(target_index)
        resolved_label = label_names[resolved_index]
        return resolved_index, resolved_label

    if target_label is not None:
        if target_label in label_names:
            resolved_index = label_names.index(target_label)
            return resolved_index, target_label

    ratios = train_stats["class_pixel_ratio"]
    if not ratios:
        return 0, label_names[0]
    smallest_index = int(np.argmin(np.asarray(ratios, dtype=np.float64)))
    return smallest_index, label_names[smallest_index]


def generate_oversampled_train_list(
        train_list_path,
        data_dir,
        target_class_idx,
        target_class_label,
        oversample_ratio,
        output_path,
        seed):
    output_path = Path(output_path)
    ensure_dir(output_path.parent)

    if oversample_ratio <= 1:
        output_path.write_text(Path(train_list_path).read_text(encoding="utf-8"), encoding="utf-8")
        return output_path, {
            "enabled": False,
            "target_class_index": int(target_class_idx),
            "target_class_label": target_class_label,
            "oversample_ratio": int(oversample_ratio),
            "effective_train_list": str(output_path),
        }

    random_state = random.Random(seed)
    lines = Path(train_list_path).read_text(encoding="utf-8").splitlines()
    expanded_lines = []
    oversampled_count = 0

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        expanded_lines.append(stripped)
        items = stripped.split()
        if len(items) != 2:
            raise ValueError(f"Invalid train list line: {stripped}")
        mask_path = Path(items[1])
        if not mask_path.is_absolute():
            mask_path = Path(data_dir) / mask_path
        mask = read_mask(mask_path)
        if np.any(mask == target_class_idx):
            expanded_lines.extend([stripped] * (oversample_ratio - 1))
            oversampled_count += 1

    random_state.shuffle(expanded_lines)
    output_path.write_text("\n".join(expanded_lines) + "\n", encoding="utf-8")
    return output_path, {
        "enabled": True,
        "target_class_index": int(target_class_idx),
        "target_class_label": target_class_label,
        "oversample_ratio": int(oversample_ratio),
        "source_train_list": str(train_list_path),
        "effective_train_list": str(output_path),
        "source_sample_count": len(lines),
        "target_positive_sample_count": oversampled_count,
        "effective_sample_count": len(expanded_lines),
    }


def load_config(config_path):
    config = read_json(config_path)
    config["_config_path"] = str(Path(config_path).resolve())
    return config


def resolve_runtime(config, args=None):
    training_cfg = config["training"]
    evaluation_cfg = config["evaluation"]

    dataset_root = resolve_dataset_root(config)
    label_list = resolve_existing_path(dataset_root, config.get("label_list"), allow_missing=True)
    class_names, num_classes = resolve_class_names(config, label_list)

    train_file_list = resolve_existing_path(dataset_root, config["train_file_list"])
    val_file_list = resolve_existing_path(dataset_root, config["val_file_list"])
    test_file_list = resolve_existing_path(
        dataset_root, config.get("test_file_list"), allow_missing=True
    )
    if test_file_list is not None and not test_file_list.exists():
        test_file_list = None

    save_dir_value = args.save_dir if args is not None and args.save_dir else config.get("save_dir", "output")
    save_dir = resolve_existing_path(WORK_DIR, save_dir_value, allow_missing=True)

    resume_checkpoint = training_cfg.get("resume_checkpoint")
    if args is not None and args.resume_checkpoint is not None:
        resume_checkpoint = args.resume_checkpoint
    if resume_checkpoint:
        resume_checkpoint = str(resolve_existing_path(WORK_DIR, resume_checkpoint, allow_missing=True))

    final_eval_split = evaluation_cfg.get("final_eval_split", "test")
    if args is not None and getattr(args, "final_eval_split", None):
        final_eval_split = args.final_eval_split
    if final_eval_split == "auto":
        final_eval_split = "test" if test_file_list is not None else "val"
    if final_eval_split == "test" and test_file_list is None:
        final_eval_split = "val"

    runtime = copy.deepcopy(config)
    runtime["_resolved"] = {
        "dataset_root": dataset_root,
        "train_file_list": train_file_list,
        "val_file_list": val_file_list,
        "test_file_list": test_file_list,
        "label_list": label_list,
        "save_dir": save_dir,
        "resume_checkpoint": resume_checkpoint,
        "final_eval_split": final_eval_split,
        "class_names": class_names,
        "num_classes": num_classes,
    }
    return runtime


def build_train_transforms(runtime):
    training_cfg = runtime["training"]
    aug_cfg = runtime["augmentation"]
    norm_cfg = runtime["normalization"]

    transforms = [T.RandomCrop(crop_size=training_cfg["crop_size"])]
    if aug_cfg.get("random_horizontal_flip", True):
        transforms.append(T.RandomHorizontalFlip())
    if aug_cfg.get("random_vertical_flip", True):
        transforms.append(T.RandomVerticalFlip())
    if aug_cfg.get("enable_distort", True):
        transforms.append(T.RandomDistort(**aug_cfg.get("distort", {})))
    if aug_cfg.get("enable_blur", False):
        transforms.append(T.RandomBlur(prob=aug_cfg.get("blur_prob", 0.1)))
    transforms.append(T.Normalize(mean=norm_cfg["mean"], std=norm_cfg["std"]))
    return T.Compose(transforms)


def build_eval_transforms(runtime):
    norm_cfg = runtime["normalization"]
    return T.Compose([T.Normalize(mean=norm_cfg["mean"], std=norm_cfg["std"])])


def build_seg_datasets(runtime, effective_train_list=None):
    resolved = runtime["_resolved"]
    data_dir = resolved["dataset_root"]
    label_list = resolved["label_list"]
    training_cfg = runtime["training"]

    train_transforms = build_train_transforms(runtime)
    eval_transforms = build_eval_transforms(runtime)

    train_pairs = read_file_pairs(resolved["train_file_list"], data_dir)
    val_pairs = read_file_pairs(resolved["val_file_list"], data_dir)
    test_pairs = (
        read_file_pairs(resolved["test_file_list"], data_dir)
        if resolved["test_file_list"] is not None
        else []
    )

    train_dataset = pdrs.datasets.SegDataset(
        data_dir=str(data_dir),
        file_list=str(effective_train_list or resolved["train_file_list"]),
        label_list=str(label_list) if label_list is not None and label_list.exists() else None,
        transforms=train_transforms,
        num_workers=training_cfg["num_workers"],
        shuffle=True,
    )
    val_dataset = pdrs.datasets.SegDataset(
        data_dir=str(data_dir),
        file_list=str(resolved["val_file_list"]),
        label_list=str(label_list) if label_list is not None and label_list.exists() else None,
        transforms=eval_transforms,
        num_workers=training_cfg["num_workers"],
        shuffle=False,
    )
    test_dataset = None
    if resolved["test_file_list"] is not None:
        test_dataset = pdrs.datasets.SegDataset(
            data_dir=str(data_dir),
            file_list=str(resolved["test_file_list"]),
            label_list=str(label_list) if label_list is not None and label_list.exists() else None,
            transforms=eval_transforms,
            num_workers=training_cfg["num_workers"],
            shuffle=False,
        )

    return {
        "train_dataset": train_dataset,
        "val_dataset": val_dataset,
        "test_dataset": test_dataset,
        "train_pairs": train_pairs,
        "val_pairs": val_pairs,
        "test_pairs": test_pairs,
        "train_transforms": train_transforms,
        "eval_transforms": eval_transforms,
    }


def build_dataset_reports(runtime, train_pairs, val_pairs, test_pairs):
    resolved = runtime["_resolved"]
    class_names = resolved["class_names"]
    num_classes = resolved["num_classes"]

    train_stats = analyze_split("train", train_pairs, IGNORE_INDEX)
    val_stats = analyze_split("val", val_pairs, IGNORE_INDEX)
    test_stats = analyze_split("test", test_pairs, IGNORE_INDEX) if test_pairs else None

    max_valid_label = max(
        train_stats["max_valid_label"],
        val_stats["max_valid_label"],
        test_stats["max_valid_label"] if test_stats is not None else -1,
    )
    inferred_num_classes = max(len(class_names), max_valid_label + 1)
    if inferred_num_classes != num_classes:
        raise ValueError(
            f"Configured num_classes={num_classes}, but labels/masks imply {inferred_num_classes}."
        )

    train_stats = finalize_split_stats(train_stats, num_classes, IGNORE_INDEX)
    val_stats = finalize_split_stats(val_stats, num_classes, IGNORE_INDEX)
    test_stats = finalize_split_stats(test_stats, num_classes, IGNORE_INDEX) if test_stats else None
    dataset_report = build_dataset_report(train_stats, val_stats, test_stats, class_names, num_classes)
    return dataset_report, train_stats


def build_model(runtime):
    resolved = runtime["_resolved"]
    model_cfg = runtime["model"]
    training_cfg = runtime["training"]

    return pdrs.tasks.seg.FarSeg(
        in_channels=model_cfg.get("in_channels", 3),
        num_classes=resolved["num_classes"],
        use_mixed_loss=training_cfg.get("use_mixed_loss", True),
        backbone=model_cfg.get("backbone", "resnet50"),
        backbone_pretrained=model_cfg.get("backbone_pretrained", True),
        fpn_out_channels=model_cfg.get("fpn_out_channels", 256),
        fsr_out_channels=model_cfg.get("fsr_out_channels", 256),
        scale_aware_proj=model_cfg.get("scale_aware_proj", True),
        decoder_out_channels=model_cfg.get("decoder_out_channels", 128),
    )


def build_runtime_train_config(runtime, effective_train_list, oversample_report):
    resolved = runtime["_resolved"]
    training_cfg = runtime["training"]

    return {
        "config_path": runtime["_config_path"],
        "dataset_root": str(resolved["dataset_root"]),
        "train_file_list": str(resolved["train_file_list"]),
        "effective_train_file_list": str(effective_train_list),
        "val_file_list": str(resolved["val_file_list"]),
        "test_file_list": str(resolved["test_file_list"]) if resolved["test_file_list"] is not None else None,
        "label_list": str(resolved["label_list"]) if resolved["label_list"] is not None else None,
        "class_names": list(resolved["class_names"]),
        "num_classes": resolved["num_classes"],
        "model": to_builtin(runtime["model"]),
        "pretrained": to_builtin(runtime["pretrained"]),
        "training": {
            **to_builtin(training_cfg),
            "resume_checkpoint": resolved["resume_checkpoint"],
        },
        "augmentation": to_builtin(runtime["augmentation"]),
        "normalization": to_builtin(runtime["normalization"]),
        "sampling": to_builtin(runtime["sampling"]),
        "evaluation": {
            **to_builtin(runtime["evaluation"]),
            "final_eval_split": resolved["final_eval_split"],
        },
        "save_dir": str(resolved["save_dir"]),
        "oversample_report": to_builtin(oversample_report),
    }


def evaluate_and_export_best_model(runtime, eval_dataset, eval_pairs, eval_transforms):
    resolved = runtime["_resolved"]
    save_dir = resolved["save_dir"]
    best_model_dir = save_dir / "best_model"
    if not best_model_dir.exists():
        raise FileNotFoundError(f"Best model directory not found: {best_model_dir}")

    training_cfg = runtime["training"]
    evaluation_cfg = runtime["evaluation"]
    eval_split = resolved["final_eval_split"]
    selection_split = evaluation_cfg.get("selection_split", "val")

    best_model = load_model(str(best_model_dir))
    metrics_from_model, details = best_model.evaluate(
        eval_dataset,
        batch_size=training_cfg["eval_batch_size"],
        return_details=True,
    )

    payload = build_evaluation_payload(
        model_name=runtime["model"]["name"],
        selection_split=selection_split,
        evaluated_split=eval_split,
        class_names=resolved["class_names"],
        metrics_from_model=metrics_from_model,
        details=details,
    )
    payload["best_model_dir"] = str(best_model_dir)

    accuracy_summary = {
        "model": runtime["model"]["name"],
        "selection_split": selection_split,
        "evaluated_split": eval_split,
        "best_model_dir": str(best_model_dir),
        "summary": payload["summary"],
    }
    write_json(save_dir / "accuracy_summary.json", accuracy_summary)
    write_json(save_dir / "best_model_metrics.json", payload)
    write_metrics_text(save_dir / "best_model_metrics.txt", payload)
    write_per_class_csv(save_dir / "per_class_metrics.csv", payload["summary"])
    save_confusion_matrix(save_dir / "confusion_matrix.npy", details["confusion_matrix"])

    pred_manifest = []
    if evaluation_cfg.get("export_predictions", True) or evaluation_cfg.get("export_visualizations", True):
        pred_manifest = export_prediction_artifacts(
            model=best_model,
            pairs=eval_pairs,
            transforms=eval_transforms,
            dataset_root=resolved["dataset_root"],
            pred_dir=save_dir / "pred",
            vis_dir=save_dir / "vis",
            num_classes=resolved["num_classes"],
            split_name=eval_split,
        )
        write_json(save_dir / "pred_manifest.json", pred_manifest)

    run_summary = {
        "model": runtime["model"]["name"],
        "selection_split": selection_split,
        "evaluated_split": eval_split,
        "best_model_dir": str(best_model_dir),
        "dataset_root": str(resolved["dataset_root"]),
        "dataset_report_path": str(save_dir / "dataset_report.json"),
        "train_config_path": str(save_dir / "train_config.json"),
        "best_model_metrics_json": str(save_dir / "best_model_metrics.json"),
        "best_model_metrics_txt": str(save_dir / "best_model_metrics.txt"),
        "per_class_metrics_csv": str(save_dir / "per_class_metrics.csv"),
        "confusion_matrix_npy": str(save_dir / "confusion_matrix.npy"),
        "pred_dir": str(save_dir / "pred"),
        "vis_dir": str(save_dir / "vis"),
        "prediction_count": len(pred_manifest),
    }
    write_json(save_dir / "run_summary.json", run_summary)
    return payload


def parse_args():
    parser = argparse.ArgumentParser(
        description="FarSeg training framework for GID-15, rebuilt on the DeepLab GID-15 template."
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--save-dir", default=None)
    parser.add_argument("--resume-checkpoint", default=None)
    parser.add_argument("--final-eval-split", choices=["auto", "val", "test"], default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    runtime = resolve_runtime(load_config(args.config), args=args)
    resolved = runtime["_resolved"]
    save_dir = resolved["save_dir"]
    ensure_dir(save_dir)

    training_cfg = runtime["training"]
    set_seed(training_cfg["seed"])
    init_dist_if_needed()

    rank = get_rank()
    world_size = get_world_size()
    main_process = rank == 0

    bundles = build_seg_datasets(runtime)
    train_pairs = bundles["train_pairs"]
    val_pairs = bundles["val_pairs"]
    test_pairs = bundles["test_pairs"]

    if main_process:
        print("[info] rank=", rank, "world_size=", world_size)
        print("[info] dataset_root=", resolved["dataset_root"])
        print("[info] train_file_list=", resolved["train_file_list"])
        print("[info] val_file_list=", resolved["val_file_list"])
        print(
            "[info] test_file_list=",
            resolved["test_file_list"] if resolved["test_file_list"] is not None else "None",
        )
        print("[info] label_list=", resolved["label_list"])
        print("[info] resume_checkpoint=", resolved["resume_checkpoint"])

        dataset_report, train_stats = build_dataset_reports(runtime, train_pairs, val_pairs, test_pairs)
        write_json(save_dir / "dataset_report.json", dataset_report)
        print("[info] num_classes=", resolved["num_classes"])
        print("[info] labels=", resolved["class_names"])
        print("[info] classes_missing_in_train=", dataset_report["classes_missing_in_train"])
        print("[info] very_rare_train_classes=", dataset_report["very_rare_train_classes"])

        sampling_cfg = runtime["sampling"]
        if sampling_cfg.get("enable_train_oversample", False):
            target_idx, target_label = resolve_target_class(
                resolved["class_names"],
                resolved["num_classes"],
                sampling_cfg.get("target_class_index"),
                sampling_cfg.get("target_class_name"),
                train_stats,
            )
            effective_train_list, oversample_report = generate_oversampled_train_list(
                train_list_path=resolved["train_file_list"],
                data_dir=resolved["dataset_root"],
                target_class_idx=target_idx,
                target_class_label=target_label,
                oversample_ratio=int(sampling_cfg.get("oversample_ratio", 1)),
                output_path=save_dir / "train_oversampled.txt",
                seed=training_cfg["seed"],
            )
        else:
            effective_train_list = resolved["train_file_list"]
            oversample_report = {
                "enabled": False,
                "target_class_index": None,
                "target_class_label": None,
                "oversample_ratio": 1,
                "source_train_list": str(resolved["train_file_list"]),
                "effective_train_list": str(effective_train_list),
            }

        write_json(save_dir / "oversample_report.json", oversample_report)
        write_json(
            save_dir / "train_config.json",
            build_runtime_train_config(runtime, effective_train_list, oversample_report),
        )
        print("[info] final_eval_split=", resolved["final_eval_split"])
        print("[info] effective_train_list=", effective_train_list)
        print("[info] start training, save_dir=", save_dir)

    dist_barrier()

    oversample_report = read_json(save_dir / "oversample_report.json")
    effective_train_list = Path(oversample_report["effective_train_list"])

    bundles = build_seg_datasets(runtime, effective_train_list=effective_train_list)
    train_dataset = bundles["train_dataset"]
    val_dataset = bundles["val_dataset"]
    test_dataset = bundles["test_dataset"]
    eval_transforms = bundles["eval_transforms"]

    model = build_model(runtime)

    pretrain_weights = runtime["pretrained"].get("model_weights_path")
    if pretrain_weights:
        pretrain_weights = str(resolve_existing_path(WORK_DIR, pretrain_weights))

    # Save a checkpoint every epoch so long-running remote jobs lose less progress when interrupted.
    model.train(
        num_epochs=training_cfg["epochs"],
        train_dataset=train_dataset,
        train_batch_size=training_cfg["train_batch_size"],
        eval_dataset=val_dataset,
        save_interval_epochs=training_cfg["save_interval_epochs"],
        log_interval_steps=training_cfg["log_interval_steps"],
        save_dir=str(save_dir),
        pretrain_weights=pretrain_weights,
        learning_rate=training_cfg["learning_rate"],
        lr_decay_power=training_cfg["lr_decay_power"],
        early_stop=training_cfg["early_stop"],
        early_stop_patience=training_cfg["early_stop_patience"],
        use_vdl=True,
        resume_checkpoint=resolved["resume_checkpoint"],
        precision=training_cfg.get("precision", "fp32"),
        amp_level=training_cfg.get("amp_level", "O1"),
    )

    dist_barrier()
    if not main_process:
        return

    eval_dataset = test_dataset if resolved["final_eval_split"] == "test" else val_dataset
    eval_pairs = bundles["test_pairs"] if resolved["final_eval_split"] == "test" else bundles["val_pairs"]
    payload = evaluate_and_export_best_model(runtime, eval_dataset, eval_pairs, eval_transforms)

    print("[info] best_model_dir=", payload["best_model_dir"])
    print("[info] best_model_metrics=", save_dir / "best_model_metrics.json")
    print("[info] per_class_metrics=", save_dir / "per_class_metrics.csv")
    print("[info] confusion_matrix=", save_dir / "confusion_matrix.npy")
    print("[info] pred_dir=", save_dir / "pred")
    print("[info] vis_dir=", save_dir / "vis")


if __name__ == "__main__":
    main()
