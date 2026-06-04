#!/usr/bin/env python3

import argparse
import copy
import json
import math
import os
import random
import re
import sys
import traceback
from pathlib import Path

import cv2
import numpy as np
import paddle


REPO_ROOT = Path(__file__).resolve().parents[2]
WORK_DIR = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(WORK_DIR) not in sys.path:
    sys.path.insert(0, str(WORK_DIR))

import paddlers as pdrs
from paddlers import transforms as T
from paddlers.models import seg_losses

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

DEFAULT_CONFIG_PATH = WORK_DIR / "train_config.json"
DEFAULT_DATA_CANDIDATES = [
    Path("${GID15_DATASET_ROOT}/gid15_seed20260320_tile640_stride448"),
    Path("${GID15_DATASET_ROOT}/paddlers_gid15_benchmark_20260320/data/gid15_seed20260320_tile640_stride448"),
    Path("${GID15_DATASET_ROOT}/gid-15"),
]
IGNORE_INDEX = 255


class TeeStream:
    def __init__(self, *streams):
        self.streams = [stream for stream in streams if stream is not None]

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()

    def isatty(self):
        return False


def setup_tee_logging(log_path):
    ensure_dir(Path(log_path).parent)
    log_handle = open(log_path, "a", encoding="utf-8", buffering=1)
    sys.stdout = TeeStream(sys.__stdout__, log_handle)
    sys.stderr = TeeStream(sys.__stderr__, log_handle)
    return log_handle


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
        idx
        for idx, ratio in enumerate(train_stats["class_pixel_ratio"])
        if ratio > 0 and ratio < 0.01
    )
    report["val_missing_classes"] = sorted(val_stats["missing_classes"])
    if test_stats is not None:
        report["test_missing_classes"] = sorted(test_stats["missing_classes"])
    return report


def resolve_focus_class_indices(class_names, requested_names):
    requested_names = list(requested_names or [])
    indices = []
    missing = []
    for name in requested_names:
        if name in class_names:
            indices.append(class_names.index(name))
        else:
            missing.append(name)
    return sorted(set(indices)), missing


def generate_focus_oversampled_train_list(
        train_list_path,
        data_dir,
        class_names,
        focus_class_indices,
        oversample_repeat,
        light_oversample_ratio,
        output_path,
        seed):
    output_path = Path(output_path)
    with open(train_list_path, "r", encoding="utf-8") as f:
        original_lines = [line if line.endswith("\n") else f"{line}\n" for line in f if line.strip()]

    light_oversample_ratio = float(light_oversample_ratio)
    if (oversample_repeat <= 1 and light_oversample_ratio <= 0.0) or not focus_class_indices:
        report = {
            "enabled": False,
            "focus_class_indices": focus_class_indices,
            "focus_class_names": [class_names[idx] for idx in focus_class_indices],
            "oversample_repeat": int(oversample_repeat),
            "light_oversample_ratio": light_oversample_ratio,
            "source_train_list": str(train_list_path),
            "effective_train_list": str(train_list_path),
            "source_sample_count": len(original_lines),
            "effective_sample_count": len(original_lines),
            "hit_image_counts": {class_names[idx]: 0 for idx in focus_class_indices},
        }
        return Path(train_list_path), report

    expanded_lines = []
    hit_image_counts = {idx: 0 for idx in focus_class_indices}
    hit_pixel_counts = {idx: 0 for idx in focus_class_indices}
    repeated_sample_count = 0
    light_selected_sample_count = 0
    added_sample_count = 0
    rng = np.random.default_rng(seed)

    for line in original_lines:
        expanded_lines.append(line)
        _, mask_rel = line.strip().split()
        mask_path = Path(mask_rel)
        if not mask_path.is_absolute():
            mask_path = Path(data_dir) / mask_path
        mask = read_mask(mask_path)

        hit_classes = []
        for idx in focus_class_indices:
            pixel_count = int((mask == idx).sum())
            if pixel_count > 0:
                hit_classes.append(idx)
                hit_image_counts[idx] += 1
                hit_pixel_counts[idx] += pixel_count

        if hit_classes:
            extra_copies = max(0, int(oversample_repeat) - 1)
            if extra_copies <= 0 and light_oversample_ratio > 0.0 and rng.random() < light_oversample_ratio:
                extra_copies = 1
                light_selected_sample_count += 1
            if extra_copies > 0:
                expanded_lines.extend([line] * extra_copies)
                repeated_sample_count += 1
                added_sample_count += extra_copies

    rng.shuffle(expanded_lines)
    ensure_dir(output_path.parent)
    with open(output_path, "w", encoding="utf-8") as f:
        f.writelines(expanded_lines)

    report = {
        "enabled": True,
        "focus_class_indices": focus_class_indices,
        "focus_class_names": [class_names[idx] for idx in focus_class_indices],
        "oversample_repeat": int(oversample_repeat),
        "light_oversample_ratio": light_oversample_ratio,
        "source_train_list": str(train_list_path),
        "effective_train_list": str(output_path),
        "source_sample_count": len(original_lines),
        "effective_sample_count": len(expanded_lines),
        "repeated_sample_count": repeated_sample_count,
        "light_selected_sample_count": light_selected_sample_count,
        "added_sample_count": added_sample_count,
        "hit_image_counts": {
            class_names[idx]: int(hit_image_counts[idx]) for idx in focus_class_indices
        },
        "hit_pixel_counts": {
            class_names[idx]: int(hit_pixel_counts[idx]) for idx in focus_class_indices
        },
    }
    return output_path, report


def load_config(config_path):
    config = read_json(config_path)
    config["_config_path"] = str(Path(config_path).resolve())
    return config


def resolve_hrnet_width(model_cfg, config=None):
    width = model_cfg.get("width")
    backbone = model_cfg.get("backbone")
    if not backbone and config is not None:
        backbone = config.get("backbone")

    if width is None and backbone is not None:
        match = re.fullmatch(r"HRNet_W(18|48)", str(backbone))
        if not match:
            raise ValueError(f"Unsupported HRNet backbone: {backbone}")
        width = int(match.group(1))

    if width is None:
        width = 48
    width = int(width)
    if width not in (18, 48):
        raise ValueError(f"HRNet width must be 18 or 48, got {width}.")
    return width, f"HRNet_W{width}"


def resolve_runtime(config, args=None):
    training_cfg = config["training"]
    evaluation_cfg = config["evaluation"]
    model_cfg = config["model"]

    model_name = model_cfg.get("name") or config.get("model_name") or "HRNet"
    if model_name != "HRNet":
        raise ValueError(f"Only HRNet is supported in this experiment, got {model_name}.")
    hrnet_width, backbone_name = resolve_hrnet_width(model_cfg, config=config)

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

    selection_metric = evaluation_cfg.get("selection_metric", "miou")
    if selection_metric != "miou":
        raise ValueError(
            "PaddleRS segmentation best-model selection follows the first eval metric. "
            "Please keep evaluation.selection_metric set to 'miou'."
        )

    eval_interval = int(training_cfg.get("eval_interval", training_cfg["save_interval_epochs"]))
    save_interval_epochs = int(training_cfg["save_interval_epochs"])
    if eval_interval != save_interval_epochs:
        raise ValueError(
            "In PaddleRS segmentation training, evaluation is tied to save_interval_epochs. "
            "Please keep training.eval_interval equal to training.save_interval_epochs."
        )

    runtime = copy.deepcopy(config)
    runtime["model"]["name"] = model_name
    runtime["model"]["backbone"] = backbone_name
    runtime["model"]["width"] = hrnet_width
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
        "hrnet_width": hrnet_width,
        "backbone_name": backbone_name,
    }
    return runtime


def build_train_transforms(runtime):
    training_cfg = runtime["training"]
    aug_cfg = runtime["augmentation"]
    norm_cfg = runtime["normalization"]
    crop_size = training_cfg.get("crop_size") or training_cfg.get("input_size")

    transforms = [T.RandomCrop(crop_size=crop_size)]
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


def compute_class_weights(train_stats, loss_cfg):
    ratios = np.asarray(train_stats["class_pixel_ratio"], dtype=np.float64)
    weights = np.ones_like(ratios, dtype=np.float64)
    valid = ratios > 0
    if not valid.any():
        return weights.tolist()

    strategy = loss_cfg.get("class_weight_strategy", "inverse_sqrt")
    if strategy == "inverse_sqrt":
        weights[valid] = 1.0 / np.sqrt(np.maximum(ratios[valid], 1e-12))
    elif strategy == "median_frequency":
        median_ratio = float(np.median(ratios[valid]))
        weights[valid] = median_ratio / np.maximum(ratios[valid], 1e-12)
    else:
        raise ValueError(f"Unsupported class_weight_strategy: {strategy}")

    weights = weights / float(weights[valid].mean())
    clip_cfg = loss_cfg.get("class_weight_clip", [0.75, 2.0])
    weights = np.clip(weights, float(clip_cfg[0]), float(clip_cfg[1]))
    return weights.astype(np.float32).tolist()


def build_model(runtime):
    resolved = runtime["_resolved"]
    model_cfg = runtime["model"]
    loss_cfg = runtime["loss"]
    model_name = model_cfg.get("name") or runtime.get("model_name") or "HRNet"
    if model_name != "HRNet":
        raise ValueError(f"Only HRNet is supported in this experiment, got {model_name}.")

    return pdrs.tasks.seg.HRNet(
        in_channels=model_cfg.get("in_channels", 3),
        num_classes=resolved["num_classes"],
        width=resolved["hrnet_width"],
        use_mixed_loss=loss_cfg.get("use_mixed_loss", True),
        align_corners=model_cfg.get("align_corners", False),
    )


def configure_model_losses(model, runtime, train_stats):
    loss_cfg = runtime["loss"]
    use_mixed_loss = bool(loss_cfg.get("use_mixed_loss", True))
    ce_lovasz_coef = list(loss_cfg.get("ce_lovasz_coef", [0.8, 0.2]))
    enable_class_weights = bool(loss_cfg.get("enable_class_weights", False))

    class_weights = None
    if enable_class_weights:
        class_weights = compute_class_weights(train_stats, loss_cfg)

    use_default_losses = (
        use_mixed_loss
        and not enable_class_weights
        and ce_lovasz_coef == [0.8, 0.2]
    )

    if not use_default_losses:
        ce_loss = seg_losses.CrossEntropyLoss(weight=class_weights)
        if use_mixed_loss:
            branch_loss = seg_losses.MixedLoss(
                losses=[ce_loss, seg_losses.LovaszSoftmaxLoss()],
                coef=ce_lovasz_coef,
            )
        else:
            branch_loss = ce_loss
        model.set_losses([branch_loss], weights=[1.0])

    return {
        "use_default_hrnet_losses": use_default_losses,
        "use_mixed_loss": use_mixed_loss,
        "ce_lovasz_coef": ce_lovasz_coef,
        "enable_class_weights": enable_class_weights,
        "class_weight_strategy": loss_cfg.get("class_weight_strategy"),
        "class_weights": class_weights,
    }


def build_optimizer(model, runtime, num_steps_each_epoch):
    training_cfg = runtime["training"]
    scheduler_cfg = runtime["scheduler"]
    total_steps = max(1, int(training_cfg["epochs"]) * num_steps_each_epoch)
    learning_rate = float(training_cfg["learning_rate"])
    scheduler_name = scheduler_cfg.get("name", "poly")
    min_lr = float(scheduler_cfg.get("min_lr", 0.0))

    if scheduler_name == "poly":
        lr = paddle.optimizer.lr.PolynomialDecay(
            learning_rate=learning_rate,
            decay_steps=total_steps,
            end_lr=min_lr,
            power=float(scheduler_cfg.get("power", 0.9)),
        )
    elif scheduler_name == "cosine":
        lr = paddle.optimizer.lr.CosineAnnealingDecay(
            learning_rate=learning_rate,
            T_max=total_steps,
            eta_min=min_lr,
        )
    elif scheduler_name == "step":
        step_epochs = int(scheduler_cfg.get("step_epochs", max(1, int(training_cfg["epochs"]) // 3)))
        lr = paddle.optimizer.lr.StepDecay(
            learning_rate=learning_rate,
            step_size=max(1, num_steps_each_epoch * step_epochs),
            gamma=float(scheduler_cfg.get("gamma", 0.5)),
        )
    elif scheduler_name == "none":
        lr = learning_rate
    else:
        raise ValueError(f"Unsupported scheduler.name: {scheduler_name}")

    warmup_steps = int(scheduler_cfg.get("warmup_steps", 0))
    warmup_epochs = float(scheduler_cfg.get("warmup_epochs", 0.0))
    if warmup_steps <= 0 and warmup_epochs > 0:
        warmup_steps = max(1, int(warmup_epochs * num_steps_each_epoch))
    if warmup_steps > 0:
        lr = paddle.optimizer.lr.LinearWarmup(
            learning_rate=lr,
            warmup_steps=warmup_steps,
            start_lr=0.0,
            end_lr=learning_rate,
        )

    optimizer = paddle.optimizer.Momentum(
        learning_rate=lr,
        parameters=model.net.parameters(),
        momentum=float(scheduler_cfg.get("momentum", 0.9)),
        weight_decay=float(scheduler_cfg.get("weight_decay", 4e-5)),
    )
    return optimizer, warmup_steps


def build_runtime_train_config(runtime, effective_train_list, oversample_report, loss_report, warmup_steps):
    resolved = runtime["_resolved"]
    training_cfg = runtime["training"]
    scheduler_cfg = runtime["scheduler"]

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
        "model_name": runtime["model"].get("name"),
        "backbone": resolved["backbone_name"],
        "model": to_builtin(runtime["model"]),
        "pretrained": to_builtin(runtime["pretrained"]),
        "training": {
            **to_builtin(training_cfg),
            "resume_checkpoint": resolved["resume_checkpoint"],
        },
        "scheduler": {
            **to_builtin(scheduler_cfg),
            "warmup_steps": warmup_steps,
        },
        "augmentation": to_builtin(runtime["augmentation"]),
        "normalization": to_builtin(runtime["normalization"]),
        "sampling": to_builtin(runtime["sampling"]),
        "loss": to_builtin(runtime["loss"]),
        "loss_report": to_builtin(loss_report),
        "evaluation": {
            **to_builtin(runtime["evaluation"]),
            "final_eval_split": resolved["final_eval_split"],
        },
        "save_dir": str(resolved["save_dir"]),
        "oversample_report": to_builtin(oversample_report),
    }


def load_runtime_model_from_checkpoint(runtime, checkpoint_dir, eval_transforms=None):
    checkpoint_dir = Path(checkpoint_dir)
    weight_path = checkpoint_dir / "model.pdparams"
    if not weight_path.exists():
        raise FileNotFoundError(f"Model weights not found: {weight_path}")

    model = build_model(runtime)
    state_dict = paddle.load(str(weight_path))
    model.net.set_state_dict(state_dict)
    model.net.eval()
    model.labels = list(runtime["_resolved"]["class_names"])
    if eval_transforms is not None:
        model.test_transforms = eval_transforms
    return model


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

    best_model = load_runtime_model_from_checkpoint(runtime, best_model_dir, eval_transforms=eval_transforms)
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
    write_metrics_text(save_dir / "eval.log", payload)

    if evaluation_cfg.get("save_per_class_metrics_csv", True):
        write_per_class_csv(save_dir / "per_class_metrics.csv", payload["summary"])
    if evaluation_cfg.get("compute_confusion_matrix", True):
        save_confusion_matrix(save_dir / "confusion_matrix.npy", details["confusion_matrix"])

    pred_manifest = []
    export_predictions = evaluation_cfg.get("export_predictions", True)
    export_visualizations = evaluation_cfg.get("export_visualizations", True)
    if export_predictions or export_visualizations:
        pred_manifest = export_prediction_artifacts(
            model=best_model,
            pairs=eval_pairs,
            transforms=eval_transforms,
            dataset_root=resolved["dataset_root"],
            pred_dir=save_dir / "pred",
            vis_dir=save_dir / "vis",
            num_classes=resolved["num_classes"],
            split_name=eval_split,
            export_predictions=export_predictions,
            export_visualizations=export_visualizations,
        )
        write_json(save_dir / "pred_manifest.json", pred_manifest)

    run_summary = {
        "model": runtime["model"]["name"],
        "backbone": resolved["backbone_name"],
        "selection_split": selection_split,
        "selection_metric": evaluation_cfg.get("selection_metric", "miou"),
        "evaluated_split": eval_split,
        "best_model_dir": str(best_model_dir),
        "dataset_root": str(resolved["dataset_root"]),
        "dataset_report_path": str(save_dir / "dataset_report.json"),
        "train_config_path": str(save_dir / "train_config.json"),
        "best_model_metrics_json": str(save_dir / "best_model_metrics.json"),
        "best_model_metrics_txt": str(save_dir / "best_model_metrics.txt"),
        "eval_log_path": str(save_dir / "eval.log"),
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
        description="HRNet training framework for GID-15 with conservative best-model export and resume support."
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
    if main_process:
        setup_tee_logging(save_dir / "train.log")

    bundles = build_seg_datasets(runtime)
    train_pairs = bundles["train_pairs"]
    val_pairs = bundles["val_pairs"]
    test_pairs = bundles["test_pairs"]

    dataset_report_path = save_dir / "dataset_report.json"
    oversample_report_path = save_dir / "oversample_report.json"
    train_config_path = save_dir / "train_config.json"
    training_status_path = save_dir / "training_status.json"
    exception = None
    exception_traceback = None
    final_status = "completed"

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
        print("[info] backbone=", resolved["backbone_name"])
        print("[info] resume_checkpoint=", resolved["resume_checkpoint"])

        dataset_report, train_stats = build_dataset_reports(runtime, train_pairs, val_pairs, test_pairs)
        sampling_cfg = runtime["sampling"]
        focus_class_indices, missing_focus_names = resolve_focus_class_indices(
            resolved["class_names"],
            sampling_cfg.get("focus_class_names", []),
        )
        if missing_focus_names:
            print("[warn] unresolved focus_class_names=", missing_focus_names)

        if sampling_cfg.get("enable_train_oversample", False):
            effective_train_list, oversample_report = generate_focus_oversampled_train_list(
                train_list_path=resolved["train_file_list"],
                data_dir=resolved["dataset_root"],
                class_names=resolved["class_names"],
                focus_class_indices=focus_class_indices,
                oversample_repeat=int(sampling_cfg.get("oversample_repeat", 1)),
                light_oversample_ratio=float(sampling_cfg.get("light_oversample_ratio", 0.0)),
                output_path=save_dir / "train_oversampled.txt",
                seed=training_cfg["seed"],
            )
        else:
            effective_train_list = resolved["train_file_list"]
            oversample_report = {
                "enabled": False,
                "focus_class_indices": focus_class_indices,
                "focus_class_names": [resolved["class_names"][idx] for idx in focus_class_indices],
                "oversample_repeat": 1,
                "light_oversample_ratio": 0.0,
                "source_train_list": str(resolved["train_file_list"]),
                "effective_train_list": str(effective_train_list),
            }

        write_json(dataset_report_path, dataset_report)
        write_json(oversample_report_path, oversample_report)
        print("[info] num_classes=", resolved["num_classes"])
        print("[info] labels=", resolved["class_names"])
        print("[info] very_rare_train_classes=", dataset_report["very_rare_train_classes"])
        print("[info] focus_class_indices=", focus_class_indices)
        print("[info] focus_class_names=", oversample_report["focus_class_names"])
        print("[info] effective_train_list=", effective_train_list)
        print("[info] final_eval_split=", resolved["final_eval_split"])

    dist_barrier()

    dataset_report = read_json(dataset_report_path)
    oversample_report = read_json(oversample_report_path)
    train_stats = dataset_report["train"]
    effective_train_list = Path(oversample_report["effective_train_list"])

    bundles = build_seg_datasets(runtime, effective_train_list=effective_train_list)
    train_dataset = bundles["train_dataset"]
    val_dataset = bundles["val_dataset"]
    test_dataset = bundles["test_dataset"]
    eval_transforms = bundles["eval_transforms"]

    model = build_model(runtime)
    loss_report = configure_model_losses(model, runtime, train_stats)
    num_steps_each_epoch = max(
        1,
        int(math.ceil(train_dataset.num_samples / training_cfg["train_batch_size"])),
    )
    optimizer, warmup_steps = build_optimizer(model, runtime, num_steps_each_epoch)

    if main_process:
        write_json(
            train_config_path,
            build_runtime_train_config(
                runtime,
                effective_train_list=effective_train_list,
                oversample_report=oversample_report,
                loss_report=loss_report,
                warmup_steps=warmup_steps,
            ),
        )
        print("[info] loss_report=", loss_report)
        print("[info] start training, save_dir=", save_dir)

    pretrain_cfg = runtime["pretrained"]
    pretrain_weights = pretrain_cfg.get("weights_path")
    if pretrain_weights:
        pretrain_weights = str(resolve_existing_path(WORK_DIR, pretrain_weights))
    else:
        pretrain_weights = pretrain_cfg.get("weights")
    if resolved["resume_checkpoint"] is not None:
        pretrain_weights = None

    train_succeeded = False
    try:
        # Save a checkpoint every epoch so SSH/tmux/nohup interruptions lose less progress.
        model.train(
            num_epochs=training_cfg["epochs"],
            train_dataset=train_dataset,
            train_batch_size=training_cfg["train_batch_size"],
            eval_dataset=val_dataset,
            optimizer=optimizer,
            save_interval_epochs=training_cfg["save_interval_epochs"],
            log_interval_steps=training_cfg["log_interval_steps"],
            save_dir=str(save_dir),
            pretrain_weights=pretrain_weights,
            learning_rate=training_cfg["learning_rate"],
            early_stop=training_cfg["early_stop"],
            early_stop_patience=training_cfg["early_stop_patience"],
            use_vdl=True,
            resume_checkpoint=resolved["resume_checkpoint"],
            precision=training_cfg.get("precision", "fp32"),
            amp_level=training_cfg.get("amp_level", "O1"),
        )
        train_succeeded = True
    except KeyboardInterrupt as exc:
        final_status = "interrupted"
        exception = exc
        exception_traceback = traceback.format_exc()
    except Exception as exc:
        final_status = "failed"
        exception = exc
        exception_traceback = traceback.format_exc()
    finally:
        if train_succeeded:
            try:
                dist_barrier()
            except Exception:
                pass

        if not main_process:
            if exception is not None:
                raise exception
            return

        eval_dataset = test_dataset if resolved["final_eval_split"] == "test" else val_dataset
        eval_pairs = bundles["test_pairs"] if resolved["final_eval_split"] == "test" else bundles["val_pairs"]

        payload = None
        if eval_dataset is not None and (save_dir / "best_model").exists():
            try:
                payload = evaluate_and_export_best_model(
                    runtime,
                    eval_dataset=eval_dataset,
                    eval_pairs=eval_pairs,
                    eval_transforms=eval_transforms,
                )
            except Exception as eval_exc:
                if exception is None:
                    final_status = "failed"
                    exception = eval_exc
                    exception_traceback = traceback.format_exc()

        training_status = {
            "status": "completed" if train_succeeded and exception is None else final_status,
            "resume_checkpoint": resolved["resume_checkpoint"],
            "backbone": resolved["backbone_name"],
            "best_model_exists": (save_dir / "best_model").exists(),
            "train_log_path": str(save_dir / "train.log"),
            "eval_log_path": str(save_dir / "eval.log"),
            "exception_type": exception.__class__.__name__ if exception is not None else None,
            "exception_message": str(exception) if exception is not None else None,
            "traceback": exception_traceback,
        }
        if payload is not None:
            training_status.update(
                {
                    "best_model_dir": payload["best_model_dir"],
                    "overall_accuracy": payload["summary"]["overall_accuracy"],
                    "mean_iou": payload["summary"]["mean_iou"],
                    "mean_accuracy": payload["summary"]["mean_accuracy"],
                }
            )
        write_json(training_status_path, training_status)

        print("[info] training_status=", training_status_path)
        if payload is not None:
            print("[info] best_model_dir=", payload["best_model_dir"])
            print("[info] best_model_metrics=", save_dir / "best_model_metrics.json")
            print("[info] per_class_metrics=", save_dir / "per_class_metrics.csv")
            print("[info] confusion_matrix=", save_dir / "confusion_matrix.npy")
            print("[info] pred_dir=", save_dir / "pred")
            print("[info] vis_dir=", save_dir / "vis")

        if exception is not None:
            raise exception


if __name__ == "__main__":
    main()
