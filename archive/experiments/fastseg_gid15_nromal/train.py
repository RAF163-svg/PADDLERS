#!/usr/bin/env python3

import argparse
import json
import math
import os
import random
import sys
import traceback
from pathlib import Path

import cv2
import numpy as np
import paddle
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import paddlers as pdrs
from paddlers import transforms as T
from paddlers.tasks.load_model import load_model


WORK_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = WORK_DIR / "output"
DEFAULT_DATA_CANDIDATES = [
    Path("${GID15_DATASET_ROOT}/gid15_seed20260320_tile640_stride448"),
    Path("${GID15_DATASET_ROOT}/paddlers_gid15_benchmark_20260320/data/gid15_seed20260320_tile640_stride448"),
    Path("${GID15_DATASET_ROOT}/gid-15"),
]
IGNORE_INDEX = 255
NEAR_ZERO_CLASS_THRESHOLD = 0.05


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def parse_bool(value):
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def parse_float_list(value, expected_len=3):
    if isinstance(value, (list, tuple)):
        floats = [float(v) for v in value]
    else:
        items = [item.strip() for item in str(value).split(",") if item.strip()]
        floats = [float(v) for v in items]
    if expected_len is not None and len(floats) != expected_len:
        raise argparse.ArgumentTypeError(
            f"Expected {expected_len} comma-separated floats, got {value}"
        )
    return floats


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


def is_main_process():
    return get_rank() == 0


def init_dist_if_needed():
    if get_world_size() > 1 and not paddle.distributed.parallel.parallel_helper._is_parallel_ctx_initialized():
        paddle.distributed.init_parallel_env()


def dist_barrier():
    if get_world_size() > 1 and paddle.distributed.parallel.parallel_helper._is_parallel_ctx_initialized():
        paddle.distributed.barrier()


def resolve_default_data_dir():
    for candidate in DEFAULT_DATA_CANDIDATES:
        if candidate.exists():
            return candidate
    return Path("/path/to/your/gid15_dataset")


def resolve_list_path(data_dir, requested, split_name):
    if requested is not None:
        return Path(requested)
    candidates = [
        Path(data_dir) / "lists" / f"{split_name}.txt",
        Path(data_dir) / f"{split_name}.txt",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def resolve_label_path(data_dir, requested):
    if requested is not None:
        return Path(requested)
    candidates = [
        Path(data_dir) / "labels.txt",
        Path(data_dir) / "label.txt",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    paddle.seed(seed)


def read_label_names(label_path):
    if not Path(label_path).exists():
        return []
    with open(label_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


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
    val_missing = set(val_stats["missing_classes"])
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
        if ratio > 0 and ratio < 0.001
    )
    report["val_missing_classes"] = sorted(val_missing)
    if test_stats is not None:
        report["test_missing_classes"] = sorted(test_stats["missing_classes"])
    return report


def resolve_target_class(label_names, num_classes, target_index, target_label, train_stats):
    if target_index is not None:
        if target_index < 0 or target_index >= num_classes:
            raise ValueError(
                f"oversample target index {target_index} is out of range for num_classes={num_classes}"
            )
        if target_index < len(label_names):
            return target_index, label_names[target_index]
        return target_index, f"class_{target_index}"

    if target_label and target_label in label_names:
        return label_names.index(target_label), target_label

    non_zero_classes = [
        (count, idx)
        for idx, count in enumerate(train_stats["class_pixel_counts"])
        if count > 0
    ]
    if not non_zero_classes:
        raise ValueError("unable to determine oversampling target class")
    _, idx = min(non_zero_classes)
    return idx, label_names[idx]


def pick_multi_rare_classes(train_stats, threshold, fallback_idx):
    selected = [
        idx
        for idx, ratio in enumerate(train_stats["class_pixel_ratio"])
        if ratio > 0 and ratio <= threshold
    ]
    if selected:
        return selected, "threshold"
    if fallback_idx is not None:
        return [fallback_idx], "fallback_single"
    non_zero = [
        idx for idx, count in enumerate(train_stats["class_pixel_counts"]) if count > 0
    ]
    if non_zero:
        smallest = min(non_zero, key=lambda idx: train_stats["class_pixel_counts"][idx])
        return [smallest], "fallback_smallest_nonzero"
    return [], "empty"


def build_single_selected_classes(target_class_idx):
    return [target_class_idx] if target_class_idx is not None else []


def compute_class_repeat_map(
    selected_class_indices,
    train_stats,
    oversample_mode,
    oversample_ratio,
    rare_class_threshold,
    max_oversample_ratio,
):
    class_repeat_map = {}
    for idx in selected_class_indices:
        ratio = train_stats["class_pixel_ratio"][idx]
        if oversample_mode == "single":
            repeat = max(1, oversample_ratio)
        else:
            if ratio <= 0:
                repeat = max_oversample_ratio
            else:
                repeat = int(math.ceil(rare_class_threshold / max(ratio, 1e-12)))
            repeat = max(2, repeat)
        repeat = min(max_oversample_ratio, repeat)
        class_repeat_map[idx] = max(1, repeat)
    return class_repeat_map


def generate_oversampled_train_list(
    train_list_path,
    data_dir,
    label_names,
    train_stats,
    oversample_mode,
    target_class_idx,
    target_class_label,
    oversample_ratio,
    rare_class_threshold,
    max_oversample_ratio,
    output_path,
    seed,
):
    output_path = Path(output_path)
    with open(train_list_path, "r", encoding="utf-8") as f:
        original_lines = [line if line.endswith("\n") else f"{line}\n" for line in f if line.strip()]

    if oversample_mode == "none":
        report = {
            "enabled": False,
            "mode": "none",
            "target_class_index": target_class_idx,
            "target_class_label": target_class_label,
            "selected_class_indices": [],
            "selected_classes": [],
            "rare_class_threshold": rare_class_threshold,
            "oversample_ratio": oversample_ratio,
            "max_oversample_ratio": max_oversample_ratio,
            "original_train_list": str(train_list_path),
            "effective_train_list": str(train_list_path),
            "original_sample_count": len(original_lines),
            "effective_sample_count": len(original_lines),
            "class_hit_stats": [],
            "repeat_histogram": {"1": len(original_lines)},
        }
        return Path(train_list_path), report

    if oversample_mode == "single":
        selected_class_indices = build_single_selected_classes(target_class_idx)
        selection_reason = "single_target"
    else:
        selected_class_indices, selection_reason = pick_multi_rare_classes(
            train_stats, rare_class_threshold, target_class_idx
        )

    class_repeat_map = compute_class_repeat_map(
        selected_class_indices,
        train_stats,
        oversample_mode,
        oversample_ratio,
        rare_class_threshold,
        max_oversample_ratio,
    )

    class_hit_stats = {
        idx: {
            "index": idx,
            "label": label_names[idx] if idx < len(label_names) else f"class_{idx}",
            "pixel_ratio": float(train_stats["class_pixel_ratio"][idx]),
            "pixel_count": int(train_stats["class_pixel_counts"][idx]),
            "image_count": int(train_stats["class_image_counts"][idx]),
            "hit_images_in_original_list": 0,
            "hit_pixels_in_original_list": 0,
            "applied_repeat": int(class_repeat_map.get(idx, 1)),
        }
        for idx in selected_class_indices
    }

    expanded_lines = []
    repeat_histogram = {}
    repeated_sample_count = 0

    for line in original_lines:
        expanded_lines.append(line)
        repeat = 1
        hit_classes = []
        _, mask_rel = line.strip().split()
        mask_path = Path(mask_rel)
        if not mask_path.is_absolute():
            mask_path = Path(data_dir) / mask_path
        mask = read_mask(mask_path)
        values, counts = np.unique(mask, return_counts=True)
        value_to_count = {int(v): int(c) for v, c in zip(values.tolist(), counts.tolist())}

        for idx in selected_class_indices:
            pixel_count = value_to_count.get(idx, 0)
            if pixel_count > 0:
                hit_classes.append(idx)
                class_hit_stats[idx]["hit_images_in_original_list"] += 1
                class_hit_stats[idx]["hit_pixels_in_original_list"] += int(pixel_count)
                repeat = max(repeat, class_repeat_map.get(idx, 1))

        if repeat > 1:
            repeated_sample_count += 1
            expanded_lines.extend([line] * (repeat - 1))

        repeat_histogram[str(repeat)] = repeat_histogram.get(str(repeat), 0) + 1

    rng = np.random.default_rng(seed)
    rng.shuffle(expanded_lines)
    ensure_dir(output_path.parent)
    with open(output_path, "w", encoding="utf-8") as f:
        f.writelines(expanded_lines)

    selected_classes = [class_hit_stats[idx] for idx in selected_class_indices]
    report = {
        "enabled": True,
        "mode": oversample_mode,
        "selection_reason": selection_reason,
        "target_class_index": target_class_idx,
        "target_class_label": target_class_label,
        "selected_class_indices": selected_class_indices,
        "selected_classes": selected_classes,
        "rare_class_threshold": rare_class_threshold,
        "oversample_ratio": oversample_ratio,
        "max_oversample_ratio": max_oversample_ratio,
        "original_train_list": str(train_list_path),
        "effective_train_list": str(output_path),
        "original_sample_count": len(original_lines),
        "effective_sample_count": len(expanded_lines),
        "repeated_sample_count": repeated_sample_count,
        "repeat_histogram": repeat_histogram,
        "class_hit_stats": selected_classes,
    }
    return output_path, report


class RareClassAwareCrop:
    def __init__(
        self,
        crop_size,
        rare_class_indices,
        crop_retry=8,
        min_target_pixels=64,
        min_target_ratio=0.001,
        ignore_index=IGNORE_INDEX,
    ):
        if isinstance(crop_size, int):
            crop_size = (crop_size, crop_size)
        self.crop_size = tuple(crop_size)
        self.rare_class_indices = sorted(set(int(i) for i in rare_class_indices))
        self.crop_retry = int(crop_retry)
        self.min_target_pixels = int(min_target_pixels)
        self.min_target_ratio = float(min_target_ratio)
        self.ignore_index = int(ignore_index)
        self.fallback_crop = T.RandomCrop(crop_size=self.crop_size)

    def get_attrs_for_serialization(self):
        return {
            "crop_size": self.crop_size,
            "rare_class_indices": self.rare_class_indices,
            "crop_retry": self.crop_retry,
            "min_target_pixels": self.min_target_pixels,
            "min_target_ratio": self.min_target_ratio,
            "ignore_index": self.ignore_index,
        }

    def _select_target_mask(self, mask):
        if self.rare_class_indices:
            rare_mask = np.isin(mask, self.rare_class_indices)
            if rare_mask.any():
                return rare_mask
        foreground_mask = np.logical_and(mask != 0, mask != self.ignore_index)
        if foreground_mask.any():
            return foreground_mask
        return None

    def _sample_crop_box(self, image_shape, target_mask):
        crop_h, crop_w = self.crop_size
        im_h, im_w = image_shape[:2]
        if im_h < crop_h or im_w < crop_w:
            return None

        ys, xs = np.where(target_mask)
        if len(ys) == 0:
            return None

        for _ in range(self.crop_retry):
            idx = np.random.randint(0, len(ys))
            cy = int(ys[idx])
            cx = int(xs[idx])
            y_low = max(0, cy - crop_h + 1)
            y_high = min(cy, im_h - crop_h)
            x_low = max(0, cx - crop_w + 1)
            x_high = min(cx, im_w - crop_w)
            if y_low > y_high or x_low > x_high:
                continue
            y1 = np.random.randint(y_low, y_high + 1)
            x1 = np.random.randint(x_low, x_high + 1)
            y2 = y1 + crop_h
            x2 = x1 + crop_w
            cropped_target = target_mask[y1:y2, x1:x2]
            target_pixels = int(cropped_target.sum())
            target_ratio = float(target_pixels / cropped_target.size)
            if target_pixels >= self.min_target_pixels and target_ratio >= self.min_target_ratio:
                return x1, y1, x2, y2
        return None

    def _crop_array(self, array, crop_box):
        x1, y1, x2, y2 = crop_box
        if array.ndim == 2:
            return array[y1:y2, x1:x2]
        return array[y1:y2, x1:x2, ...]

    def __call__(self, sample):
        mask = sample.get("mask")
        if mask is None or self.crop_retry <= 0:
            return self.fallback_crop(sample)

        target_mask = self._select_target_mask(mask)
        crop_box = None
        if target_mask is not None:
            crop_box = self._sample_crop_box(sample["image"].shape, target_mask)

        if crop_box is None:
            return self.fallback_crop(sample)

        sample["image"] = self._crop_array(sample["image"], crop_box)
        if "image2" in sample:
            sample["image2"] = self._crop_array(sample["image2"], crop_box)
        if "mask" in sample:
            sample["mask"] = self._crop_array(sample["mask"], crop_box)
        if "aux_masks" in sample:
            sample["aux_masks"] = [self._crop_array(mask_item, crop_box) for mask_item in sample["aux_masks"]]
        if "target" in sample:
            sample["target"] = self._crop_array(sample["target"], crop_box)
        return sample


def build_accuracy_summary(metrics, details, label_names):
    confusion = np.asarray(details.get("confusion_matrix", []), dtype=np.float64)
    if confusion.size == 0:
        precision = [None for _ in label_names]
        recall = [None for _ in label_names]
    else:
        pred_area = confusion.sum(axis=0)
        label_area = confusion.sum(axis=1)
        diag = np.diag(confusion)
        precision = []
        recall = []
        for idx in range(len(diag)):
            if pred_area[idx] <= 0:
                precision.append(0.0)
            else:
                precision.append(float(diag[idx] / pred_area[idx]))
            if label_area[idx] <= 0:
                recall.append(0.0)
            else:
                recall.append(float(diag[idx] / label_area[idx]))

    per_class = []
    for idx, label in enumerate(label_names):
        f1 = metrics["category_F1-score"][idx]
        if isinstance(f1, np.generic):
            f1 = f1.item()
        if isinstance(f1, float) and not math.isfinite(f1):
            f1 = None
        per_class.append(
            {
                "label": label,
                "acc": float(metrics["category_acc"][idx]),
                "precision": precision[idx] if idx < len(precision) else None,
                "recall": recall[idx] if idx < len(recall) else None,
                "iou": float(metrics["category_iou"][idx]),
                "f1": f1,
            }
        )
    return {
        "overall_accuracy": float(metrics["oacc"]),
        "mean_iou": float(metrics["miou"]),
        "kappa": float(metrics["kappa"]),
        "per_class": per_class,
    }


def to_builtin(value):
    if isinstance(value, dict):
        return {str(k): to_builtin(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_builtin(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return to_builtin(value.tolist())
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(to_builtin(payload), f, ensure_ascii=False, indent=2)


def append_jsonl(path, payload):
    ensure_dir(Path(path).parent)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(to_builtin(payload), ensure_ascii=False) + "\n")


def color_palette(num_classes):
    palette = np.zeros((num_classes, 3), dtype=np.uint8)
    for idx in range(num_classes):
        value = idx
        for bit in range(8):
            palette[idx, 0] |= ((value >> 0) & 1) << (7 - bit)
            palette[idx, 1] |= ((value >> 1) & 1) << (7 - bit)
            palette[idx, 2] |= ((value >> 2) & 1) << (7 - bit)
            value >>= 3
    return palette


def save_prediction_previews(model, pairs, transforms, output_dir, num_classes, count=8):
    if count <= 0:
        return
    ensure_dir(output_dir)
    palette = color_palette(num_classes)
    preview_manifest = []

    for idx, (image_path, mask_path) in enumerate(pairs[:count]):
        prediction = model.predict(str(image_path), transforms=transforms)
        label_map = prediction["label_map"].astype(np.uint8)
        color_mask = palette[label_map]

        stem = f"{idx:03d}_{image_path.stem}"
        gray_path = Path(output_dir) / f"{stem}_pred.png"
        color_path = Path(output_dir) / f"{stem}_pred_color.png"
        overlay_path = Path(output_dir) / f"{stem}_overlay.png"

        cv2.imwrite(str(gray_path), label_map)
        cv2.imwrite(str(color_path), cv2.cvtColor(color_mask, cv2.COLOR_RGB2BGR))

        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is not None:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            overlay = (0.6 * image_rgb + 0.4 * color_mask).astype(np.uint8)
            cv2.imwrite(str(overlay_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

        preview_manifest.append(
            {
                "image": str(image_path),
                "mask": str(mask_path),
                "prediction_gray": str(gray_path),
                "prediction_color": str(color_path),
                "overlay": str(overlay_path),
            }
        )

    write_json(Path(output_dir) / "preview_manifest.json", preview_manifest)


def build_optimizer(model, args, num_steps_each_epoch):
    total_steps = max(1, args.epochs * num_steps_each_epoch)
    if args.lr_scheduler == "poly":
        lr = paddle.optimizer.lr.PolynomialDecay(
            learning_rate=args.learning_rate,
            decay_steps=total_steps,
            end_lr=args.min_lr,
            power=0.9,
        )
    elif args.lr_scheduler == "cosine":
        lr = paddle.optimizer.lr.CosineAnnealingDecay(
            learning_rate=args.learning_rate,
            T_max=total_steps,
            eta_min=args.min_lr,
        )
    elif args.lr_scheduler == "step":
        step_epochs = max(1, args.epochs // 3)
        lr = paddle.optimizer.lr.StepDecay(
            learning_rate=args.learning_rate,
            step_size=max(1, num_steps_each_epoch * step_epochs),
            gamma=0.5,
        )
    else:
        lr = args.learning_rate

    warmup_steps = args.warmup_steps
    if warmup_steps <= 0 and args.warmup_epochs > 0:
        warmup_steps = max(1, int(args.warmup_epochs * num_steps_each_epoch))
    if warmup_steps > 0:
        lr = paddle.optimizer.lr.LinearWarmup(
            learning_rate=lr,
            warmup_steps=warmup_steps,
            start_lr=0.0,
            end_lr=args.learning_rate,
        )

    optimizer = paddle.optimizer.Momentum(
        learning_rate=lr,
        parameters=model.net.parameters(),
        momentum=0.9,
        weight_decay=args.weight_decay,
    )
    return optimizer, warmup_steps


def install_eval_history_hook(model, save_dir, label_names):
    history_path = Path(save_dir) / "metrics_history.jsonl"
    original_evaluate = model.evaluate

    def wrapped_evaluate(eval_dataset, batch_size=1, return_details=False):
        result = original_evaluate(eval_dataset, batch_size=batch_size, return_details=return_details)
        if not is_main_process():
            return result

        if return_details:
            metrics, details = result
        else:
            metrics, details = result, None

        entry = {
            "epoch": int(getattr(model, "completed_epochs", 0)),
            "miou": float(metrics["miou"]),
            "oacc": float(metrics["oacc"]),
            "kappa": float(metrics["kappa"]),
            "category_iou": [float(x) for x in metrics["category_iou"]],
            "category_acc": [float(x) for x in metrics["category_acc"]],
            "category_f1": [
                None if isinstance(x, float) and not math.isfinite(x) else float(x)
                for x in metrics["category_F1-score"]
            ],
            "near_zero_classes": [
                label_names[idx]
                for idx, acc in enumerate(metrics["category_acc"])
                if float(acc) <= NEAR_ZERO_CLASS_THRESHOLD
            ],
        }
        if details is not None and "confusion_matrix" in details:
            entry["has_confusion_matrix"] = True
        append_jsonl(history_path, entry)
        return result

    model.evaluate = wrapped_evaluate


def read_metrics_history(history_path):
    history = []
    if not Path(history_path).exists():
        return history
    with open(history_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            history.append(json.loads(line))
    history.sort(key=lambda item: item.get("epoch", 0))
    return history


def collect_epoch_metrics(save_dir):
    history = read_metrics_history(Path(save_dir) / "metrics_history.jsonl")
    if history:
        return history

    collected = []
    for epoch_dir in sorted(Path(save_dir).glob("epoch_*"), key=lambda p: int(p.name.split("_")[-1])):
        model_yml = epoch_dir / "model.yml"
        if not model_yml.exists():
            continue
        payload = yaml.unsafe_load(model_yml.read_text(encoding="utf-8"))
        attrs = payload.get("_Attributes", {})
        eval_metrics = attrs.get("eval_metrics", {})
        if not eval_metrics:
            continue
        metric_key = next(iter(eval_metrics.keys()))
        collected.append(
            {
                "epoch": int(epoch_dir.name.split("_")[-1]),
                metric_key: float(eval_metrics[metric_key]),
            }
        )
    return collected


def summarize_metrics_history(history, label_names):
    if not history:
        return {
            "history_available": False,
            "best_epoch": None,
            "best_val_miou": None,
            "best_val_oacc": None,
            "best_val_kappa": None,
            "last_epochs_declining": False,
            "near_zero_classes_recent": [],
            "history_length": 0,
        }

    history = sorted(history, key=lambda item: item.get("epoch", 0))
    best_entry = max(history, key=lambda item: item.get("miou", float("-inf")))
    recent_window = history[-min(5, len(history)) :]
    decline_window = recent_window[-min(4, len(recent_window)) :]
    declining = (
        len(decline_window) >= 3
        and all(
            decline_window[idx]["miou"] < decline_window[idx - 1]["miou"]
            for idx in range(1, len(decline_window))
        )
    )

    near_zero_recent = []
    recent_for_classes = history[-min(3, len(history)) :]
    if recent_for_classes and "category_acc" in recent_for_classes[-1]:
        for idx, label in enumerate(label_names):
            acc_values = []
            for entry in recent_for_classes:
                values = entry.get("category_acc")
                if values is None or idx >= len(values):
                    continue
                acc_values.append(float(values[idx]))
            if acc_values and max(acc_values) <= NEAR_ZERO_CLASS_THRESHOLD:
                near_zero_recent.append(label)

    return {
        "history_available": True,
        "best_epoch": int(best_entry.get("epoch")) if best_entry.get("epoch") is not None else None,
        "best_val_miou": float(best_entry.get("miou")) if best_entry.get("miou") is not None else None,
        "best_val_oacc": float(best_entry.get("oacc")) if best_entry.get("oacc") is not None else None,
        "best_val_kappa": float(best_entry.get("kappa")) if best_entry.get("kappa") is not None else None,
        "last_epochs_declining": declining,
        "near_zero_classes_recent": near_zero_recent,
        "history_length": len(history),
        "recent_epochs": recent_window,
    }


def safe_best_model_evaluation(
    save_dir,
    target_dataset,
    eval_pairs,
    eval_transforms,
    label_names,
    eval_name,
    preview_count,
    num_classes,
    model_name,
):
    best_model_dir = Path(save_dir) / "best_model"
    if not best_model_dir.exists():
        return None

    best_model = load_model(str(best_model_dir))
    best_metrics, best_details = best_model.evaluate(
        target_dataset, batch_size=1, return_details=True
    )

    accuracy_summary = {
        "evaluated_split": eval_name,
        "model": model_name,
        "best_model_dir": str(best_model_dir),
        "summary": build_accuracy_summary(best_metrics, best_details, label_names),
    }
    write_json(Path(save_dir) / "accuracy_summary.json", accuracy_summary)
    write_json(
        Path(save_dir) / "best_model_metrics.json",
        {
            "evaluated_split": eval_name,
            "metrics": best_metrics,
            "details": best_details,
        },
    )

    save_prediction_previews(
        best_model,
        eval_pairs,
        eval_transforms,
        Path(save_dir) / "best_model_previews",
        num_classes=num_classes,
        count=preview_count,
    )
    return {
        "accuracy_summary": accuracy_summary,
        "metrics": best_metrics,
        "details": best_details,
        "best_model_dir": str(best_model_dir),
    }


def write_final_status_and_summary(
    save_dir,
    status,
    exception,
    exception_traceback,
    history_summary,
    evaluation_payload,
    dataset_report_path,
    train_config_path,
):
    best_model_dir = Path(save_dir) / "best_model"
    training_status = {
        "status": status,
        "exception_type": exception.__class__.__name__ if exception is not None else None,
        "exception_message": str(exception) if exception is not None else None,
        "traceback": exception_traceback,
        "best_model_exists": best_model_dir.exists(),
    }
    write_json(Path(save_dir) / "training_status.json", training_status)

    run_summary = {
        "model_name": "FactSeg",
        "training_status_path": str(Path(save_dir) / "training_status.json"),
        "metrics_history_path": str(Path(save_dir) / "metrics_history.jsonl"),
        "epoch_metrics_path": str(Path(save_dir) / "epoch_metrics.json"),
        "dataset_report_path": str(dataset_report_path),
        "train_config_path": str(train_config_path),
        "best_epoch": history_summary.get("best_epoch"),
        "best_val_miou": history_summary.get("best_val_miou"),
        "best_val_oacc": history_summary.get("best_val_oacc"),
        "best_val_kappa": history_summary.get("best_val_kappa"),
        "last_epochs_declining": history_summary.get("last_epochs_declining"),
        "near_zero_classes_recent": history_summary.get("near_zero_classes_recent"),
        "history_length": history_summary.get("history_length"),
        "recent_epochs": history_summary.get("recent_epochs"),
    }
    if evaluation_payload is not None:
        run_summary.update(
            {
                "evaluated_split": evaluation_payload["accuracy_summary"]["evaluated_split"],
                "best_model_dir": evaluation_payload["best_model_dir"],
                "accuracy_summary_path": str(Path(save_dir) / "accuracy_summary.json"),
                "best_model_metrics_path": str(Path(save_dir) / "best_model_metrics.json"),
                "preview_dir": str(Path(save_dir) / "best_model_previews"),
                "final_overall_accuracy": float(evaluation_payload["metrics"]["oacc"]),
                "final_mean_iou": float(evaluation_payload["metrics"]["miou"]),
                "final_kappa": float(evaluation_payload["metrics"]["kappa"]),
            }
        )
    write_json(Path(save_dir) / "run_summary.json", run_summary)


def build_train_transforms(args, rare_class_indices):
    transforms = [
        RareClassAwareCrop(
            crop_size=args.crop_size,
            rare_class_indices=rare_class_indices,
            crop_retry=args.crop_retry,
            min_target_pixels=args.min_target_pixels,
            min_target_ratio=args.min_target_ratio,
        ),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
    ]
    if args.enable_distort:
        transforms.append(
            T.RandomDistort(
                brightness_range=0.1,
                brightness_prob=0.5,
                contrast_range=0.1,
                contrast_prob=0.5,
                saturation_range=0.08,
                saturation_prob=0.4,
                hue_range=5,
                hue_prob=0.2,
                random_apply=True,
                count=3,
            )
        )
    if args.enable_blur:
        transforms.append(T.RandomBlur(prob=0.05))
    transforms.append(T.Normalize(mean=args.normalize_mean, std=args.normalize_std))
    return T.Compose(transforms)


def build_eval_transforms(args):
    return T.Compose(
        [
            T.Normalize(mean=args.normalize_mean, std=args.normalize_std),
        ]
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="FactSeg training framework for GID-15 with dataset checks and best-model evaluation."
    )
    parser.add_argument("--data-dir", default=str(resolve_default_data_dir()))
    parser.add_argument("--train-list", default=None)
    parser.add_argument("--val-list", default=None)
    parser.add_argument("--test-list", default=None)
    parser.add_argument("--label-list", default=None)
    parser.add_argument("--num-classes", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--train-batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=0.0005)
    parser.add_argument("--crop-size", type=int, default=512)
    parser.add_argument("--save-interval-epochs", type=int, default=5)
    parser.add_argument("--log-interval-steps", type=int, default=20)
    parser.add_argument("--early-stop-patience", type=int, default=10)
    parser.add_argument("--num-workers", default="auto")
    parser.add_argument("--resume-checkpoint", default=None)
    parser.add_argument("--preview-count", type=int, default=8)
    parser.add_argument("--oversample-target-index", type=int, default=None)
    parser.add_argument("--oversample-target-label", default="artificial_grassland")
    parser.add_argument("--oversample-ratio", type=int, default=3)
    parser.add_argument("--oversample-mode", choices=["none", "single", "multi"], default="multi")
    parser.add_argument("--rare-class-threshold", type=float, default=0.01)
    parser.add_argument("--max-oversample-ratio", type=int, default=4)
    parser.add_argument("--crop-retry", type=int, default=8)
    parser.add_argument("--min-target-pixels", type=int, default=64)
    parser.add_argument("--min-target-ratio", type=float, default=0.001)
    parser.add_argument("--lr-scheduler", choices=["none", "poly", "cosine", "step"], default="poly")
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--warmup-epochs", type=float, default=1.0)
    parser.add_argument("--min-lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=4e-5)
    parser.add_argument("--enable-distort", type=parse_bool, default=True)
    parser.add_argument("--enable-blur", type=parse_bool, default=False)
    parser.add_argument("--normalize-mean", type=lambda v: parse_float_list(v, 3), default=[0.5, 0.5, 0.5])
    parser.add_argument("--normalize-std", type=lambda v: parse_float_list(v, 3), default=[0.5, 0.5, 0.5])
    parser.add_argument("--pretrain-weights", default="iSAID")
    parser.add_argument("--seed", type=int, default=20260325)
    parser.add_argument("--save-dir", default=str(OUTPUT_DIR))
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    init_dist_if_needed()
    rank = get_rank()
    world_size = get_world_size()
    main_process = rank == 0

    data_dir = Path(args.data_dir)
    save_dir = Path(args.save_dir)
    ensure_dir(save_dir)

    train_list = resolve_list_path(data_dir, args.train_list, "train")
    val_list = resolve_list_path(data_dir, args.val_list, "val")
    test_list = None if args.test_list == "" else resolve_list_path(data_dir, args.test_list, "test")
    if test_list is not None and not test_list.exists():
        test_list = None
    label_list = resolve_label_path(data_dir, args.label_list)

    train_pairs = read_file_pairs(train_list, data_dir)
    val_pairs = read_file_pairs(val_list, data_dir)
    test_pairs = read_file_pairs(test_list, data_dir) if test_list is not None else []

    label_names = read_label_names(label_list)
    dataset_report_path = save_dir / "dataset_report.json"
    oversample_report_path = save_dir / "oversample_report.json"
    train_config_path = save_dir / "train_config.json"
    history_path = save_dir / "metrics_history.jsonl"
    epoch_metrics_path = save_dir / "epoch_metrics.json"
    effective_train_list = train_list
    oversample_target_idx = None
    oversample_target_label = None
    eval_pairs = val_pairs
    eval_name = "val"
    val_dataset = None
    test_dataset = None
    eval_transforms = None
    exception = None
    exception_traceback = None
    final_status = "completed"
    model = None

    if main_process:
        print("[info] rank=", rank, "world_size=", world_size)
        print("[info] data_dir=", data_dir)
        print("[info] train_list=", train_list)
        print("[info] val_list=", val_list)
        print("[info] test_list=", test_list if test_list is not None else "None")
        print("[info] label_list=", label_list)

        train_stats = analyze_split("train", train_pairs, IGNORE_INDEX)
        val_stats = analyze_split("val", val_pairs, IGNORE_INDEX)
        test_stats = analyze_split("test", test_pairs, IGNORE_INDEX) if test_pairs else None

        max_valid_label = max(
            train_stats["max_valid_label"],
            val_stats["max_valid_label"],
            test_stats["max_valid_label"] if test_stats is not None else -1,
        )
        inferred_num_classes = max(len(label_names), max_valid_label + 1)
        num_classes = args.num_classes or inferred_num_classes
        if num_classes <= 0:
            raise ValueError("Failed to infer num_classes. Please pass --num-classes explicitly.")

        if not label_names:
            label_names = [f"class_{idx}" for idx in range(num_classes)]
        elif len(label_names) < num_classes:
            for idx in range(len(label_names), num_classes):
                label_names.append(f"class_{idx}")

        train_stats = finalize_split_stats(train_stats, num_classes, IGNORE_INDEX)
        val_stats = finalize_split_stats(val_stats, num_classes, IGNORE_INDEX)
        test_stats = finalize_split_stats(test_stats, num_classes, IGNORE_INDEX) if test_stats is not None else None
        dataset_report = build_dataset_report(train_stats, val_stats, test_stats, label_names, num_classes)

        if inferred_num_classes != num_classes:
            dataset_report["num_classes_warning"] = (
                f"labels/max_label inferred {inferred_num_classes}, but configured num_classes={num_classes}."
            )

        write_json(dataset_report_path, dataset_report)
        print("[info] num_classes=", num_classes)
        print("[info] labels=", label_names)
        print("[info] classes_missing_in_train=", dataset_report["classes_missing_in_train"])
        print(
            "[info] classes_present_in_val_but_missing_in_train=",
            dataset_report["classes_present_in_val_but_missing_in_train"],
        )
        if "classes_present_in_test_but_missing_in_train" in dataset_report:
            print(
                "[info] classes_present_in_test_but_missing_in_train=",
                dataset_report["classes_present_in_test_but_missing_in_train"],
            )
        print("[info] very_rare_train_classes=", dataset_report["very_rare_train_classes"])

        oversample_target_idx, oversample_target_label = resolve_target_class(
            label_names,
            num_classes,
            args.oversample_target_index,
            args.oversample_target_label,
            train_stats,
        )
        effective_train_list, oversample_report = generate_oversampled_train_list(
            train_list_path=train_list,
            data_dir=data_dir,
            label_names=label_names,
            train_stats=train_stats,
            oversample_mode=args.oversample_mode,
            target_class_idx=oversample_target_idx,
            target_class_label=oversample_target_label,
            oversample_ratio=args.oversample_ratio,
            rare_class_threshold=args.rare_class_threshold,
            max_oversample_ratio=args.max_oversample_ratio,
            output_path=save_dir / "train_oversampled.txt",
            seed=args.seed,
        )
        write_json(oversample_report_path, oversample_report)
        print("[info] oversample_mode=", args.oversample_mode)
        print("[info] oversample_target=", oversample_target_idx, oversample_target_label)
        print("[info] oversample_ratio=", args.oversample_ratio)
        print("[info] effective_train_list=", effective_train_list)

    dist_barrier()

    dataset_report = json.loads(dataset_report_path.read_text(encoding="utf-8"))
    oversample_report = json.loads(oversample_report_path.read_text(encoding="utf-8"))
    num_classes = int(dataset_report["num_classes"])
    label_names = list(dataset_report["labels"])
    oversample_target_idx = oversample_report.get("target_class_index")
    oversample_target_label = oversample_report.get("target_class_label")
    effective_train_list = Path(oversample_report["effective_train_list"])
    rare_class_indices = [int(idx) for idx in oversample_report.get("selected_class_indices", [])]

    train_transforms = build_train_transforms(args, rare_class_indices)
    eval_transforms = build_eval_transforms(args)

    train_dataset = pdrs.datasets.SegDataset(
        data_dir=str(data_dir),
        file_list=str(effective_train_list),
        label_list=str(label_list) if label_list.exists() else None,
        transforms=train_transforms,
        num_workers=args.num_workers,
        shuffle=True,
    )
    val_dataset = pdrs.datasets.SegDataset(
        data_dir=str(data_dir),
        file_list=str(val_list),
        label_list=str(label_list) if label_list.exists() else None,
        transforms=eval_transforms,
        num_workers=args.num_workers,
        shuffle=False,
    )
    if test_list is not None:
        test_dataset = pdrs.datasets.SegDataset(
            data_dir=str(data_dir),
            file_list=str(test_list),
            label_list=str(label_list) if label_list.exists() else None,
            transforms=eval_transforms,
            num_workers=args.num_workers,
            shuffle=False,
        )
        eval_pairs = test_pairs
        eval_name = "test"

    model = pdrs.tasks.seg.FactSeg(
        num_classes=num_classes,
        use_mixed_loss=True,
    )
    install_eval_history_hook(model, save_dir, label_names)
    num_steps_each_epoch = max(1, train_dataset.num_samples // args.train_batch_size)
    optimizer, warmup_steps = build_optimizer(model, args, num_steps_each_epoch)

    config = {
        "model_name": "FactSeg",
        "data_dir": str(data_dir),
        "train_list": str(train_list),
        "effective_train_list": str(effective_train_list),
        "val_list": str(val_list),
        "test_list": str(test_list) if test_list is not None else None,
        "label_list": str(label_list) if label_list.exists() else None,
        "num_classes": num_classes,
        "epochs": args.epochs,
        "train_batch_size": args.train_batch_size,
        "eval_batch_size": args.eval_batch_size,
        "learning_rate": args.learning_rate,
        "crop_size": args.crop_size,
        "save_interval_epochs": args.save_interval_epochs,
        "log_interval_steps": args.log_interval_steps,
        "early_stop_patience": args.early_stop_patience,
        "num_workers": args.num_workers,
        "resume_checkpoint": args.resume_checkpoint,
        "preview_count": args.preview_count,
        "oversample_mode": args.oversample_mode,
        "oversample_target_index": oversample_target_idx,
        "oversample_target_label": oversample_target_label,
        "oversample_ratio": args.oversample_ratio,
        "rare_class_threshold": args.rare_class_threshold,
        "max_oversample_ratio": args.max_oversample_ratio,
        "crop_retry": args.crop_retry,
        "min_target_pixels": args.min_target_pixels,
        "min_target_ratio": args.min_target_ratio,
        "selected_rare_class_indices": rare_class_indices,
        "lr_scheduler": args.lr_scheduler,
        "warmup_steps": warmup_steps,
        "warmup_epochs": args.warmup_epochs,
        "min_lr": args.min_lr,
        "weight_decay": args.weight_decay,
        "pretrain_weights": args.pretrain_weights,
        "use_mixed_loss": True,
        "enable_distort": args.enable_distort,
        "enable_blur": args.enable_blur,
        "normalize_mean": args.normalize_mean,
        "normalize_std": args.normalize_std,
        "seed": args.seed,
        "save_dir": str(save_dir),
    }
    if main_process:
        write_json(train_config_path, config)
        print("[info] start training, save_dir=", save_dir)

    train_succeeded = False
    try:
        model.train(
            num_epochs=args.epochs,
            train_dataset=train_dataset,
            train_batch_size=args.train_batch_size,
            eval_dataset=val_dataset,
            optimizer=optimizer,
            save_interval_epochs=args.save_interval_epochs,
            log_interval_steps=args.log_interval_steps,
            save_dir=str(save_dir),
            pretrain_weights=args.pretrain_weights,
            learning_rate=args.learning_rate,
            early_stop=True,
            early_stop_patience=args.early_stop_patience,
            use_vdl=True,
            resume_checkpoint=args.resume_checkpoint,
            precision="fp32",
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

        history = collect_epoch_metrics(save_dir)
        write_json(epoch_metrics_path, history)
        history_summary = summarize_metrics_history(history, label_names)

        evaluation_payload = None
        target_dataset = test_dataset if test_dataset is not None else val_dataset
        try:
            if target_dataset is not None and (save_dir / "best_model").exists():
                evaluation_payload = safe_best_model_evaluation(
                    save_dir=save_dir,
                    target_dataset=target_dataset,
                    eval_pairs=eval_pairs,
                    eval_transforms=eval_transforms,
                    label_names=label_names,
                    eval_name=eval_name,
                    preview_count=args.preview_count,
                    num_classes=num_classes,
                    model_name="FactSeg",
                )
        except Exception as eval_exc:
            if exception is None:
                final_status = "failed"
                exception = eval_exc
                exception_traceback = traceback.format_exc()

        write_final_status_and_summary(
            save_dir=save_dir,
            status="completed" if train_succeeded and exception is None else final_status,
            exception=exception,
            exception_traceback=exception_traceback,
            history_summary=history_summary,
            evaluation_payload=evaluation_payload,
            dataset_report_path=dataset_report_path,
            train_config_path=train_config_path,
        )

        print("[info] training_status=", save_dir / "training_status.json")
        print("[info] run_summary=", save_dir / "run_summary.json")
        print("[info] epoch_metrics=", save_dir / "epoch_metrics.json")
        if evaluation_payload is not None:
            print("[info] best_model_dir=", evaluation_payload["best_model_dir"])
            print("[info] accuracy_summary=", save_dir / "accuracy_summary.json")
            print("[info] best_model_metrics=", save_dir / "best_model_metrics.json")
        print("[info] dataset_report=", dataset_report_path)

        if exception is not None:
            raise exception


if __name__ == "__main__":
    main()
