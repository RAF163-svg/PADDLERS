#!/usr/bin/env python3

import argparse
import json
import math
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


WORK_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = WORK_DIR / "output"
DEFAULT_DATA_CANDIDATES = [
    Path("${GID15_DATASET_ROOT}/gid15_seed20260320_tile640_stride448"),
    Path("${GID15_DATASET_ROOT}/paddlers_gid15_benchmark_20260320/data/gid15_seed20260320_tile640_stride448"),
    Path("${GID15_DATASET_ROOT}/gid-15"),
]
IGNORE_INDEX = 255


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


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


def generate_oversampled_train_list(
    train_list_path,
    data_dir,
    target_class_idx,
    target_class_label,
    oversample_ratio,
    output_path,
    seed,
):
    output_path = Path(output_path)
    with open(train_list_path, "r", encoding="utf-8") as f:
        original_lines = [line if line.endswith("\n") else f"{line}\n" for line in f if line.strip()]

    if oversample_ratio <= 1:
        report = {
            "enabled": False,
            "target_class_index": target_class_idx,
            "target_class_label": target_class_label,
            "oversample_ratio": oversample_ratio,
            "original_train_list": str(train_list_path),
            "effective_train_list": str(train_list_path),
            "original_sample_count": len(original_lines),
            "effective_sample_count": len(original_lines),
            "target_hit_images": 0,
        }
        return Path(train_list_path), report

    expanded_lines = []
    target_hit_images = 0
    target_hit_pixels = 0

    for line in original_lines:
        expanded_lines.append(line)
        _, mask_rel = line.strip().split()
        mask_path = Path(mask_rel)
        if not mask_path.is_absolute():
            mask_path = Path(data_dir) / mask_path
        mask = read_mask(mask_path)
        target_pixels = int((mask == target_class_idx).sum())
        if target_pixels > 0:
            target_hit_images += 1
            target_hit_pixels += target_pixels
            expanded_lines.extend([line] * (oversample_ratio - 1))

    rng = np.random.default_rng(seed)
    rng.shuffle(expanded_lines)
    ensure_dir(output_path.parent)
    with open(output_path, "w", encoding="utf-8") as f:
        f.writelines(expanded_lines)

    report = {
        "enabled": True,
        "target_class_index": target_class_idx,
        "target_class_label": target_class_label,
        "oversample_ratio": oversample_ratio,
        "original_train_list": str(train_list_path),
        "effective_train_list": str(output_path),
        "original_sample_count": len(original_lines),
        "effective_sample_count": len(expanded_lines),
        "target_hit_images": target_hit_images,
        "target_hit_pixels": target_hit_pixels,
    }
    return output_path, report


def build_accuracy_summary(metrics, details, label_names):
    confusion = np.asarray(details.get("confusion_matrix", []), dtype=np.float64)
    if confusion.size == 0:
        precision = [None for _ in label_names]
    else:
        pred_area = confusion.sum(axis=0)
        diag = np.diag(confusion)
        precision = []
        for idx in range(len(diag)):
            if pred_area[idx] <= 0:
                precision.append(0.0)
            else:
                precision.append(float(diag[idx] / pred_area[idx]))

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


def parse_args():
    parser = argparse.ArgumentParser(
        description="DeepLabV3P training framework for GID-15 with dataset checks and best-model evaluation."
    )
    parser.add_argument("--data-dir", default=str(resolve_default_data_dir()))
    parser.add_argument("--train-list", default=None)
    parser.add_argument("--val-list", default=None)
    parser.add_argument("--test-list", default=None)
    parser.add_argument("--label-list", default=None)
    parser.add_argument("--num-classes", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--train-batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--crop-size", type=int, default=512)
    parser.add_argument("--save-interval-epochs", type=int, default=5)
    parser.add_argument("--log-interval-steps", type=int, default=20)
    parser.add_argument("--early-stop-patience", type=int, default=10)
    parser.add_argument("--backbone", default="ResNet50_vd")
    parser.add_argument("--num-workers", default="auto")
    parser.add_argument("--resume-checkpoint", default=None)
    parser.add_argument("--preview-count", type=int, default=8)
    parser.add_argument("--oversample-target-index", type=int, default=None)
    parser.add_argument("--oversample-target-label", default="artificial_grassland")
    parser.add_argument("--oversample-ratio", type=int, default=3)
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

        write_json(save_dir / "dataset_report.json", dataset_report)
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
            target_class_idx=oversample_target_idx,
            target_class_label=oversample_target_label,
            oversample_ratio=args.oversample_ratio,
            output_path=save_dir / "train_oversampled.txt",
            seed=args.seed,
        )
        write_json(save_dir / "oversample_report.json", oversample_report)
        print("[info] oversample_target=", oversample_target_idx, oversample_target_label)
        print("[info] oversample_ratio=", args.oversample_ratio)
        print("[info] effective_train_list=", effective_train_list)

    dist_barrier()

    dataset_report = json.loads((save_dir / "dataset_report.json").read_text(encoding="utf-8"))
    oversample_report = json.loads((save_dir / "oversample_report.json").read_text(encoding="utf-8"))
    num_classes = int(dataset_report["num_classes"])
    label_names = list(dataset_report["labels"])
    oversample_target_idx = int(oversample_report["target_class_index"])
    oversample_target_label = oversample_report["target_class_label"]
    effective_train_list = Path(oversample_report["effective_train_list"])

    train_transforms = T.Compose(
        [
            T.RandomCrop(crop_size=args.crop_size),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomDistort(
                brightness_range=0.2,
                contrast_range=0.2,
                saturation_range=0.2,
                brightness_prob=0.7,
                contrast_prob=0.7,
                saturation_prob=0.7,
                hue_prob=1.0,
            ),
            T.RandomBlur(prob=0.1),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    eval_transforms = T.Compose(
        [
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

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
    test_dataset = None
    eval_pairs = val_pairs
    eval_name = "val"
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

    model = pdrs.tasks.seg.DeepLabV3P(
        num_classes=num_classes,
        backbone=args.backbone,
        use_mixed_loss=True,
    )

    config = {
        "data_dir": str(data_dir),
        "train_list": str(train_list),
        "effective_train_list": str(effective_train_list),
        "val_list": str(val_list),
        "test_list": str(test_list) if test_list is not None else None,
        "label_list": str(label_list) if label_list.exists() else None,
        "num_classes": num_classes,
        "backbone": args.backbone,
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
        "oversample_target_index": oversample_target_idx,
        "oversample_target_label": oversample_target_label,
        "oversample_ratio": args.oversample_ratio,
        "use_mixed_loss": True,
        "seed": args.seed,
        "save_dir": str(save_dir),
    }
    if main_process:
        write_json(save_dir / "train_config.json", config)
        print("[info] start training, save_dir=", save_dir)
    model.train(
        num_epochs=args.epochs,
        train_dataset=train_dataset,
        train_batch_size=args.train_batch_size,
        eval_dataset=val_dataset,
        save_interval_epochs=args.save_interval_epochs,
        log_interval_steps=args.log_interval_steps,
        save_dir=str(save_dir),
        pretrain_weights="CITYSCAPES",
        learning_rate=args.learning_rate,
        early_stop=True,
        early_stop_patience=args.early_stop_patience,
        use_vdl=True,
        resume_checkpoint=args.resume_checkpoint,
        precision="fp32",
    )

    dist_barrier()

    if not main_process:
        return

    best_model_dir = save_dir / "best_model"
    best_model = load_model(str(best_model_dir))
    target_dataset = test_dataset if test_dataset is not None else val_dataset
    best_metrics, best_details = best_model.evaluate(
        target_dataset, batch_size=args.eval_batch_size, return_details=True
    )

    accuracy_summary = {
        "evaluated_split": eval_name,
        "model": "DeepLabV3P",
        "best_model_dir": str(best_model_dir),
        "summary": build_accuracy_summary(best_metrics, best_details, label_names),
    }
    write_json(save_dir / "accuracy_summary.json", accuracy_summary)
    write_json(
        save_dir / "best_model_metrics.json",
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
        save_dir / "best_model_previews",
        num_classes=num_classes,
        count=args.preview_count,
    )

    run_summary = {
        "evaluated_split": eval_name,
        "best_model_dir": str(best_model_dir),
        "accuracy_summary_path": str(save_dir / "accuracy_summary.json"),
        "best_model_metrics_path": str(save_dir / "best_model_metrics.json"),
        "dataset_report_path": str(save_dir / "dataset_report.json"),
        "train_config_path": str(save_dir / "train_config.json"),
        "preview_dir": str(save_dir / "best_model_previews"),
    }
    write_json(save_dir / "run_summary.json", run_summary)

    print("[info] training completed")
    print("[info] best_model_dir=", best_model_dir)
    print("[info] accuracy_summary=", save_dir / "accuracy_summary.json")
    print("[info] dataset_report=", save_dir / "dataset_report.json")


if __name__ == "__main__":
    main()
