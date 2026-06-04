#!/usr/bin/env python3

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np


def ensure_dir(path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def to_builtin(value):
    if isinstance(value, dict):
        return {str(key): to_builtin(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_builtin(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return to_builtin(value.tolist())
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path, payload) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as file:
        json.dump(to_builtin(payload), file, ensure_ascii=False, indent=2)


def read_json(path):
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)


def _safe_divide(num, den) -> float:
    if den <= 0:
        return 0.0
    return float(num / den)


def update_confusion_matrix(confusion_matrix, prediction, label, num_classes, ignore_index=255):
    prediction = np.asarray(prediction, dtype=np.int64)
    label = np.asarray(label, dtype=np.int64)
    valid_mask = (label != ignore_index) & (label >= 0) & (label < num_classes)
    if not np.any(valid_mask):
        return confusion_matrix

    encoded = num_classes * label[valid_mask] + prediction[valid_mask]
    bincount = np.bincount(encoded, minlength=num_classes * num_classes)
    confusion_matrix += bincount.reshape(num_classes, num_classes)
    return confusion_matrix


def compute_confusion_metrics(confusion_matrix, class_names, foreground_start_index=1):
    confusion = np.asarray(confusion_matrix, dtype=np.float64)
    if confusion.ndim != 2 or confusion.shape[0] != confusion.shape[1]:
        raise ValueError(f"Expected square confusion matrix, got shape={confusion.shape}")

    num_classes = confusion.shape[0]
    if len(class_names) != num_classes:
        raise ValueError(
            f"class_names length ({len(class_names)}) does not match confusion size ({num_classes})"
        )

    diag = np.diag(confusion)
    label_pixels = confusion.sum(axis=1)
    pred_pixels = confusion.sum(axis=0)
    union = label_pixels + pred_pixels - diag
    total_pixels = confusion.sum()

    class_accuracy = np.array(
        [_safe_divide(diag[idx], label_pixels[idx]) for idx in range(num_classes)],
        dtype=np.float64,
    )
    class_iou = np.array(
        [_safe_divide(diag[idx], union[idx]) for idx in range(num_classes)],
        dtype=np.float64,
    )
    valid_mask = label_pixels > 0

    fg_mask = np.zeros(num_classes, dtype=bool)
    fg_mask[foreground_start_index:] = True
    fg_valid_mask = valid_mask & fg_mask

    payload = {
        "overall_accuracy": _safe_divide(diag.sum(), total_pixels),
        "foreground_overall_accuracy": _safe_divide(diag[fg_mask].sum(), label_pixels[fg_mask].sum()),
        "mean_iou": float(class_iou[fg_valid_mask].mean()) if fg_valid_mask.any() else 0.0,
        "mean_accuracy": float(class_accuracy[fg_valid_mask].mean()) if fg_valid_mask.any() else 0.0,
        "mean_iou_with_background": float(class_iou[valid_mask].mean()) if valid_mask.any() else 0.0,
        "mean_accuracy_with_background": float(class_accuracy[valid_mask].mean()) if valid_mask.any() else 0.0,
        "foreground_class_count": int(fg_mask.sum()),
        "per_class": [],
    }

    for idx, class_name in enumerate(class_names):
        payload["per_class"].append(
            {
                "class_id": idx,
                "class_name": class_name,
                "is_background": idx < foreground_start_index,
                "class_accuracy": float(class_accuracy[idx]),
                "iou": float(class_iou[idx]),
                "label_pixels": int(label_pixels[idx]),
                "pred_pixels": int(pred_pixels[idx]),
                "true_positives": int(diag[idx]),
                "union_pixels": int(union[idx]),
            }
        )
    return payload


def write_per_class_csv(path, summary) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "class_id",
                "class_name",
                "is_background",
                "class_accuracy",
                "iou",
                "label_pixels",
                "pred_pixels",
                "true_positives",
                "union_pixels",
            ]
        )
        for item in summary["per_class"]:
            writer.writerow(
                [
                    item["class_id"],
                    item["class_name"],
                    int(bool(item["is_background"])),
                    f"{item['class_accuracy']:.10f}",
                    f"{item['iou']:.10f}",
                    item["label_pixels"],
                    item["pred_pixels"],
                    item["true_positives"],
                    item["union_pixels"],
                ]
            )


def format_train_log(epoch, total_epochs, step, steps_per_epoch, global_step, batch_size, loss, lr, step_time=None) -> str:
    message = (
        f"[train] epoch={epoch}/{total_epochs} "
        f"step={step}/{steps_per_epoch} "
        f"global_step={global_step} "
        f"batch={batch_size} "
        f"loss={loss:.6f} "
        f"lr={lr:.6f}"
    )
    if step_time is not None:
        message += f" step_time={step_time:.4f}s"
    return message


def format_val_log(epoch, total_epochs, metrics) -> str:
    return (
        f"[val] epoch={epoch}/{total_epochs} "
        f"OA={metrics['overall_accuracy']:.6f} "
        f"FG_OA={metrics['foreground_overall_accuracy']:.6f} "
        f"mIoU={metrics['mean_iou']:.6f}"
    )
