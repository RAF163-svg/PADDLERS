#!/usr/bin/env python3

import csv
import json
import math
from pathlib import Path

import numpy as np


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


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
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as file:
        json.dump(to_builtin(payload), file, ensure_ascii=False, indent=2)


def _safe_divide(num, den):
    if den <= 0:
        return 0.0
    return float(num / den)


def compute_confusion_metrics(confusion_matrix, class_names):
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

    payload = {
        "overall_accuracy": _safe_divide(diag.sum(), total_pixels),
        "mean_iou": float(class_iou[valid_mask].mean()) if valid_mask.any() else 0.0,
        "mean_accuracy": float(class_accuracy[valid_mask].mean()) if valid_mask.any() else 0.0,
        "per_class": [],
    }
    for idx, class_name in enumerate(class_names):
        payload["per_class"].append(
            {
                "class_id": idx,
                "class_name": class_name,
                "class_accuracy": float(class_accuracy[idx]),
                "iou": float(class_iou[idx]),
                "label_pixels": int(label_pixels[idx]),
                "pred_pixels": int(pred_pixels[idx]),
                "true_positives": int(diag[idx]),
                "union_pixels": int(union[idx]),
            }
        )
    return payload


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


def write_per_class_csv(path, summary):
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "class_id",
                "class_name",
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
                    f"{item['class_accuracy']:.10f}",
                    f"{item['iou']:.10f}",
                    item["label_pixels"],
                    item["pred_pixels"],
                    item["true_positives"],
                    item["union_pixels"],
                ]
            )
