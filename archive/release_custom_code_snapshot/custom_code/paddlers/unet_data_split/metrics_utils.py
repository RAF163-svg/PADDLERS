#!/usr/bin/env python3

from __future__ import annotations

import csv
import json
import logging
import math
import sys
import time
from datetime import datetime
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


def read_json(path):
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)


def write_json(path, payload) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as file:
        json.dump(to_builtin(payload), file, ensure_ascii=False, indent=2)


def format_timestamp(timestamp: float | None = None) -> str:
    if timestamp is None:
        timestamp = time.time()
    return datetime.fromtimestamp(float(timestamp)).strftime("%Y-%m-%d %H:%M:%S")


def format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "warming_up"
    seconds = max(0.0, float(seconds))
    if seconds < 60:
        return f"{seconds:.2f}s"
    minutes, sec = divmod(int(round(seconds)), 60)
    hours, minutes = divmod(minutes, 60)
    if hours > 0:
        return f"{hours}:{minutes:02d}:{sec:02d}"
    return f"{minutes}:{sec:02d}"


def format_eta(seconds: float | None) -> str:
    if seconds is None:
        return "warming_up"
    seconds = max(0, int(round(float(seconds))))
    hours, rem = divmod(seconds, 3600)
    minutes, sec = divmod(rem, 60)
    return f"{hours}:{minutes:02d}:{sec:02d}"


def format_time_each_step(seconds: float | None) -> str:
    if seconds is None:
        return "warming_up"
    return f"{float(seconds):.2f}s"


def format_end_time_from_eta(eta_seconds: float | None) -> str:
    if eta_seconds is None:
        return "warming_up"
    return format_timestamp(time.time() + max(0.0, float(eta_seconds)))


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

    class_recall = np.array(
        [_safe_divide(diag[idx], label_pixels[idx]) for idx in range(num_classes)],
        dtype=np.float64,
    )
    class_precision = np.array(
        [_safe_divide(diag[idx], pred_pixels[idx]) for idx in range(num_classes)],
        dtype=np.float64,
    )
    class_iou = np.array(
        [_safe_divide(diag[idx], union[idx]) for idx in range(num_classes)],
        dtype=np.float64,
    )
    class_f1 = np.array(
        [
            _safe_divide(2.0 * class_precision[idx] * class_recall[idx], class_precision[idx] + class_recall[idx])
            for idx in range(num_classes)
        ],
        dtype=np.float64,
    )
    valid_mask = label_pixels > 0
    fg_mask = np.zeros(num_classes, dtype=bool)
    fg_mask[foreground_start_index:] = True
    fg_valid_mask = valid_mask & fg_mask

    pe = _safe_divide(np.sum(label_pixels * pred_pixels), total_pixels * total_pixels)
    po = _safe_divide(diag.sum(), total_pixels)
    kappa = _safe_divide(po - pe, 1.0 - pe) if total_pixels > 0 else 0.0

    payload = {
        "confusion_matrix_shape": [int(num_classes), int(num_classes)],
        "total_valid_pixels": int(total_pixels),
        "overall_accuracy": po,
        "mean_accuracy": float(class_recall[valid_mask].mean()) if valid_mask.any() else 0.0,
        "mean_iou": float(class_iou[valid_mask].mean()) if valid_mask.any() else 0.0,
        "kappa": float(kappa),
        "foreground_overall_accuracy": _safe_divide(diag[fg_mask].sum(), label_pixels[fg_mask].sum()),
        "foreground_mean_accuracy": float(class_recall[fg_valid_mask].mean()) if fg_valid_mask.any() else 0.0,
        "foreground_mean_iou": float(class_iou[fg_valid_mask].mean()) if fg_valid_mask.any() else 0.0,
        "foreground_class_count": int(fg_mask.sum()),
        "per_class": [],
    }

    for idx, class_name in enumerate(class_names):
        payload["per_class"].append(
            {
                "class_id": idx,
                "class_name": class_name,
                "is_background": idx < foreground_start_index,
                "class_accuracy": float(class_recall[idx]),
                "recall": float(class_recall[idx]),
                "precision": float(class_precision[idx]),
                "iou": float(class_iou[idx]),
                "f1_score": float(class_f1[idx]),
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
                "recall",
                "precision",
                "iou",
                "f1_score",
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
                    f"{item['recall']:.10f}",
                    f"{item['precision']:.10f}",
                    f"{item['iou']:.10f}",
                    f"{item['f1_score']:.10f}",
                    item["label_pixels"],
                    item["pred_pixels"],
                    item["true_positives"],
                    item["union_pixels"],
                ]
            )


def write_confusion_csv(path, confusion_matrix, class_names) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    confusion = np.asarray(confusion_matrix, dtype=np.int64)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["gt/pred", *class_names])
        for idx, class_name in enumerate(class_names):
            writer.writerow([class_name, *confusion[idx].tolist()])


def write_metrics_text(path, metrics, *, split_name: str, selection_metric: str | None = None) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    lines = [
        f"split: {split_name}",
        f"selection_metric: {selection_metric or 'mean_iou'}",
        f"overall_accuracy: {float(metrics['overall_accuracy']):.6f}",
        f"mean_accuracy: {float(metrics['mean_accuracy']):.6f}",
        f"mean_iou: {float(metrics['mean_iou']):.6f}",
        f"kappa: {float(metrics['kappa']):.6f}",
        f"foreground_overall_accuracy: {float(metrics['foreground_overall_accuracy']):.6f}",
        f"foreground_mean_accuracy: {float(metrics['foreground_mean_accuracy']):.6f}",
        f"foreground_mean_iou: {float(metrics['foreground_mean_iou']):.6f}",
    ]
    with path.open("w", encoding="utf-8") as file:
        file.write("\n".join(lines) + "\n")


class StageFormatter(logging.Formatter):
    def format(self, record):
        if not hasattr(record, "stage"):
            record.stage = "SYSTEM"
        return super().format(record)


def setup_logger(name: str, log_path: Path) -> logging.Logger:
    ensure_dir(log_path.parent)
    logger = logging.getLogger(name)
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    logger.propagate = False

    formatter = StageFormatter(
        "%(asctime)s [%(levelname)s] [%(stage)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def build_logging_schema(model_type: str, num_heads: int) -> dict[str, object]:
    return {
        "log_format": "YYYY-MM-DD HH:MM:SS [LEVEL] [STAGE] message",
        "logger_design": {
            "single_stdout_stream_handler": True,
            "single_file_handler": True,
            "tee_stream_used": False,
            "print_used": False,
            "message_duplication_allowed": False,
            "propagate": False,
        },
        "model_type": model_type,
        "num_heads": int(num_heads),
        "required_stages": [
            "INIT",
            "SYSTEM",
            "DATA",
            "SMOKE",
            "TRAIN",
            "EVAL",
            "TEST",
            "CKPT",
        ],
    }
