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


TRAIN_LOG_STAGES = {"INIT", "DATA", "TRAIN", "CKPT", "EXPORT", "SYSTEM"}
EVAL_LOG_STAGES = {"EVAL", "TEST"}
SMOKE_LOG_STAGES = {"SMOKE"}


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
        return value.item()
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


def format_eta(seconds: float | None) -> str:
    if seconds is None:
        return "warming_up"
    seconds = max(0, int(round(float(seconds))))
    days, rem = divmod(seconds, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, sec = divmod(rem, 60)
    if days > 0:
        return f"{days} day, {hours}:{minutes:02d}:{sec:02d}"
    return f"{hours}:{minutes:02d}:{sec:02d}"


def build_log_line(level: str, stage: str, message: str, timestamp: float | None = None) -> str:
    return f"{format_timestamp(timestamp)} [{level}] [{stage}] {message}"


def _build_logger(name: str, *, path: Path | None = None, stream=None) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    logger.propagate = False

    formatter = logging.Formatter("%(message)s")
    if path is not None:
        ensure_dir(path.parent)
        handler = logging.FileHandler(path, encoding="utf-8")
    else:
        handler = logging.StreamHandler(stream or sys.stdout)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


class StageLogger:
    def __init__(self, log_dir: Path, logger_name: str = "hrnet_data_split"):
        log_dir = Path(log_dir)
        ensure_dir(log_dir)
        self.log_dir = log_dir
        self.train_log_path = log_dir / "train.log"
        self.eval_log_path = log_dir / "eval.log"
        self.smoke_log_path = log_dir / "smoke_test.log"

        self.console_logger = _build_logger(f"{logger_name}.console", stream=sys.stdout)
        self.train_logger = _build_logger(f"{logger_name}.train", path=self.train_log_path)
        self.eval_logger = _build_logger(f"{logger_name}.eval", path=self.eval_log_path)
        self.smoke_logger = _build_logger(f"{logger_name}.smoke", path=self.smoke_log_path)

    def _route_logger(self, stage: str) -> logging.Logger:
        stage = stage.upper()
        if stage in SMOKE_LOG_STAGES:
            return self.smoke_logger
        if stage in EVAL_LOG_STAGES:
            return self.eval_logger
        return self.train_logger

    def log(self, level: str, stage: str, message: str) -> str:
        stage = stage.upper()
        line = build_log_line(level.upper(), stage, message)
        levelno = getattr(logging, level.upper(), logging.INFO)
        self.console_logger.log(levelno, line)
        self._route_logger(stage).log(levelno, line)
        return line

    def info(self, stage: str, message: str) -> str:
        return self.log("INFO", stage, message)

    def warning(self, stage: str, message: str) -> str:
        return self.log("WARNING", stage, message)

    def error(self, stage: str, message: str) -> str:
        return self.log("ERROR", stage, message)


def format_train_log(
    *,
    epoch: int,
    total_epochs: int,
    step: int,
    total_steps: int,
    loss: float,
    lr: float,
    step_time_seconds: float,
    eta_seconds: float | None,
) -> str:
    return (
        f"Epoch={epoch}/{total_epochs}, "
        f"Step={step}/{total_steps}, "
        f"loss={loss:.6f}, "
        f"lr={lr:.6f}, "
        f"time_each_step={step_time_seconds:.2f}s, "
        f"eta={format_eta(eta_seconds)}"
    )


def format_eval_log(
    *,
    epoch: int,
    total_epochs: int,
    val_loss: float | None,
    oa: float,
    miou: float,
    macc: float,
    kappa: float,
) -> str:
    loss_value = 0.0 if val_loss is None else float(val_loss)
    return (
        f"Epoch={epoch}/{total_epochs}, "
        f"loss={loss_value:.6f}, "
        f"OA={oa:.6f}, "
        f"mIoU={miou:.6f}, "
        f"mAcc={macc:.6f}, "
        f"Kappa={kappa:.6f}"
    )


def format_test_log(*, test_loss: float | None, oa: float, miou: float, macc: float, kappa: float) -> str:
    loss_value = 0.0 if test_loss is None else float(test_loss)
    return (
        f"loss={loss_value:.6f}, "
        f"OA={oa:.6f}, "
        f"mIoU={miou:.6f}, "
        f"mAcc={macc:.6f}, "
        f"Kappa={kappa:.6f}"
    )


def format_checkpoint_log(*, epoch: int, path: Path) -> str:
    return f"Saved checkpoint: epoch={epoch}, path={path}"


def format_best_model_log(*, epoch: int, val_miou: float, path: Path) -> str:
    return f"Updated best model: epoch={epoch}, val_mIoU={val_miou:.6f}, path={path}"


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

    class_accuracy = np.array([_safe_divide(diag[idx], label_pixels[idx]) for idx in range(num_classes)])
    class_recall = class_accuracy.copy()
    class_precision = np.array([_safe_divide(diag[idx], pred_pixels[idx]) for idx in range(num_classes)])
    class_iou = np.array([_safe_divide(diag[idx], union[idx]) for idx in range(num_classes)])

    valid_mask = label_pixels > 0
    fg_mask = np.zeros(num_classes, dtype=bool)
    fg_mask[foreground_start_index:] = True
    fg_valid_mask = valid_mask & fg_mask

    po = _safe_divide(diag.sum(), total_pixels)
    pe = _safe_divide(np.sum(label_pixels * pred_pixels), total_pixels * total_pixels)
    kappa = _safe_divide(po - pe, 1.0 - pe) if (1.0 - pe) > 0 else 0.0

    payload = {
        "confusion_matrix_shape": [int(num_classes), int(num_classes)],
        "total_valid_pixels": int(total_pixels),
        "overall_accuracy": po,
        "foreground_overall_accuracy": _safe_divide(diag[fg_mask].sum(), label_pixels[fg_mask].sum()),
        "mean_accuracy": float(class_accuracy[valid_mask].mean()) if valid_mask.any() else 0.0,
        "foreground_mean_accuracy": float(class_accuracy[fg_valid_mask].mean()) if fg_valid_mask.any() else 0.0,
        "mean_iou": float(class_iou[valid_mask].mean()) if valid_mask.any() else 0.0,
        "foreground_mean_iou": float(class_iou[fg_valid_mask].mean()) if fg_valid_mask.any() else 0.0,
        "kappa": float(kappa),
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
                "recall": float(class_recall[idx]),
                "precision": float(class_precision[idx]),
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
                "recall",
                "precision",
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
                    f"{item['recall']:.10f}",
                    f"{item['precision']:.10f}",
                    f"{item['iou']:.10f}",
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
        writer.writerow(["gt/pred"] + list(class_names))
        for idx, class_name in enumerate(class_names):
            writer.writerow([class_name] + confusion[idx].tolist())


def write_metrics_text(path, metrics, *, split_name: str, checkpoint_path: str | None = None) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    lines = [
        f"split: {split_name}",
        f"loss: {0.0 if metrics.get('mean_loss') is None else float(metrics['mean_loss']):.6f}",
        f"OA: {float(metrics['overall_accuracy']):.6f}",
        f"Mean Accuracy: {float(metrics['mean_accuracy']):.6f}",
        f"mIoU: {float(metrics['mean_iou']):.6f}",
        f"Kappa: {float(metrics.get('kappa', 0.0)):.6f}",
        f"Foreground OA: {float(metrics.get('foreground_overall_accuracy', 0.0)):.6f}",
        f"Foreground mIoU: {float(metrics.get('foreground_mean_iou', 0.0)):.6f}",
        f"Total Valid Pixels: {int(metrics.get('total_valid_pixels', 0))}",
    ]
    if checkpoint_path:
        lines.append(f"checkpoint: {checkpoint_path}")
    lines.append("")
    lines.append("Per-class metrics:")
    for item in metrics["per_class"]:
        lines.append(
            f"- class_id={item['class_id']}, class_name={item['class_name']}, "
            f"accuracy={item['class_accuracy']:.6f}, recall={item['recall']:.6f}, "
            f"precision={item['precision']:.6f}, iou={item['iou']:.6f}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
