#!/usr/bin/env python3

from __future__ import annotations

import csv
import json
import math
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


def write_json(path, payload) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as file:
        json.dump(to_builtin(payload), file, ensure_ascii=False, indent=2)


def read_json(path):
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)


def format_timestamp(timestamp: float | None = None) -> str:
    if timestamp is None:
        timestamp = time.time()
    return datetime.fromtimestamp(float(timestamp)).strftime("%Y-%m-%d %H:%M:%S")


def format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "warming_up"
    seconds = max(0.0, float(seconds))
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, sec = divmod(int(round(seconds)), 60)
    if minutes < 60:
        return f"{minutes}m{sec:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m{sec:02d}s"


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
        "confusion_matrix_shape": [int(num_classes), int(num_classes)],
        "total_valid_pixels": int(total_pixels),
        "overall_accuracy": _safe_divide(diag.sum(), total_pixels),
        "foreground_overall_accuracy": _safe_divide(diag[fg_mask].sum(), label_pixels[fg_mask].sum()),
        "mean_accuracy": float(class_accuracy[valid_mask].mean()) if valid_mask.any() else 0.0,
        "foreground_mean_accuracy": float(class_accuracy[fg_valid_mask].mean()) if fg_valid_mask.any() else 0.0,
        "mean_iou": float(class_iou[valid_mask].mean()) if valid_mask.any() else 0.0,
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


def build_logging_schema(model_type: str, num_heads: int) -> dict[str, object]:
    return {
        "logger_design": {
            "single_file_handler": True,
            "single_stdout_stream_handler": True,
            "tee_stream_used": False,
            "stdout_stderr_redirected_to_log_file": False,
            "propagate": False,
            "message_duplication_allowed": False,
        },
        "model_type": model_type,
        "num_heads": int(num_heads),
        "records": {
            "start": {
                "prefix": "[start]",
                "required_fields": [
                    "start_time",
                    "total_epochs",
                    "train_dataset_size",
                    "val_dataset_size",
                    "steps_per_epoch",
                    "model_type",
                    "num_heads",
                    "train_batch_size",
                    "eval_batch_size",
                    "CUDA_VISIBLE_DEVICES",
                    "physical_gpu_name",
                ],
            },
            "path": {
                "prefix": "[path]",
                "required_fields": [
                    "best_model_dir",
                    "latest_model_dir",
                    "checkpoint_dir",
                    "metrics_dir",
                    "prediction_samples_dir",
                    "config_snapshot_dir",
                ],
            },
            "train": {
                "prefix": "[train]",
                "required_fields": [
                    "epoch",
                    "step",
                    "global_step",
                    "batch",
                    "loss",
                    "lr",
                    "step_time_avg",
                    "eta",
                    "end_time",
                ],
                "eta_policy": {
                    "window": 50,
                    "fallback": "warming_up",
                    "end_time_format": "YYYY-MM-DD HH:MM:SS",
                },
            },
            "val": {
                "prefix": "[val]",
                "required_fields": [
                    "epoch",
                    "OA",
                    "FG_OA",
                    "Mean_Acc",
                    "mIoU",
                    "epoch_time",
                    "eta",
                    "end_time",
                ],
            },
            "epoch_summary": {
                "prefix": "[epoch_summary]",
                "required_fields": [
                    "epoch",
                    "epoch_time",
                    "best_mIoU",
                    "eta",
                    "end_time",
                ],
            },
        },
    }


def format_startup_log(
    *,
    start_time: str,
    total_epochs: int,
    train_size: int,
    val_size: int,
    steps_per_epoch: int,
    model_type: str,
    num_heads: int,
    train_batch_size: int,
    eval_batch_size: int,
    cuda_visible_devices: str,
    physical_gpu_name: str,
) -> str:
    return (
        f"[start] start_time={start_time} "
        f"total_epochs={int(total_epochs)} "
        f"train_dataset_size={int(train_size)} "
        f"val_dataset_size={int(val_size)} "
        f"steps_per_epoch={int(steps_per_epoch)} "
        f"model_type={model_type} "
        f"num_heads={int(num_heads)} "
        f"train_batch_size={int(train_batch_size)} "
        f"eval_batch_size={int(eval_batch_size)} "
        f"CUDA_VISIBLE_DEVICES={cuda_visible_devices} "
        f"physical_gpu_name={physical_gpu_name}"
    )


def format_path_log(name: str, path) -> str:
    return f"[path] {name}={Path(path)}"


def format_train_log(
    *,
    epoch: int,
    total_epochs: int,
    step: int,
    steps_per_epoch: int,
    global_step: int,
    batch_size: int,
    loss: float,
    lr: float,
    step_time_avg_seconds: float | None,
    eta_seconds: float | None,
    end_time: str,
) -> str:
    step_time_value = format_duration(step_time_avg_seconds)
    eta_value = format_duration(eta_seconds)
    return (
        f"[train] epoch={epoch}/{total_epochs} "
        f"step={step}/{steps_per_epoch} "
        f"global_step={global_step} "
        f"batch={batch_size} "
        f"loss={loss:.6f} "
        f"lr={lr:.8f} "
        f"step_time_avg={step_time_value} "
        f"eta={eta_value} "
        f"end_time={end_time}"
    )


def format_val_log(
    *,
    epoch: int,
    total_epochs: int,
    metrics: dict[str, object],
    epoch_time_seconds: float,
    eta_seconds: float | None,
    end_time: str,
) -> str:
    return (
        f"[val] epoch={epoch}/{total_epochs} "
        f"OA={float(metrics['overall_accuracy']):.6f} "
        f"FG_OA={float(metrics['foreground_overall_accuracy']):.6f} "
        f"Mean_Acc={float(metrics['mean_accuracy']):.6f} "
        f"mIoU={float(metrics['mean_iou']):.6f} "
        f"epoch_time={format_duration(epoch_time_seconds)} "
        f"eta={format_duration(eta_seconds)} "
        f"end_time={end_time}"
    )


def format_epoch_summary_log(
    *,
    epoch: int,
    total_epochs: int,
    epoch_time_seconds: float,
    best_miou: float,
    eta_seconds: float | None,
    end_time: str,
) -> str:
    return (
        f"[epoch_summary] epoch={epoch}/{total_epochs} "
        f"epoch_time={format_duration(epoch_time_seconds)} "
        f"best_mIoU={float(best_miou):.6f} "
        f"eta={format_duration(eta_seconds)} "
        f"end_time={end_time}"
    )
