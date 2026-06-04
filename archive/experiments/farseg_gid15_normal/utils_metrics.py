#!/usr/bin/env python3

import csv
import json
import math
import shutil
from pathlib import Path

import cv2
import numpy as np


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def reset_dir(path):
    path = Path(path)
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


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


def _safe_divide(num, den):
    if den <= 0:
        return 0.0
    return float(num / den)


def compute_confusion_metrics(confusion_matrix, class_names):
    confusion = np.asarray(confusion_matrix, dtype=np.float64)
    if confusion.ndim != 2 or confusion.shape[0] != confusion.shape[1]:
        raise ValueError(
            f"Expected a square confusion matrix, but received shape={confusion.shape}."
        )

    num_classes = confusion.shape[0]
    if len(class_names) != num_classes:
        raise ValueError(
            f"class_names length ({len(class_names)}) does not match confusion size ({num_classes})."
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
    class_precision = np.array(
        [_safe_divide(diag[idx], pred_pixels[idx]) for idx in range(num_classes)],
        dtype=np.float64,
    )
    class_recall = class_accuracy.copy()
    class_f1 = np.array(
        [
            _safe_divide(2.0 * class_precision[idx] * class_recall[idx],
                         class_precision[idx] + class_recall[idx])
            if class_precision[idx] + class_recall[idx] > 0
            else 0.0
            for idx in range(num_classes)
        ],
        dtype=np.float64,
    )

    valid_mask = label_pixels > 0
    mean_accuracy = float(class_accuracy[valid_mask].mean()) if valid_mask.any() else 0.0
    mean_iou = float(class_iou[valid_mask].mean()) if valid_mask.any() else 0.0
    overall_accuracy = _safe_divide(diag.sum(), total_pixels)

    if total_pixels <= 0:
        kappa = 0.0
    else:
        expected = float(np.dot(label_pixels, pred_pixels) / (total_pixels ** 2))
        denom = 1.0 - expected
        kappa = float((overall_accuracy - expected) / denom) if abs(denom) > 1e-12 else 0.0

    per_class = []
    for idx, class_name in enumerate(class_names):
        per_class.append(
            {
                "class_id": idx,
                "class_name": class_name,
                "class_accuracy": float(class_accuracy[idx]),
                "iou": float(class_iou[idx]),
                "precision": float(class_precision[idx]),
                "recall": float(class_recall[idx]),
                "f1": float(class_f1[idx]),
                "label_pixels": int(label_pixels[idx]),
                "pred_pixels": int(pred_pixels[idx]),
                "true_positives": int(diag[idx]),
                "union_pixels": int(union[idx]),
            }
        )

    return {
        "overall_accuracy": float(overall_accuracy),
        "mean_iou": float(mean_iou),
        "mean_accuracy": float(mean_accuracy),
        "kappa": float(kappa),
        "total_pixels": int(total_pixels),
        "num_classes": int(num_classes),
        "per_class": per_class,
    }


def build_evaluation_payload(
        model_name,
        selection_split,
        evaluated_split,
        class_names,
        metrics_from_model,
        details):
    if details is None or "confusion_matrix" not in details:
        raise ValueError("Evaluation details must include `confusion_matrix`.")

    confusion_matrix = np.asarray(details["confusion_matrix"], dtype=np.int64)
    summary = compute_confusion_metrics(confusion_matrix, class_names)
    payload = {
        "model": model_name,
        "selection_split": selection_split,
        "evaluated_split": evaluated_split,
        "class_names": list(class_names),
        "summary": summary,
        "metrics_from_model": to_builtin(metrics_from_model),
        "details": {
            "confusion_matrix": to_builtin(confusion_matrix),
        },
    }
    return payload


def write_metrics_text(path, payload):
    summary = payload["summary"]
    lines = [
        f"Model: {payload['model']}",
        f"Selection Split: {payload['selection_split']}",
        f"Evaluated Split: {payload['evaluated_split']}",
        f"Overall Accuracy (OA): {summary['overall_accuracy']:.6f}",
        f"mIoU: {summary['mean_iou']:.6f}",
        f"Mean Accuracy: {summary['mean_accuracy']:.6f}",
        f"Kappa: {summary['kappa']:.6f}",
        f"Total Pixels: {summary['total_pixels']}",
        "",
        "Per-class metrics:",
    ]
    for item in summary["per_class"]:
        lines.append(
            f"[{item['class_id']:02d}] {item['class_name']}: "
            f"acc={item['class_accuracy']:.6f}, "
            f"iou={item['iou']:.6f}, "
            f"precision={item['precision']:.6f}, "
            f"recall={item['recall']:.6f}, "
            f"f1={item['f1']:.6f}"
        )

    path = Path(path)
    ensure_dir(path.parent)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_per_class_csv(path, summary):
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "class_id",
                "class_name",
                "class_accuracy",
                "iou",
                "precision",
                "recall",
                "f1",
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
                    f"{item['precision']:.10f}",
                    f"{item['recall']:.10f}",
                    f"{item['f1']:.10f}",
                    item["label_pixels"],
                    item["pred_pixels"],
                    item["true_positives"],
                    item["union_pixels"],
                ]
            )


def save_confusion_matrix(path, confusion_matrix):
    path = Path(path)
    ensure_dir(path.parent)
    np.save(path, np.asarray(confusion_matrix, dtype=np.int64))


def _relative_prediction_path(image_path, dataset_root, split_name):
    image_path = Path(image_path)
    dataset_root = Path(dataset_root)
    try:
        relative_path = image_path.resolve().relative_to(dataset_root.resolve())
    except ValueError:
        relative_path = Path(split_name) / image_path.name

    if relative_path.parts and relative_path.parts[0] == "images":
        relative_path = Path(*relative_path.parts[1:])
    return relative_path.with_suffix(".png")


def export_prediction_artifacts(
        model,
        pairs,
        transforms,
        dataset_root,
        pred_dir,
        vis_dir,
        num_classes,
        split_name):
    reset_dir(pred_dir)
    reset_dir(vis_dir)

    palette = color_palette(num_classes)
    manifest = []

    for image_path, mask_path in pairs:
        prediction = model.predict(str(image_path), transforms=transforms)
        label_map = prediction["label_map"]
        if label_map.dtype != np.uint8 and num_classes <= 256:
            label_map = label_map.astype(np.uint8)
        elif label_map.dtype != np.uint16:
            label_map = label_map.astype(np.uint16)

        relative_path = _relative_prediction_path(image_path, dataset_root, split_name)
        pred_path = Path(pred_dir) / relative_path
        vis_path = Path(vis_dir) / relative_path
        ensure_dir(pred_path.parent)
        ensure_dir(vis_path.parent)

        color_mask = palette[np.clip(label_map.astype(np.int64), 0, num_classes - 1)]
        cv2.imwrite(str(pred_path), label_map)

        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            visualization = cv2.cvtColor(color_mask, cv2.COLOR_RGB2BGR)
            vis_kind = "color_mask"
        else:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            visualization = cv2.cvtColor(
                (0.6 * image_rgb + 0.4 * color_mask).astype(np.uint8),
                cv2.COLOR_RGB2BGR,
            )
            vis_kind = "overlay"
        cv2.imwrite(str(vis_path), visualization)

        manifest.append(
            {
                "image": str(image_path),
                "mask": str(mask_path),
                "prediction": str(pred_path),
                "visualization": str(vis_path),
                "visualization_type": vis_kind,
            }
        )

    return manifest
