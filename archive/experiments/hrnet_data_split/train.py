#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
WORK_DIR = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(WORK_DIR) not in sys.path:
    sys.path.insert(0, str(WORK_DIR))

from cluster_routed_dataset import (
    export_routing_snapshot,
    load_cluster_mapping,
    preview_alignment_rows,
    read_label_names,
    split_samples,
    summarize_routed_samples,
    validate_routed_samples,
)
from hrnet_routed_model import ClusterRoutedHRNet, build_model_structure, load_pretrained_weights
from metrics_utils import (
    StageLogger,
    compute_confusion_metrics,
    ensure_dir,
    format_best_model_log,
    format_checkpoint_log,
    format_eval_log,
    format_eta,
    format_test_log,
    format_train_log,
    read_json,
    update_confusion_matrix,
    write_confusion_csv,
    write_json,
    write_metrics_text,
    write_per_class_csv,
)


DEFAULT_CONFIG_PATH = WORK_DIR / "train_config_multihead.json"
IGNORE_INDEX = 255
TRAIN_LOG_ETA_WINDOW = 50
OUTPUT_SUBDIRS = [
    "best_model",
    "latest_model",
    "checkpoints",
    "logs",
    "metrics",
    "prediction_samples",
    "config_snapshot",
]


def maybe_import_training_stack():
    try:
        import paddle
        from PIL import Image, ImageEnhance
        from paddlers.utils.checkpoint import get_pretrain_weights
        from paddlers.models import seg_losses
    except Exception as exc:
        raise RuntimeError(
            "Training dependencies are not available. Activate the paddlers conda env before running this script."
        ) from exc

    return {
        "paddle": paddle,
        "Image": Image,
        "ImageEnhance": ImageEnhance,
        "get_pretrain_weights": get_pretrain_weights,
        "seg_losses": seg_losses,
    }


def resolve_existing_path(base_dir: Path, path_value: str | None, allow_missing: bool = False) -> Path | None:
    if path_value is None:
        return None
    path = Path(path_value)
    if not path.is_absolute():
        cwd_path = path.resolve()
        base_path = (base_dir / path).resolve()
        path = cwd_path if cwd_path.exists() or allow_missing else base_path
    else:
        path = path.resolve()
    if path.exists() or allow_missing:
        return path
    raise FileNotFoundError(f"Path does not exist: {path}")


def create_output_structure(save_dir: Path) -> None:
    ensure_dir(save_dir)
    for name in OUTPUT_SUBDIRS:
        ensure_dir(save_dir / name)


def set_seed(seed: int, paddle=None) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if paddle is not None:
        paddle.seed(seed)


def query_gpu_name(device: str) -> str:
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    try:
        if visible_devices:
            physical_index = visible_devices.split(",")[0].strip()
        else:
            physical_index = "0"
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name",
                "--format=csv,noheader",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        for line in result.stdout.splitlines():
            idx, name = [item.strip() for item in line.split(",", 1)]
            if idx == physical_index:
                return f"physical_gpu={idx}, name={name}, logical_device={device}"
    except Exception:
        pass
    return f"logical_device={device}"


def build_runtime(args: argparse.Namespace) -> dict[str, object]:
    config_path = resolve_existing_path(WORK_DIR, args.config)
    assert config_path is not None
    config = read_json(config_path)

    if getattr(args, "train_batch_size", None) is not None:
        config["training"]["train_batch_size"] = int(args.train_batch_size)
    if getattr(args, "eval_batch_size", None) is not None:
        config["training"]["eval_batch_size"] = int(args.eval_batch_size)
    if getattr(args, "max_epochs", None) is not None:
        config["training"]["epochs"] = int(args.max_epochs)
    if getattr(args, "resume_checkpoint", None) is not None:
        config["training"]["resume_checkpoint"] = args.resume_checkpoint

    dataset_root = resolve_existing_path(WORK_DIR, config["dataset_root"])
    clustered_root = resolve_existing_path(WORK_DIR, config["clustered_dataset_root"])
    assert dataset_root is not None and clustered_root is not None

    mapping_path = resolve_existing_path(clustered_root, config["cluster_mapping_file"])
    label_list = resolve_existing_path(dataset_root, config["label_list"])
    assert mapping_path is not None and label_list is not None

    class_names = list(config.get("class_names", []))
    label_names = read_label_names(label_list)
    if class_names and label_names and class_names != label_names:
        raise ValueError(
            "class_names in config does not match labels.txt. "
            f"config={class_names}, labels.txt={label_names}"
        )
    class_names = class_names or label_names
    num_classes = int(config["num_classes"])
    if len(class_names) != num_classes:
        raise ValueError(f"class_names length ({len(class_names)}) does not equal num_classes ({num_classes}).")

    routing_cfg = config["routing"]
    selected_k = int(routing_cfg["selected_k"])
    num_heads = int(routing_cfg["num_heads"])
    cluster_names = list(routing_cfg["cluster_names"])

    if len(cluster_names) != num_heads:
        raise ValueError(f"cluster_names length must equal num_heads, got {len(cluster_names)} vs {num_heads}")
    if selected_k != 4 or num_heads != 4:
        raise ValueError(
            f"This formal run is fixed to K=4 and num_heads=4, got selected_k={selected_k}, num_heads={num_heads}"
        )
    if routing_cfg.get("cluster_id_handling", "route") != "route":
        raise ValueError("4-head HRNet requires routing.cluster_id_handling='route'.")
    if routing_cfg.get("sample_path_source", "input") != "input":
        raise ValueError("4-head HRNet requires routing.sample_path_source='input'.")
    if num_classes != 16:
        raise ValueError(f"4-head HRNet requires num_classes=16, got {num_classes}")

    samples = load_cluster_mapping(
        mapping_path,
        sample_path_source=routing_cfg.get("sample_path_source", "input"),
        cluster_id_handling=routing_cfg.get("cluster_id_handling", "route"),
    )
    routing_summary = summarize_routed_samples(samples)
    if routing_summary["duplicate_sample_ids_across_splits"]:
        raise ValueError("Duplicate sample ids were found across splits in cluster routing manifest.")

    routing_validation = validate_routed_samples(samples, expected_num_heads=max(num_heads, 1))
    if routing_validation["missing_files"]:
        example = routing_validation["missing_files"][0]
        raise FileNotFoundError(f"Routed sample file is missing: {example}")

    split_map = split_samples(samples)
    training_cfg = config["training"]
    full_steps_per_epoch = max(1, math.ceil(len(split_map["train"]) / int(training_cfg["train_batch_size"])))
    planned_total_steps = max(1, full_steps_per_epoch * int(training_cfg["epochs"]))
    if getattr(args, "max_train_steps", None) is not None:
        planned_total_steps = min(planned_total_steps, int(args.max_train_steps))

    save_dir_value = args.save_dir or config.get("save_dir", "output")
    save_dir = resolve_existing_path(WORK_DIR, save_dir_value, allow_missing=True)
    assert save_dir is not None

    return {
        "args": args,
        "config": config,
        "config_path": config_path,
        "dataset_root": dataset_root,
        "clustered_root": clustered_root,
        "mapping_path": mapping_path,
        "label_list": label_list,
        "class_names": class_names,
        "num_classes": num_classes,
        "selected_k": selected_k,
        "num_heads": num_heads,
        "samples": samples,
        "split_map": split_map,
        "routing_summary": routing_summary,
        "routing_validation": routing_validation,
        "full_steps_per_epoch": full_steps_per_epoch,
        "planned_total_steps": planned_total_steps,
        "save_dir": save_dir,
    }


def write_config_snapshot(runtime: dict[str, object]) -> None:
    snapshot_dir = runtime["save_dir"] / "config_snapshot"
    ensure_dir(snapshot_dir)

    config = dict(runtime["config"])
    config["resolved_dataset_root"] = str(runtime["dataset_root"])
    config["resolved_clustered_dataset_root"] = str(runtime["clustered_root"])
    config["resolved_cluster_mapping_file"] = str(runtime["mapping_path"])
    config["resolved_label_list"] = str(runtime["label_list"])
    config["resolved_save_dir"] = str(runtime["save_dir"])
    config["full_steps_per_epoch"] = int(runtime["full_steps_per_epoch"])
    config["planned_total_steps"] = int(runtime["planned_total_steps"])
    write_json(snapshot_dir / "resolved_train_config.json", config)

    write_json(snapshot_dir / "model_structure.json", build_model_structure(runtime["config"], runtime["num_classes"]))
    export_routing_snapshot(snapshot_dir / "routing_summary.json", runtime["samples"], runtime["num_heads"])
    write_json(
        snapshot_dir / "dataset_overview.json",
        {
            "dataset_root": str(runtime["dataset_root"]),
            "clustered_dataset_root": str(runtime["clustered_root"]),
            "cluster_mapping_file": str(runtime["mapping_path"]),
            "label_list": str(runtime["label_list"]),
            "class_names": runtime["class_names"],
            "num_classes": runtime["num_classes"],
            "num_heads": runtime["num_heads"],
            "selected_k": runtime["selected_k"],
            "split_counts": {split: len(rows) for split, rows in runtime["split_map"].items()},
            "routing_summary": runtime["routing_summary"],
            "routing_validation": runtime["routing_validation"],
            "alignment_preview": preview_alignment_rows(runtime["samples"], limit=8),
        },
    )


class ClusterRoutedTileDataset:
    def __init__(
        self,
        samples,
        image_module,
        image_enhance_module,
        mean,
        std,
        training,
        augmentation_cfg,
        crop_size=None,
        max_samples=None,
    ):
        self.samples = list(samples[:max_samples] if max_samples else samples)
        self.Image = image_module
        self.ImageEnhance = image_enhance_module
        self.mean = np.asarray(mean, dtype=np.float32).reshape(1, 1, 3)
        self.std = np.asarray(std, dtype=np.float32).reshape(1, 1, 3)
        self.training = bool(training)
        self.augmentation_cfg = augmentation_cfg or {}
        self.crop_size = int(crop_size) if crop_size else None
        self.enable_distort = bool(self.augmentation_cfg.get("enable_distort", False))

    def __len__(self):
        return len(self.samples)

    def _load_image(self, path: str):
        with self.Image.open(path) as image:
            return np.asarray(image.convert("RGB"), dtype=np.uint8)

    def _load_mask(self, path: str):
        with self.Image.open(path) as mask:
            array = np.asarray(mask, dtype=np.int64)
        if array.ndim == 3:
            array = array[..., 0]
        return array

    def _pad_to_crop_size(self, image, mask):
        if self.crop_size is None:
            return image, mask
        height, width = mask.shape
        pad_h = max(0, self.crop_size - height)
        pad_w = max(0, self.crop_size - width)
        if pad_h == 0 and pad_w == 0:
            return image, mask
        image = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
        mask = np.pad(mask, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=IGNORE_INDEX)
        return image, mask

    def _apply_random_crop(self, image, mask):
        if self.crop_size is None:
            return image, mask
        image, mask = self._pad_to_crop_size(image, mask)
        height, width = mask.shape
        if height == self.crop_size and width == self.crop_size:
            return image, mask
        top = random.randint(0, height - self.crop_size)
        left = random.randint(0, width - self.crop_size)
        return (
            image[top:top + self.crop_size, left:left + self.crop_size],
            mask[top:top + self.crop_size, left:left + self.crop_size],
        )

    def _apply_flips(self, image, mask):
        if self.augmentation_cfg.get("random_horizontal_flip") and random.random() < 0.5:
            image = np.flip(image, axis=1).copy()
            mask = np.flip(mask, axis=1).copy()
        if self.augmentation_cfg.get("random_vertical_flip") and random.random() < 0.5:
            image = np.flip(image, axis=0).copy()
            mask = np.flip(mask, axis=0).copy()
        return image, mask

    def _apply_distort(self, image):
        if not self.enable_distort:
            return image

        cfg = self.augmentation_cfg.get("distort", {})
        pil = self.Image.fromarray(image)

        brightness_range = float(cfg.get("brightness_range", 0.0))
        if brightness_range > 0 and random.random() < float(cfg.get("brightness_prob", 0.0)):
            factor = 1.0 + random.uniform(-brightness_range, brightness_range)
            pil = self.ImageEnhance.Brightness(pil).enhance(max(0.1, factor))

        contrast_range = float(cfg.get("contrast_range", 0.0))
        if contrast_range > 0 and random.random() < float(cfg.get("contrast_prob", 0.0)):
            factor = 1.0 + random.uniform(-contrast_range, contrast_range)
            pil = self.ImageEnhance.Contrast(pil).enhance(max(0.1, factor))

        saturation_range = float(cfg.get("saturation_range", 0.0))
        if saturation_range > 0 and random.random() < float(cfg.get("saturation_prob", 0.0)):
            factor = 1.0 + random.uniform(-saturation_range, saturation_range)
            pil = self.ImageEnhance.Color(pil).enhance(max(0.1, factor))

        return np.asarray(pil, dtype=np.uint8)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        image = self._load_image(sample.image_path)
        mask = self._load_mask(sample.mask_path)

        if self.training:
            image, mask = self._apply_random_crop(image, mask)
            image, mask = self._apply_flips(image, mask)
            image = self._apply_distort(image)

        image = image.astype(np.float32) / 255.0
        image = (image - self.mean) / self.std
        image = np.transpose(image, (2, 0, 1)).astype(np.float32, copy=False)

        return {
            "image": image,
            "label": mask.astype(np.int64, copy=False),
            "cluster_id": np.int64(sample.cluster_id),
            "raw_cluster_id": np.int64(sample.raw_cluster_id),
            "sample_id": sample.sample_id,
            "scene_id": sample.scene_id,
            "image_path": sample.image_path,
            "mask_path": sample.mask_path,
        }


def collate_samples(batch_items):
    return {
        "image": np.stack([item["image"] for item in batch_items], axis=0).astype(np.float32, copy=False),
        "label": np.stack([item["label"] for item in batch_items], axis=0).astype(np.int64, copy=False),
        "cluster_id": np.asarray([item["cluster_id"] for item in batch_items], dtype=np.int64),
        "raw_cluster_id": np.asarray([item["raw_cluster_id"] for item in batch_items], dtype=np.int64),
        "sample_id": [item["sample_id"] for item in batch_items],
        "scene_id": [item["scene_id"] for item in batch_items],
        "image_path": [item["image_path"] for item in batch_items],
        "mask_path": [item["mask_path"] for item in batch_items],
    }


def iterate_batches(dataset, batch_size: int, shuffle: bool, seed: int):
    indices = list(range(len(dataset)))
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(indices)

    for start in range(0, len(indices), batch_size):
        batch_indices = indices[start:start + batch_size]
        batch_items = [dataset[idx] for idx in batch_indices]
        yield collate_samples(batch_items)


def batch_to_tensors(batch: dict, paddle):
    return {
        "image": paddle.to_tensor(batch["image"], dtype="float32"),
        "label": paddle.to_tensor(batch["label"], dtype="int64"),
        "cluster_id": paddle.to_tensor(batch["cluster_id"], dtype="int64"),
        "raw_cluster_id": batch["raw_cluster_id"],
        "sample_id": batch["sample_id"],
        "scene_id": batch["scene_id"],
        "image_path": batch["image_path"],
        "mask_path": batch["mask_path"],
    }


def build_model(runtime: dict) -> ClusterRoutedHRNet:
    model_cfg = runtime["config"]["model"]
    return ClusterRoutedHRNet(
        num_classes=runtime["num_classes"],
        num_heads=runtime["num_heads"],
        width=int(model_cfg.get("width", 48)),
        in_channels=int(model_cfg.get("in_channels", 3)),
        head_channels=model_cfg.get("head_channels"),
        align_corners=bool(model_cfg.get("align_corners", False)),
    )


def resolve_pretrained_weights(runtime: dict, stack: dict) -> str | None:
    pretrained_cfg = runtime["config"].get("pretrained", {})
    weights_path = pretrained_cfg.get("weights_path")
    if weights_path:
        resolved = resolve_existing_path(WORK_DIR, weights_path)
        return str(resolved) if resolved is not None else None

    weights_flag = pretrained_cfg.get("weights")
    if not weights_flag:
        return None

    model_cfg = runtime["config"]["model"]
    width = int(model_cfg.get("width", 48))
    save_dir = runtime["save_dir"] / "pretrained"
    ensure_dir(save_dir)
    return stack["get_pretrain_weights"](
        flag=weights_flag,
        class_name="HRNet",
        save_dir=str(save_dir),
        backbone_name=f"HRNet_W{width}",
    )


def build_optimizer(runtime: dict, paddle, model, planned_total_steps: int):
    training_cfg = runtime["config"]["training"]
    decay_steps = max(1, int(planned_total_steps))
    lr_scheduler = paddle.optimizer.lr.PolynomialDecay(
        learning_rate=float(training_cfg["learning_rate"]),
        decay_steps=decay_steps,
        end_lr=0.0,
        power=float(training_cfg.get("lr_decay_power", 0.9)),
    )
    optimizer = paddle.optimizer.Momentum(
        learning_rate=lr_scheduler,
        parameters=model.parameters(),
        momentum=float(training_cfg.get("momentum", 0.9)),
        weight_decay=float(training_cfg.get("weight_decay", 0.00004)),
    )
    return optimizer, lr_scheduler


def build_criterion(runtime: dict, stack: dict):
    loss_cfg = runtime["config"]["loss"]
    ignore_index = int(loss_cfg.get("ignore_index", IGNORE_INDEX))
    if bool(loss_cfg.get("use_mixed_loss", False)):
        seg_losses = stack["seg_losses"]
        losses = [
            seg_losses.CrossEntropyLoss(ignore_index=ignore_index),
            seg_losses.LovaszSoftmaxLoss(ignore_index=ignore_index),
        ]
        return seg_losses.MixedLoss(losses=losses, coef=list(loss_cfg.get("ce_lovasz_coef", [0.8, 0.2])))
    return stack["paddle"].nn.CrossEntropyLoss(ignore_index=ignore_index, axis=1)


def compute_total_loss(loss_output):
    if isinstance(loss_output, (list, tuple)):
        total = None
        for item in loss_output:
            total = item if total is None else total + item
        return total
    return loss_output


def compute_model_loss(outputs, labels, criterion):
    loss = compute_total_loss(criterion(outputs["logits"], labels))
    return {"total_loss": loss}


def compute_planned_training_steps(max_epochs: int, steps_per_epoch: int, max_train_steps: int | None) -> int:
    planned = max(1, int(max_epochs) * int(steps_per_epoch))
    if max_train_steps is not None:
        planned = min(planned, int(max_train_steps))
    return max(1, planned)


def save_prediction_sample(image_module, output_dir: Path, split_name: str, sample_id: str, label, prediction):
    split_dir = output_dir / split_name
    ensure_dir(split_dir)
    image_module.fromarray(np.asarray(label, dtype=np.uint8)).save(split_dir / f"{sample_id}_label.png")
    image_module.fromarray(np.asarray(prediction, dtype=np.uint8)).save(split_dir / f"{sample_id}_pred.png")


def evaluate_model(
    model,
    dataset,
    *,
    split_name: str,
    batch_size: int,
    max_batches: int | None,
    class_names,
    paddle,
    criterion,
    image_module,
    prediction_output_dir: Path | None = None,
    prediction_sample_count: int = 0,
):
    confusion = np.zeros((len(class_names), len(class_names)), dtype=np.int64)
    losses = []
    batch_count = 0
    sample_count = 0
    exported_samples = []

    model.eval()
    with paddle.no_grad():
        for batch in iterate_batches(dataset, batch_size=batch_size, shuffle=False, seed=0):
            batch_count += 1
            sample_count += len(batch["sample_id"])
            tensors = batch_to_tensors(batch, paddle)
            outputs = model(tensors["image"], tensors["cluster_id"])
            loss_dict = compute_model_loss(outputs=outputs, labels=tensors["label"], criterion=criterion)
            losses.append(float(loss_dict["total_loss"]))

            prediction = paddle.argmax(outputs["logits"], axis=1).numpy()
            labels = tensors["label"].numpy()
            update_confusion_matrix(
                confusion_matrix=confusion,
                prediction=prediction,
                label=labels,
                num_classes=len(class_names),
                ignore_index=IGNORE_INDEX,
            )

            if prediction_output_dir is not None and prediction_sample_count > 0 and len(exported_samples) < prediction_sample_count:
                for idx, sample_id in enumerate(batch["sample_id"]):
                    if len(exported_samples) >= prediction_sample_count:
                        break
                    save_prediction_sample(
                        image_module,
                        prediction_output_dir,
                        split_name,
                        sample_id,
                        labels[idx],
                        prediction[idx],
                    )
                    exported_samples.append(
                        {
                            "sample_id": sample_id,
                            "split": split_name,
                            "image_path": batch["image_path"][idx],
                            "mask_path": batch["mask_path"][idx],
                            "prediction_path": str((prediction_output_dir / split_name / f"{sample_id}_pred.png").resolve()),
                        }
                    )

            if max_batches is not None and batch_count >= max_batches:
                break

    metrics = compute_confusion_metrics(confusion, class_names)
    metrics.update(
        {
            "mean_loss": float(np.mean(losses)) if losses else None,
            "num_batches": batch_count,
            "num_samples": sample_count,
            "ignore_index": IGNORE_INDEX,
            "split_name": split_name,
            "prediction_samples": exported_samples,
        }
    )
    return metrics, confusion


def select_smoke_probe_samples(runtime: dict, per_head: int) -> list:
    probe_samples = []
    train_samples = runtime["split_map"]["train"]
    for cluster_id in range(runtime["num_heads"]):
        matches = [item for item in train_samples if item.cluster_id == cluster_id][:per_head]
        if len(matches) < per_head:
            raise RuntimeError(f"Unable to find {per_head} train samples for cluster_id={cluster_id}")
        probe_samples.extend(matches)
    return probe_samples


def estimate_eta_seconds(step_time_history: deque[float], remaining_steps: int) -> float | None:
    if remaining_steps <= 0:
        return 0.0
    if len(step_time_history) < 5:
        return None
    return float(np.mean(step_time_history) * remaining_steps)


def save_checkpoint(model, optimizer, checkpoint_dir: Path, metadata: dict, paddle) -> Path:
    ensure_dir(checkpoint_dir)
    paddle.save(model.state_dict(), str(checkpoint_dir / "model.pdparams"))
    paddle.save(optimizer.state_dict(), str(checkpoint_dir / "model.pdopt"))
    write_json(checkpoint_dir / "meta.json", metadata)
    return checkpoint_dir


def copy_checkpoint(src_dir: Path, dst_dir: Path) -> None:
    if dst_dir.exists():
        shutil.rmtree(dst_dir)
    shutil.copytree(src_dir, dst_dir)


def load_checkpoint(model, optimizer, checkpoint_dir: Path, paddle) -> dict:
    model_path = checkpoint_dir / "model.pdparams"
    optimizer_path = checkpoint_dir / "model.pdopt"
    meta_path = checkpoint_dir / "meta.json"
    if not model_path.is_file():
        raise FileNotFoundError(f"Checkpoint model params not found: {model_path}")
    if not optimizer_path.is_file():
        raise FileNotFoundError(f"Checkpoint optimizer state not found: {optimizer_path}")

    model_state = paddle.load(str(model_path))
    optimizer_state = paddle.load(str(optimizer_path))
    model.set_state_dict(model_state)
    optimizer.set_state_dict(optimizer_state)
    if meta_path.is_file():
        return read_json(meta_path)
    return {}


def load_model_only(model, checkpoint_dir: Path, paddle) -> dict:
    model_path = checkpoint_dir / "model.pdparams"
    meta_path = checkpoint_dir / "meta.json"
    if not model_path.is_file():
        raise FileNotFoundError(f"Checkpoint model params not found: {model_path}")
    model_state = paddle.load(str(model_path))
    model.set_state_dict(model_state)
    if meta_path.is_file():
        return read_json(meta_path)
    return {}


def build_datasets(runtime: dict, stack: dict):
    training_cfg = runtime["config"]["training"]
    norm_cfg = runtime["config"]["normalization"]
    augmentation_cfg = runtime["config"]["augmentation"]

    return {
        "train": ClusterRoutedTileDataset(
            runtime["split_map"]["train"],
            image_module=stack["Image"],
            image_enhance_module=stack["ImageEnhance"],
            mean=norm_cfg["mean"],
            std=norm_cfg["std"],
            training=True,
            augmentation_cfg=augmentation_cfg,
            crop_size=training_cfg.get("crop_size"),
            max_samples=getattr(runtime["args"], "max_train_samples", None),
        ),
        "val": ClusterRoutedTileDataset(
            runtime["split_map"]["val"],
            image_module=stack["Image"],
            image_enhance_module=stack["ImageEnhance"],
            mean=norm_cfg["mean"],
            std=norm_cfg["std"],
            training=False,
            augmentation_cfg={},
            crop_size=None,
            max_samples=getattr(runtime["args"], "max_val_samples", None),
        ),
        "test": ClusterRoutedTileDataset(
            runtime["split_map"]["test"],
            image_module=stack["Image"],
            image_enhance_module=stack["ImageEnhance"],
            mean=norm_cfg["mean"],
            std=norm_cfg["std"],
            training=False,
            augmentation_cfg={},
            crop_size=None,
            max_samples=getattr(runtime["args"], "max_test_samples", None),
        ),
    }


def log_dataset_overview(runtime: dict, logger: StageLogger, device: str) -> None:
    split_counts = {split: len(rows) for split, rows in runtime["split_map"].items()}
    logger.info("INIT", f"Using device={device}")
    logger.info("DATA", f"total_samples={len(runtime['samples'])}, train_samples={split_counts['train']}, val_samples={split_counts['val']}, test_samples={split_counts['test']}")

    train_dist = {idx: 0 for idx in range(runtime["num_heads"])}
    val_dist = {idx: 0 for idx in range(runtime["num_heads"])}
    test_dist = {idx: 0 for idx in range(runtime["num_heads"])}
    for sample in runtime["split_map"]["train"]:
        train_dist[sample.cluster_id] = train_dist.get(sample.cluster_id, 0) + 1
    for sample in runtime["split_map"]["val"]:
        val_dist[sample.cluster_id] = val_dist.get(sample.cluster_id, 0) + 1
    for sample in runtime["split_map"]["test"]:
        test_dist[sample.cluster_id] = test_dist.get(sample.cluster_id, 0) + 1

    logger.info("DATA", f"cluster_distribution_train={json.dumps(train_dist, ensure_ascii=False, sort_keys=True)}")
    logger.info("DATA", f"cluster_distribution_val={json.dumps(val_dist, ensure_ascii=False, sort_keys=True)}")
    logger.info("DATA", f"cluster_distribution_test={json.dumps(test_dist, ensure_ascii=False, sort_keys=True)}")
    logger.info(
        "DATA",
        "routing_validation="
        + json.dumps(
            {
                "missing_cluster_ids": runtime["routing_validation"]["missing_cluster_ids"],
                "unexpected_cluster_ids": runtime["routing_validation"]["unexpected_cluster_ids"],
                "missing_file_count": len(runtime["routing_validation"]["missing_files"]),
                "missing_cluster_id_sample_count": len(runtime["routing_validation"]["missing_cluster_id_samples"]),
                "all_clusters_have_train_val_test": runtime["routing_validation"]["all_clusters_have_train_val_test"],
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
    )
    for preview_row in preview_alignment_rows(runtime["samples"], limit=6):
        logger.info("DATA", "alignment_check=" + json.dumps(preview_row, ensure_ascii=False))


def write_smoke_status(save_dir: Path, payload: dict) -> None:
    write_json(save_dir / "smoke_test_status.json", payload)


def ensure_smoke_passed(save_dir: Path) -> dict:
    status_path = save_dir / "smoke_test_status.json"
    if not status_path.is_file():
        raise RuntimeError(
            f"Smoke test status not found: {status_path}. Run --run-smoke-test successfully before formal training."
        )
    payload = read_json(status_path)
    if payload.get("status") != "passed":
        raise RuntimeError(f"Smoke test status is not passed: {payload}")
    return payload


def run_smoke_test(runtime: dict, stack: dict, logger: StageLogger, device: str, planned_total_steps: int) -> dict:
    paddle = stack["paddle"]
    set_seed(int(runtime["config"]["training"]["seed"]), paddle)
    model = build_model(runtime)
    optimizer, lr_scheduler = build_optimizer(runtime, paddle, model, planned_total_steps)
    criterion = build_criterion(runtime, stack)

    pretrain_path = resolve_pretrained_weights(runtime, stack)
    pretrain_report = None
    if pretrain_path:
        pretrain_report = load_pretrained_weights(model, pretrain_path)

    smoke_cfg = runtime["config"].get("smoke_test", {})
    probe_samples = select_smoke_probe_samples(runtime, int(smoke_cfg.get("probe_samples_per_head", 1)))
    smoke_dataset = ClusterRoutedTileDataset(
        probe_samples,
        image_module=stack["Image"],
        image_enhance_module=stack["ImageEnhance"],
        mean=runtime["config"]["normalization"]["mean"],
        std=runtime["config"]["normalization"]["std"],
        training=False,
        augmentation_cfg={},
        crop_size=None,
    )
    logger.info("SMOKE", f"Dataset check passed: samples={len(smoke_dataset)}")

    smoke_batch = collate_samples([smoke_dataset[idx] for idx in range(len(smoke_dataset))])
    tensors = batch_to_tensors(smoke_batch, paddle)
    active_heads = sorted(set(int(item) for item in smoke_batch["cluster_id"].tolist()))

    model.train()
    outputs = model(tensors["image"], tensors["cluster_id"], return_all_heads=True)
    logger.info("SMOKE", f"Forward passed: batch_size={len(smoke_batch['sample_id'])}, active_heads={active_heads}")

    loss_dict = compute_model_loss(outputs=outputs, labels=tensors["label"], criterion=criterion)
    total_loss = loss_dict["total_loss"]
    total_loss.backward()
    logger.info("SMOKE", "Backward passed")

    optimizer.step()
    optimizer.clear_grad()
    lr_scheduler.step()
    logger.info("SMOKE", "Optimizer step passed")

    checkpoint_dir = runtime["save_dir"] / "checkpoints" / "smoke_ckpt"
    checkpoint_metadata = {
        "stage": "smoke_test",
        "step": 1,
        "device": device,
        "batch_size": len(smoke_batch["sample_id"]),
        "cluster_ids": smoke_batch["cluster_id"].tolist(),
        "loss": float(total_loss),
        "pretrained_report": pretrain_report,
    }
    save_checkpoint(model, optimizer, checkpoint_dir, checkpoint_metadata, paddle)

    reloaded_model = build_model(runtime)
    if pretrain_path:
        load_pretrained_weights(reloaded_model, pretrain_path)
    reloaded_optimizer, _ = build_optimizer(runtime, paddle, reloaded_model, planned_total_steps)
    reload_meta = load_checkpoint(reloaded_model, reloaded_optimizer, checkpoint_dir, paddle)
    reloaded_model.eval()
    model.eval()
    with paddle.no_grad():
        saved_outputs = model(tensors["image"], tensors["cluster_id"])
        reloaded_outputs = reloaded_model(tensors["image"], tensors["cluster_id"])
    reload_main_diff = float(paddle.max(paddle.abs(saved_outputs["logits"] - reloaded_outputs["logits"])))
    logger.info("SMOKE", f"Checkpoint save/load passed: {checkpoint_dir}")

    validation_dataset = ClusterRoutedTileDataset(
        runtime["split_map"]["val"],
        image_module=stack["Image"],
        image_enhance_module=stack["ImageEnhance"],
        mean=runtime["config"]["normalization"]["mean"],
        std=runtime["config"]["normalization"]["std"],
        training=False,
        augmentation_cfg={},
        crop_size=None,
        max_samples=runtime["args"].max_val_samples,
    )
    metrics, confusion = evaluate_model(
        model=model,
        dataset=validation_dataset,
        split_name="val",
        batch_size=int(runtime["config"]["training"]["eval_batch_size"]),
        max_batches=int(smoke_cfg.get("val_batches", 2)),
        class_names=runtime["class_names"],
        paddle=paddle,
        criterion=criterion,
        image_module=stack["Image"],
        prediction_output_dir=None,
        prediction_sample_count=0,
    )
    logger.info(
        "SMOKE",
        f"Mini evaluation passed: OA={metrics['overall_accuracy']:.6f}, mIoU={metrics['mean_iou']:.6f}"
    )

    metrics_dir = runtime["save_dir"] / "metrics"
    write_json(metrics_dir / "smoke_test_metrics.json", metrics)
    write_per_class_csv(metrics_dir / "smoke_test_per_class.csv", metrics)
    write_confusion_csv(metrics_dir / "smoke_test_confusion.csv", confusion, runtime["class_names"])

    smoke_status = {
        "status": "passed",
        "device": device,
        "active_heads": active_heads,
        "checkpoint_dir": str(checkpoint_dir),
        "reload_main_max_abs_diff": reload_main_diff,
        "mini_eval": {
            "loss": metrics["mean_loss"],
            "OA": metrics["overall_accuracy"],
            "mIoU": metrics["mean_iou"],
            "mAcc": metrics["mean_accuracy"],
            "Kappa": metrics["kappa"],
        },
        "pretrained_report": pretrain_report,
        "reload_meta": reload_meta,
    }
    write_smoke_status(runtime["save_dir"], smoke_status)
    return smoke_status


def save_metric_bundle(base_path: Path, metrics: dict, confusion, class_names) -> None:
    write_json(base_path.with_suffix(".json"), metrics)
    write_per_class_csv(base_path.with_name(base_path.name + "_per_class.csv"), metrics)
    write_confusion_csv(base_path.with_name(base_path.name + "_confusion.csv"), confusion, class_names)


def evaluate_checkpoint_on_split(
    runtime: dict,
    stack: dict,
    *,
    checkpoint_dir: Path,
    split_name: str,
    logger: StageLogger | None,
    export_predictions: bool,
    prediction_sample_count: int,
):
    paddle = stack["paddle"]
    model = build_model(runtime)
    load_model_only(model, checkpoint_dir, paddle)
    criterion = build_criterion(runtime, stack)
    datasets = build_datasets(runtime, stack)
    metrics, confusion = evaluate_model(
        model=model,
        dataset=datasets[split_name],
        split_name=split_name,
        batch_size=int(runtime["config"]["training"]["eval_batch_size"]),
        max_batches=None,
        class_names=runtime["class_names"],
        paddle=paddle,
        criterion=criterion,
        image_module=stack["Image"],
        prediction_output_dir=(runtime["save_dir"] / "prediction_samples") if export_predictions else None,
        prediction_sample_count=prediction_sample_count if export_predictions else 0,
    )
    if logger is not None:
        stage = "TEST" if split_name == "test" else "EVAL"
        if split_name == "test":
            logger.info(
                stage,
                format_test_log(
                    test_loss=metrics["mean_loss"],
                    oa=metrics["overall_accuracy"],
                    miou=metrics["mean_iou"],
                    macc=metrics["mean_accuracy"],
                    kappa=metrics["kappa"],
                ),
            )
        else:
            logger.info(
                stage,
                format_eval_log(
                    epoch=int(metrics.get("epoch", 0)),
                    total_epochs=int(runtime["config"]["training"]["epochs"]),
                    val_loss=metrics["mean_loss"],
                    oa=metrics["overall_accuracy"],
                    miou=metrics["mean_iou"],
                    macc=metrics["mean_accuracy"],
                    kappa=metrics["kappa"],
                ),
            )
    return metrics, confusion


def run_training(runtime: dict, stack: dict, logger: StageLogger, device: str) -> dict:
    smoke_payload = ensure_smoke_passed(runtime["save_dir"])
    logger.info("INIT", f"Smoke gate passed: status={smoke_payload['status']}, active_heads={smoke_payload['active_heads']}")

    paddle = stack["paddle"]
    training_cfg = runtime["config"]["training"]
    eval_cfg = runtime["config"]["evaluation"]
    train_batch_size = int(training_cfg["train_batch_size"])
    eval_batch_size = int(training_cfg["eval_batch_size"])
    max_epochs = int(training_cfg["epochs"])
    max_train_steps = runtime["args"].max_train_steps
    log_interval = max(1, int(training_cfg.get("log_interval_steps", 20)))

    set_seed(int(training_cfg["seed"]), paddle)
    model = build_model(runtime)
    steps_per_epoch = max(1, math.ceil(len(runtime["split_map"]["train"]) / train_batch_size))
    planned_total_steps = compute_planned_training_steps(max_epochs, steps_per_epoch, max_train_steps)
    optimizer, lr_scheduler = build_optimizer(runtime, paddle, model, planned_total_steps)
    criterion = build_criterion(runtime, stack)

    pretrain_path = resolve_pretrained_weights(runtime, stack)
    pretrained_report = None
    if pretrain_path and not training_cfg.get("resume_checkpoint"):
        pretrained_report = load_pretrained_weights(model, pretrain_path)
        write_json(runtime["save_dir"] / "config_snapshot" / "pretrained_load_report.json", pretrained_report)

    datasets = build_datasets(runtime, stack)
    logger.info(
        "INIT",
        f"Training setup: epochs={max_epochs}, train_batch_size={train_batch_size}, eval_batch_size={eval_batch_size}, learning_rate={float(training_cfg['learning_rate']):.6f}, crop_size={int(training_cfg['crop_size'])}, steps_per_epoch={steps_per_epoch}, total_steps={planned_total_steps}",
    )
    if pretrained_report is not None:
        logger.info(
            "INIT",
            f"Loaded pretrained weights: path={pretrained_report['pretrained_path']}, num_loaded_tensors={pretrained_report['num_loaded_tensors']}/{pretrained_report['num_model_tensors']}",
        )

    best_score = float("-inf")
    best_epoch = 0
    global_step = 0
    start_epoch = 1
    resumed_from = None
    recent_step_times = deque(maxlen=TRAIN_LOG_ETA_WINDOW)

    resume_checkpoint = training_cfg.get("resume_checkpoint")
    if resume_checkpoint:
        resume_path = resolve_existing_path(runtime["save_dir"], resume_checkpoint)
        if resume_path is None:
            raise FileNotFoundError(f"Unable to resolve resume checkpoint: {resume_checkpoint}")
        resume_meta = load_checkpoint(model, optimizer, resume_path, paddle)
        resumed_from = str(resume_path)
        global_step = int(resume_meta.get("global_step", 0))
        best_score = float(resume_meta.get("best_score", best_score))
        best_epoch = int(resume_meta.get("best_epoch", 0))
        start_epoch = int(resume_meta.get("epoch", 0)) + 1
        logger.info("INIT", f"Resumed from checkpoint: {resume_path}")

    best_model_dir = runtime["save_dir"] / "best_model"
    latest_model_dir = runtime["save_dir"] / "latest_model"
    metrics_dir = runtime["save_dir"] / "metrics"
    status_path = runtime["save_dir"] / "training_status.json"

    final_status = "running"
    last_completed_epoch = start_epoch - 1
    for epoch in range(start_epoch, max_epochs + 1):
        model.train()
        epoch_loss_values = []
        epoch_start = time.perf_counter()

        for step_in_epoch, batch in enumerate(
            iterate_batches(
                datasets["train"],
                batch_size=train_batch_size,
                shuffle=True,
                seed=int(training_cfg["seed"]) + epoch,
            ),
            start=1,
        ):
            global_step += 1
            step_start = time.perf_counter()
            tensors = batch_to_tensors(batch, paddle)
            outputs = model(tensors["image"], tensors["cluster_id"])
            loss_dict = compute_model_loss(outputs=outputs, labels=tensors["label"], criterion=criterion)
            total_loss = loss_dict["total_loss"]
            total_loss.backward()
            optimizer.step()
            optimizer.clear_grad()
            lr_scheduler.step()

            step_time = time.perf_counter() - step_start
            recent_step_times.append(step_time)
            loss_value = float(total_loss)
            epoch_loss_values.append(loss_value)

            will_stop = max_train_steps is not None and global_step >= max_train_steps
            should_log = (
                global_step == 1
                or global_step % log_interval == 0
                or step_in_epoch == steps_per_epoch
                or will_stop
            )
            if should_log:
                remaining_steps = max(0, planned_total_steps - global_step)
                eta_seconds = estimate_eta_seconds(recent_step_times, remaining_steps)
                logger.info(
                    "TRAIN",
                    format_train_log(
                        epoch=epoch,
                        total_epochs=max_epochs,
                        step=step_in_epoch,
                        total_steps=steps_per_epoch,
                        loss=loss_value,
                        lr=float(optimizer.get_lr()),
                        step_time_seconds=float(np.mean(recent_step_times)) if recent_step_times else step_time,
                        eta_seconds=eta_seconds,
                    ),
                )
            if will_stop:
                break

        epoch_time = time.perf_counter() - epoch_start
        last_completed_epoch = epoch
        val_metrics, val_confusion = evaluate_model(
            model=model,
            dataset=datasets["val"],
            split_name="val",
            batch_size=eval_batch_size,
            max_batches=runtime["args"].max_val_batches,
            class_names=runtime["class_names"],
            paddle=paddle,
            criterion=criterion,
            image_module=stack["Image"],
            prediction_output_dir=None,
            prediction_sample_count=0,
        )
        val_metrics["epoch"] = epoch
        val_metrics["global_step"] = global_step
        val_metrics["train_loss_mean"] = float(np.mean(epoch_loss_values)) if epoch_loss_values else None
        logger.info(
            "EVAL",
            format_eval_log(
                epoch=epoch,
                total_epochs=max_epochs,
                val_loss=val_metrics["mean_loss"],
                oa=val_metrics["overall_accuracy"],
                miou=val_metrics["mean_iou"],
                macc=val_metrics["mean_accuracy"],
                kappa=val_metrics["kappa"],
            ),
        )
        save_metric_bundle(metrics_dir / f"val_epoch_{epoch:03d}", val_metrics, val_confusion, runtime["class_names"])

        checkpoint_dir = runtime["save_dir"] / "checkpoints" / f"epoch_{epoch:03d}_step_{global_step:06d}"
        checkpoint_metadata = {
            "stage": "train",
            "epoch": epoch,
            "global_step": global_step,
            "device": device,
            "best_score": max(best_score, float(val_metrics["mean_iou"])),
            "best_epoch": best_epoch,
            "train_loss_mean": val_metrics["train_loss_mean"],
            "val_mean_iou": val_metrics["mean_iou"],
            "val_oa": val_metrics["overall_accuracy"],
            "resumed_from": resumed_from,
            "pretrained_report": pretrained_report,
        }
        save_checkpoint(model, optimizer, checkpoint_dir, checkpoint_metadata, paddle)
        copy_checkpoint(checkpoint_dir, latest_model_dir)
        logger.info("CKPT", format_checkpoint_log(epoch=epoch, path=checkpoint_dir))

        current_score = float(val_metrics["mean_iou"])
        if current_score > best_score:
            best_score = current_score
            best_epoch = epoch
            copy_checkpoint(checkpoint_dir, best_model_dir)
            logger.info("CKPT", format_best_model_log(epoch=epoch, val_miou=best_score, path=best_model_dir))

            if bool(eval_cfg.get("evaluate_test_on_best", True)):
                test_metrics, test_confusion = evaluate_model(
                    model=model,
                    dataset=datasets["test"],
                    split_name="test",
                    batch_size=eval_batch_size,
                    max_batches=runtime["args"].max_test_batches,
                    class_names=runtime["class_names"],
                    paddle=paddle,
                    criterion=criterion,
                    image_module=stack["Image"],
                    prediction_output_dir=(runtime["save_dir"] / "prediction_samples")
                    if bool(eval_cfg.get("export_prediction_samples", True))
                    else None,
                    prediction_sample_count=int(eval_cfg.get("prediction_sample_count", 12)),
                )
                test_metrics["epoch"] = epoch
                test_metrics["global_step"] = global_step
                logger.info(
                    "TEST",
                    format_test_log(
                        test_loss=test_metrics["mean_loss"],
                        oa=test_metrics["overall_accuracy"],
                        miou=test_metrics["mean_iou"],
                        macc=test_metrics["mean_accuracy"],
                        kappa=test_metrics["kappa"],
                    ),
                )
                save_metric_bundle(metrics_dir / "best_model_test", test_metrics, test_confusion, runtime["class_names"])
                write_metrics_text(
                    runtime["save_dir"] / "best_model_metrics.txt",
                    test_metrics,
                    split_name="test",
                    checkpoint_path=str(best_model_dir),
                )
                write_per_class_csv(runtime["save_dir"] / "per_class_metrics.csv", test_metrics)
                write_confusion_csv(runtime["save_dir"] / "confusion_matrix.csv", test_confusion, runtime["class_names"])
                write_json(runtime["save_dir"] / "metrics" / "best_model_test_predictions.json", test_metrics["prediction_samples"])

        remaining_steps = max(0, planned_total_steps - global_step)
        logger.info(
            "TRAIN",
            f"Epoch Summary: epoch={epoch}/{max_epochs}, mean_loss={float(np.mean(epoch_loss_values)):.6f}, epoch_time={epoch_time:.2f}s, best_mIoU={best_score:.6f}, eta={format_eta(estimate_eta_seconds(recent_step_times, remaining_steps))}",
        )

        write_json(
            status_path,
            {
                "status": final_status,
                "current_epoch": epoch,
                "global_step": global_step,
                "best_epoch": best_epoch,
                "best_score": best_score,
                "latest_checkpoint_dir": str(checkpoint_dir),
                "latest_model_dir": str(latest_model_dir),
                "best_model_dir": str(best_model_dir),
                "resumed_from": resumed_from,
            },
        )

        if max_train_steps is not None and global_step >= max_train_steps:
            break

    final_status = "completed"
    write_json(
        status_path,
        {
            "status": final_status,
            "current_epoch": last_completed_epoch,
            "global_step": global_step,
            "best_epoch": best_epoch,
            "best_score": best_score,
            "best_model_dir": str(best_model_dir),
            "latest_model_dir": str(latest_model_dir),
            "resumed_from": resumed_from,
        },
    )
    return {
        "status": final_status,
        "epochs_completed": last_completed_epoch,
        "global_steps": global_step,
        "best_epoch": best_epoch,
        "best_score": best_score,
        "best_model_dir": str(best_model_dir),
        "latest_model_dir": str(latest_model_dir),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="4-head cluster-routed HRNet training for GID15.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--save-dir", default=None)
    parser.add_argument("--device", default="gpu")
    parser.add_argument("--run-smoke-test", action="store_true")
    parser.add_argument("--run-train", action="store_true")
    parser.add_argument("--resume-checkpoint", default=None)
    parser.add_argument("--train-batch-size", type=int, default=None)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--max-train-steps", type=int, default=None)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    parser.add_argument("--max-test-batches", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.run_smoke_test and not args.run_train:
        args.run_smoke_test = True

    stack = maybe_import_training_stack()
    runtime = build_runtime(args)
    create_output_structure(runtime["save_dir"])
    write_config_snapshot(runtime)

    logger = StageLogger(runtime["save_dir"] / "logs")
    paddle = stack["paddle"]
    paddle.set_device(args.device)
    device = paddle.get_device()
    logger.info("INIT", f"Environment ready: device={device}, {query_gpu_name(device)}")
    log_dataset_overview(runtime, logger, device)

    if args.run_smoke_test:
        smoke_status = run_smoke_test(runtime, stack, logger, device, runtime["planned_total_steps"])
        logger.info(
            "SMOKE",
            f"Smoke test completed: status={smoke_status['status']}, active_heads={smoke_status['active_heads']}, checkpoint_dir={smoke_status['checkpoint_dir']}",
        )

    if args.run_train:
        result = run_training(runtime, stack, logger, device)
        logger.info(
            "SYSTEM",
            f"Training finished: status={result['status']}, epochs_completed={result['epochs_completed']}, best_epoch={result['best_epoch']}, best_mIoU={result['best_score']:.6f}",
        )


if __name__ == "__main__":
    main()
