#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import shutil
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
    read_label_names,
    split_samples,
    summarize_routed_samples,
    validate_routed_samples,
)
from metrics_utils import (
    build_logging_schema,
    compute_confusion_metrics,
    ensure_dir,
    format_end_time_from_eta,
    format_epoch_summary_log,
    format_path_log,
    format_startup_log,
    format_timestamp,
    format_train_log,
    format_val_log,
    read_json,
    update_confusion_matrix,
    write_json,
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


def resolve_existing_path(base_dir: Path, path_value: str | None, allow_missing: bool = False) -> Path | None:
    if path_value is None:
        return None
    path = Path(path_value)
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    else:
        path = path.resolve()
    if path.exists() or allow_missing:
        return path
    raise FileNotFoundError(f"Path does not exist: {path}")


def create_output_structure(save_dir: Path) -> None:
    ensure_dir(save_dir)
    for name in OUTPUT_SUBDIRS:
        ensure_dir(save_dir / name)


def setup_logger(log_path: Path) -> logging.Logger:
    ensure_dir(log_path.parent)
    logger = logging.getLogger("fastscnn_data_split")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    logger.propagate = False

    formatter = logging.Formatter("%(message)s")

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def maybe_import_training_stack():
    try:
        import paddle
        import paddlers
        from PIL import Image, ImageEnhance
        from paddlers.models import seg_losses
        from paddlers.models.paddleseg.models.fast_scnn import FastSCNN as PaddleFastSCNN
        from fastscnn_routed_model import ClusterRoutedFastSCNN
    except Exception as exc:
        raise RuntimeError(
            "Training dependencies are not available. Activate the paddlers conda env before running this script."
        ) from exc

    return {
        "paddle": paddle,
        "paddlers": paddlers,
        "Image": Image,
        "ImageEnhance": ImageEnhance,
        "seg_losses": seg_losses,
        "PaddleFastSCNN": PaddleFastSCNN,
        "ClusterRoutedFastSCNN": ClusterRoutedFastSCNN,
    }


def set_seed(seed: int, paddle=None) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if paddle is not None:
        paddle.seed(seed)


def build_model_structure_from_config(config: dict, num_classes: int) -> dict[str, object]:
    from fastscnn_routed_model import build_model_structure

    return build_model_structure(config, num_classes)


def build_runtime(args: argparse.Namespace) -> dict[str, object]:
    config_path = resolve_existing_path(WORK_DIR, args.config)
    assert config_path is not None
    config = read_json(config_path)

    if args.train_batch_size is not None:
        config["training"]["train_batch_size"] = int(args.train_batch_size)
    if args.eval_batch_size is not None:
        config["training"]["eval_batch_size"] = int(args.eval_batch_size)
    if args.max_epochs is not None:
        config["training"]["epochs"] = int(args.max_epochs)
    if args.resume_checkpoint is not None:
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
    if args.run_smoke_test or args.run_train:
        if selected_k != 4 or num_heads != 4:
            raise ValueError(
                f"This acceptance run is fixed to K=4 and num_heads=4, got selected_k={selected_k}, num_heads={num_heads}"
            )
        if routing_cfg.get("cluster_id_handling", "route") != "route":
            raise ValueError("4-head acceptance requires routing.cluster_id_handling='route'.")
        if routing_cfg.get("sample_path_source", "input") != "input":
            raise ValueError("4-head acceptance requires routing.sample_path_source='input'.")
        if num_classes != 16:
            raise ValueError(f"4-head acceptance requires num_classes=16, got {num_classes}")

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
    if args.max_train_steps is not None:
        planned_total_steps = min(planned_total_steps, int(args.max_train_steps))

    save_dir_value = args.save_dir or config.get("save_dir", "output")
    save_dir = resolve_existing_path(WORK_DIR, save_dir_value, allow_missing=True)
    assert save_dir is not None

    run_name = args.run_name or time.strftime("acceptance_k4_%Y%m%d_%H%M%S")
    log_path = save_dir / "logs" / f"{run_name}.log"

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
        "log_path": log_path,
        "run_name": run_name,
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
    write_json(snapshot_dir / "model_structure.json", build_model_structure_from_config(runtime["config"], runtime["num_classes"]))
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
        },
    )
    write_json(
        snapshot_dir / "logging_schema.json",
        build_logging_schema("ClusterRoutedFastSCNN", runtime["num_heads"]),
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


def build_model(runtime: dict, stack: dict):
    model_cfg = runtime["config"]["model"]
    return stack["ClusterRoutedFastSCNN"](
        num_classes=runtime["num_classes"],
        num_heads=runtime["num_heads"],
        in_channels=int(model_cfg.get("in_channels", 3)),
        enable_auxiliary_loss=bool(model_cfg.get("enable_auxiliary_loss", True)),
        align_corners=bool(model_cfg.get("align_corners", False)),
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


def build_branch_criterion(stack: dict, runtime: dict):
    loss_cfg = runtime["config"]["loss"]
    seg_losses = stack["seg_losses"]
    ignore_index = int(loss_cfg.get("ignore_index", IGNORE_INDEX))
    use_mixed_loss = bool(loss_cfg.get("use_mixed_loss", True))

    if use_mixed_loss:
        coefs = list(loss_cfg.get("ce_lovasz_coef", [0.8, 0.2]))
        losses = [
            seg_losses.CrossEntropyLoss(ignore_index=ignore_index),
            seg_losses.LovaszSoftmaxLoss(ignore_index=ignore_index),
        ]
        return seg_losses.MixedLoss(losses=losses, coef=coefs)
    return seg_losses.CrossEntropyLoss(ignore_index=ignore_index)


def compute_total_loss(loss_output):
    if isinstance(loss_output, (list, tuple)):
        total = None
        for item in loss_output:
            total = item if total is None else total + item
        return total
    return loss_output


def compute_model_loss(outputs, labels, criterion, aux_loss_weight: float):
    main_loss = compute_total_loss(criterion(outputs["logits"], labels))
    aux_logits = outputs.get("aux_logits")
    if aux_logits is None:
        total_loss = main_loss
        aux_loss = None
    else:
        aux_loss = compute_total_loss(criterion(aux_logits, labels))
        total_loss = main_loss + float(aux_loss_weight) * aux_loss
    return {
        "total_loss": total_loss,
        "main_loss": main_loss,
        "aux_loss": aux_loss,
    }


def compute_module_grad_norm(layer) -> float:
    total = 0.0
    for parameter in layer.parameters():
        grad = parameter.grad
        if grad is None:
            continue
        grad_array = grad.numpy().astype(np.float64, copy=False)
        total += float(np.sum(np.square(grad_array)))
    return float(total ** 0.5)


def collect_gradient_report(model) -> dict[str, float]:
    report = {
        "learning_to_downsample": compute_module_grad_norm(model.learning_to_downsample),
        "global_feature_extractor": compute_module_grad_norm(model.global_feature_extractor),
        "feature_fusion": compute_module_grad_norm(model.feature_fusion),
    }
    for idx, head in enumerate(model.main_heads):
        report[f"main_head_{idx}"] = compute_module_grad_norm(head)
    if model.aux_heads is not None:
        for idx, head in enumerate(model.aux_heads):
            report[f"aux_head_{idx}"] = compute_module_grad_norm(head)
    report["model_total"] = compute_module_grad_norm(model)
    return report


def assert_gradient_report(gradient_report: dict[str, float], num_heads: int, enable_auxiliary_loss: bool) -> None:
    required_keys = [
        "learning_to_downsample",
        "global_feature_extractor",
        "feature_fusion",
    ] + [f"main_head_{idx}" for idx in range(num_heads)]
    if enable_auxiliary_loss:
        required_keys.extend([f"aux_head_{idx}" for idx in range(num_heads)])

    zero_grad_keys = [name for name in required_keys if float(gradient_report.get(name, 0.0)) <= 0.0]
    if zero_grad_keys:
        raise RuntimeError(f"Expected non-zero gradients for all routed branches, but found zeros in: {zero_grad_keys}")


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


def verify_checkpoint_reload(runtime: dict, stack: dict, checkpoint_dir: Path, probe_batch: dict, planned_total_steps: int) -> dict:
    paddle = stack["paddle"]
    model = build_model(runtime, stack)
    optimizer, _ = build_optimizer(runtime, paddle, model, planned_total_steps)
    meta = load_checkpoint(model, optimizer, checkpoint_dir, paddle)
    model.eval()
    with paddle.no_grad():
        reloaded_outputs = model(
            probe_batch["image"],
            probe_batch["cluster_id"],
            return_all_heads=True,
        )
    return {
        "meta": meta,
        "outputs": reloaded_outputs,
    }


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
            max_samples=runtime["args"].max_train_samples,
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
            max_samples=runtime["args"].max_val_samples,
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
            max_samples=runtime["args"].max_test_samples,
        ),
    }


def compute_planned_training_steps(max_epochs: int, steps_per_epoch: int, max_train_steps: int | None) -> int:
    planned = max(1, int(max_epochs) * int(steps_per_epoch))
    if max_train_steps is not None:
        planned = min(planned, int(max_train_steps))
    return max(1, planned)


def evaluate_model(
    model,
    dataset,
    batch_size: int,
    max_batches: int | None,
    class_names,
    paddle,
    criterion,
    aux_loss_weight: float,
):
    confusion = np.zeros((len(class_names), len(class_names)), dtype=np.int64)
    losses = []
    main_losses = []
    aux_losses = []
    batch_count = 0
    sample_count = 0

    model.eval()
    with paddle.no_grad():
        for batch in iterate_batches(dataset, batch_size=batch_size, shuffle=False, seed=0):
            batch_count += 1
            sample_count += len(batch["sample_id"])
            tensors = batch_to_tensors(batch, paddle)
            outputs = model(tensors["image"], tensors["cluster_id"])
            loss_dict = compute_model_loss(
                outputs=outputs,
                labels=tensors["label"],
                criterion=criterion,
                aux_loss_weight=aux_loss_weight,
            )
            losses.append(float(loss_dict["total_loss"]))
            main_losses.append(float(loss_dict["main_loss"]))
            if loss_dict["aux_loss"] is not None:
                aux_losses.append(float(loss_dict["aux_loss"]))

            prediction = paddle.argmax(outputs["logits"], axis=1).numpy()
            update_confusion_matrix(
                confusion_matrix=confusion,
                prediction=prediction,
                label=tensors["label"].numpy(),
                num_classes=len(class_names),
                ignore_index=IGNORE_INDEX,
            )
            if max_batches is not None and batch_count >= max_batches:
                break

    metrics = compute_confusion_metrics(confusion, class_names)
    metrics.update(
        {
            "mean_loss": float(np.mean(losses)) if losses else None,
            "mean_main_loss": float(np.mean(main_losses)) if main_losses else None,
            "mean_aux_loss": float(np.mean(aux_losses)) if aux_losses else None,
            "num_batches": batch_count,
            "num_samples": sample_count,
            "ignore_index": IGNORE_INDEX,
        }
    )
    return metrics, confusion


def select_smoke_probe_samples(runtime: dict):
    probe_samples = []
    train_samples = runtime["split_map"]["train"]
    for cluster_id in range(runtime["num_heads"]):
        sample = next((item for item in train_samples if item.cluster_id == cluster_id), None)
        if sample is None:
            raise RuntimeError(f"Unable to find a train sample for cluster_id={cluster_id}")
        probe_samples.append(sample)
    return probe_samples


def estimate_eta_seconds(step_time_history: deque[float], remaining_steps: int) -> float | None:
    if remaining_steps <= 0:
        return 0.0
    if len(step_time_history) < 5:
        return None
    return float(np.mean(step_time_history) * remaining_steps)


def log_startup_summary(
    runtime: dict,
    logger: logging.Logger,
    start_time: str,
    train_size: int,
    val_size: int,
    steps_per_epoch: int,
    env_report: dict[str, object],
) -> None:
    training_cfg = runtime["config"]["training"]
    logger.info(
        format_startup_log(
            start_time=start_time,
            total_epochs=int(training_cfg["epochs"]),
            train_size=train_size,
            val_size=val_size,
            steps_per_epoch=steps_per_epoch,
            model_type="ClusterRoutedFastSCNN",
            num_heads=int(runtime["num_heads"]),
            train_batch_size=int(training_cfg["train_batch_size"]),
            eval_batch_size=int(training_cfg["eval_batch_size"]),
            cuda_visible_devices=str(env_report.get("cuda_visible_devices", "not_set")),
            physical_gpu_name=str(env_report.get("gpu_name") or "unknown"),
        )
    )
    logger.info(format_path_log("best_model_dir", runtime["save_dir"] / "best_model"))
    logger.info(format_path_log("latest_model_dir", runtime["save_dir"] / "latest_model"))
    logger.info(format_path_log("checkpoint_dir", runtime["save_dir"] / "checkpoints"))
    logger.info(format_path_log("metrics_dir", runtime["save_dir"] / "metrics"))
    logger.info(format_path_log("prediction_samples_dir", runtime["save_dir"] / "prediction_samples"))
    logger.info(format_path_log("config_snapshot_dir", runtime["save_dir"] / "config_snapshot"))


def run_smoke_test(runtime: dict, stack: dict, logger: logging.Logger, device: str, planned_total_steps: int) -> dict:
    paddle = stack["paddle"]
    model = build_model(runtime, stack)
    optimizer, lr_scheduler = build_optimizer(runtime, paddle, model, planned_total_steps)
    criterion = build_branch_criterion(stack, runtime)
    aux_loss_weight = float(runtime["config"]["loss"].get("aux_loss_weight", 0.4))

    model.train()

    smoke_dataset = ClusterRoutedTileDataset(
        select_smoke_probe_samples(runtime),
        image_module=stack["Image"],
        image_enhance_module=stack["ImageEnhance"],
        mean=runtime["config"]["normalization"]["mean"],
        std=runtime["config"]["normalization"]["std"],
        training=False,
        augmentation_cfg={},
        crop_size=None,
    )
    smoke_batch = collate_samples([smoke_dataset[idx] for idx in range(len(smoke_dataset))])
    tensors = batch_to_tensors(smoke_batch, paddle)

    outputs = model(tensors["image"], tensors["cluster_id"], return_all_heads=True)
    routed_logits = outputs["logits"]
    routed_aux_logits = outputs["aux_logits"]
    all_head_logits = outputs["all_head_logits"]
    all_aux_head_logits = outputs.get("all_aux_head_logits")

    route_diffs = []
    aux_route_diffs = []
    for sample_index, cluster_id in enumerate(smoke_batch["cluster_id"].tolist()):
        main_diff = paddle.max(
            paddle.abs(routed_logits[sample_index] - all_head_logits[cluster_id][sample_index])
        )
        route_diffs.append(float(main_diff))
        if routed_aux_logits is not None and all_aux_head_logits is not None:
            aux_diff = paddle.max(
                paddle.abs(routed_aux_logits[sample_index] - all_aux_head_logits[cluster_id][sample_index])
            )
            aux_route_diffs.append(float(aux_diff))

    loss_dict = compute_model_loss(
        outputs=outputs,
        labels=tensors["label"],
        criterion=criterion,
        aux_loss_weight=aux_loss_weight,
    )
    total_loss = loss_dict["total_loss"]
    total_loss.backward()
    gradient_report = collect_gradient_report(model)
    assert_gradient_report(
        gradient_report=gradient_report,
        num_heads=runtime["num_heads"],
        enable_auxiliary_loss=bool(runtime["config"]["model"].get("enable_auxiliary_loss", True)),
    )
    optimizer.step()
    optimizer.clear_grad()
    lr_scheduler.step()

    checkpoint_dir = runtime["save_dir"] / "checkpoints" / "smoke_test_step_0001"
    checkpoint_metadata = {
        "stage": "smoke_test",
        "step": 1,
        "device": device,
        "batch_size": len(smoke_batch["sample_id"]),
        "cluster_ids": smoke_batch["cluster_id"].tolist(),
        "loss": float(total_loss),
        "main_loss": float(loss_dict["main_loss"]),
        "aux_loss": float(loss_dict["aux_loss"]) if loss_dict["aux_loss"] is not None else None,
        "gradient_report": gradient_report,
        "max_route_diff": max(route_diffs) if route_diffs else None,
        "max_aux_route_diff": max(aux_route_diffs) if aux_route_diffs else None,
    }
    save_checkpoint(model, optimizer, checkpoint_dir, checkpoint_metadata, paddle)

    model.eval()
    with paddle.no_grad():
        saved_outputs = model(tensors["image"], tensors["cluster_id"], return_all_heads=True)
    reload_result = verify_checkpoint_reload(runtime, stack, checkpoint_dir, tensors, planned_total_steps)
    reloaded_outputs = reload_result["outputs"]
    reload_main_diff = float(
        paddle.max(paddle.abs(saved_outputs["logits"] - reloaded_outputs["logits"]))
    )
    reload_aux_diff = None
    if saved_outputs.get("aux_logits") is not None and reloaded_outputs.get("aux_logits") is not None:
        reload_aux_diff = float(
            paddle.max(paddle.abs(saved_outputs["aux_logits"] - reloaded_outputs["aux_logits"]))
        )

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
        batch_size=int(runtime["config"]["training"]["eval_batch_size"]),
        max_batches=runtime["args"].smoke_val_batches,
        class_names=runtime["class_names"],
        paddle=paddle,
        criterion=criterion,
        aux_loss_weight=aux_loss_weight,
    )
    metrics["routing_probe_cluster_ids"] = smoke_batch["cluster_id"].tolist()
    metrics["routing_probe_sample_ids"] = smoke_batch["sample_id"]
    metrics["routing_probe_scene_ids"] = smoke_batch["scene_id"]
    metrics["routing_probe_image_paths"] = smoke_batch["image_path"]
    metrics["routing_probe_mask_paths"] = smoke_batch["mask_path"]
    metrics["routing_probe_max_abs_diff"] = max(route_diffs) if route_diffs else None
    metrics["routing_probe_aux_max_abs_diff"] = max(aux_route_diffs) if aux_route_diffs else None
    metrics["reload_main_max_abs_diff"] = reload_main_diff
    metrics["reload_aux_max_abs_diff"] = reload_aux_diff
    metrics["checkpoint_dir"] = str(checkpoint_dir)
    metrics["device"] = device
    metrics["gradient_report"] = gradient_report
    metrics["confusion_matrix_shape"] = list(confusion.shape)

    metrics_dir = runtime["save_dir"] / "metrics"
    write_json(metrics_dir / "smoke_test_metrics.json", metrics)
    np.save(metrics_dir / "smoke_test_confusion.npy", confusion)
    write_per_class_csv(metrics_dir / "smoke_test_per_class.csv", metrics)

    logger.info(
        "[smoke] "
        f"status=passed "
        f"cluster_ids={smoke_batch['cluster_id'].tolist()} "
        f"loss={float(total_loss):.6f} "
        f"main_loss={float(loss_dict['main_loss']):.6f} "
        f"aux_loss={(float(loss_dict['aux_loss']) if loss_dict['aux_loss'] is not None else 0.0):.6f} "
        f"max_route_diff={(max(route_diffs) if route_diffs else 0.0):.10f} "
        f"max_aux_route_diff={(max(aux_route_diffs) if aux_route_diffs else 0.0):.10f} "
        f"reload_main_diff={reload_main_diff:.10f} "
        f"reload_aux_diff={(reload_aux_diff if reload_aux_diff is not None else 0.0):.10f} "
        f"checkpoint={checkpoint_dir}"
    )
    logger.info(
        "[smoke] "
        f"val_batches={metrics['num_batches']} "
        f"OA={metrics['overall_accuracy']:.6f} "
        f"FG_OA={metrics['foreground_overall_accuracy']:.6f} "
        f"mIoU={metrics['mean_iou']:.6f}"
    )

    return {
        "status": "passed",
        "checkpoint_dir": str(checkpoint_dir),
        "routing_probe_cluster_ids": smoke_batch["cluster_id"].tolist(),
        "routing_probe_max_abs_diff": max(route_diffs) if route_diffs else None,
        "routing_probe_aux_max_abs_diff": max(aux_route_diffs) if aux_route_diffs else None,
        "reload_main_max_abs_diff": reload_main_diff,
        "reload_aux_max_abs_diff": reload_aux_diff,
        "validation_metrics_path": str(metrics_dir / "smoke_test_metrics.json"),
    }


def run_short_training(runtime: dict, stack: dict, logger: logging.Logger, device: str, datasets: dict) -> dict:
    paddle = stack["paddle"]
    model = build_model(runtime, stack)

    training_cfg = runtime["config"]["training"]
    train_batch_size = int(training_cfg["train_batch_size"])
    eval_batch_size = int(training_cfg["eval_batch_size"])
    max_epochs = int(training_cfg["epochs"])
    max_train_steps = runtime["args"].max_train_steps
    max_val_batches = runtime["args"].max_val_batches
    log_interval = max(1, int(training_cfg.get("log_interval_steps", 20)))
    steps_per_epoch = max(1, math.ceil(len(datasets["train"]) / train_batch_size))
    planned_total_steps = compute_planned_training_steps(max_epochs, steps_per_epoch, max_train_steps)

    optimizer, lr_scheduler = build_optimizer(runtime, paddle, model, planned_total_steps)
    criterion = build_branch_criterion(stack, runtime)
    aux_loss_weight = float(runtime["config"]["loss"].get("aux_loss_weight", 0.4))

    best_score = float("-inf")
    global_step = 0
    latest_checkpoint_dir = None
    best_metrics_path = None
    best_epoch = None
    stop_requested = False
    resumed_from = None
    last_epoch = 0
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

    for epoch in range(1, max_epochs + 1):
        last_epoch = epoch
        epoch_start = time.perf_counter()
        model.train()
        epoch_loss_values = []
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
            loss_dict = compute_model_loss(
                outputs=outputs,
                labels=tensors["label"],
                criterion=criterion,
                aux_loss_weight=aux_loss_weight,
            )
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
                step_time_avg = float(np.mean(recent_step_times)) if recent_step_times else None
                eta_seconds = estimate_eta_seconds(recent_step_times, remaining_steps)
                end_time = format_end_time_from_eta(eta_seconds)
                logger.info(
                    format_train_log(
                        epoch=epoch,
                        total_epochs=max_epochs,
                        step=step_in_epoch,
                        steps_per_epoch=steps_per_epoch,
                        global_step=global_step,
                        batch_size=len(batch["sample_id"]),
                        loss=loss_value,
                        lr=float(optimizer.get_lr()),
                        step_time_avg_seconds=step_time_avg,
                        eta_seconds=eta_seconds,
                        end_time=end_time,
                    )
                )

            if will_stop:
                stop_requested = True
                break

        epoch_time = time.perf_counter() - epoch_start
        val_metrics, val_confusion = evaluate_model(
            model=model,
            dataset=datasets["val"],
            batch_size=eval_batch_size,
            max_batches=max_val_batches,
            class_names=runtime["class_names"],
            paddle=paddle,
            criterion=criterion,
            aux_loss_weight=aux_loss_weight,
        )

        remaining_steps = max(0, planned_total_steps - global_step)
        eta_seconds = estimate_eta_seconds(recent_step_times, remaining_steps)
        end_time = format_end_time_from_eta(eta_seconds)
        logger.info(
            format_val_log(
                epoch=epoch,
                total_epochs=max_epochs,
                metrics=val_metrics,
                epoch_time_seconds=epoch_time,
                eta_seconds=eta_seconds,
                end_time=end_time,
            )
        )

        metrics_dir = runtime["save_dir"] / "metrics"
        epoch_prefix = metrics_dir / f"short_train_epoch_{epoch:02d}"
        epoch_metrics_payload = {
            **val_metrics,
            "epoch": epoch,
            "global_step": global_step,
            "train_loss_mean": float(np.mean(epoch_loss_values)) if epoch_loss_values else None,
            "device": device,
            "resumed_from": resumed_from,
            "epoch_time_seconds": float(epoch_time),
        }
        write_json(epoch_prefix.with_suffix(".json"), epoch_metrics_payload)
        np.save(epoch_prefix.with_name(epoch_prefix.name + "_confusion.npy"), val_confusion)
        write_per_class_csv(epoch_prefix.with_name(epoch_prefix.name + "_per_class.csv"), val_metrics)

        current_score = float(val_metrics["mean_iou"])
        checkpoint_dir = runtime["save_dir"] / "checkpoints" / f"short_train_epoch_{epoch:02d}_step_{global_step:04d}"
        checkpoint_metadata = {
            "stage": "short_train",
            "epoch": epoch,
            "global_step": global_step,
            "device": device,
            "train_loss_mean": epoch_metrics_payload["train_loss_mean"],
            "val_mean_iou": current_score,
            "val_overall_accuracy": val_metrics["overall_accuracy"],
            "best_score": max(best_score, current_score),
            "resumed_from": resumed_from,
        }
        save_checkpoint(model, optimizer, checkpoint_dir, checkpoint_metadata, paddle)
        copy_checkpoint(checkpoint_dir, runtime["save_dir"] / "latest_model")

        latest_checkpoint_dir = checkpoint_dir
        if current_score > best_score:
            best_score = current_score
            best_metrics_path = str(epoch_prefix.with_suffix(".json"))
            best_epoch = epoch
            copy_checkpoint(checkpoint_dir, runtime["save_dir"] / "best_model")

        logger.info(
            format_epoch_summary_log(
                epoch=epoch,
                total_epochs=max_epochs,
                epoch_time_seconds=epoch_time,
                best_miou=best_score if best_score != float("-inf") else current_score,
                eta_seconds=eta_seconds,
                end_time=end_time,
            )
        )

        if stop_requested:
            break

    if latest_checkpoint_dir is None:
        raise RuntimeError("Short training did not complete any optimization step.")

    return {
        "status": "passed",
        "epochs_completed": last_epoch,
        "global_steps": global_step,
        "latest_checkpoint_dir": str(latest_checkpoint_dir),
        "latest_model_dir": str(runtime["save_dir"] / "latest_model"),
        "best_model_dir": str(runtime["save_dir"] / "best_model"),
        "best_epoch": best_epoch,
        "best_metrics_path": best_metrics_path,
        "planned_total_steps": planned_total_steps,
        "steps_per_epoch": steps_per_epoch,
    }


def collect_env_report(stack: dict) -> dict[str, object]:
    paddle = stack["paddle"]
    gpu_count = int(paddle.device.cuda.device_count()) if paddle.device.is_compiled_with_cuda() else 0
    gpu_name = None
    if gpu_count > 0:
        try:
            gpu_name = paddle.device.cuda.get_device_name(0)
        except Exception:
            gpu_name = None
    return {
        "paddle_import": True,
        "paddlers_import": True,
        "fastscnn_import": True,
        "paddle_version": getattr(paddle, "__version__", "unknown"),
        "paddlers_version": getattr(stack["paddlers"], "__version__", "unknown"),
        "cuda_compiled": bool(paddle.device.is_compiled_with_cuda()),
        "gpu_count": gpu_count,
        "gpu_name": gpu_name,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "not_set"),
    }


def run_acceptance(runtime: dict) -> dict:
    stack = maybe_import_training_stack()
    paddle = stack["paddle"]

    requested_device = runtime["args"].device
    if requested_device == "auto":
        if paddle.device.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0:
            requested_device = "gpu"
        else:
            requested_device = "cpu"
    device = paddle.set_device(requested_device)

    seed = int(runtime["config"]["training"]["seed"])
    set_seed(seed, paddle=paddle)

    datasets = build_datasets(runtime, stack)
    steps_per_epoch = max(1, math.ceil(len(datasets["train"]) / int(runtime["config"]["training"]["train_batch_size"])))
    planned_total_steps = compute_planned_training_steps(
        max_epochs=int(runtime["config"]["training"]["epochs"]),
        steps_per_epoch=steps_per_epoch,
        max_train_steps=runtime["args"].max_train_steps,
    )

    logger = setup_logger(runtime["log_path"])
    start_time = format_timestamp()
    env_report = collect_env_report(stack)
    log_startup_summary(
        runtime,
        logger,
        start_time,
        len(datasets["train"]),
        len(datasets["val"]),
        steps_per_epoch,
        env_report,
    )
    logger.info(
        "[env] "
        f"paddle_import={str(env_report['paddle_import']).lower()} "
        f"paddlers_import={str(env_report['paddlers_import']).lower()} "
        f"fastscnn_import={str(env_report['fastscnn_import']).lower()} "
        f"paddle_version={env_report['paddle_version']} "
        f"paddlers_version={env_report['paddlers_version']} "
        f"cuda_compiled={str(env_report['cuda_compiled']).lower()} "
        f"gpu_count={env_report['gpu_count']} "
        f"gpu_name={env_report['gpu_name']} "
        f"cuda_visible_devices={env_report['cuda_visible_devices']} "
        f"device={device}"
    )
    logger.info(
        "[routing] "
        f"observed_cluster_ids={runtime['routing_validation']['observed_cluster_ids']} "
        f"missing_cluster_ids={runtime['routing_validation']['missing_cluster_ids']} "
        f"all_clusters_have_train_val_test={str(runtime['routing_validation']['all_clusters_have_train_val_test']).lower()}"
    )

    results = {
        "status": "running",
        "device": str(device),
        "log_path": str(runtime["log_path"]),
        "env_report": env_report,
        "routing_validation": runtime["routing_validation"],
        "smoke_test": None,
        "short_training": None,
    }

    if runtime["args"].run_smoke_test:
        results["smoke_test"] = run_smoke_test(runtime, stack, logger, str(device), planned_total_steps)

    if runtime["args"].run_train:
        results["short_training"] = run_short_training(runtime, stack, logger, str(device), datasets)

    results["status"] = "passed"
    status_path = runtime["save_dir"] / "metrics" / f"{runtime['run_name']}_status.json"
    write_json(status_path, results)
    logger.info(f"[status] acceptance_status={results['status']}")
    logger.info(format_path_log("log_path", runtime["log_path"]))
    logger.info(format_path_log("status_path", status_path))
    return results


def parse_args():
    parser = argparse.ArgumentParser(
        description="Smoke test and short-training acceptance for cluster-routed 4-head FastSCNN on GID15."
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--save-dir", default=None)
    parser.add_argument("--resume-checkpoint", default=None)
    parser.add_argument("--run-smoke-test", action="store_true")
    parser.add_argument("--run-train", action="store_true")
    parser.add_argument("--device", choices=["auto", "gpu", "cpu"], default="auto")
    parser.add_argument("--train-batch-size", type=int, default=None)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--max-train-steps", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    parser.add_argument("--smoke-val-batches", type=int, default=1)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)
    parser.add_argument("--run-name", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    runtime = build_runtime(args)
    create_output_structure(runtime["save_dir"])
    write_config_snapshot(runtime)

    if args.run_smoke_test or args.run_train:
        results = run_acceptance(runtime)
        print(json.dumps(results, ensure_ascii=False, indent=2))
    else:
        logger = setup_logger(runtime["save_dir"] / "logs" / "prepare.log")
        logger.info("[status] prepare_only=true training_started=false smoke_test_started=false")
        logger.info(format_path_log("config_snapshot_dir", runtime["save_dir"] / "config_snapshot"))


if __name__ == "__main__":
    main()
