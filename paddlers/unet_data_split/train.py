#!/usr/bin/env python3

from __future__ import annotations

import argparse
import math
import os
import random
import shutil
import sys
import time
import traceback
from collections import deque
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
WORK_DIR = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(WORK_DIR) not in sys.path:
    sys.path.insert(0, str(WORK_DIR))

from cluster_routed_dataset import load_cluster_mapping, read_label_names, split_samples, summarize_routed_samples, validate_routed_samples
from metrics_utils import (
    build_logging_schema,
    compute_confusion_metrics,
    ensure_dir,
    format_end_time_from_eta,
    format_eta,
    format_time_each_step,
    format_timestamp,
    read_json,
    setup_logger,
    update_confusion_matrix,
    write_confusion_csv,
    write_json,
    write_metrics_text,
    write_per_class_csv,
)
from unet_routed_model import ClusterRoutedUNet, build_model_structure, load_pretrained_weights


DEFAULT_CONFIG_PATH = WORK_DIR / "train_config_singlehead_baseline.json"
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


def log_message(logger, stage: str, message: str, level: int = 20) -> None:
    logger.log(level, message, extra={"stage": stage})


def expand_path_value(path_value: str | None) -> str | None:
    if path_value is None:
        return None
    return os.path.expanduser(os.path.expandvars(str(path_value)))


def resolve_existing_path(
    base_dir: Path,
    path_value: str | None,
    allow_missing: bool = False,
    extra_base_dirs: list[Path] | None = None,
) -> Path | None:
    if path_value is None:
        return None
    expanded_value = expand_path_value(path_value)
    assert expanded_value is not None
    path = Path(expanded_value)
    candidates: list[Path] = []
    if path.is_absolute():
        candidates.append(path.resolve())
    else:
        search_bases = [Path.cwd(), base_dir]
        if extra_base_dirs:
            search_bases.extend(extra_base_dirs)
        seen_bases: set[Path] = set()
        for item in search_bases:
            resolved_base = item.resolve()
            if resolved_base in seen_bases:
                continue
            seen_bases.add(resolved_base)
            candidates.append((resolved_base / path).resolve())
    for candidate in candidates:
        if candidate.exists():
            return candidate
    if allow_missing:
        return candidates[0]
    raise FileNotFoundError(f"Path does not exist: {candidates[0]}")


def create_output_structure(save_dir: Path) -> None:
    ensure_dir(save_dir)
    for name in OUTPUT_SUBDIRS:
        ensure_dir(save_dir / name)


def create_phase_logger(save_dir: Path, phase: str):
    log_path = save_dir / "logs" / f"{phase}.log"
    logger_name = f"unet_data_split.{save_dir.name}.{phase}"
    return setup_logger(logger_name, log_path), log_path


def maybe_import_training_stack():
    try:
        import paddle
        import paddlers
        from PIL import Image, ImageEnhance
        from paddlers.models import seg_losses
        from paddlers.utils.checkpoint import get_pretrain_weights
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
        "get_pretrain_weights": get_pretrain_weights,
    }


def set_seed(seed: int, paddle=None) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if paddle is not None:
        paddle.seed(seed)


def resolve_save_dir(args: argparse.Namespace, config: dict, config_base_dir: Path) -> Path:
    save_dir_value = args.save_dir or config.get("save_dir", "output")
    save_dir_path = Path(save_dir_value)
    if save_dir_path.is_absolute():
        return save_dir_path.resolve()
    if args.save_dir:
        resolved = resolve_existing_path(config_base_dir, save_dir_value, allow_missing=True)
        assert resolved is not None
        return resolved
    return (config_base_dir / save_dir_path).resolve()


def load_cached_routing_validation(cache_path: Path, routing_summary: dict, expected_num_heads: int) -> dict | None:
    if not cache_path.exists():
        return None
    payload = read_json(cache_path)
    expected_cluster_ids = list(range(expected_num_heads))
    if payload.get("num_samples") != routing_summary.get("num_samples"):
        return None
    if payload.get("split_counts") != routing_summary.get("split_counts"):
        return None
    if payload.get("raw_cluster_total_counts") != routing_summary.get("raw_cluster_total_counts"):
        return None
    if payload.get("expected_cluster_ids") != expected_cluster_ids:
        return None
    return payload


def build_runtime(args: argparse.Namespace) -> dict[str, object]:
    config_path = resolve_existing_path(WORK_DIR, args.config)
    assert config_path is not None
    config_base_dir = config_path.parent
    config = read_json(config_path)
    save_dir = resolve_save_dir(args, config, config_base_dir)

    if getattr(args, "train_batch_size", None) is not None:
        config["training"]["train_batch_size"] = int(args.train_batch_size)
    if getattr(args, "eval_batch_size", None) is not None:
        config["training"]["eval_batch_size"] = int(args.eval_batch_size)
    if getattr(args, "max_epochs", None) is not None:
        config["training"]["epochs"] = int(args.max_epochs)
    if getattr(args, "resume_checkpoint", None) is not None:
        config["training"]["resume_checkpoint"] = args.resume_checkpoint

    dataset_root = resolve_existing_path(config_base_dir, config["dataset_root"])
    clustered_root = resolve_existing_path(config_base_dir, config["clustered_dataset_root"])
    assert dataset_root is not None and clustered_root is not None

    mapping_path = resolve_existing_path(
        clustered_root,
        config["cluster_mapping_file"],
        extra_base_dirs=[config_base_dir],
    )
    label_list = resolve_existing_path(
        dataset_root,
        config["label_list"],
        extra_base_dirs=[config_base_dir],
    )
    cluster_summary_path = resolve_existing_path(
        clustered_root,
        "stats/cluster_summary.json",
        extra_base_dirs=[config_base_dir],
    )
    assert mapping_path is not None and label_list is not None and cluster_summary_path is not None

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

    cluster_summary = read_json(cluster_summary_path)
    routing_cfg = config["routing"]
    selected_k = int(routing_cfg["selected_k"])
    actual_selected_k = int(cluster_summary["selected_k"])
    if actual_selected_k != selected_k:
        raise ValueError(f"Config selected_k={selected_k} does not match cluster_summary selected_k={actual_selected_k}.")
    if routing_cfg.get("sample_path_source", "input") != "input":
        raise ValueError("UNet data split experiments require routing.sample_path_source='input'.")

    num_heads = int(routing_cfg["num_heads"])
    cluster_names = list(routing_cfg["cluster_names"])
    cluster_id_handling = str(routing_cfg.get("cluster_id_handling", "route"))
    if num_heads not in {1, actual_selected_k}:
        raise ValueError(f"Only num_heads=1 or num_heads={actual_selected_k} is supported, got {num_heads}.")
    if num_heads == 1 and cluster_id_handling != "ignore":
        raise ValueError("single-head baseline requires routing.cluster_id_handling='ignore'.")
    if num_heads > 1 and cluster_id_handling != "route":
        raise ValueError("multi-head routed UNet requires routing.cluster_id_handling='route'.")
    if len(cluster_names) != num_heads:
        raise ValueError(f"cluster_names length must equal num_heads, got {len(cluster_names)} vs {num_heads}")

    samples = load_cluster_mapping(
        mapping_path,
        sample_path_source="input",
        cluster_id_handling=cluster_id_handling,
        dataset_root=dataset_root,
        clustered_root=clustered_root,
    )
    routing_summary = summarize_routed_samples(samples)
    if routing_summary["duplicate_sample_ids_across_splits"]:
        raise ValueError("Duplicate sample ids were found across splits in cluster routing manifest.")

    routing_cache_path = save_dir / "config_snapshot" / "routing_summary.json"
    routing_validation = load_cached_routing_validation(
        routing_cache_path,
        routing_summary=routing_summary,
        expected_num_heads=max(num_heads, 1),
    )
    routing_validation_source = "cache" if routing_validation is not None else "fresh"
    if routing_validation is None:
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

    return {
        "args": args,
        "config": config,
        "config_path": config_path,
        "config_base_dir": config_base_dir,
        "dataset_root": dataset_root,
        "clustered_root": clustered_root,
        "mapping_path": mapping_path,
        "label_list": label_list,
        "cluster_summary_path": cluster_summary_path,
        "cluster_summary": cluster_summary,
        "class_names": class_names,
        "num_classes": num_classes,
        "selected_k": selected_k,
        "num_heads": num_heads,
        "samples": samples,
        "split_map": split_map,
        "routing_summary": routing_summary,
        "routing_validation": routing_validation,
        "routing_validation_source": routing_validation_source,
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
    config["resolved_cluster_summary_path"] = str(runtime["cluster_summary_path"])
    config["resolved_save_dir"] = str(runtime["save_dir"])
    config["full_steps_per_epoch"] = int(runtime["full_steps_per_epoch"])
    config["planned_total_steps"] = int(runtime["planned_total_steps"])

    write_json(snapshot_dir / "resolved_train_config.json", config)
    write_json(snapshot_dir / "model_structure.json", build_model_structure(runtime["config"], runtime["num_classes"]))
    write_json(snapshot_dir / "routing_summary.json", runtime["routing_validation"])
    write_json(
        snapshot_dir / "dataset_overview.json",
        {
            "dataset_root": str(runtime["dataset_root"]),
            "clustered_dataset_root": str(runtime["clustered_root"]),
            "cluster_mapping_file": str(runtime["mapping_path"]),
            "label_list": str(runtime["label_list"]),
            "cluster_summary_path": str(runtime["cluster_summary_path"]),
            "selected_k": runtime["selected_k"],
            "num_heads": runtime["num_heads"],
            "num_classes": runtime["num_classes"],
            "class_names": runtime["class_names"],
            "split_counts": {split: len(rows) for split, rows in runtime["split_map"].items()},
            "routing_summary": runtime["routing_summary"],
            "routing_validation": runtime["routing_validation"],
        },
    )
    write_json(
        snapshot_dir / "cluster_summary_snapshot.json",
        runtime["cluster_summary"],
    )
    write_json(
        snapshot_dir / "logging_schema.json",
        build_logging_schema("ClusterRoutedUNet", runtime["num_heads"]),
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
            "cluster_name": sample.cluster_name,
            "raw_cluster_name": sample.raw_cluster_name,
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
        "cluster_name": [item["cluster_name"] for item in batch_items],
        "raw_cluster_name": [item["raw_cluster_name"] for item in batch_items],
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
        "cluster_name": batch["cluster_name"],
        "raw_cluster_name": batch["raw_cluster_name"],
        "sample_id": batch["sample_id"],
        "scene_id": batch["scene_id"],
        "image_path": batch["image_path"],
        "mask_path": batch["mask_path"],
    }


def build_model(runtime: dict):
    model_cfg = runtime["config"]["model"]
    return ClusterRoutedUNet(
        num_classes=runtime["num_classes"],
        num_heads=runtime["num_heads"],
        in_channels=int(model_cfg.get("in_channels", 3)),
        use_deconv=bool(model_cfg.get("use_deconv", False)),
        align_corners=bool(model_cfg.get("align_corners", False)),
    )


def build_optimizer(runtime: dict, paddle, model, planned_total_steps: int):
    training_cfg = runtime["config"]["training"]
    lr_scheduler = paddle.optimizer.lr.PolynomialDecay(
        learning_rate=float(training_cfg["learning_rate"]),
        decay_steps=max(1, int(planned_total_steps)),
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


def build_criterion(stack: dict, runtime: dict):
    loss_cfg = runtime["config"]["loss"]
    seg_losses = stack["seg_losses"]
    ignore_index = int(loss_cfg.get("ignore_index", IGNORE_INDEX))
    use_mixed_loss = bool(loss_cfg.get("use_mixed_loss", True))
    if use_mixed_loss:
        losses = [
            seg_losses.CrossEntropyLoss(ignore_index=ignore_index),
            seg_losses.LovaszSoftmaxLoss(ignore_index=ignore_index),
        ]
        coef = list(loss_cfg.get("ce_lovasz_coef", [0.8, 0.2]))
        return seg_losses.MixedLoss(losses=losses, coef=coef)
    return seg_losses.CrossEntropyLoss(ignore_index=ignore_index)


def compute_total_loss(loss_output):
    if isinstance(loss_output, (list, tuple)):
        total = None
        for item in loss_output:
            total = item if total is None else total + item
        return total
    return loss_output


def compute_model_loss(outputs, labels, criterion):
    total_loss = compute_total_loss(criterion(outputs["logits"], labels))
    return {
        "total_loss": total_loss,
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


def collect_gradient_report(model: ClusterRoutedUNet) -> dict[str, float]:
    report = {
        "encoder": compute_module_grad_norm(model.encode),
        "decoder": compute_module_grad_norm(model.decode),
    }
    for idx, head in enumerate(model.heads):
        report[f"head_{idx}"] = compute_module_grad_norm(head)
    report["model_total"] = compute_module_grad_norm(model)
    return report


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


def resolve_pretrained_weights(runtime: dict, stack: dict) -> str | None:
    pre_cfg = runtime["config"].get("pretrained", {})
    explicit_path = pre_cfg.get("weights_path")
    if explicit_path:
        resolved = resolve_existing_path(WORK_DIR, explicit_path)
        return str(resolved) if resolved is not None else None

    weights_flag = pre_cfg.get("weights")
    if weights_flag is None:
        return None
    save_dir = runtime["save_dir"] / "pretrained"
    ensure_dir(save_dir)
    return str(stack["get_pretrain_weights"](weights_flag, "UNet", str(save_dir)))


def maybe_load_pretrained(runtime: dict, stack: dict, model: ClusterRoutedUNet, logger) -> dict[str, object]:
    pretrained_path = resolve_pretrained_weights(runtime, stack)
    if pretrained_path is None:
        report = {
            "pretrained_enabled": False,
            "pretrained_path": None,
            "note": "No pretrained weights requested.",
        }
    else:
        report = {
            "pretrained_enabled": True,
            **load_pretrained_weights(model, pretrained_path),
        }
        log_message(
            logger,
            "INIT",
            (
                f"Loaded pretrained trunk from {report['pretrained_path']}, "
                f"num_loaded_tensors={report['num_loaded_tensors']}/{report['num_model_tensors']}, "
                f"broadcast_head_tensors={len(report['head_broadcast_keys_preview'])}"
            ),
        )

    write_json(runtime["save_dir"] / "config_snapshot" / "pretrained_load_report.json", report)
    return report


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


def evaluate_model(
    model,
    dataset,
    batch_size: int,
    max_batches: int | None,
    class_names,
    paddle,
    criterion,
):
    confusion = np.zeros((len(class_names), len(class_names)), dtype=np.int64)
    losses = []
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
            )
            losses.append(float(loss_dict["total_loss"]))

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
            "num_batches": batch_count,
            "num_samples": sample_count,
            "ignore_index": IGNORE_INDEX,
        }
    )
    return metrics, confusion


def write_epoch_metrics(save_dir: Path, split_name: str, epoch: int, metrics: dict, confusion: np.ndarray) -> Path:
    metrics_dir = save_dir / "metrics"
    ensure_dir(metrics_dir)
    prefix = metrics_dir / f"{split_name}_epoch_{epoch:03d}"
    write_json(prefix.with_suffix(".json"), metrics)
    write_per_class_csv(prefix.with_name(prefix.name + "_per_class.csv"), metrics)
    write_confusion_csv(prefix.with_name(prefix.name + "_confusion_matrix.csv"), confusion, [item["class_name"] for item in metrics["per_class"]])
    return prefix.with_suffix(".json")


def write_best_artifacts(save_dir: Path, split_name: str, metrics: dict, confusion: np.ndarray, selection_metric: str) -> None:
    write_json(save_dir / "best_model_metrics.json", metrics)
    write_metrics_text(
        save_dir / "best_model_metrics.txt",
        metrics,
        split_name=split_name,
        selection_metric=selection_metric,
    )
    write_per_class_csv(save_dir / "per_class_metrics.csv", metrics)
    write_confusion_csv(save_dir / "confusion_matrix.csv", confusion, [item["class_name"] for item in metrics["per_class"]])


def export_prediction_samples(model, dataset, save_dir: Path, split_name: str, count: int, stack: dict, paddle) -> list[str]:
    prediction_dir = save_dir / "prediction_samples" / split_name
    ensure_dir(prediction_dir)
    saved_paths: list[str] = []

    model.eval()
    with paddle.no_grad():
        for index in range(min(int(count), len(dataset))):
            sample = dataset[index]
            batch = collate_samples([sample])
            tensors = batch_to_tensors(batch, paddle)
            outputs = model(tensors["image"], tensors["cluster_id"])
            prediction = paddle.argmax(outputs["logits"], axis=1).numpy()[0].astype(np.uint8)
            label = batch["label"][0].astype(np.uint8)
            sample_id = batch["sample_id"][0]

            pred_path = prediction_dir / f"{sample_id}_pred.png"
            label_path = prediction_dir / f"{sample_id}_label.png"
            stack["Image"].fromarray(prediction).save(pred_path)
            stack["Image"].fromarray(label).save(label_path)
            saved_paths.extend([str(pred_path), str(label_path)])
    return saved_paths


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
        "paddle_version": getattr(paddle, "__version__", "unknown"),
        "paddlers_version": getattr(stack["paddlers"], "__version__", "unknown"),
        "cuda_compiled": bool(paddle.device.is_compiled_with_cuda()),
        "gpu_count": gpu_count,
        "gpu_name": gpu_name,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "not_set"),
    }


def estimate_eta_seconds(step_time_history: deque[float], remaining_steps: int) -> float | None:
    if remaining_steps <= 0:
        return 0.0
    if len(step_time_history) < 5:
        return None
    return float(np.mean(step_time_history) * remaining_steps)


def selection_metric_value(runtime: dict, metrics: dict) -> float:
    selection_metric = str(runtime["config"]["evaluation"].get("selection_metric", "miou")).lower()
    if selection_metric == "miou":
        return float(metrics["mean_iou"])
    if selection_metric == "kappa":
        return float(metrics["kappa"])
    if selection_metric == "oa":
        return float(metrics["overall_accuracy"])
    if selection_metric == "foreground_miou":
        return float(metrics["foreground_mean_iou"])
    raise ValueError(f"Unsupported selection metric: {selection_metric}")


def update_training_status(path: Path, payload: dict) -> None:
    write_json(path, payload)


def select_smoke_probe_samples(runtime: dict):
    train_samples = runtime["split_map"]["train"]
    if runtime["num_heads"] == 1:
        return train_samples[: min(4, len(train_samples))]

    probe_samples = []
    for cluster_id in range(runtime["num_heads"]):
        sample = next((item for item in train_samples if item.cluster_id == cluster_id), None)
        if sample is None:
            raise RuntimeError(f"Unable to find a train sample for cluster_id={cluster_id}")
        probe_samples.append(sample)
    return probe_samples


def compute_logit_gradient_report(all_head_logits_tensor_grad: np.ndarray, active_cluster_ids: list[int]) -> dict[str, object]:
    grad_payload = {
        "per_sample_head_grad_norms": [],
        "max_inactive_head_grad_norm": 0.0,
        "all_inactive_head_grads_zero": True,
    }
    for sample_index, active_head in enumerate(active_cluster_ids):
        sample_grad = all_head_logits_tensor_grad[sample_index]
        head_norms = [float(np.sqrt(np.sum(np.square(sample_grad[head_idx])))) for head_idx in range(sample_grad.shape[0])]
        inactive_norms = [value for head_idx, value in enumerate(head_norms) if head_idx != int(active_head)]
        active_norm = head_norms[int(active_head)]
        inactive_max = max(inactive_norms) if inactive_norms else 0.0
        grad_payload["per_sample_head_grad_norms"].append(
            {
                "sample_index": sample_index,
                "active_head": int(active_head),
                "head_grad_norms": head_norms,
                "active_head_grad_norm": active_norm,
                "inactive_head_max_grad_norm": inactive_max,
            }
        )
        grad_payload["max_inactive_head_grad_norm"] = max(grad_payload["max_inactive_head_grad_norm"], inactive_max)
        if active_norm <= 0.0 or inactive_max > 1e-12:
            grad_payload["all_inactive_head_grads_zero"] = False
    return grad_payload


def verify_checkpoint_reload(runtime: dict, stack: dict, checkpoint_dir: Path, probe_batch: dict, planned_total_steps: int) -> dict:
    paddle = stack["paddle"]
    model = build_model(runtime)
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


def run_smoke_test(runtime: dict) -> dict:
    stack = maybe_import_training_stack()
    paddle = stack["paddle"]

    requested_device = runtime["args"].device
    if requested_device == "auto":
        if paddle.device.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0:
            requested_device = "gpu"
        else:
            requested_device = "cpu"
    device = paddle.set_device(requested_device)
    set_seed(int(runtime["config"]["training"]["seed"]), paddle=paddle)

    logger, log_path = create_phase_logger(runtime["save_dir"], "smoke_test")
    log_message(logger, "INIT", f"Starting smoke test for ClusterRoutedUNet with num_heads={runtime['num_heads']}.")
    log_message(
        logger,
        "DATA",
        f"dataset_root={runtime['dataset_root']} clustered_root={runtime['clustered_root']} sample_path_source=input",
    )
    log_message(
        logger,
        "SYSTEM",
        (
            f"device={device} cuda_visible_devices={os.environ.get('CUDA_VISIBLE_DEVICES', 'not_set')} "
            f"pid={os.getpid()}"
        ),
    )

    datasets = build_datasets(runtime, stack)
    model = build_model(runtime)
    pretrained_report = maybe_load_pretrained(runtime, stack, model, logger)
    optimizer, lr_scheduler = build_optimizer(runtime, paddle, model, runtime["planned_total_steps"])
    criterion = build_criterion(stack, runtime)

    probe_samples = select_smoke_probe_samples(runtime)
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
    smoke_batch = collate_samples([smoke_dataset[idx] for idx in range(len(smoke_dataset))])
    tensors = batch_to_tensors(smoke_batch, paddle)

    model.train()
    outputs = model(
        tensors["image"],
        tensors["cluster_id"],
        return_all_heads=True,
        retain_all_head_grads=True,
    )
    routed_logits = outputs["logits"]
    all_head_logits = outputs["all_head_logits"]
    all_head_logits_tensor = outputs["all_head_logits_tensor"]

    route_diffs = []
    active_heads = sorted(set(int(item) for item in smoke_batch["cluster_id"].tolist()))
    for sample_index, cluster_id in enumerate(smoke_batch["cluster_id"].tolist()):
        main_diff = paddle.max(
            paddle.abs(routed_logits[sample_index] - all_head_logits[cluster_id][sample_index])
        )
        route_diffs.append(float(main_diff))

    loss_dict = compute_model_loss(
        outputs=outputs,
        labels=tensors["label"],
        criterion=criterion,
    )
    total_loss = loss_dict["total_loss"]
    total_loss.backward()
    gradient_report = collect_gradient_report(model)

    if all_head_logits_tensor.grad is None:
        raise RuntimeError("Expected retained gradients for all_head_logits_tensor during smoke test.")
    per_sample_logit_grad_report = compute_logit_gradient_report(
        all_head_logits_tensor.grad.numpy(),
        smoke_batch["cluster_id"].tolist(),
    )
    if runtime["num_heads"] > 1 and not per_sample_logit_grad_report["all_inactive_head_grads_zero"]:
        raise RuntimeError("Inactive routed heads received non-zero sample gradients during smoke test.")

    optimizer.step()
    optimizer.clear_grad()
    lr_scheduler.step()

    checkpoint_dir = runtime["save_dir"] / "checkpoints" / "smoke_test_step_0001"
    checkpoint_metadata = {
        "stage": "smoke_test",
        "step": 1,
        "device": str(device),
        "batch_size": len(smoke_batch["sample_id"]),
        "cluster_ids": smoke_batch["cluster_id"].tolist(),
        "loss": float(total_loss),
        "gradient_report": gradient_report,
        "max_route_diff": max(route_diffs) if route_diffs else None,
        "pretrained_report": pretrained_report,
        "per_sample_logit_grad_report": per_sample_logit_grad_report,
    }
    save_checkpoint(model, optimizer, checkpoint_dir, checkpoint_metadata, paddle)
    log_message(logger, "CKPT", f"Saved smoke checkpoint to {checkpoint_dir}.")

    model.eval()
    with paddle.no_grad():
        saved_outputs = model(tensors["image"], tensors["cluster_id"], return_all_heads=True)
    reload_result = verify_checkpoint_reload(runtime, stack, checkpoint_dir, tensors, runtime["planned_total_steps"])
    reloaded_outputs = reload_result["outputs"]
    reload_main_diff = float(
        paddle.max(paddle.abs(saved_outputs["logits"] - reloaded_outputs["logits"]))
    )

    val_metrics, val_confusion = evaluate_model(
        model=model,
        dataset=datasets["val"],
        batch_size=int(runtime["config"]["training"]["eval_batch_size"]),
        max_batches=runtime["args"].smoke_val_batches,
        class_names=runtime["class_names"],
        paddle=paddle,
        criterion=criterion,
    )
    write_json(runtime["save_dir"] / "metrics" / "smoke_test_metrics.json", val_metrics)
    write_per_class_csv(runtime["save_dir"] / "metrics" / "smoke_test_per_class.csv", val_metrics)
    write_confusion_csv(
        runtime["save_dir"] / "metrics" / "smoke_test_confusion_matrix.csv",
        val_confusion,
        runtime["class_names"],
    )

    smoke_status = {
        "status": "passed",
        "device": str(device),
        "log_path": str(log_path),
        "checkpoint_dir": str(checkpoint_dir),
        "active_heads": active_heads,
        "cluster_ids": smoke_batch["cluster_id"].tolist(),
        "max_route_diff": max(route_diffs) if route_diffs else None,
        "reload_main_diff": reload_main_diff,
        "gradient_report": gradient_report,
        "per_sample_logit_grad_report": per_sample_logit_grad_report,
        "val_metrics": val_metrics,
        "routing_summary_path": str(runtime["save_dir"] / "config_snapshot" / "routing_summary.json"),
    }
    write_json(runtime["save_dir"] / "smoke_test_status.json", smoke_status)

    log_message(
        logger,
        "SMOKE",
        (
            f"status=passed active_heads={active_heads} cluster_ids={smoke_batch['cluster_id'].tolist()} "
            f"loss={float(total_loss):.6f} max_route_diff={(max(route_diffs) if route_diffs else 0.0):.10f} "
            f"max_inactive_head_grad_norm={per_sample_logit_grad_report['max_inactive_head_grad_norm']:.10f}"
        ),
    )
    log_message(
        logger,
        "EVAL",
        (
            f"Split=val, OA={val_metrics['overall_accuracy']:.6f}, "
            f"mAcc={val_metrics['mean_accuracy']:.6f}, mIoU={val_metrics['mean_iou']:.6f}, "
            f"Kappa={val_metrics['kappa']:.6f}"
        ),
    )
    return smoke_status


def run_training(runtime: dict) -> dict:
    stack = maybe_import_training_stack()
    paddle = stack["paddle"]

    requested_device = runtime["args"].device
    if requested_device == "auto":
        if paddle.device.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0:
            requested_device = "gpu"
        else:
            requested_device = "cpu"
    device = paddle.set_device(requested_device)

    logger, log_path = create_phase_logger(runtime["save_dir"], "train")
    env_report = collect_env_report(stack)
    log_message(
        logger,
        "INIT",
        (
            f"model=ClusterRoutedUNet num_heads={runtime['num_heads']} num_classes={runtime['num_classes']} "
            f"epochs={runtime['config']['training']['epochs']} train_batch_size={runtime['config']['training']['train_batch_size']} "
            f"eval_batch_size={runtime['config']['training']['eval_batch_size']}"
        ),
    )
    log_message(
        logger,
        "DATA",
        (
            f"dataset_root={runtime['dataset_root']} clustered_root={runtime['clustered_root']} "
            f"train_samples={len(runtime['split_map']['train'])} val_samples={len(runtime['split_map']['val'])} "
            f"test_samples={len(runtime['split_map']['test'])} selected_k={runtime['selected_k']}"
        ),
    )
    log_message(
        logger,
        "SYSTEM",
        (
            f"device={device} cuda_visible_devices={env_report['cuda_visible_devices']} gpu_name={env_report['gpu_name']} "
            f"paddle_version={env_report['paddle_version']} paddlers_version={env_report['paddlers_version']} pid={os.getpid()}"
        ),
    )

    set_seed(int(runtime["config"]["training"]["seed"]), paddle=paddle)
    datasets = build_datasets(runtime, stack)
    model = build_model(runtime)
    criterion = build_criterion(stack, runtime)
    training_cfg = runtime["config"]["training"]
    train_batch_size = int(training_cfg["train_batch_size"])
    eval_batch_size = int(training_cfg["eval_batch_size"])
    max_epochs = int(training_cfg["epochs"])
    max_train_steps = runtime["args"].max_train_steps
    log_interval = max(1, int(training_cfg.get("log_interval_steps", 20)))
    steps_per_epoch = max(1, math.ceil(len(datasets["train"]) / train_batch_size))
    planned_total_steps = max(1, steps_per_epoch * max_epochs)
    if max_train_steps is not None:
        planned_total_steps = min(planned_total_steps, int(max_train_steps))

    optimizer, lr_scheduler = build_optimizer(runtime, paddle, model, planned_total_steps)

    resume_checkpoint = training_cfg.get("resume_checkpoint")
    resumed_from = None
    pretrained_report = None
    start_epoch = 1
    global_step = 0
    best_score = float("-inf")
    best_epoch = None
    best_metrics = None

    if resume_checkpoint:
        resume_path = resolve_existing_path(runtime["save_dir"], resume_checkpoint)
        if resume_path is None:
            raise FileNotFoundError(f"Unable to resolve resume checkpoint: {resume_checkpoint}")
        resume_meta = load_checkpoint(model, optimizer, resume_path, paddle)
        resumed_from = str(resume_path)
        start_epoch = int(resume_meta.get("epoch", 0)) + 1
        global_step = int(resume_meta.get("global_step", 0))
        best_score = float(resume_meta.get("best_score", best_score))
        best_epoch = resume_meta.get("best_epoch")
        log_message(logger, "INIT", f"Resumed from checkpoint {resumed_from}.")
    else:
        pretrained_report = maybe_load_pretrained(runtime, stack, model, logger)

    status_path = runtime["save_dir"] / "training_status.json"
    status_payload = {
        "status": "running",
        "pid": os.getpid(),
        "device": str(device),
        "log_path": str(log_path),
        "started_at": format_timestamp(),
        "resumed_from": resumed_from,
        "pretrained_report": pretrained_report,
        "best_epoch": best_epoch,
        "best_score": None if best_score == float("-inf") else best_score,
    }
    update_training_status(status_path, status_payload)

    stop_requested = False
    recent_step_times = deque(maxlen=TRAIN_LOG_ETA_WINDOW)

    try:
        for epoch in range(start_epoch, max_epochs + 1):
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

                remaining_steps = max(0, planned_total_steps - global_step)
                eta_seconds = estimate_eta_seconds(recent_step_times, remaining_steps)
                should_log = (
                    global_step == 1
                    or global_step % log_interval == 0
                    or step_in_epoch == steps_per_epoch
                    or (max_train_steps is not None and global_step >= max_train_steps)
                )
                if should_log:
                    log_message(
                        logger,
                        "TRAIN",
                        (
                            f"Epoch={epoch}/{max_epochs}, Step={step_in_epoch}/{steps_per_epoch}, "
                            f"loss={loss_value:.6f}, lr={float(optimizer.get_lr()):.6f}, "
                            f"time_each_step={format_time_each_step(float(np.mean(recent_step_times)) if recent_step_times else None)}, "
                            f"eta={format_eta(eta_seconds)}"
                        ),
                    )

                if max_train_steps is not None and global_step >= max_train_steps:
                    stop_requested = True
                    break

            epoch_time = time.perf_counter() - epoch_start
            val_metrics, val_confusion = evaluate_model(
                model=model,
                dataset=datasets["val"],
                batch_size=eval_batch_size,
                max_batches=runtime["args"].max_val_batches,
                class_names=runtime["class_names"],
                paddle=paddle,
                criterion=criterion,
            )
            val_metrics.update(
                {
                    "epoch": epoch,
                    "global_step": global_step,
                    "train_loss_mean": float(np.mean(epoch_loss_values)) if epoch_loss_values else None,
                    "epoch_time_seconds": float(epoch_time),
                }
            )
            val_metrics_path = write_epoch_metrics(runtime["save_dir"], "val", epoch, val_metrics, val_confusion)

            remaining_steps = max(0, planned_total_steps - global_step)
            eta_seconds = estimate_eta_seconds(recent_step_times, remaining_steps)
            log_message(
                logger,
                "EVAL",
                (
                    f"Split=val, Epoch={epoch}/{max_epochs}, OA={val_metrics['overall_accuracy']:.6f}, "
                    f"mAcc={val_metrics['mean_accuracy']:.6f}, mIoU={val_metrics['mean_iou']:.6f}, "
                    f"Kappa={val_metrics['kappa']:.6f}, epoch_time={format_time_each_step(epoch_time)}, "
                    f"eta={format_eta(eta_seconds)}"
                ),
            )

            checkpoint_dir = runtime["save_dir"] / "checkpoints" / f"epoch_{epoch:03d}_step_{global_step:06d}"
            checkpoint_metadata = {
                "stage": "train",
                "epoch": epoch,
                "global_step": global_step,
                "best_score": best_score,
                "best_epoch": best_epoch,
                "device": str(device),
                "resumed_from": resumed_from,
                "val_metrics_path": str(val_metrics_path),
            }
            save_checkpoint(model, optimizer, checkpoint_dir, checkpoint_metadata, paddle)
            copy_checkpoint(checkpoint_dir, runtime["save_dir"] / "latest_model")
            log_message(logger, "CKPT", f"Saved latest checkpoint to {checkpoint_dir}.")

            current_score = selection_metric_value(runtime, val_metrics)
            if current_score > best_score:
                best_score = current_score
                best_epoch = epoch
                best_metrics = val_metrics
                copy_checkpoint(checkpoint_dir, runtime["save_dir"] / "best_model")
                write_best_artifacts(
                    runtime["save_dir"],
                    "val",
                    val_metrics,
                    val_confusion,
                    str(runtime["config"]["evaluation"].get("selection_metric", "miou")),
                )
                log_message(
                    logger,
                    "CKPT",
                    (
                        f"Updated best_model at epoch={epoch}, "
                        f"selection_metric={runtime['config']['evaluation'].get('selection_metric', 'miou')}, "
                        f"score={best_score:.6f}"
                    ),
                )

            status_payload = {
                "status": "running",
                "pid": os.getpid(),
                "device": str(device),
                "log_path": str(log_path),
                "started_at": status_payload["started_at"],
                "updated_at": format_timestamp(),
                "current_epoch": epoch,
                "global_step": global_step,
                "steps_per_epoch": steps_per_epoch,
                "latest_checkpoint_dir": str(checkpoint_dir),
                "latest_model_dir": str(runtime["save_dir"] / "latest_model"),
                "best_model_dir": str(runtime["save_dir"] / "best_model"),
                "best_epoch": best_epoch,
                "best_score": None if best_score == float("-inf") else best_score,
                "best_metrics_path": str(runtime["save_dir"] / "best_model_metrics.json") if best_metrics else None,
                "latest_val_metrics_path": str(val_metrics_path),
                "resumed_from": resumed_from,
                "pretrained_report": pretrained_report,
            }
            update_training_status(status_path, status_payload)

            if stop_requested:
                break

        status_payload["status"] = "completed"
        status_payload["updated_at"] = format_timestamp()
        update_training_status(status_path, status_payload)
        log_message(logger, "SYSTEM", "Training loop completed.")
        return status_payload
    except Exception as exc:
        status_payload["status"] = "failed"
        status_payload["updated_at"] = format_timestamp()
        status_payload["error"] = repr(exc)
        status_payload["traceback"] = traceback.format_exc()
        update_training_status(status_path, status_payload)
        log_message(logger, "SYSTEM", f"Training failed: {exc}", level=40)
        raise


def run_best_model_evaluation(runtime: dict, split_name: str, log_file_name: str = "eval") -> dict:
    stack = maybe_import_training_stack()
    paddle = stack["paddle"]

    requested_device = runtime["args"].device
    if requested_device == "auto":
        if paddle.device.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0:
            requested_device = "gpu"
        else:
            requested_device = "cpu"
    device = paddle.set_device(requested_device)
    logger, log_path = create_phase_logger(runtime["save_dir"], log_file_name)
    log_stage = "TEST" if split_name == "test" else "EVAL"
    log_message(
        logger,
        "INIT",
        f"Starting best-model evaluation for split={split_name} using ClusterRoutedUNet num_heads={runtime['num_heads']}.",
    )
    log_message(
        logger,
        "SYSTEM",
        (
            f"device={device} cuda_visible_devices={os.environ.get('CUDA_VISIBLE_DEVICES', 'not_set')} "
            f"pid={os.getpid()}"
        ),
    )

    set_seed(int(runtime["config"]["training"]["seed"]), paddle=paddle)
    datasets = build_datasets(runtime, stack)
    dataset = datasets[split_name]
    log_message(
        logger,
        "DATA",
        (
            f"dataset_root={runtime['dataset_root']} clustered_root={runtime['clustered_root']} "
            f"split={split_name} num_samples={len(dataset)} eval_batch_size={runtime['config']['training']['eval_batch_size']}"
        ),
    )

    model = build_model(runtime)
    optimizer, _ = build_optimizer(runtime, paddle, model, runtime["planned_total_steps"])
    best_model_dir = runtime["save_dir"] / "best_model"
    meta = load_checkpoint(model, optimizer, best_model_dir, paddle)
    log_message(logger, "CKPT", f"Loaded best model from {best_model_dir}.")
    criterion = build_criterion(stack, runtime)
    log_message(logger, log_stage, f"Running evaluation over split={split_name}.")
    metrics, confusion = evaluate_model(
        model=model,
        dataset=dataset,
        batch_size=int(runtime["config"]["training"]["eval_batch_size"]),
        max_batches=runtime["args"].max_val_batches if split_name == "val" else runtime["args"].max_test_batches,
        class_names=runtime["class_names"],
        paddle=paddle,
        criterion=criterion,
    )
    metrics["evaluated_split"] = split_name
    metrics["best_model_dir"] = str(best_model_dir)
    metrics["device"] = str(device)
    metrics["checkpoint_meta"] = meta

    write_json(runtime["save_dir"] / "best_model_metrics.json", metrics)
    write_metrics_text(
        runtime["save_dir"] / "best_model_metrics.txt",
        metrics,
        split_name=split_name,
        selection_metric=str(runtime["config"]["evaluation"].get("selection_metric", "miou")),
    )
    write_per_class_csv(runtime["save_dir"] / "per_class_metrics.csv", metrics)
    write_confusion_csv(runtime["save_dir"] / "confusion_matrix.csv", confusion, runtime["class_names"])
    write_json(runtime["save_dir"] / "metrics" / f"best_model_{split_name}_metrics.json", metrics)
    write_per_class_csv(runtime["save_dir"] / "metrics" / f"best_model_{split_name}_per_class.csv", metrics)
    write_confusion_csv(
        runtime["save_dir"] / "metrics" / f"best_model_{split_name}_confusion_matrix.csv",
        confusion,
        runtime["class_names"],
    )

    exported_samples = []
    if runtime["config"]["evaluation"].get("export_prediction_samples", True):
        exported_samples = export_prediction_samples(
            model,
            dataset,
            runtime["save_dir"],
            split_name,
            int(runtime["config"]["evaluation"].get("prediction_sample_count", 12)),
            stack,
            paddle,
        )

    log_message(
        logger,
        log_stage,
        (
            f"Split={split_name}, OA={metrics['overall_accuracy']:.6f}, "
            f"mAcc={metrics['mean_accuracy']:.6f}, mIoU={metrics['mean_iou']:.6f}, "
            f"Kappa={metrics['kappa']:.6f}, exported_prediction_files={len(exported_samples)}"
        ),
    )
    return {
        "status": "passed",
        "log_path": str(log_path),
        "metrics": metrics,
        "exported_prediction_files": exported_samples,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="UNet single-head and routed multi-head training for GID15.")
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
    parser.add_argument("--max-test-batches", type=int, default=None)
    parser.add_argument("--smoke-val-batches", type=int, default=1)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    runtime = build_runtime(args)
    create_output_structure(runtime["save_dir"])
    write_config_snapshot(runtime)

    if args.run_smoke_test:
        run_smoke_test(runtime)
        return

    if args.run_train:
        run_training(runtime)
        return

    logger, _ = create_phase_logger(runtime["save_dir"], "prepare")
    log_message(logger, "INIT", "prepare_only=true training_started=false smoke_test_started=false")
    log_message(logger, "DATA", f"config_snapshot_dir={runtime['save_dir'] / 'config_snapshot'}")


if __name__ == "__main__":
    main()
