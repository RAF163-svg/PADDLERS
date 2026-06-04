#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import shutil
import sys
import time
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
)
from farseg_routed_model import build_model_structure
from metrics_utils import (
    compute_confusion_metrics,
    ensure_dir,
    format_train_log,
    format_val_log,
    update_confusion_matrix,
    write_json,
    write_per_class_csv,
)


DEFAULT_CONFIG_PATH = WORK_DIR / "train_config_multihead.json"
IGNORE_INDEX = 255
OUTPUT_SUBDIRS = [
    "best_model",
    "latest_model",
    "checkpoints",
    "logs",
    "metrics",
    "prediction_samples",
    "config_snapshot",
]


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


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
    logger = logging.getLogger("farseg_data_split")
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
    }


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def build_runtime(args: argparse.Namespace) -> dict:
    config_path = resolve_existing_path(WORK_DIR, args.config)
    assert config_path is not None
    config = read_json(config_path)

    if args.disable_backbone_pretrained:
        config["model"]["backbone_pretrained"] = False
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
            "class_names in train_config_multihead.json does not match labels.txt. "
            f"config={class_names}, labels.txt={label_names}"
        )
    class_names = class_names or label_names
    num_classes = int(config["num_classes"])
    if len(class_names) != num_classes:
        raise ValueError(
            f"class_names length ({len(class_names)}) does not equal num_classes ({num_classes})."
        )

    routing_cfg = config["routing"]
    num_heads = int(routing_cfg["num_heads"])
    selected_k = int(routing_cfg["selected_k"])
    cluster_names = list(routing_cfg["cluster_names"])
    if selected_k != 4 or num_heads != 4:
        raise ValueError(f"This project is fixed to K=4 / num_heads=4, got selected_k={selected_k}, num_heads={num_heads}")
    if num_heads != selected_k:
        raise ValueError(f"num_heads must equal selected_k, got {num_heads} vs {selected_k}")
    if len(cluster_names) != num_heads:
        raise ValueError(f"cluster_names length must equal num_heads, got {len(cluster_names)} vs {num_heads}")

    samples = load_cluster_mapping(
        mapping_path,
        sample_path_source=routing_cfg.get("sample_path_source", "clustered_output"),
    )
    routing_summary = summarize_routed_samples(samples)
    if routing_summary["duplicate_sample_ids_across_splits"]:
        raise ValueError("Duplicate sample ids were found across splits in cluster routing manifest.")

    split_map = split_samples(samples)
    training_cfg = config["training"]
    full_steps_per_epoch = math.ceil(len(split_map["train"]) / int(training_cfg["train_batch_size"]))
    planned_total_steps = max(1, full_steps_per_epoch * int(training_cfg["epochs"]))
    if args.max_train_steps is not None:
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
        "num_heads": num_heads,
        "samples": samples,
        "split_map": split_map,
        "routing_summary": routing_summary,
        "full_steps_per_epoch": full_steps_per_epoch,
        "planned_total_steps": planned_total_steps,
        "save_dir": save_dir,
    }


def write_config_snapshot(runtime: dict) -> None:
    snapshot_dir = runtime["save_dir"] / "config_snapshot"
    ensure_dir(snapshot_dir)

    resolved_config = {
        **runtime["config"],
        "resolved_dataset_root": str(runtime["dataset_root"]),
        "resolved_clustered_dataset_root": str(runtime["clustered_root"]),
        "resolved_cluster_mapping_file": str(runtime["mapping_path"]),
        "resolved_label_list": str(runtime["label_list"]),
        "resolved_save_dir": str(runtime["save_dir"]),
        "num_heads": runtime["num_heads"],
        "segmentation_num_classes": runtime["num_classes"],
        "full_steps_per_epoch": runtime["full_steps_per_epoch"],
        "planned_total_steps": runtime["planned_total_steps"],
    }
    write_json(snapshot_dir / "resolved_train_config.json", resolved_config)
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
            "routing_summary": runtime["routing_summary"],
            "split_counts": {split: len(rows) for split, rows in runtime["split_map"].items()},
        },
    )
    write_json(
        snapshot_dir / "dry_run_status.json",
        {
            "status": "prepared_only",
            "training_started": False,
            "smoke_test_started": False,
            "note": "The config snapshot is ready. No optimization step has been executed yet.",
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
            "sample_id": sample.sample_id,
            "scene_id": sample.scene_id,
        }


def collate_samples(batch_items):
    return {
        "image": np.stack([item["image"] for item in batch_items], axis=0).astype(np.float32, copy=False),
        "label": np.stack([item["label"] for item in batch_items], axis=0).astype(np.int64, copy=False),
        "cluster_id": np.asarray([item["cluster_id"] for item in batch_items], dtype=np.int64),
        "sample_id": [item["sample_id"] for item in batch_items],
        "scene_id": [item["scene_id"] for item in batch_items],
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
        "sample_id": batch["sample_id"],
        "scene_id": batch["scene_id"],
    }


def compute_gradient_norm(model) -> float:
    total = 0.0
    for parameter in model.parameters():
        grad = parameter.grad
        if grad is None:
            continue
        grad_array = grad.numpy().astype(np.float64, copy=False)
        total += float(np.sum(np.square(grad_array)))
    return float(total ** 0.5)


def build_model(runtime: dict):
    from farseg_routed_model import ClusterRoutedFarSeg

    model_cfg = runtime["config"]["model"]
    return ClusterRoutedFarSeg(
        in_channels=int(model_cfg.get("in_channels", 3)),
        num_classes=runtime["num_classes"],
        num_heads=runtime["num_heads"],
        backbone=model_cfg.get("backbone", "resnet50"),
        backbone_pretrained=bool(model_cfg.get("backbone_pretrained", True)),
        fpn_out_channels=int(model_cfg.get("fpn_out_channels", 256)),
        fsr_out_channels=int(model_cfg.get("fsr_out_channels", 256)),
        scale_aware_proj=bool(model_cfg.get("scale_aware_proj", True)),
        decoder_out_channels=int(model_cfg.get("decoder_out_channels", 128)),
    )


def build_optimizer(runtime: dict, paddle, model):
    training_cfg = runtime["config"]["training"]
    decay_steps = max(1, int(runtime["planned_total_steps"]))
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


def build_criterion(stack: dict, use_mixed_loss: bool = True):
    seg_losses = stack["seg_losses"]
    if use_mixed_loss:
        losses = [
            seg_losses.CrossEntropyLoss(ignore_index=IGNORE_INDEX),
            seg_losses.LovaszSoftmaxLoss(ignore_index=IGNORE_INDEX),
        ]
        return seg_losses.MixedLoss(losses=losses, coef=[0.8, 0.2])
    return seg_losses.CrossEntropyLoss(ignore_index=IGNORE_INDEX)


def compute_total_loss(criterion, logits, labels):
    loss_output = criterion(logits, labels)
    if isinstance(loss_output, (list, tuple)):
        total = None
        for item in loss_output:
            total = item if total is None else total + item
        return total
    return loss_output


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
            logits = model(tensors["image"], tensors["cluster_id"])
            loss = compute_total_loss(criterion, logits, tensors["label"])
            losses.append(float(loss))

            prediction = paddle.argmax(logits, axis=1).numpy()
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


def select_smoke_probe_samples(runtime: dict):
    probe_samples = []
    train_samples = runtime["split_map"]["train"]
    for cluster_id in range(runtime["num_heads"]):
        sample = next((item for item in train_samples if item.cluster_id == cluster_id), None)
        if sample is None:
            raise RuntimeError(f"Unable to find a train sample for cluster_id={cluster_id}")
        probe_samples.append(sample)
    return probe_samples


def log_startup_summary(runtime: dict, logger: logging.Logger, train_size: int, val_size: int, steps_per_epoch: int) -> None:
    training_cfg = runtime["config"]["training"]
    logger.info(
        "[start] "
        f"total_epochs={training_cfg['epochs']} "
        f"train_dataset_size={train_size} "
        f"val_dataset_size={val_size} "
        f"steps_per_epoch={steps_per_epoch} "
        f"model_type=ClusterRoutedFarSeg "
        f"num_heads={runtime['num_heads']}"
    )
    logger.info(f"[path] best_model_dir={runtime['save_dir'] / 'best_model'}")
    logger.info(f"[path] latest_model_dir={runtime['save_dir'] / 'latest_model'}")
    logger.info(f"[path] checkpoint_dir={runtime['save_dir'] / 'checkpoints'}")
    logger.info(f"[path] metrics_dir={runtime['save_dir'] / 'metrics'}")
    logger.info(f"[path] prediction_samples_dir={runtime['save_dir'] / 'prediction_samples'}")


def run_smoke_test(runtime: dict, stack: dict, logger: logging.Logger, device: str) -> dict:
    paddle = stack["paddle"]
    model = build_model(runtime)
    optimizer, lr_scheduler = build_optimizer(runtime, paddle, model)
    criterion = build_criterion(stack, use_mixed_loss=bool(runtime["config"]["training"].get("use_mixed_loss", True)))

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
    routed_logits = outputs["routed_logits"]
    all_head_logits = outputs["all_head_logits"]
    route_diffs = []
    for sample_index, cluster_id in enumerate(smoke_batch["cluster_id"].tolist()):
        diff = paddle.max(
            paddle.abs(routed_logits[sample_index] - all_head_logits[cluster_id][sample_index])
        )
        route_diffs.append(float(diff))

    loss = compute_total_loss(criterion, routed_logits, tensors["label"])
    loss.backward()
    grad_norm = compute_gradient_norm(model)
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
        "loss": float(loss),
        "grad_norm": grad_norm,
        "max_route_diff": max(route_diffs) if route_diffs else None,
    }
    save_checkpoint(model, optimizer, checkpoint_dir, checkpoint_metadata, paddle)

    validation_dataset = ClusterRoutedTileDataset(
        runtime["split_map"]["val"],
        image_module=stack["Image"],
        image_enhance_module=stack["ImageEnhance"],
        mean=runtime["config"]["normalization"]["mean"],
        std=runtime["config"]["normalization"]["std"],
        training=False,
        augmentation_cfg={},
        crop_size=None,
        max_samples=max(runtime["args"].max_val_samples or 0, len(smoke_dataset.samples)) or None,
    )
    metrics, confusion = evaluate_model(
        model=model,
        dataset=validation_dataset,
        batch_size=int(runtime["config"]["training"]["eval_batch_size"]),
        max_batches=runtime["args"].smoke_val_batches,
        class_names=runtime["class_names"],
        paddle=paddle,
        criterion=criterion,
    )
    metrics["routing_probe_cluster_ids"] = smoke_batch["cluster_id"].tolist()
    metrics["routing_probe_max_abs_diff"] = max(route_diffs) if route_diffs else None
    metrics["checkpoint_dir"] = str(checkpoint_dir)
    metrics["device"] = device

    metrics_dir = runtime["save_dir"] / "metrics"
    write_json(metrics_dir / "smoke_test_metrics.json", metrics)
    np.save(metrics_dir / "smoke_test_confusion.npy", confusion)
    write_per_class_csv(metrics_dir / "smoke_test_per_class.csv", metrics)

    logger.info(
        "[smoke] "
        f"status=passed "
        f"cluster_ids={smoke_batch['cluster_id'].tolist()} "
        f"loss={float(loss):.6f} "
        f"grad_norm={grad_norm:.6f} "
        f"max_route_diff={(max(route_diffs) if route_diffs else 0.0):.10f} "
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
        "validation_metrics_path": str(metrics_dir / "smoke_test_metrics.json"),
    }


def run_short_training(runtime: dict, stack: dict, logger: logging.Logger, device: str) -> dict:
    paddle = stack["paddle"]
    datasets = build_datasets(runtime, stack)
    model = build_model(runtime)
    optimizer, lr_scheduler = build_optimizer(runtime, paddle, model)
    criterion = build_criterion(stack, use_mixed_loss=bool(runtime["config"]["training"].get("use_mixed_loss", True)))

    training_cfg = runtime["config"]["training"]
    train_batch_size = int(training_cfg["train_batch_size"])
    eval_batch_size = int(training_cfg["eval_batch_size"])
    max_epochs = int(training_cfg["epochs"])
    max_train_steps = runtime["args"].max_train_steps
    max_val_batches = runtime["args"].max_val_batches
    log_interval = max(1, int(training_cfg.get("log_interval_steps", 20)))

    steps_per_epoch = max(1, math.ceil(len(datasets["train"]) / train_batch_size))
    best_score = float("-inf")
    global_step = 0
    latest_checkpoint_dir = None
    best_metrics_path = None
    best_epoch = None
    stop_requested = False
    resumed_from = None

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
            logits = model(tensors["image"], tensors["cluster_id"])
            loss = compute_total_loss(criterion, logits, tensors["label"])
            loss.backward()
            optimizer.step()
            optimizer.clear_grad()
            lr_scheduler.step()
            step_time = time.perf_counter() - step_start

            loss_value = float(loss)
            epoch_loss_values.append(loss_value)

            if global_step == 1 or global_step % log_interval == 0 or step_in_epoch == steps_per_epoch:
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
                        step_time=step_time,
                    )
                )

            if max_train_steps is not None and global_step >= max_train_steps:
                stop_requested = True
                break

        val_metrics, val_confusion = evaluate_model(
            model=model,
            dataset=datasets["val"],
            batch_size=eval_batch_size,
            max_batches=max_val_batches,
            class_names=runtime["class_names"],
            paddle=paddle,
            criterion=criterion,
        )
        logger.info(format_val_log(epoch=epoch, total_epochs=max_epochs, metrics=val_metrics))

        metrics_dir = runtime["save_dir"] / "metrics"
        epoch_prefix = metrics_dir / f"short_train_epoch_{epoch:02d}"
        epoch_metrics_payload = {
            **val_metrics,
            "epoch": epoch,
            "global_step": global_step,
            "train_loss_mean": float(np.mean(epoch_loss_values)) if epoch_loss_values else None,
            "device": device,
            "resumed_from": resumed_from,
        }
        write_json(epoch_prefix.with_suffix(".json"), epoch_metrics_payload)
        np.save(epoch_prefix.with_name(epoch_prefix.name + "_confusion.npy"), val_confusion)
        write_per_class_csv(epoch_prefix.with_name(epoch_prefix.name + "_per_class.csv"), val_metrics)

        checkpoint_dir = runtime["save_dir"] / "checkpoints" / f"short_train_epoch_{epoch:02d}_step_{global_step:04d}"
        checkpoint_metadata = {
            "stage": "short_train",
            "epoch": epoch,
            "global_step": global_step,
            "device": device,
            "train_loss_mean": epoch_metrics_payload["train_loss_mean"],
            "val_mean_iou": val_metrics["mean_iou"],
            "val_overall_accuracy": val_metrics["overall_accuracy"],
            "best_score": max(best_score, float(val_metrics["mean_iou"])),
            "resumed_from": resumed_from,
        }
        save_checkpoint(model, optimizer, checkpoint_dir, checkpoint_metadata, paddle)
        copy_checkpoint(checkpoint_dir, runtime["save_dir"] / "latest_model")

        latest_checkpoint_dir = checkpoint_dir
        if float(val_metrics["mean_iou"]) > best_score:
            best_score = float(val_metrics["mean_iou"])
            best_metrics_path = str(epoch_prefix.with_suffix(".json"))
            best_epoch = epoch
            copy_checkpoint(checkpoint_dir, runtime["save_dir"] / "best_model")

        if stop_requested:
            break

    if latest_checkpoint_dir is None:
        raise RuntimeError("Short training did not complete any optimization step.")

    return {
        "status": "passed",
        "epochs_completed": epoch,
        "global_steps": global_step,
        "latest_checkpoint_dir": str(latest_checkpoint_dir),
        "latest_model_dir": str(runtime["save_dir"] / "latest_model"),
        "best_model_dir": str(runtime["save_dir"] / "best_model"),
        "best_epoch": best_epoch,
        "best_metrics_path": best_metrics_path,
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
    set_seed(seed)
    paddle.seed(seed)

    datasets = build_datasets(runtime, stack)
    steps_per_epoch = max(1, math.ceil(len(datasets["train"]) / int(runtime["config"]["training"]["train_batch_size"])))

    run_name = runtime["args"].run_name or time.strftime("smoke_short_%Y%m%d_%H%M%S")
    log_path = runtime["save_dir"] / "logs" / f"{run_name}.log"
    logger = setup_logger(log_path)

    log_startup_summary(runtime, logger, len(datasets["train"]), len(datasets["val"]), steps_per_epoch)
    logger.info(f"[env] device={device}")
    logger.info(f"[env] paddle_version={stack['paddle'].__version__}")
    logger.info(f"[env] paddlers_version={stack['paddlers'].__version__}")

    results = {
        "status": "running",
        "device": str(device),
        "log_path": str(log_path),
        "smoke_test": None,
        "short_training": None,
    }

    if runtime["args"].run_smoke_test:
        results["smoke_test"] = run_smoke_test(runtime, stack, logger, str(device))

    if runtime["args"].run_train:
        results["short_training"] = run_short_training(runtime, stack, logger, str(device))

    results["status"] = "passed"
    write_json(runtime["save_dir"] / "metrics" / f"{run_name}_status.json", results)
    logger.info(f"[status] acceptance_status={results['status']}")
    logger.info(f"[status] log_path={log_path}")
    return results


def parse_args():
    parser = argparse.ArgumentParser(
        description="Smoke test and short-training acceptance for cluster-routed 4-head FarSeg on GID15."
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--save-dir", default=None)
    parser.add_argument("--resume-checkpoint", default=None)
    parser.add_argument("--run-smoke-test", action="store_true")
    parser.add_argument("--run-train", action="store_true")
    parser.add_argument("--device", choices=["auto", "gpu", "cpu"], default="auto")
    parser.add_argument("--disable-backbone-pretrained", action="store_true")
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
        logger.info(f"[status] config_snapshot={runtime['save_dir'] / 'config_snapshot'}")


if __name__ == "__main__":
    main()
