#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import os
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
    resolve_public_path,
    split_samples,
    summarize_routed_samples,
)
from metrics_utils import (
    compute_confusion_metrics,
    ensure_dir,
    update_confusion_matrix,
    write_json,
    write_per_class_csv,
)


DEFAULT_CONFIG_PATH = WORK_DIR / "train_config.json"
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


class TeeStream:
    def __init__(self, *streams):
        self.streams = [stream for stream in streams if stream is not None]

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()

    def isatty(self):
        return False


def setup_tee_logging(log_path: Path):
    ensure_dir(log_path.parent)
    log_handle = open(log_path, "a", encoding="utf-8", buffering=1)
    sys.stdout = TeeStream(sys.__stdout__, log_handle)
    sys.stderr = TeeStream(sys.__stderr__, log_handle)
    return log_handle


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def resolve_existing_path(base_dir: Path, path_value: str | None, allow_missing: bool = False) -> Path | None:
    if path_value is None:
        return None
    path = Path(path_value)
    if str(path).startswith("/public/"):
        path = resolve_public_path(str(path))
    elif not path.is_absolute():
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


def build_model_structure(runtime: dict) -> dict:
    model_cfg = runtime["config"]["model"]
    routing_cfg = runtime["config"]["routing"]
    return {
        "model_name": "ClusterRoutedDeepLabV3",
        "backbone": model_cfg["backbone"],
        "backbone_pretrained": model_cfg["backbone_pretrained"],
        "shared_encoder": True,
        "shared_aspp": True,
        "shared_decoder_trunk": True,
        "num_heads": routing_cfg["num_heads"],
        "head_names": routing_cfg["cluster_names"],
        "segmentation_num_classes": runtime["config"]["num_classes"],
        "routing_rule": "Each sample uses exactly one segmentation head selected by cluster_id.",
        "supervision_rule": "The selected head is supervised by the sample's original segmentation mask.",
        "evaluation_rule": "Metrics are computed on original GID15 segmentation classes, not cluster ids.",
        "backbone_indices": model_cfg["backbone_indices"],
        "aspp_ratios": model_cfg["aspp_ratios"],
        "aspp_out_channels": model_cfg["aspp_out_channels"],
        "trunk_channels": model_cfg["trunk_channels"],
    }


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

    save_dir_value = args.save_dir or config.get("save_dir", "output")
    save_dir = resolve_existing_path(WORK_DIR, save_dir_value, allow_missing=True)
    assert save_dir is not None

    class_names = list(config.get("class_names", []))
    label_names = read_label_names(label_list)
    if class_names and label_names and class_names != label_names:
        raise ValueError(
            "class_names in train_config.json does not match labels.txt. "
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
    if num_heads != 4 or selected_k != 4:
        raise ValueError(f"This project is fixed to K=4 / num_heads=4, got selected_k={selected_k}, num_heads={num_heads}")
    if num_heads != selected_k:
        raise ValueError(f"num_heads must equal selected_k, got {num_heads} vs {selected_k}")
    if len(cluster_names) != num_heads:
        raise ValueError(f"cluster_names length must equal num_heads, got {len(cluster_names)} vs {num_heads}")

    samples = load_cluster_mapping(mapping_path, sample_path_source=routing_cfg.get("sample_path_source", "input"))
    routing_summary = summarize_routed_samples(samples)
    if routing_summary["duplicate_sample_ids_across_splits"]:
        raise ValueError("Duplicate sample ids were found across splits in cluster routing manifest.")

    training_cfg = config["training"]
    split_map = split_samples(samples)
    steps_per_epoch = math.ceil(len(split_map["train"]) / int(training_cfg["train_batch_size"]))
    planned_total_steps = max(1, steps_per_epoch * int(training_cfg["epochs"]))
    if args.max_train_steps is not None:
        planned_total_steps = min(planned_total_steps, int(args.max_train_steps))

    runtime = {
        "args": args,
        "config_path": config_path,
        "config": config,
        "dataset_root": dataset_root,
        "clustered_root": clustered_root,
        "mapping_path": mapping_path,
        "label_list": label_list,
        "class_names": class_names,
        "num_classes": num_classes,
        "num_heads": num_heads,
        "save_dir": save_dir,
        "samples": samples,
        "split_map": split_map,
        "routing_summary": routing_summary,
        "steps_per_epoch": steps_per_epoch,
        "planned_total_steps": planned_total_steps,
    }
    return runtime


def write_config_snapshot(runtime: dict) -> None:
    save_dir = runtime["save_dir"]
    snapshot_dir = save_dir / "config_snapshot"
    ensure_dir(snapshot_dir)

    resolved_config = {
        **runtime["config"],
        "resolved_dataset_root": str(runtime["dataset_root"]),
        "resolved_clustered_dataset_root": str(runtime["clustered_root"]),
        "resolved_cluster_mapping_file": str(runtime["mapping_path"]),
        "resolved_label_list": str(runtime["label_list"]),
        "resolved_save_dir": str(save_dir),
        "head_count": runtime["num_heads"],
        "segmentation_num_classes": runtime["num_classes"],
        "steps_per_epoch": runtime["steps_per_epoch"],
        "planned_total_steps": runtime["planned_total_steps"],
    }
    write_json(snapshot_dir / "resolved_train_config.json", resolved_config)
    write_json(snapshot_dir / "model_structure.json", build_model_structure(runtime))
    export_routing_snapshot(snapshot_dir / "routing_summary.json", runtime["samples"], runtime["num_heads"])

    dataset_overview = {
        "dataset_root": str(runtime["dataset_root"]),
        "clustered_dataset_root": str(runtime["clustered_root"]),
        "cluster_mapping_file": str(runtime["mapping_path"]),
        "label_list": str(runtime["label_list"]),
        "class_names": runtime["class_names"],
        "num_classes": runtime["num_classes"],
        "num_heads": runtime["num_heads"],
        "routing_summary": runtime["routing_summary"],
        "split_counts": {split: len(rows) for split, rows in runtime["split_map"].items()},
    }
    write_json(snapshot_dir / "dataset_overview.json", dataset_overview)

    status_payload = {
        "status": "prepared_only",
        "training_started": False,
        "note": "Dry-run completed; no optimization step has been executed yet.",
    }
    write_json(snapshot_dir / "dry_run_status.json", status_payload)


def maybe_import_training_stack():
    try:
        import paddle
        import paddlers
        from PIL import Image, ImageEnhance
    except Exception as exc:
        raise RuntimeError(
            "Training dependencies are not available. Activate the conda env that contains paddle/paddlers "
            "before running this script."
        ) from exc
    return {
        "paddle": paddle,
        "paddlers": paddlers,
        "Image": Image,
        "ImageEnhance": ImageEnhance,
    }


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
        max_samples=None,
    ):
        self.samples = list(samples[:max_samples] if max_samples else samples)
        self.Image = image_module
        self.ImageEnhance = image_enhance_module
        self.mean = np.asarray(mean, dtype=np.float32).reshape(1, 1, 3)
        self.std = np.asarray(std, dtype=np.float32).reshape(1, 1, 3)
        self.training = bool(training)
        self.augmentation_cfg = augmentation_cfg or {}
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
    from multihead_deeplabv3 import ClusterRoutedDeepLabV3

    model_cfg = runtime["config"]["model"]
    return ClusterRoutedDeepLabV3(
        num_classes=runtime["num_classes"],
        num_heads=runtime["num_heads"],
        backbone_name=model_cfg["backbone"],
        backbone_pretrained=model_cfg["backbone_pretrained"],
        backbone_indices=tuple(model_cfg["backbone_indices"]),
        aspp_ratios=tuple(model_cfg["aspp_ratios"]),
        aspp_out_channels=int(model_cfg["aspp_out_channels"]),
        trunk_channels=int(model_cfg["trunk_channels"]),
        dropout_prob=float(model_cfg["dropout_prob"]),
        align_corners=bool(model_cfg["align_corners"]),
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
        weight_decay=float(training_cfg.get("weight_decay", 0.0)),
    )
    return optimizer, lr_scheduler


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
            max_samples=runtime["args"].max_test_samples,
        ),
    }


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


def evaluate_model(
    model,
    dataset,
    batch_size: int,
    max_batches: int | None,
    class_names,
    paddle,
):
    criterion = paddle.nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX, axis=1)
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
            loss = criterion(logits, tensors["label"])
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


def run_smoke_test(runtime: dict, stack: dict, device: str) -> dict:
    paddle = stack["paddle"]
    paddle.seed(int(runtime["config"]["training"]["seed"]))

    norm_cfg = runtime["config"]["normalization"]
    model = build_model(runtime)
    model.train()
    optimizer, lr_scheduler = build_optimizer(runtime, paddle, model)
    criterion = paddle.nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX, axis=1)

    smoke_dataset = ClusterRoutedTileDataset(
        select_smoke_probe_samples(runtime),
        image_module=stack["Image"],
        image_enhance_module=stack["ImageEnhance"],
        mean=norm_cfg["mean"],
        std=norm_cfg["std"],
        training=False,
        augmentation_cfg={},
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

    loss = criterion(routed_logits, tensors["label"])
    loss.backward()
    grad_norm = compute_gradient_norm(model)
    optimizer.step()
    optimizer.clear_grad()
    lr_scheduler.step()

    checkpoint_dir = runtime["save_dir"] / "checkpoints" / "smoke_test_step_1"
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
        mean=norm_cfg["mean"],
        std=norm_cfg["std"],
        training=False,
        augmentation_cfg={},
        max_samples=max(runtime["args"].max_val_samples or 0, len(smoke_dataset.samples)) or None,
    )
    metrics, confusion = evaluate_model(
        model=model,
        dataset=validation_dataset,
        batch_size=int(runtime["config"]["training"]["eval_batch_size"]),
        max_batches=runtime["args"].smoke_val_batches,
        class_names=runtime["class_names"],
        paddle=paddle,
    )
    metrics["routing_probe_cluster_ids"] = smoke_batch["cluster_id"].tolist()
    metrics["routing_probe_max_abs_diff"] = max(route_diffs) if route_diffs else None
    metrics["checkpoint_dir"] = str(checkpoint_dir)
    metrics["device"] = device

    metrics_dir = runtime["save_dir"] / "metrics"
    write_json(metrics_dir / "smoke_test_metrics.json", metrics)
    np.save(metrics_dir / "smoke_test_confusion.npy", confusion)
    write_per_class_csv(metrics_dir / "smoke_test_per_class.csv", metrics)

    return {
        "status": "passed",
        "loss": float(loss),
        "grad_norm": grad_norm,
        "checkpoint_dir": str(checkpoint_dir),
        "routing_probe_cluster_ids": smoke_batch["cluster_id"].tolist(),
        "routing_probe_max_abs_diff": max(route_diffs) if route_diffs else None,
        "validation_metrics_path": str(metrics_dir / "smoke_test_metrics.json"),
    }


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

    set_seed(int(runtime["config"]["training"]["seed"]))
    paddle.seed(int(runtime["config"]["training"]["seed"]))

    run_name = runtime["args"].run_name or time.strftime("%Y%m%d_%H%M%S")
    log_path = runtime["save_dir"] / "logs" / f"{run_name}.log"
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    log_handle = setup_tee_logging(log_path)

    training_status_path = runtime["save_dir"] / "metrics" / f"{run_name}_status.json"
    results = {
        "device": str(device),
        "log_path": str(log_path),
        "smoke_test": None,
        "short_training": None,
    }

    try:
        print("[info] device=", device)
        print("[info] paddle_version=", stack["paddle"].__version__)
        print("[info] paddlers_version=", stack["paddlers"].__version__)
        print("[info] train_samples=", len(runtime["split_map"]["train"]))
        print("[info] val_samples=", len(runtime["split_map"]["val"]))
        print("[info] test_samples=", len(runtime["split_map"]["test"]))
        print("[info] planned_total_steps=", runtime["planned_total_steps"])

        if runtime["args"].run_smoke_test:
            print("[info] starting smoke test")
            smoke_result = run_smoke_test(runtime, stack, str(device))
            results["smoke_test"] = smoke_result
            write_json(training_status_path, results)
            print("[info] smoke test completed:", smoke_result)

        if runtime["args"].run_train:
            datasets = build_datasets(runtime, stack)
            model = build_model(runtime)
            optimizer, lr_scheduler = build_optimizer(runtime, paddle, model)
            criterion = paddle.nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX, axis=1)

            training_cfg = runtime["config"]["training"]
            train_batch_size = int(training_cfg["train_batch_size"])
            eval_batch_size = int(training_cfg["eval_batch_size"])
            max_epochs = int(training_cfg["epochs"])
            max_train_steps = runtime["args"].max_train_steps
            max_val_batches = runtime["args"].max_val_batches
            log_interval = max(1, int(training_cfg.get("log_interval_steps", 20)))

            best_score = float("-inf")
            global_step = 0
            start_epoch = 1
            epoch_losses = []
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
                start_epoch = int(resume_meta.get("epoch", 0)) + 1
                best_score = float(resume_meta.get("best_score", best_score))
                print(f"[info] resumed from checkpoint={resume_path} start_epoch={start_epoch} global_step={global_step}")

            print("[info] starting training loop")
            for epoch in range(start_epoch, max_epochs + 1):
                model.train()
                epoch_loss_values = []
                epoch_start = time.time()
                for batch in iterate_batches(
                    datasets["train"],
                    batch_size=train_batch_size,
                    shuffle=True,
                    seed=int(training_cfg["seed"]) + epoch,
                ):
                    global_step += 1
                    tensors = batch_to_tensors(batch, paddle)
                    logits = model(tensors["image"], tensors["cluster_id"])
                    loss = criterion(logits, tensors["label"])
                    loss.backward()
                    optimizer.step()
                    optimizer.clear_grad()
                    lr_scheduler.step()

                    loss_value = float(loss)
                    epoch_losses.append(loss_value)
                    epoch_loss_values.append(loss_value)

                    if global_step % log_interval == 0 or global_step == 1:
                        print(
                            f"[train] epoch={epoch} step={global_step} batch={len(batch['sample_id'])} "
                            f"loss={loss_value:.6f} lr={optimizer.get_lr():.8f}"
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
                )
                val_score = float(val_metrics["mean_iou"])
                epoch_summary = {
                    "epoch": epoch,
                    "global_step": global_step,
                    "train_loss_mean": float(np.mean(epoch_loss_values)) if epoch_loss_values else None,
                    "val_overall_accuracy": val_metrics["overall_accuracy"],
                    "val_mean_iou": val_metrics["mean_iou"],
                    "val_mean_accuracy": val_metrics["mean_accuracy"],
                    "epoch_seconds": time.time() - epoch_start,
                }
                print("[eval]", json.dumps(epoch_summary, ensure_ascii=False))

                metrics_dir = runtime["save_dir"] / "metrics"
                epoch_prefix = metrics_dir / f"epoch_{epoch:02d}"
                write_json(epoch_prefix.with_suffix(".json"), {**val_metrics, **epoch_summary})
                np.save(epoch_prefix.with_name(epoch_prefix.name + "_confusion.npy"), val_confusion)
                write_per_class_csv(epoch_prefix.with_name(epoch_prefix.name + "_per_class.csv"), val_metrics)

                checkpoint_dir = runtime["save_dir"] / "checkpoints" / f"epoch_{epoch:02d}_step_{global_step:04d}"
                checkpoint_metadata = {
                    "stage": "train",
                    "epoch": epoch,
                    "global_step": global_step,
                    "device": str(device),
                    "train_loss_mean": epoch_summary["train_loss_mean"],
                    "val_mean_iou": val_metrics["mean_iou"],
                    "val_overall_accuracy": val_metrics["overall_accuracy"],
                    "best_score": max(best_score, val_score),
                    "resumed_from": resumed_from,
                }
                save_checkpoint(model, optimizer, checkpoint_dir, checkpoint_metadata, paddle)
                copy_checkpoint(checkpoint_dir, runtime["save_dir"] / "latest_model")

                if val_score > best_score:
                    best_score = val_score
                    copy_checkpoint(checkpoint_dir, runtime["save_dir"] / "best_model")

                results["short_training"] = {
                    "status": "passed",
                    "epochs_completed": epoch,
                    "global_steps": global_step,
                    "latest_checkpoint_dir": str(checkpoint_dir),
                    "best_model_dir": str(runtime["save_dir"] / "best_model"),
                    "latest_model_dir": str(runtime["save_dir"] / "latest_model"),
                    "val_metrics_path": str(epoch_prefix.with_suffix(".json")),
                    "train_loss_mean": epoch_summary["train_loss_mean"],
                    "val_overall_accuracy": val_metrics["overall_accuracy"],
                    "val_mean_iou": val_metrics["mean_iou"],
                    "resumed_from": resumed_from,
                }
                write_json(training_status_path, results)

                if stop_requested:
                    print(f"[info] stopping early after {global_step} steps as requested")
                    break

            if results["short_training"] is None:
                raise RuntimeError("Training loop exited without completing any epoch.")

            if epoch_losses:
                results["short_training"]["global_train_loss_mean"] = float(np.mean(epoch_losses))

        write_json(training_status_path, results)
        return results
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        log_handle.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train or validate the 4-head cluster-routed DeepLabV3 model for GID15."
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--save-dir", default=None)
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
    parser.add_argument("--resume-checkpoint", default=None)
    parser.add_argument("--run-name", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runtime = build_runtime(args)
    create_output_structure(runtime["save_dir"])
    write_config_snapshot(runtime)

    print("[info] project_dir=", WORK_DIR)
    print("[info] save_dir=", runtime["save_dir"])
    print("[info] dataset_root=", runtime["dataset_root"])
    print("[info] clustered_dataset_root=", runtime["clustered_root"])
    print("[info] cluster_mapping_file=", runtime["mapping_path"])
    print("[info] num_classes=", runtime["num_classes"])
    print("[info] num_heads=", runtime["num_heads"])
    print("[info] split_counts=", runtime["routing_summary"]["split_counts"])
    print("[info] cluster_total_counts=", runtime["routing_summary"]["cluster_total_counts"])
    print("[info] config_snapshot=", runtime["save_dir"] / "config_snapshot")

    if args.run_smoke_test or args.run_train:
        results = run_training(runtime)
        print("[info] runtime_status=", json.dumps(results, ensure_ascii=False))
    else:
        print("[info] dry-run only; training not started.")


if __name__ == "__main__":
    main()
