#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import shutil
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image


DEFAULT_RAW_CANDIDATES = [
    "${GID15_DATASET_ROOT}/gid-15/GID",
    "${NAS_ROOT}/gid-15/GID",
]

DEFAULT_PREPARED_CANDIDATES = [
    "${GID15_DATASET_ROOT}/gid15_seed20260320_tile640_stride448",
    "${GID15_DATASET_ROOT}/paddlers_gid15_benchmark_20260320/data/gid15_seed20260320_tile640_stride448",
]

DEFAULT_OUTPUT_ROOT = "${GID15_DATASET_ROOT}/gid15_split_dataset"
DEFAULT_SEED = 2026
DEFAULT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}


@dataclass(frozen=True)
class SampleRecord:
    sample_id: str
    scene_id: str
    source_split: str
    src_image: str
    src_mask: str
    assigned_class_id: int
    assigned_class_name: str
    dominant_pixels: int
    foreground_pixels: int
    total_pixels: int
    dominant_foreground_ratio: float
    dominant_total_ratio: float


def log(message: str) -> None:
    print(message, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a reproducible GID15 classification dataset from the prepared "
            "tile cache using dominant foreground class assignment."
        )
    )
    parser.add_argument("--raw-root", default=None, help="Raw GID15 root (img_dir/ann_dir).")
    parser.add_argument(
        "--prepared-root",
        default=None,
        help="Prepared GID15 tile root (images/masks/lists/labels.txt).",
    )
    parser.add_argument(
        "--output-root",
        default=DEFAULT_OUTPUT_ROOT,
        help="Requested output root. /public/... is auto-mapped to ${NAS_ROOT}/... on this host.",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed.")
    parser.add_argument("--preferred-threshold", type=int, default=500)
    parser.add_argument("--fallback-threshold", type=int, default=300)
    parser.add_argument("--absolute-min-threshold", type=int, default=200)
    parser.add_argument("--copy-workers", type=int, default=8)
    parser.add_argument("--force", action="store_true", help="Overwrite output root if it exists.")
    return parser.parse_args()


def resolve_existing_root(provided: str | None, candidates: list[str], label: str) -> Path:
    if provided:
        root = Path(provided).expanduser().resolve()
        if not root.exists():
            raise FileNotFoundError(f"{label} does not exist: {root}")
        return root

    for candidate in candidates:
        root = Path(candidate)
        if root.exists():
            return root.resolve()

    raise FileNotFoundError(f"Unable to locate {label}. Checked: {candidates}")


def resolve_output_root(requested_path: str) -> tuple[Path, dict[str, str]]:
    requested = Path(requested_path)
    if str(requested).startswith("/public/"):
        mapped = Path("${NAS_ROOT}") / requested.relative_to("/public")
        return mapped.resolve(), {
            "requested_output_root": str(requested),
            "resolved_output_root": str(mapped.resolve()),
            "output_resolution_note": (
                "The host does not expose /public directly. ${NAS_ROOT} is mounted from "
                "//10.62.192.79/public, so the requested /public path was resolved "
                "to the equivalent ${NAS_ROOT} path."
            ),
        }

    return requested.expanduser().resolve(), {
        "requested_output_root": str(requested),
        "resolved_output_root": str(requested.expanduser().resolve()),
        "output_resolution_note": "The requested output path is used directly.",
    }


def ensure_output_root(output_root: Path, force: bool) -> None:
    if output_root.exists() and not force:
        raise FileExistsError(
            f"Output root already exists: {output_root}. Re-run with --force to overwrite."
        )

    if output_root.exists() and force:
        log(f"[info] Removing existing output root because --force was supplied: {output_root}")
        shutil.rmtree(output_root)

    output_root.mkdir(parents=True, exist_ok=True)


def load_labels(prepared_root: Path) -> list[str]:
    labels_path = prepared_root / "labels.txt"
    labels = [line.strip() for line in labels_path.read_text().splitlines() if line.strip()]
    if not labels or labels[0] != "background":
        raise RuntimeError(
            f"Expected labels.txt at {labels_path} to start with 'background', got: {labels[:3]}"
        )
    return labels


def scan_raw_dataset(raw_root: Path) -> dict[str, object]:
    img_dir = raw_root / "img_dir"
    ann_dir = raw_root / "ann_dir"
    if not img_dir.is_dir() or not ann_dir.is_dir():
        raise RuntimeError(
            f"Raw root {raw_root} does not look like GID15. Expected img_dir/ and ann_dir/."
        )

    image_counts = {}
    mask_counts = {}
    for split in ("train", "val", "test"):
        image_counts[split] = len(list((img_dir / split).glob("*.tif")))
        mask_counts[split] = len(list((ann_dir / split).glob("*_15label.png")))

    return {
        "raw_root": str(raw_root),
        "raw_layout": "image + annotation masks (semantic segmentation), not pre-classified folders",
        "raw_image_counts": image_counts,
        "raw_mask_counts": mask_counts,
        "raw_labeled_scene_count": image_counts["train"] + image_counts["val"],
        "raw_unlabeled_test_image_count": image_counts["test"],
    }


def iter_mask_paths(prepared_root: Path) -> Iterable[tuple[str, Path]]:
    for split in ("train", "val", "test"):
        mask_dir = prepared_root / "masks" / split
        if not mask_dir.is_dir():
            continue
        for mask_path in sorted(mask_dir.glob("*.png")):
            yield split, mask_path


def scan_prepared_tiles(
    prepared_root: Path,
    labels: list[str],
) -> tuple[list[SampleRecord], Counter, Counter, dict[int, set[str]], int, int]:
    records: list[SampleRecord] = []
    dominant_non_background_counts: Counter = Counter()
    dominant_any_counts: Counter = Counter()
    scene_support: dict[int, set[str]] = defaultdict(set)
    total_mask_count = 0
    background_only_tile_count = 0

    num_scanned = 0
    for split, mask_path in iter_mask_paths(prepared_root):
        total_mask_count += 1
        sample_id = mask_path.stem
        scene_id = sample_id.rsplit("_", 2)[0]
        image_path = prepared_root / "images" / split / f"{sample_id}.tif"
        if not image_path.is_file():
            raise FileNotFoundError(f"Missing image for mask {mask_path}: {image_path}")

        arr = np.array(Image.open(mask_path), dtype=np.uint8)
        hist = np.bincount(arr.reshape(-1), minlength=len(labels))
        dominant_any = int(hist.argmax())
        dominant_any_counts[dominant_any] += 1

        foreground_hist = hist.copy()
        foreground_hist[0] = 0
        foreground_pixels = int(foreground_hist.sum())
        if foreground_pixels == 0:
            background_only_tile_count += 1
            num_scanned += 1
            continue

        class_id = int(foreground_hist.argmax())
        dominant_pixels = int(foreground_hist[class_id])
        total_pixels = int(arr.size)
        dominant_foreground_ratio = dominant_pixels / foreground_pixels
        dominant_total_ratio = dominant_pixels / total_pixels

        record = SampleRecord(
            sample_id=sample_id,
            scene_id=scene_id,
            source_split=split,
            src_image=str(image_path.resolve()),
            src_mask=str(mask_path.resolve()),
            assigned_class_id=class_id,
            assigned_class_name=labels[class_id],
            dominant_pixels=dominant_pixels,
            foreground_pixels=foreground_pixels,
            total_pixels=total_pixels,
            dominant_foreground_ratio=dominant_foreground_ratio,
            dominant_total_ratio=dominant_total_ratio,
        )
        records.append(record)
        dominant_non_background_counts[class_id] += 1
        scene_support[class_id].add(scene_id)

        num_scanned += 1
        if num_scanned % 5000 == 0:
            log(f"[scan] Processed {num_scanned} tiles...")

    return (
        records,
        dominant_non_background_counts,
        dominant_any_counts,
        scene_support,
        total_mask_count,
        background_only_tile_count,
    )


def build_before_rows(
    labels: list[str],
    class_counts: Counter,
    dominant_any_counts: Counter,
    scene_support: dict[int, set[str]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for class_id in range(1, len(labels)):
        rows.append(
            {
                "class_id": class_id,
                "class_name": labels[class_id],
                "sample_count": int(class_counts[class_id]),
                "scene_support": len(scene_support.get(class_id, set())),
                "dominant_overall_count": int(dominant_any_counts[class_id]),
            }
        )
    rows.sort(key=lambda item: (-int(item["sample_count"]), int(item["class_id"])))
    return rows


def choose_threshold(
    before_rows: list[dict[str, object]],
    preferred_threshold: int,
    fallback_threshold: int,
    absolute_min_threshold: int,
) -> tuple[int, str]:
    preferred_rows = [row for row in before_rows if int(row["sample_count"]) >= preferred_threshold]
    fallback_rows = [row for row in before_rows if int(row["sample_count"]) >= fallback_threshold]

    minimum_preferred_classes = max(5, math.ceil(len(before_rows) * 0.40))
    if len(preferred_rows) >= minimum_preferred_classes:
        return (
            preferred_threshold,
            (
                f"{len(preferred_rows)} classes already satisfy the preferred >= {preferred_threshold} "
                f"sample rule, which is enough semantic coverage to avoid relaxing to {fallback_threshold}. "
                "Using the stricter threshold keeps validation/test subsets healthier for later training."
            ),
        )

    if fallback_rows:
        return (
            fallback_threshold,
            (
                f"Only {len(preferred_rows)} classes satisfy >= {preferred_threshold}, which is too few. "
                f"Relaxing to >= {fallback_threshold} keeps the dataset usable while still respecting the "
                f"hard floor of >= {absolute_min_threshold}."
            ),
        )

    raise RuntimeError(
        "No foreground classes satisfy the fallback threshold. Aborting instead of producing a low-quality dataset."
    )


def build_excluded_rows(
    before_rows: list[dict[str, object]],
    adopted_threshold: int,
    fallback_threshold: int,
    absolute_min_threshold: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for row in before_rows:
        sample_count = int(row["sample_count"])
        if sample_count >= adopted_threshold:
            continue

        if sample_count < absolute_min_threshold:
            reason = f"Excluded because {sample_count} < hard minimum {absolute_min_threshold}."
        elif sample_count < fallback_threshold:
            reason = (
                f"Excluded because {sample_count} is below the relaxed threshold {fallback_threshold}; "
                "keeping it would leave too few samples for stable validation/test splits."
            )
        else:
            reason = (
                f"Excluded because the adopted threshold is {adopted_threshold}; although {sample_count} "
                f">= {fallback_threshold}, the data distribution was strong enough to keep the stricter rule."
            )

        rows.append(
            {
                "class_id": row["class_id"],
                "class_name": row["class_name"],
                "sample_count": sample_count,
                "scene_support": row["scene_support"],
                "reason": reason,
            }
        )

    rows.sort(key=lambda item: (-int(item["sample_count"]), int(item["class_id"])))
    return rows


def allocate_group_counts(size: int, ratios: dict[str, float]) -> dict[str, int]:
    ordered_splits = ["train", "val", "test"]
    exact = {split: size * ratios[split] for split in ordered_splits}
    counts = {split: int(math.floor(exact[split])) for split in ordered_splits}
    remainder = size - sum(counts.values())

    ranked = sorted(
        ordered_splits,
        key=lambda split: (exact[split] - counts[split], ratios[split], split),
        reverse=True,
    )
    for split in ranked[:remainder]:
        counts[split] += 1

    if size >= 3:
        missing = [split for split in ordered_splits if counts[split] == 0]
        for split in missing:
            donors = [candidate for candidate in ordered_splits if counts[candidate] > 1]
            if not donors:
                break
            donor = max(
                donors,
                key=lambda candidate: (counts[candidate] - exact[candidate], counts[candidate], candidate),
            )
            counts[donor] -= 1
            counts[split] += 1

    return counts


def split_scenes(
    records: list[SampleRecord],
    retained_class_ids: list[int],
    ratios: dict[str, float],
    seed: int,
) -> tuple[dict[str, str], dict[str, list[str]], dict[int, int]]:
    scene_class_counts: dict[str, Counter] = defaultdict(Counter)
    scene_primary_support: Counter = Counter()

    for record in records:
        if record.assigned_class_id not in retained_class_ids:
            continue
        scene_class_counts[record.scene_id][record.assigned_class_id] += 1

    if not scene_class_counts:
        raise RuntimeError("No scenes contain retained classes after thresholding.")

    primary_groups: dict[int, list[str]] = defaultdict(list)
    for scene_id, class_counter in scene_class_counts.items():
        primary_class_id = max(
            class_counter.items(),
            key=lambda item: (item[1], -item[0]),
        )[0]
        primary_groups[primary_class_id].append(scene_id)
        scene_primary_support[primary_class_id] += 1

    scene_to_split: dict[str, str] = {}
    split_to_scenes: dict[str, list[str]] = {"train": [], "val": [], "test": []}

    for primary_class_id in sorted(primary_groups):
        group = sorted(primary_groups[primary_class_id])
        group_rng = random.Random(seed + primary_class_id)
        group_rng.shuffle(group)
        allocation = allocate_group_counts(len(group), ratios)

        cursor = 0
        for split in ("train", "val", "test"):
            next_cursor = cursor + allocation[split]
            for scene_id in group[cursor:next_cursor]:
                scene_to_split[scene_id] = split
                split_to_scenes[split].append(scene_id)
            cursor = next_cursor

    return scene_to_split, split_to_scenes, dict(scene_primary_support)


def compute_split_class_counts(
    records: list[SampleRecord],
    retained_class_ids: set[int],
    scene_to_split: dict[str, str],
) -> dict[str, Counter]:
    split_counts = {"train": Counter(), "val": Counter(), "test": Counter()}
    for record in records:
        if record.assigned_class_id not in retained_class_ids:
            continue
        split_counts[scene_to_split[record.scene_id]][record.assigned_class_id] += 1
    return split_counts


def repair_missing_classes(
    records: list[SampleRecord],
    retained_class_ids: list[int],
    scene_to_split: dict[str, str],
    split_to_scenes: dict[str, list[str]],
) -> None:
    retained_set = set(retained_class_ids)
    scene_class_counts: dict[str, Counter] = defaultdict(Counter)
    for record in records:
        if record.assigned_class_id in retained_set:
            scene_class_counts[record.scene_id][record.assigned_class_id] += 1

    for _ in range(100):
        split_counts = compute_split_class_counts(records, retained_set, scene_to_split)
        missing = []
        for split in ("train", "val", "test"):
            for class_id in retained_class_ids:
                if split_counts[split][class_id] == 0:
                    missing.append((split, class_id))
        if not missing:
            return

        repaired = False
        for target_split, class_id in missing:
            donors = sorted(
                ("train", "val", "test"),
                key=lambda split: split_counts[split][class_id],
                reverse=True,
            )
            for donor_split in donors:
                if donor_split == target_split or split_counts[donor_split][class_id] <= 1:
                    continue

                candidate_scenes = [
                    scene_id
                    for scene_id in split_to_scenes[donor_split]
                    if scene_class_counts[scene_id][class_id] > 0
                ]
                candidate_scenes.sort(
                    key=lambda scene_id: (
                        scene_class_counts[scene_id][class_id],
                        sum(scene_class_counts[scene_id].values()),
                        scene_id,
                    )
                )
                if not candidate_scenes:
                    continue

                scene_id = candidate_scenes[0]
                split_to_scenes[donor_split].remove(scene_id)
                split_to_scenes[target_split].append(scene_id)
                scene_to_split[scene_id] = target_split
                repaired = True
                break

            if repaired:
                break

        if not repaired:
            break

    split_counts = compute_split_class_counts(records, retained_set, scene_to_split)
    missing_messages = []
    for split in ("train", "val", "test"):
        for class_id in retained_class_ids:
            if split_counts[split][class_id] == 0:
                missing_messages.append(f"{split}:{class_id}")
    if missing_messages:
        raise RuntimeError(
            "Unable to guarantee all retained classes appear in every split. Missing: "
            + ", ".join(missing_messages)
        )


def build_retained_records(
    records: list[SampleRecord],
    retained_class_ids: set[int],
    scene_to_split: dict[str, str],
) -> list[dict[str, object]]:
    retained_records: list[dict[str, object]] = []
    for record in records:
        if record.assigned_class_id not in retained_class_ids:
            continue
        split = scene_to_split[record.scene_id]
        retained_records.append(
            {
                "sample_id": record.sample_id,
                "scene_id": record.scene_id,
                "source_split": record.source_split,
                "src_image": record.src_image,
                "src_mask": record.src_mask,
                "assigned_class_id": record.assigned_class_id,
                "assigned_class_name": record.assigned_class_name,
                "dominant_pixels": record.dominant_pixels,
                "foreground_pixels": record.foreground_pixels,
                "total_pixels": record.total_pixels,
                "dominant_foreground_ratio": f"{record.dominant_foreground_ratio:.6f}",
                "dominant_total_ratio": f"{record.dominant_total_ratio:.6f}",
                "split": split,
            }
        )
    retained_records.sort(key=lambda item: (item["split"], item["assigned_class_name"], item["sample_id"]))
    return retained_records


def copy_file(task: tuple[str, str]) -> None:
    src, dst = task
    shutil.copy2(src, dst)


def copy_dataset(
    retained_records: list[dict[str, object]],
    output_root: Path,
    retained_class_names: list[str],
    workers: int,
) -> None:
    for split in ("train", "val", "test"):
        for class_name in retained_class_names:
            (output_root / split / class_name).mkdir(parents=True, exist_ok=True)

    tasks: list[tuple[str, str]] = []
    for record in retained_records:
        src = record["src_image"]
        dst = str(output_root / record["split"] / record["assigned_class_name"] / Path(src).name)
        record["output_image"] = dst
        tasks.append((src, dst))

    log(f"[copy] Copying {len(tasks)} image files...")
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        for index, _ in enumerate(executor.map(copy_file, tasks), start=1):
            if index % 2000 == 0 or index == len(tasks):
                log(f"[copy] Copied {index}/{len(tasks)} files")


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_after_rows(
    labels: list[str],
    retained_class_ids: list[int],
    split_counts: dict[str, Counter],
    scene_support: dict[int, set[str]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for class_id in retained_class_ids:
        train_count = int(split_counts["train"][class_id])
        val_count = int(split_counts["val"][class_id])
        test_count = int(split_counts["test"][class_id])
        rows.append(
            {
                "class_id": class_id,
                "class_name": labels[class_id],
                "total_count": train_count + val_count + test_count,
                "train_count": train_count,
                "val_count": val_count,
                "test_count": test_count,
                "scene_support": len(scene_support.get(class_id, set())),
            }
        )
    rows.sort(key=lambda item: (-int(item["total_count"]), int(item["class_id"])))
    return rows


def compute_actual_ratio(split_totals: dict[str, int]) -> dict[str, float]:
    total = sum(split_totals.values())
    return {split: (count / total if total else 0.0) for split, count in split_totals.items()}


def leak_checks(retained_records: list[dict[str, object]]) -> dict[str, int]:
    sample_to_splits: dict[str, set[str]] = defaultdict(set)
    scene_to_splits: dict[str, set[str]] = defaultdict(set)
    for record in retained_records:
        sample_to_splits[str(record["sample_id"])].add(str(record["split"]))
        scene_to_splits[str(record["scene_id"])].add(str(record["split"]))

    duplicate_samples = sum(1 for splits in sample_to_splits.values() if len(splits) > 1)
    duplicate_scenes = sum(1 for splits in scene_to_splits.values() if len(splits) > 1)
    return {
        "duplicate_sample_ids_across_splits": duplicate_samples,
        "duplicate_scene_ids_across_splits": duplicate_scenes,
    }


def build_summary(
    output_info: dict[str, str],
    raw_summary: dict[str, object],
    prepared_root: Path,
    total_mask_count: int,
    background_only_tile_count: int,
    labels: list[str],
    before_rows: list[dict[str, object]],
    after_rows: list[dict[str, object]],
    excluded_rows: list[dict[str, object]],
    retained_records: list[dict[str, object]],
    foreground_bearing_sample_count: int,
    adopted_threshold: int,
    threshold_reason: str,
    split_to_scenes: dict[str, list[str]],
    split_counts: dict[str, Counter],
    scene_primary_support: dict[int, int],
    preferred_threshold: int,
    fallback_threshold: int,
    absolute_min_threshold: int,
    seed: int,
) -> dict[str, object]:
    split_sample_totals = {
        split: int(sum(split_counts[split].values())) for split in ("train", "val", "test")
    }
    split_ratios = compute_actual_ratio(split_sample_totals)

    retained_class_ids = [int(row["class_id"]) for row in after_rows]
    retained_class_names = [str(row["class_name"]) for row in after_rows]

    return {
        **output_info,
        "raw_summary": raw_summary,
        "prepared_root": str(prepared_root),
        "prepared_tile_count": total_mask_count,
        "background_only_tile_count": background_only_tile_count,
        "foreground_bearing_tile_count": foreground_bearing_sample_count,
        "candidate_label_names": labels[1:],
        "existing_dorgm_logic": {
            "entrypoint": "<PROJECT_ROOT>/DORGM/main.py",
            "logic": (
                "The existing DORGM entry is an unsupervised crack-image clustering script "
                "using grayscale features + GMM/SpectralClustering. It is not a semantic "
                "GID15 labeler, so the new builder keeps it untouched and uses GID15 masks "
                "for deterministic dominant-class assignment instead."
            ),
        },
        "classification_logic": {
            "sample_unit": "640x640 prepared GID15 tile",
            "class_assignment": "dominant non-background class by mask pixel count",
            "background_handling": (
                "Mask id 0 (background) is not treated as a classification target. "
                "GID15 candidate classes come from the 15 foreground land-cover labels."
            ),
        },
        "threshold_selection": {
            "preferred_threshold": preferred_threshold,
            "fallback_threshold": fallback_threshold,
            "absolute_min_threshold": absolute_min_threshold,
            "adopted_threshold": adopted_threshold,
            "reason": threshold_reason,
        },
        "split_strategy": {
            "seed": seed,
            "requested_ratio": DEFAULT_RATIOS,
            "strategy": (
                "Scene-level split with deterministic per-primary-class shuffling so that "
                "patches from the same source scene never cross train/val/test."
            ),
            "scene_counts": {split: len(split_to_scenes[split]) for split in ("train", "val", "test")},
            "sample_counts": split_sample_totals,
            "sample_ratios": split_ratios,
        },
        "retained_class_count": len(retained_class_ids),
        "retained_class_names": retained_class_names,
        "excluded_class_count": len(excluded_rows),
        "retained_sample_count": len(retained_records),
        "excluded_sample_count": foreground_bearing_sample_count - len(retained_records),
        "scene_primary_support": {
            labels[class_id]: count for class_id, count in sorted(scene_primary_support.items())
        },
        "class_count_before": before_rows,
        "class_count_after": after_rows,
        "excluded_classes": excluded_rows,
        "leak_checks": leak_checks(retained_records),
    }


def build_readme(summary: dict[str, object]) -> str:
    retained_names = ", ".join(summary["retained_class_names"])
    threshold = summary["threshold_selection"]["adopted_threshold"]
    split_counts = summary["split_strategy"]["sample_counts"]
    split_ratios = summary["split_strategy"]["sample_ratios"]
    raw_summary = summary["raw_summary"]

    return f"""# GID15 Split Classification Dataset

## Output Location

- Requested path: `{summary["requested_output_root"]}`
- Actual path on this host: `{summary["resolved_output_root"]}`
- Path note: {summary["output_resolution_note"]}

## Source Data

- Raw GID15 root: `{raw_summary["raw_root"]}`
- Prepared tile root: `{summary["prepared_root"]}`
- Raw layout: {raw_summary["raw_layout"]}
- Raw labeled scenes: {raw_summary["raw_labeled_scene_count"]}
- Raw official unlabeled test images: {raw_summary["raw_unlabeled_test_image_count"]}
- Prepared tiles scanned: {summary["prepared_tile_count"]}
- Background-only prepared tiles skipped from candidate classes: {summary["background_only_tile_count"]}
- Foreground-bearing prepared tiles considered for class counting: {summary["foreground_bearing_tile_count"]}

## Existing DORGM Logic

- Entrypoint: `{summary["existing_dorgm_logic"]["entrypoint"]}`
- Note: {summary["existing_dorgm_logic"]["logic"]}

## Classification Logic Used Here

- Sample unit: {summary["classification_logic"]["sample_unit"]}
- Assignment rule: {summary["classification_logic"]["class_assignment"]}
- Background handling: {summary["classification_logic"]["background_handling"]}

## Threshold Decision

- Preferred threshold: >= {summary["threshold_selection"]["preferred_threshold"]}
- Fallback threshold: >= {summary["threshold_selection"]["fallback_threshold"]}
- Hard minimum: >= {summary["threshold_selection"]["absolute_min_threshold"]}
- Adopted threshold: >= {threshold}
- Reason: {summary["threshold_selection"]["reason"]}

## Final Retained Classes

- Count: {summary["retained_class_count"]}
- Classes: {retained_names}

## Split Strategy

- Seed: {summary["split_strategy"]["seed"]}
- Strategy: {summary["split_strategy"]["strategy"]}
- Requested ratio: train={DEFAULT_RATIOS["train"]:.2f}, val={DEFAULT_RATIOS["val"]:.2f}, test={DEFAULT_RATIOS["test"]:.2f}
- Actual sample counts: train={split_counts["train"]}, val={split_counts["val"]}, test={split_counts["test"]}
- Actual sample ratios: train={split_ratios["train"]:.4f}, val={split_ratios["val"]:.4f}, test={split_ratios["test"]:.4f}
- Scene leakage check: duplicate scene ids across splits = {summary["leak_checks"]["duplicate_scene_ids_across_splits"]}
- Sample leakage check: duplicate sample ids across splits = {summary["leak_checks"]["duplicate_sample_ids_across_splits"]}

## Output Structure

```
{summary["resolved_output_root"]}/
    train/<class_name>/*.tif
    val/<class_name>/*.tif
    test/<class_name>/*.tif
    meta/
        class_count_before.csv
        class_count_after.csv
        excluded_classes.csv
        split_manifest.csv
        split_summary.json
        README.md
```

## Notes

- Images are copied, not moved or linked, so the new dataset can be used independently.
- Classes below the adopted threshold are excluded instead of being merged into unrelated semantics.
- Background-only tiles are reported separately and are not counted as candidate class samples.
- The official raw `img_dir/test` images are unlabeled, so they are reported but not used in this supervised classification dataset.
"""


def main() -> int:
    args = parse_args()

    raw_root = resolve_existing_root(args.raw_root, DEFAULT_RAW_CANDIDATES, "raw GID15 root")
    prepared_root = resolve_existing_root(
        args.prepared_root, DEFAULT_PREPARED_CANDIDATES, "prepared GID15 tile root"
    )
    output_root, output_info = resolve_output_root(args.output_root)

    log(f"[info] Raw GID15 root: {raw_root}")
    log(f"[info] Prepared GID15 tile root: {prepared_root}")
    log(f"[info] Requested output root: {output_info['requested_output_root']}")
    log(f"[info] Resolved output root: {output_info['resolved_output_root']}")

    ensure_output_root(output_root, args.force)
    meta_dir = output_root / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    raw_summary = scan_raw_dataset(raw_root)
    labels = load_labels(prepared_root)

    log("[scan] Scanning prepared tiles and assigning dominant foreground classes...")
    (
        records,
        class_counts,
        dominant_any_counts,
        scene_support,
        total_mask_count,
        background_only_tile_count,
    ) = scan_prepared_tiles(prepared_root, labels)
    before_rows = build_before_rows(labels, class_counts, dominant_any_counts, scene_support)

    log(f"[scan] Total prepared tiles scanned: {total_mask_count}")
    log(f"[scan] Background-only tiles skipped: {background_only_tile_count}")
    log(f"[scan] Foreground-bearing candidate samples: {len(records)}")
    for row in before_rows:
        log(
            f"[scan] {row['class_name']}: samples={row['sample_count']}, "
            f"scene_support={row['scene_support']}"
        )

    adopted_threshold, threshold_reason = choose_threshold(
        before_rows,
        args.preferred_threshold,
        args.fallback_threshold,
        args.absolute_min_threshold,
    )
    retained_class_ids = [
        int(row["class_id"]) for row in before_rows if int(row["sample_count"]) >= adopted_threshold
    ]
    retained_class_names = [labels[class_id] for class_id in retained_class_ids]
    excluded_rows = build_excluded_rows(
        before_rows,
        adopted_threshold,
        args.fallback_threshold,
        args.absolute_min_threshold,
    )

    log(f"[threshold] Adopted threshold: >= {adopted_threshold}")
    log(f"[threshold] Reason: {threshold_reason}")
    log(f"[threshold] Retained {len(retained_class_ids)} classes: {', '.join(retained_class_names)}")

    scene_to_split, split_to_scenes, scene_primary_support = split_scenes(
        records,
        retained_class_ids,
        DEFAULT_RATIOS,
        args.seed,
    )
    repair_missing_classes(records, retained_class_ids, scene_to_split, split_to_scenes)

    retained_set = set(retained_class_ids)
    split_counts = compute_split_class_counts(records, retained_set, scene_to_split)
    after_rows = build_after_rows(labels, retained_class_ids, split_counts, scene_support)

    retained_records = build_retained_records(records, retained_set, scene_to_split)
    log(f"[split] Retained sample count: {len(retained_records)}")
    for split in ("train", "val", "test"):
        log(
            f"[split] {split}: scenes={len(split_to_scenes[split])}, "
            f"samples={sum(split_counts[split].values())}"
        )

    copy_dataset(retained_records, output_root, retained_class_names, args.copy_workers)

    before_fields = ["class_id", "class_name", "sample_count", "scene_support", "dominant_overall_count"]
    after_fields = ["class_id", "class_name", "total_count", "train_count", "val_count", "test_count", "scene_support"]
    excluded_fields = ["class_id", "class_name", "sample_count", "scene_support", "reason"]
    manifest_fields = [
        "sample_id",
        "scene_id",
        "source_split",
        "src_image",
        "src_mask",
        "assigned_class_id",
        "assigned_class_name",
        "dominant_pixels",
        "foreground_pixels",
        "total_pixels",
        "dominant_foreground_ratio",
        "dominant_total_ratio",
        "split",
        "output_image",
    ]

    write_csv(meta_dir / "class_count_before.csv", before_rows, before_fields)
    write_csv(meta_dir / "class_count_after.csv", after_rows, after_fields)
    write_csv(meta_dir / "excluded_classes.csv", excluded_rows, excluded_fields)
    write_csv(meta_dir / "split_manifest.csv", retained_records, manifest_fields)

    summary = build_summary(
        output_info=output_info,
        raw_summary=raw_summary,
        prepared_root=prepared_root,
        total_mask_count=total_mask_count,
        background_only_tile_count=background_only_tile_count,
        labels=labels,
        before_rows=before_rows,
        after_rows=after_rows,
        excluded_rows=excluded_rows,
        retained_records=retained_records,
        foreground_bearing_sample_count=len(records),
        adopted_threshold=adopted_threshold,
        threshold_reason=threshold_reason,
        split_to_scenes=split_to_scenes,
        split_counts=split_counts,
        scene_primary_support=scene_primary_support,
        preferred_threshold=args.preferred_threshold,
        fallback_threshold=args.fallback_threshold,
        absolute_min_threshold=args.absolute_min_threshold,
        seed=args.seed,
    )

    (meta_dir / "split_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (meta_dir / "README.md").write_text(build_readme(summary), encoding="utf-8")

    log(f"[done] Dataset created at: {output_root}")
    log(f"[done] Metadata written to: {meta_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
