#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import shutil
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image


Image.MAX_IMAGE_PIXELS = None

LABELS = [
    "background",
    "industrial_land",
    "urban_residential",
    "rural_residential",
    "traffic_land",
    "paddy_field",
    "irrigated_land",
    "dry_cropland",
    "garden_plot",
    "arbor_woodland",
    "shrub_land",
    "natural_grassland",
    "artificial_grassland",
    "river",
    "lake",
    "pond",
]

DEFAULT_RAW_CANDIDATES = [
    "${GID15_DATASET_ROOT}/gid-15/GID",
    "${NAS_ROOT}/gid-15/GID",
]
DEFAULT_OUTPUT_ROOT = "${GID15_DATASET_ROOT}/gid15_dataset"
DEFAULT_SEED = 20260403
DEFAULT_TILE_SIZE = 640
DEFAULT_STRIDE = 640
DEFAULT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}


@dataclass(frozen=True)
class SceneRecord:
    scene_id: str
    source_split: str
    src_image: str
    src_mask: str
    width: int
    height: int
    class_ids: tuple[int, ...]


def log(message: str) -> None:
    print(message, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a standard GID15 semantic-segmentation dataset by re-splitting the "
            "110 labeled scenes at scene level and then cutting them into image/mask tiles."
        )
    )
    parser.add_argument("--raw-root", default=None, help="Raw GID15 root (img_dir/ann_dir).")
    parser.add_argument(
        "--output-root",
        default=DEFAULT_OUTPUT_ROOT,
        help="Requested output root. /public/... is auto-mapped to ${NAS_ROOT}/... on this host.",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed.")
    parser.add_argument("--tile-size", type=int, default=DEFAULT_TILE_SIZE)
    parser.add_argument("--stride", type=int, default=DEFAULT_STRIDE)
    parser.add_argument("--train-ratio", type=float, default=DEFAULT_RATIOS["train"])
    parser.add_argument("--val-ratio", type=float, default=DEFAULT_RATIOS["val"])
    parser.add_argument("--test-ratio", type=float, default=DEFAULT_RATIOS["test"])
    parser.add_argument(
        "--filter-pure-background",
        action="store_true",
        help="Drop tiles whose mask contains only class 0.",
    )
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

    resolved = requested.expanduser().resolve()
    return resolved, {
        "requested_output_root": str(requested),
        "resolved_output_root": str(resolved),
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


def validate_ratios(train_ratio: float, val_ratio: float, test_ratio: float) -> dict[str, float]:
    ratios = {"train": train_ratio, "val": val_ratio, "test": test_ratio}
    total = sum(ratios.values())
    if any(value <= 0 for value in ratios.values()):
        raise ValueError(f"Ratios must all be > 0, got: {ratios}")
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(f"Ratios must sum to 1.0, got {ratios} (sum={total})")
    return ratios


def allocate_counts(size: int, ratios: dict[str, float]) -> dict[str, int]:
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


def load_scene_records(raw_root: Path) -> list[SceneRecord]:
    img_dir = raw_root / "img_dir"
    ann_dir = raw_root / "ann_dir"
    if not img_dir.is_dir() or not ann_dir.is_dir():
        raise RuntimeError(f"Raw root {raw_root} does not look like GID15. Expected img_dir/ and ann_dir/.")

    records: list[SceneRecord] = []
    for source_split in ("train", "val"):
        split_img_dir = img_dir / source_split
        split_mask_dir = ann_dir / source_split
        for img_path in sorted(split_img_dir.glob("*.tif")):
            scene_id = img_path.stem
            mask_path = split_mask_dir / f"{scene_id}_15label.png"
            if not mask_path.is_file():
                raise FileNotFoundError(f"Missing training mask for scene {scene_id}: {mask_path}")

            with Image.open(img_path) as image:
                width, height = image.size
            with Image.open(mask_path) as mask:
                if mask.mode != "L":
                    raise RuntimeError(f"Expected grayscale class-id mask at {mask_path}, got mode={mask.mode}")
                if mask.size != (width, height):
                    raise RuntimeError(
                        f"Image/mask size mismatch for {scene_id}: image={(width, height)} mask={mask.size}"
                    )
                class_ids = tuple(int(v) for v in np.unique(np.array(mask, dtype=np.uint8)).tolist())

            unexpected_ids = [class_id for class_id in class_ids if class_id < 0 or class_id >= len(LABELS)]
            if unexpected_ids:
                raise RuntimeError(f"Scene {scene_id} contains unexpected class ids: {unexpected_ids}")

            records.append(
                SceneRecord(
                    scene_id=scene_id,
                    source_split=source_split,
                    src_image=str(img_path.resolve()),
                    src_mask=str(mask_path.resolve()),
                    width=width,
                    height=height,
                    class_ids=class_ids,
                )
            )

    if len(records) != 110:
        raise RuntimeError(f"Expected 110 labeled scenes from raw train+val, got {len(records)}")
    return records


def split_scene_records(
    scenes: list[SceneRecord], ratios: dict[str, float], seed: int
) -> tuple[dict[str, list[SceneRecord]], dict[str, int]]:
    items = sorted(scenes, key=lambda item: item.scene_id)
    random.Random(seed).shuffle(items)
    counts = allocate_counts(len(items), ratios)

    split_map: dict[str, list[SceneRecord]] = {}
    cursor = 0
    for split in ("train", "val", "test"):
        next_cursor = cursor + counts[split]
        split_map[split] = items[cursor:next_cursor]
        cursor = next_cursor

    return split_map, counts


def sliding_positions(size: int, tile_size: int, stride: int) -> list[int]:
    if size <= tile_size:
        return [0]

    count = int(math.ceil((size - tile_size) / stride)) + 1
    values = []
    for index in range(count):
        end = min(index * stride + tile_size, size)
        start = max(end - tile_size, 0)
        values.append(start)

    deduped: list[int] = []
    seen: set[int] = set()
    for value in values:
        if value not in seen:
            deduped.append(value)
            seen.add(value)
    return deduped


def ensure_dirs(output_root: Path) -> None:
    for split in ("train", "val", "test"):
        (output_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_root / "masks" / split).mkdir(parents=True, exist_ok=True)
    (output_root / "lists").mkdir(parents=True, exist_ok=True)
    (output_root / "manifests").mkdir(parents=True, exist_ok=True)
    (output_root / "stats").mkdir(parents=True, exist_ok=True)


def write_labels(output_root: Path) -> None:
    (output_root / "labels.txt").write_text("\n".join(LABELS) + "\n", encoding="utf-8")


def crop_split(
    split: str,
    scenes: list[SceneRecord],
    output_root: Path,
    tile_size: int,
    stride: int,
    filter_pure_background: bool,
) -> tuple[list[str], list[dict[str, object]], dict[str, object]]:
    list_entries: list[str] = []
    manifest_rows: list[dict[str, object]] = []
    split_stats = {
        "scene_count": len(scenes),
        "tile_count": 0,
        "generated_tile_count_before_filter": 0,
        "pure_background_tile_count": 0,
        "source_split_histogram": dict(Counter(scene.source_split for scene in scenes)),
        "scene_ids": sorted(scene.scene_id for scene in scenes),
    }

    for scene_index, scene in enumerate(sorted(scenes, key=lambda item: item.scene_id), start=1):
        log(f"[crop] split={split} scene={scene.scene_id} ({scene_index}/{len(scenes)})")
        image_path = Path(scene.src_image)
        mask_path = Path(scene.src_mask)

        with Image.open(image_path) as image, Image.open(mask_path) as mask:
            if image.size != mask.size:
                raise RuntimeError(f"Image/mask size mismatch while cropping {scene.scene_id}")

            xs = sliding_positions(scene.width, tile_size, stride)
            ys = sliding_positions(scene.height, tile_size, stride)
            for row_offset in ys:
                for col_offset in xs:
                    split_stats["generated_tile_count_before_filter"] += 1
                    sample_id = f"{scene.scene_id}_{row_offset}_{col_offset}"
                    box = (col_offset, row_offset, col_offset + tile_size, row_offset + tile_size)
                    image_crop = image.crop(box)
                    mask_crop = mask.crop(box)
                    mask_array = np.array(mask_crop, dtype=np.uint8)
                    unique_ids = [int(v) for v in np.unique(mask_array).tolist()]
                    pure_background = len(unique_ids) == 1 and unique_ids[0] == 0

                    if pure_background:
                        split_stats["pure_background_tile_count"] += 1
                        if filter_pure_background:
                            continue

                    image_rel = Path("images") / split / f"{sample_id}.tif"
                    mask_rel = Path("masks") / split / f"{sample_id}.png"
                    image_out = output_root / image_rel
                    mask_out = output_root / mask_rel

                    image_crop.save(image_out)
                    mask_crop.save(mask_out)

                    list_entries.append(f"{image_rel.as_posix()} {mask_rel.as_posix()}")
                    manifest_rows.append(
                        {
                            "sample_id": sample_id,
                            "split": split,
                            "scene_id": scene.scene_id,
                            "source_split": scene.source_split,
                            "row_offset": row_offset,
                            "col_offset": col_offset,
                            "tile_size": tile_size,
                            "src_image": scene.src_image,
                            "src_mask": scene.src_mask,
                            "output_image": str(image_out),
                            "output_mask": str(mask_out),
                            "image_rel": image_rel.as_posix(),
                            "mask_rel": mask_rel.as_posix(),
                            "pure_background": pure_background,
                            "mask_unique_ids": ";".join(str(v) for v in unique_ids),
                        }
                    )
                    split_stats["tile_count"] += 1

    list_entries.sort()
    manifest_rows.sort(key=lambda row: (str(row["split"]), str(row["sample_id"])))
    return list_entries, manifest_rows, split_stats


def write_list(path: Path, rows: list[str]) -> None:
    path.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_pair_checks(output_root: Path, split_manifest_rows: list[dict[str, object]]) -> dict[str, object]:
    sample_to_splits: dict[str, set[str]] = {}
    scene_to_splits: dict[str, set[str]] = {}
    missing_image_count = 0
    missing_mask_count = 0

    for row in split_manifest_rows:
        sample_id = str(row["sample_id"])
        split = str(row["split"])
        scene_id = str(row["scene_id"])
        sample_to_splits.setdefault(sample_id, set()).add(split)
        scene_to_splits.setdefault(scene_id, set()).add(split)
        if not Path(str(row["output_image"])).is_file():
            missing_image_count += 1
        if not Path(str(row["output_mask"])).is_file():
            missing_mask_count += 1

    per_split = {}
    for split in ("train", "val", "test"):
        image_paths = sorted((output_root / "images" / split).glob("*.tif"))
        mask_paths = sorted((output_root / "masks" / split).glob("*.png"))
        list_path = output_root / "lists" / f"{split}.txt"
        list_count = sum(1 for _ in list_path.open(encoding="utf-8")) if list_path.is_file() else 0
        image_stems = {path.stem for path in image_paths}
        mask_stems = {path.stem for path in mask_paths}
        per_split[split] = {
            "image_count": len(image_paths),
            "mask_count": len(mask_paths),
            "list_count": list_count,
            "image_mask_stem_mismatch_count": len(image_stems ^ mask_stems),
            "images_missing_mask": sorted(image_stems - mask_stems)[:20],
            "masks_missing_image": sorted(mask_stems - image_stems)[:20],
        }

    duplicate_sample_ids = sum(1 for splits in sample_to_splits.values() if len(splits) > 1)
    duplicate_scene_ids = sum(1 for splits in scene_to_splits.values() if len(splits) > 1)
    return {
        "duplicate_sample_ids_across_splits": duplicate_sample_ids,
        "duplicate_scene_ids_across_splits": duplicate_scene_ids,
        "missing_manifest_image_files": missing_image_count,
        "missing_manifest_mask_files": missing_mask_count,
        "per_split": per_split,
    }


def build_scene_split_manifest(
    split_to_scenes: dict[str, list[SceneRecord]],
    counts: dict[str, int],
    seed: int,
    ratios: dict[str, float],
) -> dict[str, object]:
    return {
        "seed": seed,
        "target_scene_ratios": ratios,
        "actual_scene_counts": counts,
        "splits": {
            split: [asdict(scene) for scene in sorted(scenes, key=lambda item: item.scene_id)]
            for split, scenes in split_to_scenes.items()
        },
    }


def build_summary(
    raw_root: Path,
    output_info: dict[str, str],
    split_to_scenes: dict[str, list[SceneRecord]],
    counts: dict[str, int],
    split_stats: dict[str, dict[str, object]],
    pair_checks: dict[str, object],
    tile_size: int,
    stride: int,
    filter_pure_background: bool,
    seed: int,
    ratios: dict[str, float],
) -> dict[str, object]:
    actual_scene_ratios = {
        split: (counts[split] / sum(counts.values()) if counts else 0.0) for split in ("train", "val", "test")
    }
    class_coverage = {}
    for split in ("train", "val", "test"):
        class_ids = sorted({class_id for scene in split_to_scenes[split] for class_id in scene.class_ids})
        class_coverage[split] = {
            "class_ids": class_ids,
            "class_names": [LABELS[class_id] for class_id in class_ids],
        }

    return {
        **output_info,
        "raw_root": str(raw_root),
        "seed": seed,
        "target_scene_ratios": ratios,
        "actual_scene_counts": counts,
        "actual_scene_ratios": actual_scene_ratios,
        "tile_size": tile_size,
        "stride": stride,
        "filter_pure_background_tiles": filter_pure_background,
        "labels": LABELS,
        "labeled_source_scene_count": sum(counts.values()),
        "source_scene_size": {"width": 7200, "height": 6800},
        "class_coverage_by_split": class_coverage,
        "per_split": split_stats,
        "pair_checks": pair_checks,
    }


def main() -> None:
    args = parse_args()
    ratios = validate_ratios(args.train_ratio, args.val_ratio, args.test_ratio)
    if args.tile_size <= 0 or args.stride <= 0:
        raise ValueError(f"tile_size and stride must be > 0, got {args.tile_size}, {args.stride}")
    if args.stride > args.tile_size:
        raise ValueError(f"stride must be <= tile_size to avoid gaps, got {args.stride} > {args.tile_size}")

    raw_root = resolve_existing_root(args.raw_root, DEFAULT_RAW_CANDIDATES, "raw GID15 root")
    output_root, output_info = resolve_output_root(args.output_root)

    log(f"[info] Raw root: {raw_root}")
    log(f"[info] Requested output root: {output_info['requested_output_root']}")
    log(f"[info] Resolved output root: {output_info['resolved_output_root']}")
    log(f"[info] Seed: {args.seed}")
    log(f"[info] Tile size: {args.tile_size}")
    log(f"[info] Stride: {args.stride}")
    log(f"[info] Filter pure background tiles: {args.filter_pure_background}")
    log(f"[info] Target scene ratios: {ratios}")

    ensure_output_root(output_root, args.force)
    ensure_dirs(output_root)
    write_labels(output_root)

    scenes = load_scene_records(raw_root)
    split_to_scenes, counts = split_scene_records(scenes, ratios, args.seed)
    log(f"[split] Scene counts: {counts}")

    scene_split_manifest = build_scene_split_manifest(split_to_scenes, counts, args.seed, ratios)
    (output_root / "manifests" / "scene_split.json").write_text(
        json.dumps(scene_split_manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    all_manifest_rows: list[dict[str, object]] = []
    split_stats: dict[str, dict[str, object]] = {}
    for split in ("train", "val", "test"):
        list_entries, manifest_rows, stats = crop_split(
            split=split,
            scenes=split_to_scenes[split],
            output_root=output_root,
            tile_size=args.tile_size,
            stride=args.stride,
            filter_pure_background=args.filter_pure_background,
        )
        write_list(output_root / "lists" / f"{split}.txt", list_entries)
        all_manifest_rows.extend(manifest_rows)
        split_stats[split] = stats
        log(
            f"[done] split={split} scenes={stats['scene_count']} tiles={stats['tile_count']} "
            f"pure_background_tiles={stats['pure_background_tile_count']}"
        )

    manifest_fields = [
        "sample_id",
        "split",
        "scene_id",
        "source_split",
        "row_offset",
        "col_offset",
        "tile_size",
        "src_image",
        "src_mask",
        "output_image",
        "output_mask",
        "image_rel",
        "mask_rel",
        "pure_background",
        "mask_unique_ids",
    ]
    write_csv(output_root / "manifests" / "tile_manifest.csv", all_manifest_rows, manifest_fields)

    pair_checks = build_pair_checks(output_root, all_manifest_rows)
    (output_root / "manifests" / "pair_checks.json").write_text(
        json.dumps(pair_checks, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    summary = build_summary(
        raw_root=raw_root,
        output_info=output_info,
        split_to_scenes=split_to_scenes,
        counts=counts,
        split_stats=split_stats,
        pair_checks=pair_checks,
        tile_size=args.tile_size,
        stride=args.stride,
        filter_pure_background=args.filter_pure_background,
        seed=args.seed,
        ratios=ratios,
    )
    (output_root / "stats" / "dataset_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    log("[complete] Dataset build finished.")
    log(f"[complete] Scene manifest: {output_root / 'manifests' / 'scene_split.json'}")
    log(f"[complete] Tile manifest: {output_root / 'manifests' / 'tile_manifest.csv'}")
    log(f"[complete] Summary: {output_root / 'stats' / 'dataset_summary.json'}")


if __name__ == "__main__":
    main()
