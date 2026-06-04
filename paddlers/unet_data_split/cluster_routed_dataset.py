#!/usr/bin/env python3

from __future__ import annotations

import csv
import os
from dataclasses import asdict, dataclass
from pathlib import Path

from metrics_utils import ensure_dir, write_json


@dataclass(frozen=True)
class RoutedSample:
    sample_id: str
    scene_id: str
    split: str
    raw_cluster_id: int
    raw_cluster_name: str
    cluster_id: int
    cluster_name: str
    image_path: str
    mask_path: str
    input_image: str
    input_mask: str
    output_image: str
    output_mask: str


def read_label_names(label_path: Path) -> list[str]:
    if not label_path.exists():
        return []
    with label_path.open("r", encoding="utf-8") as file:
        return [line.strip() for line in file if line.strip()]


def expand_row_path(path_value: str | None) -> str | None:
    if path_value is None:
        return None
    value = str(path_value).strip()
    if not value:
        return None
    return os.path.expanduser(os.path.expandvars(value))


def resolve_manifest_path(
    raw_value: str | None,
    rel_value: str | None,
    *,
    root_dir: Path | None,
    mapping_dir: Path,
) -> str:
    preferred = expand_row_path(rel_value)
    fallback = expand_row_path(raw_value)
    selected = preferred or fallback
    if selected is None:
        raise ValueError("Manifest row is missing a required path field.")

    path = Path(selected)
    if path.is_absolute():
        return str(path.resolve())
    if root_dir is not None:
        return str((root_dir / path).resolve())
    return str((mapping_dir / path).resolve())


def load_cluster_mapping(
    mapping_path: Path,
    sample_path_source: str = "input",
    cluster_id_handling: str = "route",
    dataset_root: Path | None = None,
    clustered_root: Path | None = None,
) -> list[RoutedSample]:
    if sample_path_source not in {"input", "clustered_output"}:
        raise ValueError(
            f"sample_path_source must be 'input' or 'clustered_output', got {sample_path_source}"
        )
    if cluster_id_handling not in {"route", "ignore"}:
        raise ValueError(
            f"cluster_id_handling must be 'route' or 'ignore', got {cluster_id_handling}"
        )

    samples: list[RoutedSample] = []
    mapping_dir = mapping_path.parent.resolve()
    with mapping_path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            input_image = resolve_manifest_path(
                row.get("input_image"),
                row.get("input_image_rel"),
                root_dir=dataset_root,
                mapping_dir=mapping_dir,
            )
            input_mask = resolve_manifest_path(
                row.get("input_mask"),
                row.get("input_mask_rel"),
                root_dir=dataset_root,
                mapping_dir=mapping_dir,
            )
            output_image = resolve_manifest_path(
                row.get("output_image"),
                row.get("output_image_rel") or row.get("cluster_image_rel"),
                root_dir=clustered_root,
                mapping_dir=mapping_dir,
            )
            output_mask = resolve_manifest_path(
                row.get("output_mask"),
                row.get("output_mask_rel") or row.get("cluster_mask_rel"),
                root_dir=clustered_root,
                mapping_dir=mapping_dir,
            )

            if sample_path_source == "input":
                image_path = input_image
                mask_path = input_mask
            else:
                image_path = output_image
                mask_path = output_mask

            raw_cluster_id = int(row["cluster_id"])
            raw_cluster_name = str(row["cluster_name"])
            if cluster_id_handling == "ignore":
                cluster_id = 0
                cluster_name = "shared_head"
            else:
                cluster_id = raw_cluster_id
                cluster_name = raw_cluster_name

            samples.append(
                RoutedSample(
                    sample_id=str(row["sample_id"]),
                    scene_id=str(row["scene_id"]),
                    split=str(row["split"]),
                    raw_cluster_id=raw_cluster_id,
                    raw_cluster_name=raw_cluster_name,
                    cluster_id=cluster_id,
                    cluster_name=cluster_name,
                    image_path=image_path,
                    mask_path=mask_path,
                    input_image=input_image,
                    input_mask=input_mask,
                    output_image=output_image,
                    output_mask=output_mask,
                )
            )

    if not samples:
        raise RuntimeError(f"No routed samples loaded from {mapping_path}")
    return samples


def split_samples(samples: list[RoutedSample]) -> dict[str, list[RoutedSample]]:
    split_map = {"train": [], "val": [], "test": []}
    for sample in samples:
        split_map.setdefault(sample.split, []).append(sample)
    return split_map


def summarize_routed_samples(samples: list[RoutedSample]) -> dict[str, object]:
    split_map = split_samples(samples)
    cluster_names = sorted({sample.cluster_name for sample in samples})
    raw_cluster_names = sorted({sample.raw_cluster_name for sample in samples})

    cluster_split_counts = {
        cluster_name: {"train": 0, "val": 0, "test": 0} for cluster_name in cluster_names
    }
    raw_cluster_split_counts = {
        cluster_name: {"train": 0, "val": 0, "test": 0} for cluster_name in raw_cluster_names
    }

    duplicate_sample_ids = {}
    sample_to_splits = {}
    for sample in samples:
        sample_to_splits.setdefault(sample.sample_id, set()).add(sample.split)
        cluster_split_counts[sample.cluster_name][sample.split] += 1
        raw_cluster_split_counts[sample.raw_cluster_name][sample.split] += 1

    for sample_id, splits in sample_to_splits.items():
        if len(splits) > 1:
            duplicate_sample_ids[sample_id] = sorted(splits)

    return {
        "num_samples": len(samples),
        "split_counts": {split: len(rows) for split, rows in split_map.items()},
        "cluster_names": cluster_names,
        "raw_cluster_names": raw_cluster_names,
        "cluster_total_counts": {
            cluster_name: int(sum(cluster_split_counts[cluster_name].values()))
            for cluster_name in cluster_names
        },
        "raw_cluster_total_counts": {
            cluster_name: int(sum(raw_cluster_split_counts[cluster_name].values()))
            for cluster_name in raw_cluster_names
        },
        "cluster_split_counts": cluster_split_counts,
        "raw_cluster_split_counts": raw_cluster_split_counts,
        "duplicate_sample_ids_across_splits": duplicate_sample_ids,
    }


def validate_routed_samples(samples: list[RoutedSample], expected_num_heads: int) -> dict[str, object]:
    summary = summarize_routed_samples(samples)
    observed_cluster_ids = sorted({sample.cluster_id for sample in samples})
    observed_raw_cluster_ids = sorted({sample.raw_cluster_id for sample in samples})
    expected_cluster_ids = list(range(expected_num_heads))

    missing_files = []
    for sample in samples:
        if not Path(sample.image_path).is_file():
            missing_files.append({"type": "image", "path": sample.image_path, "sample_id": sample.sample_id})
        if not Path(sample.mask_path).is_file():
            missing_files.append({"type": "mask", "path": sample.mask_path, "sample_id": sample.sample_id})

    summary.update(
        {
            "observed_cluster_ids": observed_cluster_ids,
            "observed_raw_cluster_ids": observed_raw_cluster_ids,
            "expected_cluster_ids": expected_cluster_ids,
            "missing_cluster_ids": sorted(set(expected_cluster_ids) - set(observed_cluster_ids)),
            "unexpected_cluster_ids": sorted(set(observed_cluster_ids) - set(expected_cluster_ids)),
            "missing_files": missing_files[:50],
            "all_clusters_have_train_val_test": all(
                count > 0
                for cluster_name in summary["cluster_names"]
                for count in summary["cluster_split_counts"][cluster_name].values()
            ),
        }
    )
    return summary


def export_routing_snapshot(output_path: Path, samples: list[RoutedSample], expected_num_heads: int) -> dict[str, object]:
    payload = validate_routed_samples(samples, expected_num_heads)
    ensure_dir(output_path.parent)
    write_json(output_path, payload)
    return payload


def samples_to_rows(samples: list[RoutedSample]) -> list[dict[str, object]]:
    return [asdict(sample) for sample in samples]
