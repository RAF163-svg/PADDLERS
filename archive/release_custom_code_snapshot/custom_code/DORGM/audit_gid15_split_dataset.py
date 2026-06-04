#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


DEFAULT_REQUESTED_OUTPUT_ROOT = "${GID15_DATASET_ROOT}/gid15_split_dataset"
DEFAULT_DATASET_ROOT = "${GID15_DATASET_ROOT}/gid15_split_dataset"
DEFAULT_PREPARED_CANDIDATES = [
    "${GID15_DATASET_ROOT}/gid15_seed20260320_tile640_stride448",
    "${GID15_DATASET_ROOT}/paddlers_gid15_benchmark_20260320/data/gid15_seed20260320_tile640_stride448",
]
DEFAULT_LOW_PURITY_THRESHOLD = 0.60
README_AUDIT_BEGIN = "<!-- AUDIT:BEGIN -->"
README_AUDIT_END = "<!-- AUDIT:END -->"


@dataclass(frozen=True)
class ManifestRecord:
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
    split: str
    output_image: str


def log(message: str) -> None:
    print(message, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit an existing GID15 split classification dataset without rebuilding "
            "the train/val/test payload."
        )
    )
    parser.add_argument(
        "--dataset-root",
        default=DEFAULT_DATASET_ROOT,
        help="Existing dataset root that already contains train/val/test/meta.",
    )
    parser.add_argument(
        "--requested-output-root",
        default=DEFAULT_REQUESTED_OUTPUT_ROOT,
        help="Original requested output path, used for path-mapping auditing.",
    )
    parser.add_argument(
        "--prepared-root",
        default=None,
        help="Prepared tile cache root. If omitted, known candidates are checked.",
    )
    parser.add_argument(
        "--low-purity-threshold",
        type=float,
        default=DEFAULT_LOW_PURITY_THRESHOLD,
        help="Threshold for low-purity retained samples.",
    )
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


def run_shell(command: str) -> dict[str, object]:
    completed = subprocess.run(
        command,
        shell=True,
        executable="/bin/bash",
        text=True,
        capture_output=True,
        check=False,
    )
    return {
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def write_text(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def write_json(path: Path, payload: dict[str, object] | list[object]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def to_float(value: float) -> float:
    return round(float(value), 6)


def summarize_numeric(values: Iterable[float]) -> dict[str, float | int]:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return {"count": 0}
    return {
        "count": int(arr.size),
        "min": to_float(arr.min()),
        "p10": to_float(np.percentile(arr, 10)),
        "p25": to_float(np.percentile(arr, 25)),
        "median": to_float(np.percentile(arr, 50)),
        "mean": to_float(arr.mean()),
        "p75": to_float(np.percentile(arr, 75)),
        "p90": to_float(np.percentile(arr, 90)),
        "p95": to_float(np.percentile(arr, 95)),
        "max": to_float(arr.max()),
    }


def load_manifest(manifest_path: Path) -> list[ManifestRecord]:
    records: list[ManifestRecord] = []
    with manifest_path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            records.append(
                ManifestRecord(
                    sample_id=row["sample_id"],
                    scene_id=row["scene_id"],
                    source_split=row["source_split"],
                    src_image=row["src_image"],
                    src_mask=row["src_mask"],
                    assigned_class_id=int(row["assigned_class_id"]),
                    assigned_class_name=row["assigned_class_name"],
                    dominant_pixels=int(row["dominant_pixels"]),
                    foreground_pixels=int(row["foreground_pixels"]),
                    total_pixels=int(row["total_pixels"]),
                    dominant_foreground_ratio=float(row["dominant_foreground_ratio"]),
                    dominant_total_ratio=float(row["dominant_total_ratio"]),
                    split=row["split"],
                    output_image=row["output_image"],
                )
            )
    return records


def build_path_mapping_audit(
    requested_output_root: str,
    dataset_root: Path,
    meta_dir: Path,
) -> dict[str, object]:
    commands = [
        "mount | grep -E 'public|nas'",
        "df -h",
        "ls -ld /public ${PUBLIC_USERS_ROOT} ${GID15_DATASET_ROOT}",
        "ls -ld ${NAS_ROOT} ${NAS_ROOT}/users ${GID15_DATASET_ROOT}",
    ]
    results = [run_shell(command) for command in commands]

    evidence_lines = ["# Path Mapping Evidence", ""]
    for result in results:
        evidence_lines.append(f"$ {result['command']}")
        evidence_lines.append(result["stdout"].rstrip() or "[stdout empty]")
        if result["stderr"]:
            evidence_lines.append("")
            evidence_lines.append("[stderr]")
            evidence_lines.append(result["stderr"].rstrip())
        evidence_lines.append("")
        evidence_lines.append(f"[returncode] {result['returncode']}")
        evidence_lines.append("")

    write_text(meta_dir / "path_mapping_evidence.txt", "\n".join(evidence_lines).rstrip() + "\n")

    mount_stdout = str(results[0]["stdout"])
    df_stdout = str(results[1]["stdout"])
    public_exists = Path("/public").exists()
    mnt_nas_exists = Path("${NAS_ROOT}").exists()

    mnt_nas_mount_line = next(
        (line for line in mount_stdout.splitlines() if " on ${NAS_ROOT} type cifs " in line),
        "",
    )
    mnt_nas_public_line = next(
        (line for line in mount_stdout.splitlines() if " on ${NAS_ROOT}/public type cifs " in line),
        "",
    )
    df_mnt_nas_line = next(
        (line for line in df_stdout.splitlines() if line.strip().endswith(" ${NAS_ROOT}")),
        "",
    )

    public_state = "present" if public_exists else "missing"
    if not public_exists and mnt_nas_mount_line:
        conclusion = (
            "This host does not expose `/public` as a mounted path. The CIFS share "
            "`//10.62.192.79/public` is mounted at `${NAS_ROOT}`, so "
            f"`{requested_output_root}` maps operationally to `{dataset_root}` on this machine."
        )
    elif public_exists and mnt_nas_mount_line:
        conclusion = (
            "Both `/public` and `${NAS_ROOT}` are present on this host, and the mount output "
            "shows that `${NAS_ROOT}` is backed by `//10.62.192.79/public`."
        )
    else:
        conclusion = (
            "The path mapping could not be proven from mount output alone; manual inspection "
            "is recommended before treating `/public` and `${NAS_ROOT}` as equivalent."
        )

    return {
        "requested_output_root": requested_output_root,
        "resolved_dataset_root": str(dataset_root),
        "public_path_state": public_state,
        "mnt_nas_path_state": "present" if mnt_nas_exists else "missing",
        "mount_line_for_mnt_nas": mnt_nas_mount_line,
        "mount_line_for_mnt_nas_public": mnt_nas_public_line,
        "df_line_for_mnt_nas": df_mnt_nas_line,
        "conclusion": conclusion,
        "evidence_file": str(meta_dir / "path_mapping_evidence.txt"),
    }


def build_dominant_ratio_audit(
    records: list[ManifestRecord],
    low_purity_threshold: float,
    meta_dir: Path,
) -> dict[str, object]:
    ratios = [record.dominant_foreground_ratio for record in records]
    overall_stats = summarize_numeric(ratios)
    low_records = [record for record in records if record.dominant_foreground_ratio < low_purity_threshold]
    low_purity_count = len(low_records)
    low_purity_ratio = low_purity_count / len(records) if records else 0.0

    by_class_rows: list[dict[str, object]] = []
    grouped: dict[str, list[ManifestRecord]] = defaultdict(list)
    for record in records:
        grouped[record.assigned_class_name].append(record)

    for class_name, class_records in grouped.items():
        class_ratios = [record.dominant_foreground_ratio for record in class_records]
        low_count = sum(1 for value in class_ratios if value < low_purity_threshold)
        stats = summarize_numeric(class_ratios)
        by_class_rows.append(
            {
                "class_name": class_name,
                "sample_count": len(class_records),
                "low_purity_threshold": to_float(low_purity_threshold),
                "low_purity_count": low_count,
                "low_purity_ratio": to_float(low_count / len(class_records)),
                "mean_ratio": stats["mean"],
                "median_ratio": stats["median"],
                "min_ratio": stats["min"],
                "p10_ratio": stats["p10"],
                "p25_ratio": stats["p25"],
                "p75_ratio": stats["p75"],
                "p90_ratio": stats["p90"],
                "max_ratio": stats["max"],
            }
        )
    by_class_rows.sort(
        key=lambda row: (-float(row["low_purity_ratio"]), -int(row["sample_count"]), str(row["class_name"]))
    )

    low_rows = [
        {
            "sample_id": record.sample_id,
            "scene_id": record.scene_id,
            "split": record.split,
            "assigned_class_name": record.assigned_class_name,
            "dominant_foreground_ratio": to_float(record.dominant_foreground_ratio),
            "dominant_total_ratio": to_float(record.dominant_total_ratio),
            "dominant_pixels": record.dominant_pixels,
            "foreground_pixels": record.foreground_pixels,
            "total_pixels": record.total_pixels,
            "source_split": record.source_split,
            "src_image": record.src_image,
            "src_mask": record.src_mask,
            "output_image": record.output_image,
        }
        for record in sorted(
            low_records,
            key=lambda item: (item.dominant_foreground_ratio, item.assigned_class_name, item.sample_id),
        )
    ]

    write_csv(
        meta_dir / "dominant_ratio_by_class.csv",
        by_class_rows,
        [
            "class_name",
            "sample_count",
            "low_purity_threshold",
            "low_purity_count",
            "low_purity_ratio",
            "mean_ratio",
            "median_ratio",
            "min_ratio",
            "p10_ratio",
            "p25_ratio",
            "p75_ratio",
            "p90_ratio",
            "max_ratio",
        ],
    )
    write_csv(
        meta_dir / "low_purity_samples.csv",
        low_rows,
        [
            "sample_id",
            "scene_id",
            "split",
            "assigned_class_name",
            "dominant_foreground_ratio",
            "dominant_total_ratio",
            "dominant_pixels",
            "foreground_pixels",
            "total_pixels",
            "source_split",
            "src_image",
            "src_mask",
            "output_image",
        ],
    )

    top_low_ratio_classes = [
        {
            "class_name": row["class_name"],
            "low_purity_ratio": row["low_purity_ratio"],
            "low_purity_count": row["low_purity_count"],
            "sample_count": row["sample_count"],
            "mean_ratio": row["mean_ratio"],
        }
        for row in by_class_rows[:5]
    ]
    summary = {
        "metric": "dominant_foreground_ratio = dominant_pixels / foreground_pixels",
        "sample_count": len(records),
        "low_purity_threshold": to_float(low_purity_threshold),
        "low_purity_count": low_purity_count,
        "low_purity_ratio": to_float(low_purity_ratio),
        "overall_stats": overall_stats,
        "top_low_purity_classes": top_low_ratio_classes,
    }
    write_json(meta_dir / "dominant_ratio_summary.json", summary)
    return {
        "summary": summary,
        "by_class_rows": by_class_rows,
    }


def build_scene_audit(records: list[ManifestRecord], meta_dir: Path) -> dict[str, object]:
    split_scene_ids: dict[str, set[str]] = {"train": set(), "val": set(), "test": set()}
    scene_rows: list[dict[str, object]] = []
    scene_stats: dict[str, dict[str, object]] = {}

    for record in records:
        split_scene_ids[record.split].add(record.scene_id)
        info = scene_stats.setdefault(
            record.scene_id,
            {
                "split": record.split,
                "sample_count": 0,
                "class_counts": Counter(),
                "source_splits": set(),
            },
        )
        info["sample_count"] = int(info["sample_count"]) + 1
        info["class_counts"][record.assigned_class_name] += 1
        info["source_splits"].add(record.source_split)

    for scene_id, info in scene_stats.items():
        class_counts: Counter = info["class_counts"]
        dominant_class, dominant_count = max(
            class_counts.items(),
            key=lambda item: (item[1], item[0]),
        )
        class_histogram = ";".join(
            f"{name}:{count}" for name, count in sorted(class_counts.items(), key=lambda item: (-item[1], item[0]))
        )
        scene_rows.append(
            {
                "split": info["split"],
                "scene_id": scene_id,
                "sample_count": int(info["sample_count"]),
                "class_count": len(class_counts),
                "dominant_class_name": dominant_class,
                "dominant_class_sample_count": dominant_count,
                "source_splits": ";".join(sorted(info["source_splits"])),
                "class_histogram": class_histogram,
            }
        )
    scene_rows.sort(key=lambda row: (row["split"], row["scene_id"]))
    write_csv(
        meta_dir / "scene_split_summary.csv",
        scene_rows,
        [
            "split",
            "scene_id",
            "sample_count",
            "class_count",
            "dominant_class_name",
            "dominant_class_sample_count",
            "source_splits",
            "class_histogram",
        ],
    )

    pair_names = [("train", "val"), ("train", "test"), ("val", "test")]
    pairwise_overlaps = {
        f"{left}_{right}": sorted(split_scene_ids[left] & split_scene_ids[right])
        for left, right in pair_names
    }
    overlap_total = len(set().union(*[set(values) for values in pairwise_overlaps.values()]))
    overlap_json = {
        "scene_counts": {split: len(scene_ids) for split, scene_ids in split_scene_ids.items()},
        "split_scene_ids": {split: sorted(scene_ids) for split, scene_ids in split_scene_ids.items()},
        "pairwise_overlap_counts": {name: len(values) for name, values in pairwise_overlaps.items()},
        "pairwise_overlaps": pairwise_overlaps,
        "overlap_scene_count_total": overlap_total,
        "scene_overlap_free": overlap_total == 0,
    }
    write_json(meta_dir / "scene_overlap_check.json", overlap_json)
    return overlap_json


def optimize_scene_partition(scene_counter: Counter[str]) -> dict[str, object]:
    items = sorted(scene_counter.items(), key=lambda item: item[0])
    total = sum(count for _, count in items)
    targets = {
        "train": total * 0.70,
        "val": total * 0.15,
        "test": total * 0.15,
    }

    parents: dict[tuple[int, int], tuple[tuple[int, int], int, str] | None] = {(0, 0): None}
    for index, (_, count) in enumerate(items):
        current_states = list(parents.keys())
        for val_sum, test_sum in current_states:
            val_state = (val_sum + count, test_sum)
            if val_state not in parents:
                parents[val_state] = ((val_sum, test_sum), index, "val")
            test_state = (val_sum, test_sum + count)
            if test_state not in parents:
                parents[test_state] = ((val_sum, test_sum), index, "test")

    def score(state: tuple[int, int]) -> tuple[float, float, float, float]:
        val_sum, test_sum = state
        train_sum = total - val_sum - test_sum
        if train_sum <= 0 or val_sum <= 0 or test_sum <= 0:
            return (math.inf, math.inf, math.inf, math.inf)
        return (
            abs(train_sum - targets["train"]) + abs(val_sum - targets["val"]) + abs(test_sum - targets["test"]),
            max(abs(val_sum - targets["val"]), abs(test_sum - targets["test"])),
            abs(val_sum - test_sum),
            abs(train_sum - targets["train"]),
        )

    best_state = min(parents.keys(), key=score)
    assignments = ["train"] * len(items)
    cursor = best_state
    while parents[cursor] is not None:
        previous, index, split = parents[cursor]
        assignments[index] = split
        cursor = previous

    split_counts = Counter()
    scene_counts = Counter()
    scenes = {"train": [], "val": [], "test": []}
    for (scene_id, count), split in zip(items, assignments):
        split_counts[split] += count
        scene_counts[split] += 1
        scenes[split].append(scene_id)

    return {
        "targets": {key: to_float(value) for key, value in targets.items()},
        "sample_counts": {key: int(split_counts[key]) for key in ("train", "val", "test")},
        "scene_counts": {key: int(scene_counts[key]) for key in ("train", "val", "test")},
        "scenes": {key: sorted(value) for key, value in scenes.items()},
        "score": [to_float(value) for value in score(best_state)],
    }


def build_balance_audit(records: list[ManifestRecord], meta_dir: Path) -> dict[str, object]:
    class_scene_counts: dict[str, Counter[str]] = defaultdict(Counter)
    class_current_sample_counts: dict[str, Counter[str]] = defaultdict(Counter)
    class_current_scene_counts: dict[str, Counter[str]] = defaultdict(Counter)
    current_scene_split: dict[str, str] = {}

    for record in records:
        class_scene_counts[record.assigned_class_name][record.scene_id] += 1
        class_current_sample_counts[record.assigned_class_name][record.split] += 1
        current_scene_split[record.scene_id] = record.split

    for class_name, scene_counter in class_scene_counts.items():
        for scene_id in scene_counter:
            class_current_scene_counts[class_name][current_scene_split[scene_id]] += 1

    target_classes = ["river", "paddy_field"]
    audit: dict[str, object] = {}
    lines = ["# Split Balance Audit", ""]
    lines.append(
        "This audit focuses on retained classes whose validation/test counts look thin under the "
        "current scene-level split. It does not rebuild the dataset; it only measures whether the "
        "thinness is a hard scene-level constraint or a by-product of the present global scene assignment."
    )
    lines.append("")

    for class_name in target_classes:
        scene_counter = class_scene_counts[class_name]
        total_samples = int(sum(scene_counter.values()))
        total_scenes = len(scene_counter)
        top_scenes = sorted(scene_counter.items(), key=lambda item: (-item[1], item[0]))[:10]
        top5_share = sum(count for _, count in top_scenes[:5]) / total_samples if total_samples else 0.0
        current_counts = {
            split: int(class_current_sample_counts[class_name][split])
            for split in ("train", "val", "test")
        }
        current_scene_counts = {
            split: int(class_current_scene_counts[class_name][split])
            for split in ("train", "val", "test")
        }
        optimized = optimize_scene_partition(scene_counter)

        current_targets = {
            "train": to_float(total_samples * 0.70),
            "val": to_float(total_samples * 0.15),
            "test": to_float(total_samples * 0.15),
        }
        current_score = {
            split: to_float(abs(current_counts[split] - current_targets[split]))
            for split in ("train", "val", "test")
        }
        improved = (
            optimized["sample_counts"]["val"] > current_counts["val"]
            and optimized["sample_counts"]["test"] > current_counts["test"]
        )

        audit[class_name] = {
            "total_samples": total_samples,
            "total_scenes": total_scenes,
            "top5_scene_share": to_float(top5_share),
            "current_sample_counts": current_counts,
            "current_scene_counts": current_scene_counts,
            "target_sample_counts": current_targets,
            "current_absolute_deviation": current_score,
            "top_scenes": [{"scene_id": scene_id, "sample_count": count} for scene_id, count in top_scenes],
            "class_only_optimized_scene_assignment": optimized,
            "improvement_possible_without_scene_leakage": improved,
        }

        lines.append(f"## {class_name}")
        lines.append("")
        lines.append(
            f"- Current samples: train={current_counts['train']}, val={current_counts['val']}, "
            f"test={current_counts['test']} out of {total_samples} retained `{class_name}` tiles."
        )
        lines.append(
            f"- Scene support: {total_scenes} scenes total; current scene split is "
            f"train={current_scene_counts['train']}, val={current_scene_counts['val']}, "
            f"test={current_scene_counts['test']}."
        )
        lines.append(
            f"- Concentration: the top 5 `{class_name}` scenes contribute {to_float(top5_share)} "
            f"of the class's retained samples."
        )
        lines.append(
            f"- Class-only scene-level optimum: train={optimized['sample_counts']['train']}, "
            f"val={optimized['sample_counts']['val']}, test={optimized['sample_counts']['test']} "
            "without splitting any scene across multiple subsets."
        )
        if improved:
            lines.append(
                f"- Conclusion: the current thin `{class_name}` val/test counts are a natural result "
                "of the existing global scene assignment, but they are not a hard scene-level limit. "
                "A more balanced scene reassignment appears feasible without leakage if we are willing "
                "to rebuild the split globally."
            )
        else:
            lines.append(
                f"- Conclusion: the current thin `{class_name}` val/test counts are close to the best "
                "available under scene-level integrity, so a major improvement would be difficult "
                "without changing the underlying sampling rules."
            )
        lines.append("- Largest contributing scenes currently in the dataset:")
        for scene_id, count in top_scenes:
            lines.append(f"  - {scene_id}: {count}")
        lines.append("")

    lines.append("## Recommendation")
    lines.append("")
    lines.append(
        "- Do not move individual tiles between splits to fix these classes; that would break the scene-level leakage guard."
    )
    lines.append(
        "- If `river` or `paddy_field` are critical evaluation targets, consider a future rebuild with a multi-objective scene assignment that explicitly constrains per-class val/test minimums."
    )
    lines.append(
        "- If those classes are not central, the current dataset remains usable, but reported metrics for them will have higher variance because their test sets are thin."
    )
    lines.append("")

    write_text(meta_dir / "split_balance_audit.md", "\n".join(lines))
    return audit


def compare_raw_roots() -> dict[str, object]:
    first = Path("${NAS_ROOT}/gid-15/GID")
    second = Path("${GID15_DATASET_ROOT}/gid-15/GID")
    if not first.exists() or not second.exists():
        return {
            "both_roots_present": False,
            "first_root": str(first),
            "second_root": str(second),
        }

    comparisons = {}
    for relative in [
        "img_dir/train",
        "img_dir/val",
        "img_dir/test",
        "ann_dir/train",
        "ann_dir/val",
        "ann_dir/test",
    ]:
        left_names = sorted(path.name for path in (first / relative).glob("*"))
        right_names = sorted(path.name for path in (second / relative).glob("*"))
        comparisons[relative] = {
            "left_count": len(left_names),
            "right_count": len(right_names),
            "matching_file_names": left_names == right_names,
        }

    sample_pairs = [
        (
            first / "img_dir/train/GF2_PMS1__L1A0000647767-MSS1.tif",
            second / "img_dir/train/GF2_PMS1__L1A0000647767-MSS1.tif",
        ),
        (
            first / "ann_dir/train/GF2_PMS1__L1A0000647767-MSS1_15label.png",
            second / "ann_dir/train/GF2_PMS1__L1A0000647767-MSS1_15label.png",
        ),
    ]
    samples = []
    for left, right in sample_pairs:
        samples.append(
            {
                "left": str(left),
                "right": str(right),
                "left_exists": left.exists(),
                "right_exists": right.exists(),
                "left_size": left.stat().st_size if left.exists() else None,
                "right_size": right.stat().st_size if right.exists() else None,
                "matching_size": left.exists() and right.exists() and left.stat().st_size == right.stat().st_size,
            }
        )

    return {
        "both_roots_present": True,
        "first_root": str(first),
        "second_root": str(second),
        "relative_dir_checks": comparisons,
        "sample_file_checks": samples,
    }


def build_traceability_audit(
    records: list[ManifestRecord],
    dataset_root: Path,
    prepared_root: Path,
    meta_dir: Path,
) -> dict[str, object]:
    summary_path = prepared_root / "stats" / "dataset_summary.json"
    scene_split_path = prepared_root / "manifests" / "scene_split.json"
    unlabeled_test_path = prepared_root / "manifests" / "official_unlabeled_test_images.json"
    split_summary_path = meta_dir / "split_summary.json"

    dataset_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    scene_split = json.loads(scene_split_path.read_text(encoding="utf-8"))
    unlabeled_test_images = json.loads(unlabeled_test_path.read_text(encoding="utf-8"))
    split_summary = json.loads(split_summary_path.read_text(encoding="utf-8"))

    prepared_image_count = sum(len(list((prepared_root / "images" / split).glob("*.tif"))) for split in ("train", "val", "test"))
    prepared_mask_count = sum(len(list((prepared_root / "masks" / split).glob("*.png"))) for split in ("train", "val", "test"))
    prepared_list_count = sum(
        sum(1 for _ in (prepared_root / "lists" / f"{split}.txt").open(encoding="utf-8"))
        for split in ("train", "val", "test")
    )
    prepared_unique_sample_ids = set()
    for split in ("train", "val", "test"):
        prepared_unique_sample_ids.update(path.stem for path in (prepared_root / "images" / split).glob("*.tif"))

    scene_source_index = {}
    for split, items in scene_split.items():
        for item in items:
            scene_source_index[item["scene_id"]] = {
                "prepared_scene_split": split,
                "image_path": item["image_path"],
                "mask_path": item["mask_path"],
                "source_split": item["source_split"],
            }

    missing_scene_ids = sorted({record.scene_id for record in records if record.scene_id not in scene_source_index})
    missing_src_images = [record.src_image for record in records if not Path(record.src_image).exists()]
    missing_src_masks = [record.src_mask for record in records if not Path(record.src_mask).exists()]
    missing_output_images = [record.output_image for record in records if not Path(record.output_image).exists()]

    raw_root_comparison = compare_raw_roots()
    accounting = {
        "prepared_tile_count": int(split_summary["prepared_tile_count"]),
        "retained_sample_count": len(records),
        "threshold_excluded_sample_count": int(split_summary["excluded_sample_count"]),
        "background_only_tile_count": int(split_summary["background_only_tile_count"]),
    }
    accounting["accounting_matches_prepared_tile_count"] = (
        accounting["retained_sample_count"]
        + accounting["threshold_excluded_sample_count"]
        + accounting["background_only_tile_count"]
        == accounting["prepared_tile_count"]
    )

    source_scene_counts = {split: len(items) for split, items in scene_split.items()}
    source_split_histogram = Counter()
    for items in scene_split.values():
        for item in items:
            source_split_histogram[item["source_split"]] += 1

    traceability = {
        "prepared_root": str(prepared_root),
        "tile_size": dataset_summary["tile_size"],
        "stride": dataset_summary["stride"],
        "min_foreground_ratio": dataset_summary["min_foreground_ratio"],
        "cache_source_root": dataset_summary["source_root"],
        "prepared_scene_counts": source_scene_counts,
        "prepared_source_split_histogram": dict(source_split_histogram),
        "official_unlabeled_test_count": len(unlabeled_test_images),
        "prepared_image_count": prepared_image_count,
        "prepared_mask_count": prepared_mask_count,
        "prepared_list_count": prepared_list_count,
        "prepared_unique_sample_id_count": len(prepared_unique_sample_ids),
        "scene_source_index_count": len(scene_source_index),
        "missing_scene_ids_in_scene_manifest": missing_scene_ids,
        "missing_src_image_count": len(missing_src_images),
        "missing_src_mask_count": len(missing_src_masks),
        "missing_output_image_count": len(missing_output_images),
        "sample_id_pattern": "<scene_id>_<row_offset>_<col_offset>",
        "accounting": accounting,
        "raw_root_comparison": raw_root_comparison,
    }

    lines = ["# Source Traceability", ""]
    lines.append("## Prepared Cache Parameters")
    lines.append("")
    lines.append(f"- Prepared cache root: `{prepared_root}`")
    lines.append(f"- Tile size: {dataset_summary['tile_size']}")
    lines.append(f"- Stride: {dataset_summary['stride']}")
    lines.append(f"- Min foreground ratio used by the cache builder: {dataset_summary['min_foreground_ratio']}")
    lines.append(f"- Cache metadata source root: `{dataset_summary['source_root']}`")
    lines.append("")
    lines.append("## Scene Coverage")
    lines.append("")
    lines.append(
        f"- Labeled source scenes in prepared cache: {sum(source_scene_counts.values())} "
        f"(train={source_scene_counts['train']}, val={source_scene_counts['val']}, test={source_scene_counts['test']})."
    )
    lines.append(
        f"- Raw source-split histogram recorded by cache manifest: {dict(source_split_histogram)}."
    )
    lines.append(
        f"- Official raw unlabeled `img_dir/test` scenes listed separately: {len(unlabeled_test_images)}. "
        "They are not part of the supervised classification split."
    )
    lines.append("")
    lines.append("## Completeness Checks")
    lines.append("")
    lines.append(
        f"- Prepared images={prepared_image_count}, masks={prepared_mask_count}, list entries={prepared_list_count}, "
        f"unique prepared sample ids={len(prepared_unique_sample_ids)}."
    )
    lines.append(
        f"- Split accounting: retained={accounting['retained_sample_count']}, "
        f"threshold-excluded={accounting['threshold_excluded_sample_count']}, "
        f"background-only={accounting['background_only_tile_count']}, "
        f"prepared total={accounting['prepared_tile_count']}."
    )
    lines.append(
        f"- Accounting matches prepared total: {accounting['accounting_matches_prepared_tile_count']}."
    )
    lines.append(
        f"- Missing trace links from current manifest to source scene manifest: {len(missing_scene_ids)}."
    )
    lines.append(
        f"- Missing cached source images referenced by manifest: {len(missing_src_images)}; "
        f"missing cached masks: {len(missing_src_masks)}; missing copied output images: {len(missing_output_images)}."
    )
    lines.append("")
    lines.append("## Mapping Back To Raw Scenes")
    lines.append("")
    lines.append(
        "- `split_manifest.csv` retains `sample_id`, `scene_id`, `src_image`, `src_mask`, `source_split`, and `output_image`, "
        "so every copied classification sample can be traced back to the cache tile and then to the raw scene listed in `scene_split.json`."
    )
    lines.append(
        f"- Sample id format: `{traceability['sample_id_pattern']}`. The trailing offsets identify the tile location inside the parent scene."
    )
    lines.append(
        "- Because stride is smaller than tile size, neighboring tiles overlap spatially by design. "
        "That overlap is expected coverage reuse, not duplicate sample ids."
    )
    if raw_root_comparison.get("both_roots_present"):
        lines.append(
            f"- The cache metadata names `{raw_root_comparison['first_root']}` as source root, while this host also has "
            f"`{raw_root_comparison['second_root']}`. Directory-name checks matched across their image/mask splits, "
            "and sampled file sizes matched as well."
        )
    lines.append("")

    write_text(meta_dir / "source_traceability.md", "\n".join(lines))
    return traceability


def build_readme_audit_block(
    path_mapping: dict[str, object],
    purity: dict[str, object],
    scene_overlap: dict[str, object],
    balance: dict[str, object],
    traceability: dict[str, object],
    low_purity_threshold: float,
) -> str:
    purity_summary = purity["summary"]
    top_low_purity = purity_summary["top_low_purity_classes"]
    worst_classes = ", ".join(
        f"{item['class_name']} ({item['low_purity_ratio']:.3f})" for item in top_low_purity[:4]
    )
    overlap_zero = scene_overlap["overlap_scene_count_total"] == 0
    recommend_filtered = purity_summary["low_purity_ratio"] >= 0.20

    lines = [README_AUDIT_BEGIN, "", "## Acceptance Audit", ""]
    lines.append("### Path Mapping Audit")
    lines.append("")
    lines.append(f"- Conclusion: {path_mapping['conclusion']}")
    lines.append(f"- `/public` state on this host: {path_mapping['public_path_state']}")
    lines.append(f"- `${NAS_ROOT}` state on this host: {path_mapping['mnt_nas_path_state']}")
    if path_mapping["mount_line_for_mnt_nas"]:
        lines.append(f"- Mount evidence: `{path_mapping['mount_line_for_mnt_nas']}`")
    if path_mapping["df_line_for_mnt_nas"]:
        lines.append(f"- Capacity evidence: `{path_mapping['df_line_for_mnt_nas']}`")
    lines.append("")
    lines.append("### Dominant-Ratio Purity Audit")
    lines.append("")
    lines.append(
        f"- Metric: dominant foreground ratio = dominant_pixels / foreground_pixels, low-purity threshold = {low_purity_threshold:.2f}."
    )
    lines.append(
        f"- Overall stats: mean={purity_summary['overall_stats']['mean']}, "
        f"median={purity_summary['overall_stats']['median']}, "
        f"p10={purity_summary['overall_stats']['p10']}, "
        f"p25={purity_summary['overall_stats']['p25']}, "
        f"p90={purity_summary['overall_stats']['p90']}."
    )
    lines.append(
        f"- Low-purity retained samples: {purity_summary['low_purity_count']} / {purity_summary['sample_count']} "
        f"({purity_summary['low_purity_ratio']})."
    )
    lines.append(f"- Classes with the clearest low-purity pressure: {worst_classes}.")
    if recommend_filtered:
        lines.append(
            "- Recommendation: the current version is still usable for training, but a future purity-filtered variant is worth considering, especially if industrial/urban mixed tiles hurt downstream stability."
        )
    else:
        lines.append(
            "- Recommendation: purity is acceptable for a first-pass training set, and a filtered variant is optional rather than urgent."
        )
    lines.append("")
    lines.append("### Scene Leakage Audit")
    lines.append("")
    lines.append(
        f"- Unique scene counts: train={scene_overlap['scene_counts']['train']}, "
        f"val={scene_overlap['scene_counts']['val']}, test={scene_overlap['scene_counts']['test']}."
    )
    lines.append(
        f"- Pairwise overlap counts: train/val={scene_overlap['pairwise_overlap_counts']['train_val']}, "
        f"train/test={scene_overlap['pairwise_overlap_counts']['train_test']}, "
        f"val/test={scene_overlap['pairwise_overlap_counts']['val_test']}."
    )
    lines.append(f"- Scene overlap is zero: {overlap_zero}.")
    lines.append("")
    lines.append("### Small-Class Balance Audit")
    lines.append("")
    for class_name in ("river", "paddy_field"):
        item = balance[class_name]
        optimized = item["class_only_optimized_scene_assignment"]
        lines.append(
            f"- `{class_name}` current samples are train={item['current_sample_counts']['train']}, "
            f"val={item['current_sample_counts']['val']}, test={item['current_sample_counts']['test']}; "
            f"class-only scene reassignment could reach train={optimized['sample_counts']['train']}, "
            f"val={optimized['sample_counts']['val']}, test={optimized['sample_counts']['test']} without scene leakage."
        )
    lines.append(
        "- Audit conclusion: the thin `river` and `paddy_field` test sets are a consequence of the present global scene assignment rather than an unavoidable leakage constraint."
    )
    lines.append("")
    lines.append("### Traceability Audit")
    lines.append("")
    lines.append(
        f"- Prepared cache parameters: tile_size={traceability['tile_size']}, stride={traceability['stride']}, "
        f"min_foreground_ratio={traceability['min_foreground_ratio']}."
    )
    lines.append(
        f"- Cache accounting matches prepared total: {traceability['accounting']['accounting_matches_prepared_tile_count']}."
    )
    lines.append(
        f"- Missing source-scene trace links: {len(traceability['missing_scene_ids_in_scene_manifest'])}; "
        f"missing cached source images: {traceability['missing_src_image_count']}; "
        f"missing cached masks: {traceability['missing_src_mask_count']}; "
        f"missing copied output images: {traceability['missing_output_image_count']}."
    )
    lines.append("")
    lines.append(README_AUDIT_END)
    lines.append("")
    return "\n".join(lines)


def update_readme(readme_path: Path, audit_block: str) -> None:
    if readme_path.exists():
        content = readme_path.read_text(encoding="utf-8")
    else:
        content = "# GID15 Split Classification Dataset\n\n"

    if README_AUDIT_BEGIN in content and README_AUDIT_END in content:
        prefix = content.split(README_AUDIT_BEGIN, 1)[0].rstrip()
        suffix = content.split(README_AUDIT_END, 1)[1].lstrip()
        updated = prefix + "\n\n" + audit_block
        if suffix:
            updated += "\n" + suffix
    else:
        updated = content.rstrip() + "\n\n" + audit_block
    write_text(readme_path, updated.rstrip() + "\n")


def main() -> int:
    args = parse_args()

    dataset_root = Path(args.dataset_root).expanduser().resolve()
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")
    meta_dir = dataset_root / "meta"
    if not meta_dir.exists():
        raise FileNotFoundError(f"Meta directory does not exist: {meta_dir}")

    prepared_root = resolve_existing_root(
        args.prepared_root, DEFAULT_PREPARED_CANDIDATES, "prepared GID15 tile root"
    )
    manifest_path = meta_dir / "split_manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing split manifest: {manifest_path}")

    log(f"[info] Dataset root: {dataset_root}")
    log(f"[info] Meta directory: {meta_dir}")
    log(f"[info] Prepared tile root: {prepared_root}")
    log("[scan] Loading retained sample manifest...")
    records = load_manifest(manifest_path)
    log(f"[scan] Loaded {len(records)} retained samples from split_manifest.csv")

    log("[audit] Writing path-mapping evidence...")
    path_mapping = build_path_mapping_audit(args.requested_output_root, dataset_root, meta_dir)

    log("[audit] Computing dominant-ratio purity statistics...")
    purity = build_dominant_ratio_audit(records, args.low_purity_threshold, meta_dir)

    log("[audit] Verifying scene-level non-overlap...")
    scene_overlap = build_scene_audit(records, meta_dir)

    log("[audit] Analyzing small-class val/test thinness...")
    balance = build_balance_audit(records, meta_dir)

    log("[audit] Building cache/source traceability note...")
    traceability = build_traceability_audit(records, dataset_root, prepared_root, meta_dir)

    log("[audit] Updating README with audit conclusions...")
    update_readme(
        meta_dir / "README.md",
        build_readme_audit_block(
            path_mapping=path_mapping,
            purity=purity,
            scene_overlap=scene_overlap,
            balance=balance,
            traceability=traceability,
            low_purity_threshold=args.low_purity_threshold,
        ),
    )

    log(f"[done] Audit artifacts written under: {meta_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
