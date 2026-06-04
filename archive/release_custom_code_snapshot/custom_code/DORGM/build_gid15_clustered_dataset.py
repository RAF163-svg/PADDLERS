#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
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

DEFAULT_INPUT_ROOT = "${GID15_DATASET_ROOT}/gid15_dataset"
DEFAULT_OUTPUT_ROOT = "${GID15_DATASET_ROOT}/gid15_clustered_dataset"
DEFAULT_SEED = 20260403
DEFAULT_CANDIDATE_K = [3, 4, 5]
DEFAULT_THUMB_SIZE = 16
DEFAULT_PCA_DIM = 64


@dataclass(frozen=True)
class TileRecord:
    sample_id: str
    split: str
    scene_id: str
    image_abs: str
    mask_abs: str
    image_rel: str
    mask_rel: str


def log(message: str) -> None:
    print(message, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Cluster GID15 tiles in a DORGM-style unsupervised workflow and build an "
            "independent clustered segmentation dataset."
        )
    )
    parser.add_argument("--input-root", default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--candidate-k", nargs="+", type=int, default=DEFAULT_CANDIDATE_K)
    parser.add_argument("--selected-k", type=int, default=None)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--thumb-size", type=int, default=DEFAULT_THUMB_SIZE)
    parser.add_argument("--pca-dim", type=int, default=DEFAULT_PCA_DIM)
    parser.add_argument("--kmeans-iters", type=int, default=50)
    parser.add_argument("--kmeans-restarts", type=int, default=5)
    parser.add_argument("--feature-workers", type=int, default=8)
    parser.add_argument("--copy-workers", type=int, default=8)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def resolve_path(requested_path: str) -> tuple[Path, dict[str, str]]:
    requested = Path(requested_path)
    if str(requested).startswith("/public/"):
        mapped = Path("${NAS_ROOT}") / requested.relative_to("/public")
        resolved = mapped.resolve()
        return resolved, {
            "requested_path": str(requested),
            "resolved_path": str(resolved),
            "resolution_note": (
                "The host does not expose /public directly. ${NAS_ROOT} is mounted from "
                "//10.62.192.79/public, so the requested /public path was resolved "
                "to the equivalent ${NAS_ROOT} path."
            ),
        }

    resolved = requested.expanduser().resolve()
    return resolved, {
        "requested_path": str(requested),
        "resolved_path": str(resolved),
        "resolution_note": "The requested path is used directly.",
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


def load_tile_records(input_root: Path) -> list[TileRecord]:
    manifest_path = input_root / "manifests" / "tile_manifest.csv"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Expected tile manifest at {manifest_path}")

    records: list[TileRecord] = []
    with manifest_path.open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            records.append(
                TileRecord(
                    sample_id=str(row["sample_id"]),
                    split=str(row["split"]),
                    scene_id=str(row["scene_id"]),
                    image_abs=str(row["output_image"]),
                    mask_abs=str(row["output_mask"]),
                    image_rel=str(row["image_rel"]),
                    mask_rel=str(row["mask_rel"]),
                )
            )

    if not records:
        raise RuntimeError(f"No tiles were loaded from {manifest_path}")
    return records


def rgb_histogram_features(rgb: np.ndarray, bins: int = 16) -> np.ndarray:
    parts = []
    for channel in range(3):
        hist, _ = np.histogram(rgb[..., channel], bins=bins, range=(0.0, 1.0))
        hist = hist.astype(np.float32)
        hist /= max(float(hist.sum()), 1.0)
        parts.append(hist)
    return np.concatenate(parts, axis=0)


def gradient_features(gray: np.ndarray, bins: int = 16) -> np.ndarray:
    gx = np.zeros_like(gray, dtype=np.float32)
    gy = np.zeros_like(gray, dtype=np.float32)
    gx[:, :-1] = gray[:, 1:] - gray[:, :-1]
    gy[:-1, :] = gray[1:, :] - gray[:-1, :]
    mag = np.sqrt(gx * gx + gy * gy)

    stats = np.array(
        [
            float(mag.mean()),
            float(mag.std()),
            float(np.quantile(mag, 0.5)),
            float(np.quantile(mag, 0.9)),
        ],
        dtype=np.float32,
    )
    hist, _ = np.histogram(mag, bins=bins, range=(0.0, 1.5))
    hist = hist.astype(np.float32)
    hist /= max(float(hist.sum()), 1.0)
    return np.concatenate([stats, hist], axis=0)


def extract_feature(record: TileRecord, thumb_size: int) -> np.ndarray:
    with Image.open(record.image_abs) as image:
        image = image.convert("RGB")
        rgb = np.asarray(image, dtype=np.float32) / 255.0
        hsv = np.asarray(image.convert("HSV"), dtype=np.float32) / 255.0
        thumb = image.resize((thumb_size, thumb_size), resample=Image.BILINEAR)
        thumb_arr = np.asarray(thumb, dtype=np.float32) / 255.0

    gray = rgb.mean(axis=2)
    rgb_stats = np.concatenate([rgb.mean(axis=(0, 1)), rgb.std(axis=(0, 1))], axis=0)
    hsv_stats = np.concatenate([hsv.mean(axis=(0, 1)), hsv.std(axis=(0, 1))], axis=0)
    gray_hist, _ = np.histogram(gray, bins=32, range=(0.0, 1.0))
    gray_hist = gray_hist.astype(np.float32)
    gray_hist /= max(float(gray_hist.sum()), 1.0)

    parts = [
        thumb_arr.reshape(-1).astype(np.float32),
        rgb_histogram_features(rgb),
        rgb_stats.astype(np.float32),
        hsv_stats.astype(np.float32),
        gray_hist,
        gradient_features(gray),
    ]
    return np.concatenate(parts, axis=0)


def build_features(records: list[TileRecord], thumb_size: int, workers: int) -> np.ndarray:
    def _job(record: TileRecord) -> np.ndarray:
        return extract_feature(record, thumb_size)

    features: list[np.ndarray] = [None] * len(records)  # type: ignore[assignment]
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        for index, feature in enumerate(executor.map(_job, records), start=1):
            features[index - 1] = feature
            if index % 1000 == 0 or index == len(records):
                log(f"[feature] extracted {index}/{len(records)}")
    return np.stack(features, axis=0).astype(np.float32)


def standardize(features: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = features.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = features.std(axis=0, dtype=np.float64).astype(np.float32)
    std[std < 1e-6] = 1.0
    standardized = (features - mean) / std
    return standardized.astype(np.float32), mean, std


def pca_reduce(features: np.ndarray, n_components: int) -> tuple[np.ndarray, dict[str, object]]:
    feature_dim = int(features.shape[1])
    n_components = min(int(n_components), feature_dim)
    covariance = np.cov(features, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(covariance)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    components = eigvecs[:, :n_components].astype(np.float32)
    reduced = features @ components
    total_var = float(np.clip(eigvals.sum(), a_min=1e-12, a_max=None))
    explained = eigvals[:n_components] / total_var
    info = {
        "original_dim": feature_dim,
        "reduced_dim": n_components,
        "explained_variance_ratio_sum": float(explained.sum()),
        "explained_variance_ratio_head": [float(v) for v in explained[:10]],
    }
    return reduced.astype(np.float32), info


def pairwise_sq_dists(points: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    x2 = np.sum(points * points, axis=1, keepdims=True)
    c2 = np.sum(centroids * centroids, axis=1, keepdims=True).T
    cross = points @ centroids.T
    dists = x2 + c2 - 2.0 * cross
    return np.maximum(dists, 0.0)


def kmeans_plus_plus_init(features: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    n_samples = features.shape[0]
    centroids = np.empty((k, features.shape[1]), dtype=np.float32)
    first_idx = int(rng.integers(0, n_samples))
    centroids[0] = features[first_idx]

    closest_dist_sq = pairwise_sq_dists(features, centroids[:1]).reshape(-1)
    for index in range(1, k):
        probs = closest_dist_sq / np.clip(closest_dist_sq.sum(), a_min=1e-12, a_max=None)
        next_idx = int(rng.choice(n_samples, p=probs))
        centroids[index] = features[next_idx]
        dist_sq = pairwise_sq_dists(features, centroids[index : index + 1]).reshape(-1)
        closest_dist_sq = np.minimum(closest_dist_sq, dist_sq)
    return centroids


def run_kmeans(
    features: np.ndarray,
    k: int,
    seed: int,
    max_iters: int,
    restarts: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    best_labels = None
    best_centroids = None
    best_inertia = None
    n_samples = features.shape[0]

    for restart in range(restarts):
        rng = np.random.default_rng(seed + 97 * k + restart)
        centroids = kmeans_plus_plus_init(features, k, rng)
        labels = np.full(n_samples, -1, dtype=np.int32)

        for _ in range(max_iters):
            dists = pairwise_sq_dists(features, centroids)
            new_labels = np.argmin(dists, axis=1).astype(np.int32)
            if np.array_equal(labels, new_labels):
                labels = new_labels
                break
            labels = new_labels

            new_centroids = centroids.copy()
            for cluster_id in range(k):
                mask = labels == cluster_id
                if np.any(mask):
                    new_centroids[cluster_id] = features[mask].mean(axis=0)
                else:
                    farthest_idx = int(np.argmax(np.min(dists, axis=1)))
                    new_centroids[cluster_id] = features[farthest_idx]
            centroids = new_centroids

        final_dists = pairwise_sq_dists(features, centroids)
        inertia = float(final_dists[np.arange(n_samples), labels].sum())
        if best_inertia is None or inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels.copy()
            best_centroids = centroids.copy()

    assert best_labels is not None and best_centroids is not None and best_inertia is not None
    return best_labels, best_centroids, best_inertia


def calinski_harabasz_score(features: np.ndarray, labels: np.ndarray, centroids: np.ndarray) -> float:
    n_samples = features.shape[0]
    k = centroids.shape[0]
    if k <= 1 or n_samples <= k:
        return 0.0

    overall_mean = features.mean(axis=0)
    counts = np.bincount(labels, minlength=k).astype(np.float64)
    between = 0.0
    within = 0.0
    for cluster_id in range(k):
        mask = labels == cluster_id
        if not np.any(mask):
            continue
        diff_center = centroids[cluster_id] - overall_mean
        between += counts[cluster_id] * float(np.dot(diff_center, diff_center))
        diff = features[mask] - centroids[cluster_id]
        within += float(np.sum(diff * diff))
    if within <= 0.0:
        return 0.0
    return float((between / (k - 1)) / (within / (n_samples - k)))


def davies_bouldin_score(features: np.ndarray, labels: np.ndarray, centroids: np.ndarray) -> float:
    k = centroids.shape[0]
    scatters = np.zeros(k, dtype=np.float64)
    for cluster_id in range(k):
        mask = labels == cluster_id
        if not np.any(mask):
            return float("inf")
        diff = features[mask] - centroids[cluster_id]
        scatters[cluster_id] = float(np.sqrt(np.sum(diff * diff, axis=1)).mean())

    centroid_dists = np.sqrt(pairwise_sq_dists(centroids, centroids))
    centroid_dists[centroid_dists < 1e-12] = np.inf
    scores = []
    for i in range(k):
        ratios = (scatters[i] + scatters) / centroid_dists[i]
        ratios[i] = -np.inf
        scores.append(float(np.max(ratios)))
    return float(np.mean(scores))


def cluster_split_counts(labels: np.ndarray, records: list[TileRecord], k: int) -> list[dict[str, int]]:
    counts = [{"train": 0, "val": 0, "test": 0} for _ in range(k)]
    for label, record in zip(labels.tolist(), records):
        counts[int(label)][record.split] += 1
    return counts


def evaluate_candidate(
    k: int,
    labels: np.ndarray,
    centroids: np.ndarray,
    inertia: float,
    reduced_features: np.ndarray,
    records: list[TileRecord],
) -> dict[str, object]:
    counts = np.bincount(labels, minlength=k).astype(int)
    split_counts = cluster_split_counts(labels, records, k)
    nonzero_splits = [split_count for row in split_counts for split_count in row.values()]
    min_total = int(counts.min())
    max_total = int(counts.max())
    result = {
        "k": k,
        "inertia": float(inertia),
        "calinski_harabasz": calinski_harabasz_score(reduced_features, labels, centroids),
        "davies_bouldin": davies_bouldin_score(reduced_features, labels, centroids),
        "cluster_sizes": counts.tolist(),
        "min_cluster_size": min_total,
        "max_cluster_size": max_total,
        "size_imbalance_ratio": float(max_total / max(min_total, 1)),
        "cluster_split_counts": split_counts,
        "all_clusters_have_all_splits": all(value > 0 for value in nonzero_splits),
        "min_cluster_split_count": int(min(nonzero_splits)),
    }
    return result


def choose_k(results: list[dict[str, object]]) -> int:
    valid = [row for row in results if bool(row["all_clusters_have_all_splits"])]
    if not valid:
        raise RuntimeError("No candidate K keeps train/val/test present in every cluster.")

    valid.sort(
        key=lambda row: (
            -int(row["min_cluster_size"]),
            -int(row["min_cluster_split_count"]),
            -float(row["calinski_harabasz"]),
            float(row["davies_bouldin"]),
        )
    )

    shortlist = [row for row in valid if int(row["min_cluster_size"]) >= 1000]
    if shortlist:
        shortlist.sort(
            key=lambda row: (
                -int(row["k"]),
                float(row["size_imbalance_ratio"]),
                -float(row["calinski_harabasz"]),
                float(row["davies_bouldin"]),
            )
        )
        return int(shortlist[0]["k"])

    return int(valid[0]["k"])


def make_cluster_dirs(output_root: Path, k: int) -> None:
    for cluster_id in range(k):
        cluster_root = output_root / "clusters" / f"cluster_{cluster_id}"
        for split in ("train", "val", "test"):
            (cluster_root / "images" / split).mkdir(parents=True, exist_ok=True)
            (cluster_root / "masks" / split).mkdir(parents=True, exist_ok=True)
        (cluster_root / "lists").mkdir(parents=True, exist_ok=True)
        (cluster_root / "labels.txt").write_text("\n".join(LABELS) + "\n", encoding="utf-8")


def copy_file(task: tuple[str, str]) -> None:
    src, dst = task
    shutil.copy2(src, dst)


def build_cluster_dataset(
    output_root: Path,
    records: list[TileRecord],
    labels: np.ndarray,
    selected_k: int,
    copy_workers: int,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    make_cluster_dirs(output_root, selected_k)
    mapping_rows: list[dict[str, object]] = []
    copy_tasks: list[tuple[str, str]] = []
    cluster_list_entries = {
        cluster_id: {"train": [], "val": [], "test": []} for cluster_id in range(selected_k)
    }

    for record, cluster_id in zip(records, labels.tolist()):
        cluster_id = int(cluster_id)
        cluster_name = f"cluster_{cluster_id}"
        cluster_root = output_root / "clusters" / cluster_name
        image_out = cluster_root / "images" / record.split / Path(record.image_rel).name
        mask_out = cluster_root / "masks" / record.split / Path(record.mask_rel).name
        image_rel = Path("images") / record.split / Path(record.image_rel).name
        mask_rel = Path("masks") / record.split / Path(record.mask_rel).name

        copy_tasks.append((record.image_abs, str(image_out)))
        copy_tasks.append((record.mask_abs, str(mask_out)))
        cluster_list_entries[cluster_id][record.split].append(f"{image_rel.as_posix()} {mask_rel.as_posix()}")

        mapping_rows.append(
            {
                "sample_id": record.sample_id,
                "scene_id": record.scene_id,
                "split": record.split,
                "cluster_id": cluster_id,
                "cluster_name": cluster_name,
                "input_image": record.image_abs,
                "input_mask": record.mask_abs,
                "output_image": str(image_out),
                "output_mask": str(mask_out),
                "cluster_image_rel": image_rel.as_posix(),
                "cluster_mask_rel": mask_rel.as_posix(),
            }
        )

    log(f"[copy] copying {len(copy_tasks)} files into clustered dataset")
    with ThreadPoolExecutor(max_workers=max(1, copy_workers)) as executor:
        for index, _ in enumerate(executor.map(copy_file, copy_tasks), start=1):
            if index % 5000 == 0 or index == len(copy_tasks):
                log(f"[copy] copied {index}/{len(copy_tasks)}")

    per_cluster_split_counts: dict[str, dict[str, int]] = {}
    for cluster_id in range(selected_k):
        cluster_name = f"cluster_{cluster_id}"
        cluster_root = output_root / "clusters" / cluster_name
        per_cluster_split_counts[cluster_name] = {}
        for split in ("train", "val", "test"):
            rows = sorted(cluster_list_entries[cluster_id][split])
            (cluster_root / "lists" / f"{split}.txt").write_text(
                "\n".join(rows) + ("\n" if rows else ""),
                encoding="utf-8",
            )
            per_cluster_split_counts[cluster_name][split] = len(rows)

    mapping_rows.sort(key=lambda row: (int(row["cluster_id"]), str(row["split"]), str(row["sample_id"])))
    return mapping_rows, {"per_cluster_split_counts": per_cluster_split_counts}


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_pair_checks(output_root: Path, selected_k: int) -> dict[str, object]:
    checks = {}
    for cluster_id in range(selected_k):
        cluster_name = f"cluster_{cluster_id}"
        cluster_root = output_root / "clusters" / cluster_name
        cluster_checks = {}
        for split in ("train", "val", "test"):
            image_paths = sorted((cluster_root / "images" / split).glob("*.tif"))
            mask_paths = sorted((cluster_root / "masks" / split).glob("*.png"))
            list_path = cluster_root / "lists" / f"{split}.txt"
            list_count = sum(1 for _ in list_path.open(encoding="utf-8")) if list_path.is_file() else 0
            image_stems = {path.stem for path in image_paths}
            mask_stems = {path.stem for path in mask_paths}
            cluster_checks[split] = {
                "image_count": len(image_paths),
                "mask_count": len(mask_paths),
                "list_count": list_count,
                "image_mask_stem_mismatch_count": len(image_stems ^ mask_stems),
                "images_missing_mask": sorted(image_stems - mask_stems)[:20],
                "masks_missing_image": sorted(mask_stems - image_stems)[:20],
            }
        checks[cluster_name] = cluster_checks
    return checks


def main() -> None:
    args = parse_args()
    if not args.candidate_k:
        raise ValueError("At least one candidate K is required.")
    candidate_k = sorted({int(k) for k in args.candidate_k})
    if any(k <= 1 for k in candidate_k):
        raise ValueError(f"Candidate K must all be >= 2, got {candidate_k}")

    input_root, input_info = resolve_path(args.input_root)
    output_root, output_info = resolve_path(args.output_root)
    if not input_root.is_dir():
        raise FileNotFoundError(f"Input dataset root does not exist: {input_root}")

    log(f"[info] Input requested path: {input_info['requested_path']}")
    log(f"[info] Input resolved path: {input_info['resolved_path']}")
    log(f"[info] Output requested path: {output_info['requested_path']}")
    log(f"[info] Output resolved path: {output_info['resolved_path']}")
    log(f"[info] Candidate K: {candidate_k}")

    ensure_output_root(output_root, args.force)
    (output_root / "analysis").mkdir(parents=True, exist_ok=True)
    (output_root / "manifests").mkdir(parents=True, exist_ok=True)
    (output_root / "stats").mkdir(parents=True, exist_ok=True)

    records = load_tile_records(input_root)
    log(f"[info] Loaded {len(records)} tile records from {input_root}")

    features = build_features(records, args.thumb_size, args.feature_workers)
    standardized, feature_mean, feature_std = standardize(features)
    reduced, pca_info = pca_reduce(standardized, args.pca_dim)
    np.save(output_root / "analysis" / "features_standardized.npy", standardized)
    np.save(output_root / "analysis" / "features_reduced.npy", reduced)

    feature_info = {
        "thumb_size": args.thumb_size,
        "feature_dim_before_pca": int(features.shape[1]),
        "feature_dim_after_pca": int(reduced.shape[1]),
        "feature_components": {
            "rgb_thumbnail_flatten": args.thumb_size * args.thumb_size * 3,
            "rgb_histogram_bins": 48,
            "rgb_stats": 6,
            "hsv_stats": 6,
            "gray_histogram_bins": 32,
            "gradient_features": 20,
        },
        "pca": pca_info,
    }
    (output_root / "analysis" / "feature_info.json").write_text(
        json.dumps(feature_info, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    sweep_results: list[dict[str, object]] = []
    assignments_by_k: dict[int, np.ndarray] = {}
    for k in candidate_k:
        log(f"[cluster] evaluating K={k}")
        labels, centroids, inertia = run_kmeans(
            reduced,
            k=k,
            seed=args.seed,
            max_iters=args.kmeans_iters,
            restarts=args.kmeans_restarts,
        )
        np.save(output_root / "analysis" / f"labels_k{k}.npy", labels)
        assignments_by_k[k] = labels
        result = evaluate_candidate(k, labels, centroids, inertia, reduced, records)
        sweep_results.append(result)
        log(
            f"[cluster] K={k} sizes={result['cluster_sizes']} "
            f"CH={result['calinski_harabasz']:.2f} DB={result['davies_bouldin']:.4f}"
        )

    selected_k = int(args.selected_k) if args.selected_k is not None else choose_k(sweep_results)
    if selected_k not in assignments_by_k:
        raise ValueError(f"Selected K={selected_k} is not in candidate K list {candidate_k}")
    log(f"[select] selected K={selected_k}")

    k_comparison = {
        "candidate_k": candidate_k,
        "selected_k": selected_k,
        "seed": args.seed,
        "results": sweep_results,
    }
    (output_root / "stats" / "k_comparison.json").write_text(
        json.dumps(k_comparison, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    selected_labels = assignments_by_k[selected_k]
    mapping_rows, build_info = build_cluster_dataset(
        output_root=output_root,
        records=records,
        labels=selected_labels,
        selected_k=selected_k,
        copy_workers=args.copy_workers,
    )

    mapping_fields = [
        "sample_id",
        "scene_id",
        "split",
        "cluster_id",
        "cluster_name",
        "input_image",
        "input_mask",
        "output_image",
        "output_mask",
        "cluster_image_rel",
        "cluster_mask_rel",
    ]
    write_csv(output_root / "manifests" / "sample_cluster_mapping.csv", mapping_rows, mapping_fields)

    pair_checks = build_pair_checks(output_root, selected_k)
    (output_root / "manifests" / "pair_checks.json").write_text(
        json.dumps(pair_checks, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    selected_result = next(row for row in sweep_results if int(row["k"]) == selected_k)
    cluster_summary = {
        "input_requested_root": input_info["requested_path"],
        "input_resolved_root": input_info["resolved_path"],
        "output_requested_root": output_info["requested_path"],
        "output_resolved_root": output_info["resolved_path"],
        "seed": args.seed,
        "selected_k": selected_k,
        "selected_result": selected_result,
        "feature_info": feature_info,
        "pair_checks": pair_checks,
        **build_info,
    }
    (output_root / "stats" / "cluster_summary.json").write_text(
        json.dumps(cluster_summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    log("[complete] Clustered dataset build finished.")
    log(f"[complete] Mapping: {output_root / 'manifests' / 'sample_cluster_mapping.csv'}")
    log(f"[complete] K comparison: {output_root / 'stats' / 'k_comparison.json'}")
    log(f"[complete] Summary: {output_root / 'stats' / 'cluster_summary.json'}")
    log(f"[complete] Pair checks: {output_root / 'manifests' / 'pair_checks.json'}")


if __name__ == "__main__":
    main()
