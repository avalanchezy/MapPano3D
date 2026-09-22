#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree


DYNAMIC_PREFIXES = ("vehicle.", "human.", "animal", "movable_object.")


def local_path(value: str | Path) -> Path:
    text = str(value).replace("\\", "/")
    if os.name == "nt" and text.startswith("/mnt/") and len(text) > 7 and text[6] == "/":
        return Path(f"{text[5].upper()}:/{text[7:]}")
    return Path(text)


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def by_token(rows: list[dict]) -> dict[str, dict]:
    return {row["token"]: row for row in rows}


def rotation_matrix(quaternion: list[float]) -> np.ndarray:
    w, x, y, z = np.asarray(quaternion, dtype=np.float64)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def transform_points(points: np.ndarray, record: dict) -> np.ndarray:
    rotation = rotation_matrix(record["rotation"])
    translation = np.asarray(record["translation"], dtype=np.float64)
    return points @ rotation.T + translation


def scene_samples(scene: dict, sample_index: dict[str, dict]) -> list[dict]:
    samples = []
    token = scene["first_sample_token"]
    while token:
        sample = sample_index[token]
        samples.append(sample)
        token = sample["next"]
    return samples


def points_outside_boxes(
    points_global: np.ndarray,
    annotations: list[dict],
    instance_index: dict[str, dict],
    category_index: dict[str, dict],
    padding_m: float,
) -> np.ndarray:
    keep = np.ones(len(points_global), dtype=bool)
    for annotation in annotations:
        instance = instance_index[annotation["instance_token"]]
        category = category_index[instance["category_token"]]["name"]
        if not category.startswith(DYNAMIC_PREFIXES):
            continue
        active = np.flatnonzero(keep)
        if len(active) == 0:
            break
        center = np.asarray(annotation["translation"], dtype=np.float64)
        local = (points_global[active] - center) @ rotation_matrix(annotation["rotation"])
        width, length, height = np.asarray(annotation["size"], dtype=np.float64)
        # nuScenes stores size as (width, length, height), while box-local
        # coordinates are x-forward, y-left, z-up.
        half_extent = np.array([length, width, height], dtype=np.float64) / 2.0 + padding_m
        inside = np.all(np.abs(local) <= half_extent, axis=1)
        keep[active[inside]] = False
    return keep


def load_lidar_reference(
    dataroot: Path,
    version_dir: Path,
    scene_name: str,
    dynamic_padding_m: float,
    min_sensor_range_m: float,
    max_sensor_range_m: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    scenes = load_json(version_dir / "scene.json")
    scene = next(row for row in scenes if row["name"] == scene_name)
    samples = scene_samples(scene, by_token(load_json(version_dir / "sample.json")))
    sample_tokens = {sample["token"] for sample in samples}

    sensors = by_token(load_json(version_dir / "sensor.json"))
    calibrations = by_token(load_json(version_dir / "calibrated_sensor.json"))
    ego_poses = by_token(load_json(version_dir / "ego_pose.json"))
    sample_data = [
        row
        for row in load_json(version_dir / "sample_data.json")
        if row["sample_token"] in sample_tokens
        and row["is_key_frame"]
        and sensors[calibrations[row["calibrated_sensor_token"]]["sensor_token"]]["channel"] == "LIDAR_TOP"
    ]
    lidar_by_sample = {row["sample_token"]: row for row in sample_data}

    instance_index = by_token(load_json(version_dir / "instance.json"))
    category_index = by_token(load_json(version_dir / "category.json"))
    annotations_by_sample: dict[str, list[dict]] = {}
    for annotation in load_json(version_dir / "sample_annotation.json"):
        if annotation["sample_token"] in sample_tokens:
            annotations_by_sample.setdefault(annotation["sample_token"], []).append(annotation)

    point_sets = []
    trajectory = []
    raw_count = 0
    range_count = 0
    static_count = 0
    for sample in samples:
        lidar = lidar_by_sample.get(sample["token"])
        if lidar is None:
            continue
        raw = np.fromfile(dataroot / lidar["filename"], dtype=np.float32)
        if raw.size % 5 != 0:
            raise ValueError(f"Unexpected nuScenes LiDAR shape: {lidar['filename']}")
        sensor_points = raw.reshape(-1, 5)[:, :3].astype(np.float64)
        raw_count += len(sensor_points)
        radius = np.linalg.norm(sensor_points[:, :2], axis=1)
        sensor_points = sensor_points[(radius >= min_sensor_range_m) & (radius <= max_sensor_range_m)]
        range_count += len(sensor_points)

        calibration = calibrations[lidar["calibrated_sensor_token"]]
        ego_pose = ego_poses[lidar["ego_pose_token"]]
        ego_points = transform_points(sensor_points, calibration)
        global_points = transform_points(ego_points, ego_pose)
        keep = points_outside_boxes(
            global_points,
            annotations_by_sample.get(sample["token"], []),
            instance_index,
            category_index,
            dynamic_padding_m,
        )
        global_points = global_points[keep]
        static_count += len(global_points)
        point_sets.append(global_points)
        trajectory.append(np.asarray(ego_pose["translation"], dtype=np.float64))

    if not point_sets:
        raise RuntimeError(f"No keyframe LIDAR_TOP data found for {scene_name}")
    counts = {
        "keyframes": len(point_sets),
        "raw_points": raw_count,
        "range_filtered_points": range_count,
        "dynamic_filtered_points": static_count,
    }
    return np.concatenate(point_sets), np.stack(trajectory), counts


def crop_to_trajectory(
    points: np.ndarray,
    trajectory: np.ndarray,
    corridor_m: float,
    min_relative_z_m: float,
    max_relative_z_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    tree = cKDTree(trajectory[:, :2])
    distance, nearest = tree.query(points[:, :2], workers=-1)
    relative_z = points[:, 2] - trajectory[nearest, 2]
    keep = (
        (distance <= corridor_m)
        & (relative_z >= min_relative_z_m)
        & (relative_z <= max_relative_z_m)
        & np.all(np.isfinite(points), axis=1)
    )
    return points[keep], relative_z[keep]


def voxel_downsample(points: np.ndarray, voxel_m: float) -> np.ndarray:
    if len(points) == 0:
        return points
    keys = np.floor(points / voxel_m).astype(np.int64)
    _, first = np.unique(keys, axis=0, return_index=True)
    return points[np.sort(first)]


def load_cloud(path: Path) -> np.ndarray:
    cloud = o3d.io.read_point_cloud(str(path))
    points = np.asarray(cloud.points, dtype=np.float64)
    if len(points) == 0:
        raise RuntimeError(f"Point cloud is empty: {path}")
    return points


def distance_summary(distances: np.ndarray, prefix: str) -> dict[str, float]:
    return {
        f"{prefix}_mean_m": float(np.mean(distances)),
        f"{prefix}_median_m": float(np.median(distances)),
        f"{prefix}_p90_m": float(np.percentile(distances, 90)),
    }


def geometry_metrics(
    prediction: np.ndarray,
    reference: np.ndarray,
    thresholds_m: list[float],
    reference_tree: cKDTree | None = None,
) -> dict[str, float | int]:
    if len(prediction) == 0 or len(reference) == 0:
        raise RuntimeError("Evaluation crop produced an empty point set")
    reference_tree = reference_tree or cKDTree(reference)
    prediction_tree = cKDTree(prediction)
    accuracy, nearest_reference = reference_tree.query(prediction, workers=-1)
    completeness, _ = prediction_tree.query(reference, workers=-1)
    result: dict[str, float | int] = {
        "prediction_points": int(len(prediction)),
        "reference_points": int(len(reference)),
        "chamfer_l1_m": float(np.mean(accuracy) + np.mean(completeness)),
    }
    result.update(distance_summary(accuracy, "accuracy"))
    result.update(distance_summary(completeness, "completeness"))
    vertical = prediction[:, 2] - reference[nearest_reference, 2]
    result.update(
        {
            "vertical_signed_median_m": float(np.median(vertical)),
            "vertical_abs_median_m": float(np.median(np.abs(vertical))),
            "vertical_abs_p90_m": float(np.percentile(np.abs(vertical), 90)),
        }
    )
    for threshold in thresholds_m:
        precision = float(np.mean(accuracy <= threshold))
        recall = float(np.mean(completeness <= threshold))
        fscore = 2.0 * precision * recall / max(precision + recall, 1e-12)
        suffix = str(threshold).replace(".", "p")
        result[f"precision_{suffix}m"] = precision
        result[f"recall_{suffix}m"] = recall
        result[f"fscore_{suffix}m"] = fscore
    return result


def save_cloud(path: Path, points: np.ndarray, color: tuple[float, float, float]) -> None:
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    cloud.colors = o3d.utility.Vector3dVector(np.tile(np.asarray(color), (len(points), 1)))
    o3d.io.write_point_cloud(str(path), cloud, write_ascii=False)


def save_overlay(path: Path, prediction: np.ndarray, reference: np.ndarray) -> None:
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(np.concatenate([reference, prediction]))
    reference_colors = np.tile(np.array([[0.15, 0.55, 0.95]]), (len(reference), 1))
    prediction_colors = np.tile(np.array([[0.95, 0.25, 0.15]]), (len(prediction), 1))
    cloud.colors = o3d.utility.Vector3dVector(np.concatenate([reference_colors, prediction_colors]))
    o3d.io.write_point_cloud(str(path), cloud, write_ascii=False)


def write_csv(path: Path, rows: list[dict]) -> None:
    keys = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate fixed-pose nuScenes reconstructions against held-out LiDAR.")
    parser.add_argument("--dataroot", type=Path, required=True)
    parser.add_argument("--version", default="v1.0-mini")
    parser.add_argument("--scene", required=True)
    parser.add_argument("--pose-init-cloud", type=Path, required=True)
    parser.add_argument("--refined-cloud", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--voxel-m", type=float, default=0.25)
    parser.add_argument("--corridor-m", type=float, default=35.0)
    parser.add_argument("--min-relative-z-m", type=float, default=-2.5)
    parser.add_argument("--max-relative-z-m", type=float, default=6.0)
    parser.add_argument("--non-ground-relative-z-m", type=float, default=0.35)
    parser.add_argument("--dynamic-padding-m", type=float, default=0.30)
    parser.add_argument("--min-sensor-range-m", type=float, default=1.5)
    parser.add_argument("--max-sensor-range-m", type=float, default=50.0)
    parser.add_argument("--threshold-m", type=float, nargs="+", default=[0.5, 1.0])
    args = parser.parse_args()

    version_dir = args.dataroot / args.version
    args.out_dir.mkdir(parents=True, exist_ok=True)
    reference, trajectory, reference_counts = load_lidar_reference(
        args.dataroot,
        version_dir,
        args.scene,
        args.dynamic_padding_m,
        args.min_sensor_range_m,
        args.max_sensor_range_m,
    )
    reference, reference_relative_z = crop_to_trajectory(
        reference,
        trajectory,
        args.corridor_m,
        args.min_relative_z_m,
        args.max_relative_z_m,
    )

    clouds = {
        "pose_init": load_cloud(args.pose_init_cloud),
        "bev_refined": load_cloud(args.refined_cloud),
    }
    rows = []
    cached_references: dict[str, np.ndarray] = {}
    for variant, variant_reference in {
        "all_static": reference,
        "non_ground_static": reference[reference_relative_z > args.non_ground_relative_z_m],
    }.items():
        variant_reference = voxel_downsample(variant_reference, args.voxel_m)
        cached_references[variant] = variant_reference
        save_cloud(args.out_dir / f"lidar_{variant}.ply", variant_reference, (0.15, 0.55, 0.95))
        for stage, points in clouds.items():
            points, relative_z = crop_to_trajectory(
                points,
                trajectory,
                args.corridor_m,
                args.min_relative_z_m,
                args.max_relative_z_m,
            )
            if variant == "non_ground_static":
                points = points[relative_z > args.non_ground_relative_z_m]
            points = voxel_downsample(points, args.voxel_m)
            metrics = geometry_metrics(points, variant_reference, args.threshold_m)
            row = {"scene": args.scene, "stage": stage, "variant": variant, **metrics}
            rows.append(row)
            save_overlay(args.out_dir / f"overlay_{stage}_{variant}.ply", points, variant_reference)

    write_csv(args.out_dir / "metrics.csv", rows)
    report = {
        "protocol": {
            "description": "Controlled before/after geometry evaluation under the same strong nuScenes GT camera-pose anchor.",
            "optimization_inputs": ["camera images", "nuScenes GT camera pose anchor", "nuScenes semantic road map for BEV refinement"],
            "evaluation_only": ["LIDAR_TOP keyframes", "sample annotations for dynamic-object filtering"],
            "reference_provenance": {
                "reference_used_for": ["evaluation"],
                "reference_not_used_for": [
                    "initialization",
                    "map_selection",
                    "refinement",
                    "acceptance",
                    "parameter_tuning",
                    "scene_selection",
                    "per_method_alignment",
                ],
                "method_gt_inputs": ["nuScenes GT ego pose anchor"],
            },
            "forbidden_operation": "No LiDAR-based ICP, Sim(3), tuning, crop selection, or acceptance decision is performed.",
            "primary_variant": "non_ground_static",
            "voxel_m": args.voxel_m,
            "corridor_m": args.corridor_m,
            "relative_z_m": [args.min_relative_z_m, args.max_relative_z_m],
            "non_ground_relative_z_m": args.non_ground_relative_z_m,
            "thresholds_m": args.threshold_m,
        },
        "reference_counts": reference_counts,
        "trajectory_samples": int(len(trajectory)),
        "rows": rows,
    }
    (args.out_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(args.out_dir / "metrics.csv")


if __name__ == "__main__":
    main()
