#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

import evaluate_nuscenes_lidar as evaluator


METRICS = (
    "accuracy_mean_m",
    "accuracy_median_m",
    "accuracy_p90_m",
    "completeness_mean_m",
    "completeness_median_m",
    "completeness_p90_m",
    "chamfer_l1_m",
    "precision_0p5m",
    "recall_0p5m",
    "fscore_0p5m",
    "precision_1p0m",
    "recall_1p0m",
    "fscore_1p0m",
    "vertical_signed_median_m",
    "vertical_abs_median_m",
    "vertical_abs_p90_m",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise RuntimeError(f"No rows to write: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_").lower()


def save_overlay(path: Path, prediction: np.ndarray, reference: np.ndarray) -> None:
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(np.concatenate([reference, prediction]))
    reference_colors = np.tile(np.array([[0.12, 0.45, 0.82]]), (len(reference), 1))
    prediction_colors = np.tile(np.array([[0.90, 0.24, 0.16]]), (len(prediction), 1))
    cloud.colors = o3d.utility.Vector3dVector(np.concatenate([reference_colors, prediction_colors]))
    o3d.io.write_point_cloud(str(path), cloud, write_ascii=False)


def bootstrap_mean(values: np.ndarray, seed: int = 2026, samples: int = 10000) -> tuple[float, float]:
    if len(values) == 0:
        return float("nan"), float("nan")
    if len(values) == 1:
        return float(values[0]), float(values[0])
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(samples, len(values)), replace=True).mean(axis=1)
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare complete nuScenes reconstruction clouds against one shared LiDAR protocol."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataroot", type=Path, required=True)
    parser.add_argument("--version", default="v1.0-mini")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--voxel-m", type=float, default=0.25)
    parser.add_argument("--corridor-m", type=float, default=35.0)
    parser.add_argument("--min-relative-z-m", type=float, default=-2.5)
    parser.add_argument("--max-relative-z-m", type=float, default=6.0)
    parser.add_argument("--non-ground-relative-z-m", type=float, default=0.35)
    parser.add_argument("--dynamic-padding-m", type=float, default=0.30)
    parser.add_argument("--min-sensor-range-m", type=float, default=1.5)
    parser.add_argument("--max-sensor-range-m", type=float, default=50.0)
    parser.add_argument("--min-target-path-m", type=float, default=5.0)
    parser.add_argument("--threshold-m", type=float, nargs="+", default=[0.5, 1.0])
    parser.add_argument("--save-overlays", action="store_true")
    args = parser.parse_args()

    manifest = read_csv(args.manifest)
    required = {"scene", "method", "cloud_path"}
    if not manifest or not required.issubset(manifest[0]):
        raise ValueError(f"Manifest must contain {sorted(required)}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    version_dir = args.dataroot / args.version

    references: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    rows = []
    reference_counts = []
    for item in manifest:
        scene = item["scene"]
        if scene not in references:
            reference, trajectory, counts = evaluator.load_lidar_reference(
                args.dataroot,
                version_dir,
                scene,
                args.dynamic_padding_m,
                args.min_sensor_range_m,
                args.max_sensor_range_m,
            )
            reference, relative_z = evaluator.crop_to_trajectory(
                reference,
                trajectory,
                args.corridor_m,
                args.min_relative_z_m,
                args.max_relative_z_m,
            )
            references[scene] = (reference, relative_z, trajectory)
            reference_counts.append({"scene": scene, **counts})

        reference, reference_relative_z, trajectory = references[scene]
        prediction = evaluator.load_cloud(evaluator.local_path(item["cloud_path"]))
        prediction, prediction_relative_z = evaluator.crop_to_trajectory(
            prediction,
            trajectory,
            args.corridor_m,
            args.min_relative_z_m,
            args.max_relative_z_m,
        )
        for variant in ("all_static", "non_ground_static"):
            variant_reference = reference
            variant_prediction = prediction
            if variant == "non_ground_static":
                variant_reference = reference[reference_relative_z > args.non_ground_relative_z_m]
                variant_prediction = prediction[prediction_relative_z > args.non_ground_relative_z_m]
            variant_reference = evaluator.voxel_downsample(variant_reference, args.voxel_m)
            variant_prediction = evaluator.voxel_downsample(variant_prediction, args.voxel_m)
            target_path_m = float(item.get("target_path_m", "inf"))
            alignment_identifiable = (
                str(item.get("alignment_identifiable", "true")).lower() == "true"
                and target_path_m >= args.min_target_path_m
            )
            valid = alignment_identifiable and len(variant_reference) > 0 and len(variant_prediction) > 0
            if valid:
                metrics = evaluator.geometry_metrics(
                    variant_prediction,
                    variant_reference,
                    args.threshold_m,
                    reference_tree=cKDTree(variant_reference),
                )
                invalid_reason = ""
                if args.save_overlays:
                    overlay_dir = args.output_dir / "overlays" / scene
                    overlay_dir.mkdir(parents=True, exist_ok=True)
                    save_overlay(
                        overlay_dir / f"{safe_name(item['method'])}_{variant}.ply",
                        variant_prediction,
                        variant_reference,
                    )
            else:
                metrics = {
                    "prediction_points": int(len(variant_prediction)),
                    "reference_points": int(len(variant_reference)),
                    **{metric: "" for metric in METRICS},
                }
                if not alignment_identifiable:
                    invalid_reason = "camera_trajectory_scale_unidentifiable"
                else:
                    invalid_reason = "empty_prediction" if len(variant_prediction) == 0 else "empty_reference"
            metadata = {
                key: value
                for key, value in item.items()
                if key not in {"scene", "method", "cloud_path"}
            }
            rows.append(
                {
                    "scene": scene,
                    "method": item["method"],
                    "variant": variant,
                    **metadata,
                    "evaluation_valid": valid,
                    "invalid_reason": invalid_reason,
                    **metrics,
                }
            )
        print(f"[evaluated] {scene}: {item['method']}")

    write_csv(args.output_dir / "per_scene_metrics.csv", rows)
    write_csv(args.output_dir / "reference_counts.csv", reference_counts)

    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        groups[(row["method"], row["variant"])].append(row)
    summaries = []
    for (method, variant), group in sorted(groups.items()):
        valid_rows = [row for row in group if row["evaluation_valid"]]
        summary = {
            "method": method,
            "variant": variant,
            "reported_scenes": len(group),
            "valid_scenes": len(valid_rows),
            "invalid_scenes": len(group) - len(valid_rows),
        }
        for metric in METRICS:
            values = np.asarray([float(row[metric]) for row in valid_rows], dtype=np.float64)
            summary[metric] = float(np.mean(values)) if len(values) else ""
            lo, hi = bootstrap_mean(values)
            summary[f"{metric}_ci_low"] = lo if len(values) else ""
            summary[f"{metric}_ci_high"] = hi if len(values) else ""
        summaries.append(summary)
    write_csv(args.output_dir / "summary_metrics.csv", summaries)

    index = {(row["scene"], row["method"], row["variant"]): row for row in rows}
    methods = sorted({row["method"] for row in rows})
    pair_rows = []
    comparisons = (
        ("MapPano3D", "VGGT-Long"),
        ("MapPano3D", "MapPano3D w/o BEV"),
    )
    for proposed_method, baseline_method in comparisons:
        if proposed_method not in methods or baseline_method not in methods:
            continue
        for variant in ("all_static", "non_ground_static"):
            for metric in METRICS:
                deltas = []
                scenes = []
                for scene in sorted({row["scene"] for row in rows}):
                    baseline = index.get((scene, baseline_method, variant))
                    proposed = index.get((scene, proposed_method, variant))
                    if not baseline or not proposed or not baseline["evaluation_valid"] or not proposed["evaluation_valid"]:
                        continue
                    deltas.append(float(proposed[metric]) - float(baseline[metric]))
                    scenes.append(scene)
                values = np.asarray(deltas, dtype=np.float64)
                lo, hi = bootstrap_mean(values)
                pair_rows.append(
                    {
                        "proposed": proposed_method,
                        "baseline": baseline_method,
                        "variant": variant,
                        "metric": metric,
                        "paired_scenes": len(values),
                        "proposed_minus_baseline": float(np.mean(values)) if len(values) else "",
                        "ci_low": lo if len(values) else "",
                        "ci_high": hi if len(values) else "",
                        "scenes": ";".join(scenes),
                    }
                )
    if pair_rows:
        write_csv(args.output_dir / "paired_method_deltas.csv", pair_rows)

    report = {
        "protocol": {
            "prediction_alignment": "single upright camera-trajectory Sim(3) only",
            "lidar_alignment": "none",
            "reference": "dynamic-filtered LIDAR_TOP keyframes",
            "minimum_camera_target_path_m": args.min_target_path_m,
            "crop": {
                "corridor_m": args.corridor_m,
                "relative_z_m": [args.min_relative_z_m, args.max_relative_z_m],
                "non_ground_relative_z_m": args.non_ground_relative_z_m,
                "voxel_m": args.voxel_m,
            },
        },
        "manifest": str(args.manifest),
        "methods": methods,
        "scenes": sorted({row["scene"] for row in rows}),
    }
    (args.output_dir / "protocol.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(args.output_dir / "summary_metrics.csv")


if __name__ == "__main__":
    main()
