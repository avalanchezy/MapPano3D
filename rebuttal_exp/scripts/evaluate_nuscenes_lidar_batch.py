#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

import evaluate_nuscenes_lidar as evaluator


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise RuntimeError(f"No rows to write: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def mean_rows(rows: list[dict], group_keys: tuple[str, ...], metric_keys: list[str]) -> list[dict]:
    groups: dict[tuple[str, ...], list[dict]] = defaultdict(list)
    for row in rows:
        groups[tuple(str(row[key]) for key in group_keys)].append(row)
    output = []
    for key, group in sorted(groups.items()):
        record = {name: value for name, value in zip(group_keys, key)}
        record["units"] = len(group)
        record["valid_units"] = sum(str(row.get("evaluation_valid", "true")).lower() == "true" for row in group)
        for metric in metric_keys:
            values = [float(row[metric]) for row in group if row.get(metric, "") != ""]
            record[metric] = float(np.mean(values)) if values else ""
        output.append(record)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch held-out nuScenes LiDAR evaluation with per-scene reference caching.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataroot", type=Path, required=True)
    parser.add_argument("--version", default="v1.0-mini")
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

    manifest = read_csv(args.manifest)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    version_dir = args.dataroot / args.version
    references: dict[str, tuple[dict[str, tuple[np.ndarray, cKDTree]], np.ndarray]] = {}
    reference_count_rows: list[dict] = []
    rows: list[dict] = []

    for item_index, item in enumerate(manifest, start=1):
        scene = item["scene"]
        if scene not in references:
            reference, trajectory, reference_counts = evaluator.load_lidar_reference(
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
            scene_references = {
                "all_static": evaluator.voxel_downsample(reference, args.voxel_m),
                "non_ground_static": evaluator.voxel_downsample(
                    reference[relative_z > args.non_ground_relative_z_m], args.voxel_m
                ),
            }
            references[scene] = (
                {
                    variant: (points, cKDTree(points))
                    for variant, points in scene_references.items()
                },
                trajectory,
            )
            reference_count_rows.append({"scene": scene, **reference_counts})
            print(
                f"[reference] {scene}: {reference_counts['keyframes']} keyframes, "
                f"{reference_counts['dynamic_filtered_points']} static points"
            )

        scene_references, trajectory = references[scene]
        alignment_identifiable = str(item.get("sim3_identifiable", "true")).lower() == "true"
        for stage, path_key in (("pose_init", "pose_init_cloud"), ("bev_refined", "refined_cloud")):
            points = evaluator.load_cloud(evaluator.local_path(item[path_key]))
            points, relative_z = evaluator.crop_to_trajectory(
                points,
                trajectory,
                args.corridor_m,
                args.min_relative_z_m,
                args.max_relative_z_m,
            )
            for variant, (reference, reference_tree) in scene_references.items():
                variant_points = points
                if variant == "non_ground_static":
                    variant_points = points[relative_z > args.non_ground_relative_z_m]
                variant_points = evaluator.voxel_downsample(variant_points, args.voxel_m)
                valid = alignment_identifiable and len(variant_points) > 0 and len(reference) > 0
                if valid:
                    metrics = evaluator.geometry_metrics(
                        variant_points,
                        reference,
                        args.threshold_m,
                        reference_tree=reference_tree,
                    )
                    invalid_reason = ""
                else:
                    metrics = {
                        "prediction_points": int(len(variant_points)),
                        "reference_points": int(len(reference)),
                        **{
                            metric: ""
                            for metric in (
                                "chamfer_l1_m",
                                "accuracy_mean_m",
                                "accuracy_median_m",
                                "accuracy_p90_m",
                                "completeness_mean_m",
                                "completeness_median_m",
                                "completeness_p90_m",
                                "vertical_signed_median_m",
                                "vertical_abs_median_m",
                                "vertical_abs_p90_m",
                                "precision_0p5m",
                                "recall_0p5m",
                                "fscore_0p5m",
                                "precision_1p0m",
                                "recall_1p0m",
                                "fscore_1p0m",
                            )
                        },
                    }
                    if not alignment_identifiable:
                        invalid_reason = "camera_trajectory_scale_unidentifiable"
                    else:
                        invalid_reason = (
                            f"empty_prediction_{variant}"
                            if len(variant_points) == 0
                            else f"empty_reference_{variant}"
                        )
                metadata = {
                    key: item[key]
                    for key in item
                    if key not in {"pose_init_cloud", "refined_cloud"}
                }
                rows.append(
                    {
                        **metadata,
                        "stage": stage,
                        "variant": variant,
                        "evaluation_valid": valid,
                        "invalid_reason": invalid_reason,
                        **metrics,
                    }
                )
        if item_index == 1 or item_index % 10 == 0 or item_index == len(manifest):
            print(f"[progress] {item_index}/{len(manifest)} runs")

    write_csv(args.out_dir / "metrics_all.csv", rows)
    write_csv(args.out_dir / "reference_counts.csv", reference_count_rows)
    metric_keys = [
        "chamfer_l1_m",
        "accuracy_mean_m",
        "accuracy_median_m",
        "accuracy_p90_m",
        "completeness_mean_m",
        "completeness_median_m",
        "completeness_p90_m",
        "precision_0p5m",
        "recall_0p5m",
        "fscore_0p5m",
        "precision_1p0m",
        "recall_1p0m",
        "fscore_1p0m",
        "vertical_signed_median_m",
        "vertical_abs_median_m",
        "vertical_abs_p90_m",
    ]
    summary = mean_rows(rows, ("level", "variant", "stage"), metric_keys)
    write_csv(args.out_dir / "metrics_summary.csv", summary)

    indexed = {(row["run_id"], row["variant"], row["stage"]): row for row in rows}
    deltas = []
    for item in manifest:
        for variant in ("all_static", "non_ground_static"):
            before = indexed[(item["run_id"], variant, "pose_init")]
            after = indexed[(item["run_id"], variant, "bev_refined")]
            metadata = {
                key: item[key]
                for key in item
                if key not in {"pose_init_cloud", "refined_cloud"}
            }
            valid = (
                str(before.get("evaluation_valid", "true")).lower() == "true"
                and str(after.get("evaluation_valid", "true")).lower() == "true"
            )
            deltas.append(
                {
                    **metadata,
                    "variant": variant,
                    "evaluation_valid": valid,
                    "invalid_reason": "" if valid else (before.get("invalid_reason") or after.get("invalid_reason")),
                    **{
                        f"delta_{metric}": float(after[metric]) - float(before[metric])
                        if valid and before.get(metric, "") != "" and after.get(metric, "") != ""
                        else ""
                        for metric in metric_keys
                    },
                }
            )
    write_csv(args.out_dir / "paired_deltas.csv", deltas)
    print(args.out_dir / "metrics_summary.csv")


if __name__ == "__main__":
    main()
