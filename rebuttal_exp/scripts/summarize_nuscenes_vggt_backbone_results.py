#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


METHODS = ("VGGT-Long", "MapPano3D")
VARIANTS = ("all_static", "non_ground_static")
METRICS = (
    "prediction_points",
    "reference_points",
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
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(
    path: Path,
    rows: list[dict[str, object]],
    chunk_size: int,
    overlap: int,
) -> None:
    lines = [
        "# nuScenes mini shared-VGGT point-cloud quality",
        "",
        (
            "Both methods use the same six native camera images, "
            f"{chunk_size}-image VGGT chunks with {overlap}-image overlap, "
            "semantic filtering, and camera-only world-frame alignment. "
            "MapPano3D adds BEV updates only to chunks supported by official "
            "intersection polygons. LiDAR is evaluation-only."
        ),
        "",
        "| Scene | Location | Path (m) | Cross/accepted | All Chamfer VGGT/Map | NG accuracy VGGT/Map | NG completeness VGGT/Map | NG Chamfer VGGT/Map | NG F@1 VGGT/Map | Map score before/after |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {scene} | {location} | {path:.1f} ({motion}) | {crossing}/{accepted} | "
            "{all_v:.3f}/{all_m:.3f} | {acc_v:.3f}/{acc_m:.3f} | "
            "{comp_v:.3f}/{comp_m:.3f} | {ng_v:.3f}/{ng_m:.3f} | "
            "{f_v:.3f}/{f_m:.3f} | {map_before:.3f}/{map_after:.3f} |".format(
                scene=row["scene"],
                location=row["location"],
                path=float(row["physical_path_m"]),
                motion=row["motion_status"],
                crossing=int(row["crossing_supported_chunks"]),
                accepted=int(row["accepted_chunks"]),
                all_v=float(row["vggt_long_all_chamfer_l1_m"]),
                all_m=float(row["mappano3d_all_chamfer_l1_m"]),
                acc_v=float(row["vggt_long_non_ground_accuracy_mean_m"]),
                acc_m=float(row["mappano3d_non_ground_accuracy_mean_m"]),
                comp_v=float(row["vggt_long_non_ground_completeness_mean_m"]),
                comp_m=float(row["mappano3d_non_ground_completeness_mean_m"]),
                ng_v=float(row["vggt_long_non_ground_chamfer_l1_m"]),
                ng_m=float(row["mappano3d_non_ground_chamfer_l1_m"]),
                f_v=float(row["vggt_long_non_ground_fscore_1p0m"]),
                f_m=float(row["mappano3d_non_ground_fscore_1p0m"]),
                map_before=float(row["map_score_before"]),
                map_after=float(row["map_score_after"]),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def bootstrap_mean(values: np.ndarray, seed: int = 2026) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(10000, len(values)), replace=True).mean(axis=1)
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def locations(version_dir: Path) -> dict[str, str]:
    scenes = json.loads((version_dir / "scene.json").read_text(encoding="utf-8"))
    logs = json.loads((version_dir / "log.json").read_text(encoding="utf-8"))
    log_locations = {row["token"]: row["location"] for row in logs}
    return {row["name"]: log_locations[row["log_token"]] for row in scenes}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create the shared-VGGT nuScenes all-scene rebuttal tables."
    )
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--version-dir", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--moving-path-m", type=float, default=5.0)
    args = parser.parse_args()

    metric_rows = [
        row
        for row in read_csv(args.metrics)
        if row["method"] in METHODS and row["evaluation_valid"].lower() == "true"
    ]
    index = {
        (row["scene"], row["method"], row["variant"]): row for row in metric_rows
    }
    manifest_rows = read_csv(args.manifest)
    paths = {row["scene"]: float(row["target_path_m"]) for row in manifest_rows}
    chunk_sizes = {int(row["chunk_size_views"]) for row in manifest_rows}
    overlaps = {int(row["overlap_views"]) for row in manifest_rows}
    if len(chunk_sizes) != 1 or len(overlaps) != 1:
        raise ValueError("All manifest rows must use one shared chunk protocol")
    chunk_size = chunk_sizes.pop()
    overlap = overlaps.pop()
    scene_locations = locations(args.version_dir)
    scenes = sorted(paths)

    detailed: list[dict[str, object]] = []
    map_records: dict[str, dict[str, float | int]] = {}
    for scene in scenes:
        report = json.loads(
            (args.run_root / scene / "metrics.json").read_text(encoding="utf-8")
        )
        before = np.asarray(
            [row["score_before"]["route_fit_score"] for row in report["chunk_metrics"]],
            dtype=np.float64,
        )
        after = np.asarray(
            [row["score_after"]["route_fit_score"] for row in report["chunk_metrics"]],
            dtype=np.float64,
        )
        map_records[scene] = {
            "chunks": int(report["chunks"]),
            "crossing_supported_chunks": int(report["crossing_supported_chunks"]),
            "accepted_chunks": int(report["accepted_chunks"]),
            "map_score_before": float(before.mean()),
            "map_score_after": float(after.mean()),
            "map_score_reduction_pct": float(100.0 * (before.mean() - after.mean()) / before.mean()),
        }
        record: dict[str, object] = {
            "scene": scene,
            "location": scene_locations[scene],
            "physical_path_m": paths[scene],
            "motion_status": "moving" if paths[scene] >= args.moving_path_m else "near_stationary",
            **map_records[scene],
        }
        for variant in VARIANTS:
            variant_tag = "all" if variant == "all_static" else "non_ground"
            for method in METHODS:
                method_tag = "vggt_long" if method == "VGGT-Long" else "mappano3d"
                row = index[(scene, method, variant)]
                for metric in METRICS:
                    value = row[metric]
                    record[f"{method_tag}_{variant_tag}_{metric}"] = (
                        int(value) if metric.endswith("_points") else float(value)
                    )
        detailed.append(record)
    write_csv(args.output_csv, detailed)
    write_markdown(
        args.output_csv.with_suffix(".md"),
        detailed,
        chunk_size,
        overlap,
    )

    summary: dict[str, object] = {
        "protocol": {
            "backbone": "VGGT for both methods",
            "inputs": "six native nuScenes perspective cameras",
            "local_chunks": (
                f"identical {chunk_size}-image chunks with "
                f"{overlap}-image overlap"
            ),
            "semantic_filtering": "identical Mask2Former static masks",
            "alignment": "one shared camera-trajectory upright Sim(3)",
            "refinement_scope": "official nuScenes intersection polygons only; zero margin",
            "lidar_use": "evaluation only",
            "moving_path_threshold_m": args.moving_path_m,
        },
        "groups": {},
    }
    for group_name, group_scenes in (
        ("all10", scenes),
        ("moving8", [scene for scene in scenes if paths[scene] >= args.moving_path_m]),
    ):
        group: dict[str, object] = {
            "scenes": group_scenes,
            "methods": {},
            "mappano3d_minus_vggt_long": {},
        }
        for method in METHODS:
            method_summary: dict[str, float] = {}
            for variant in VARIANTS:
                for metric in METRICS:
                    if metric.endswith("_points"):
                        continue
                    values = np.asarray(
                        [float(index[(scene, method, variant)][metric]) for scene in group_scenes],
                        dtype=np.float64,
                    )
                    method_summary[f"{variant}_{metric}"] = float(values.mean())
                    method_summary[f"{variant}_{metric}_median_across_scenes"] = float(
                        np.median(values)
                    )
            group["methods"][method] = method_summary
        for variant in VARIANTS:
            for metric in METRICS:
                if metric.endswith("_points"):
                    continue
                deltas = np.asarray(
                    [
                        float(index[(scene, "MapPano3D", variant)][metric])
                        - float(index[(scene, "VGGT-Long", variant)][metric])
                        for scene in group_scenes
                    ],
                    dtype=np.float64,
                )
                low, high = bootstrap_mean(deltas)
                group["mappano3d_minus_vggt_long"][f"{variant}_{metric}"] = {
                    "mean": float(deltas.mean()),
                    "median": float(np.median(deltas)),
                    "ci95": [low, high],
                }
        before = np.asarray([map_records[scene]["map_score_before"] for scene in group_scenes])
        after = np.asarray([map_records[scene]["map_score_after"] for scene in group_scenes])
        group["map_score"] = {
            "before": float(before.mean()),
            "after": float(after.mean()),
            "reduction_pct": float(100.0 * (before.mean() - after.mean()) / before.mean()),
        }
        summary["groups"][group_name] = group

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(args.output_csv)
    print(args.output_json)


if __name__ == "__main__":
    main()
