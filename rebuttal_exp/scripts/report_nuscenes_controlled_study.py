#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path


LEVEL_ORDER = (
    "clean",
    "rigid_mild",
    "rigid_medium",
    "rigid_severe",
    "scale_mild",
    "scale_medium",
    "scale_severe",
)

PERTURBATION = {
    "clean": "none",
    "rigid_mild": "yaw=2deg, horizontal=0.5m",
    "rigid_medium": "yaw=5deg, horizontal=1m",
    "rigid_severe": "yaw=10deg, horizontal=2m",
    "scale_mild": "scale=2pct",
    "scale_medium": "scale=5pct",
    "scale_severe": "scale=10pct",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def finite_numeric_rows(rows: list[dict[str, str]], ignored: set[str]) -> None:
    for row in rows:
        for key, value in row.items():
            if key in ignored or value == "":
                continue
            try:
                number = float(value)
            except ValueError:
                continue
            if not math.isfinite(number):
                raise RuntimeError(f"Non-finite value in {key}: {value}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate and summarize the frozen nuScenes controlled study.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--qa", type=Path, required=True)
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    scene_names = {item["name"] for item in config["scenes"]}
    primary_levels = set(config["primary_evaluation_levels"])
    extension_levels = set(config["robustness_extension_levels"])
    levels = primary_levels | extension_levels
    if levels != set(LEVEL_ORDER):
        raise RuntimeError(f"Unexpected configured levels: {sorted(levels)}")

    manifest = read_csv(args.root / "perturbation_manifest.csv")
    metrics = read_csv(args.root / "lidar_eval" / "metrics_all.csv")
    deltas = read_csv(args.root / "lidar_eval" / "paired_deltas.csv")
    references = read_csv(args.root / "lidar_eval" / "reference_counts.csv")
    summary = read_csv(args.root / "lidar_eval" / "metrics_summary.csv")
    intervals = read_csv(args.root / "statistics" / "summary_ci.csv")

    expected_counts = {level: len(scene_names) * (1 if level == "clean" else 10) for level in levels}
    actual_counts = Counter(row["level"] for row in manifest)
    if actual_counts != expected_counts:
        raise RuntimeError(f"Manifest level counts differ: {dict(actual_counts)}")
    if {row["scene"] for row in manifest} != scene_names:
        raise RuntimeError("Manifest scene set differs from the frozen config")
    if len({row["run_id"] for row in manifest}) != len(manifest):
        raise RuntimeError("Duplicate run IDs in manifest")
    if len(metrics) != 4 * len(manifest) or len(deltas) != 2 * len(manifest):
        raise RuntimeError("Metric/delta row count does not match two stages and two variants")
    if len(references) != len(scene_names):
        raise RuntimeError("Reference count table does not cover every frozen scene")

    finite_numeric_rows(
        metrics,
        {
            "run_id", "scene", "level", "seed", "refine_accepted", "refine_fallback_reason",
            "stage", "variant", "evaluation_valid", "invalid_reason",
        },
    )
    finite_numeric_rows(
        deltas,
        {
            "run_id", "scene", "level", "seed", "refine_accepted", "refine_fallback_reason",
            "variant", "evaluation_valid", "invalid_reason",
        },
    )

    fallback_ids = {row["run_id"] for row in manifest if row["refine_accepted"].lower() != "true"}
    for row in deltas:
        if row["run_id"] not in fallback_ids:
            continue
        values = [float(value) for key, value in row.items() if key.startswith("delta_") and value != ""]
        if any(abs(value) > 1e-12 for value in values):
            raise RuntimeError(f"Fallback is not identity: {row['run_id']}")

    def build_variant_table(variant: str) -> list[dict]:
        mean_index = {
            (row["level"], row["stage"]): row
            for row in summary
            if row["variant"] == variant
        }
        ci_index = {
            (row["level"], row["metric"]): row
            for row in intervals
            if row["variant"] == variant
        }
        table_rows = []
        for level in LEVEL_ORDER:
            before = mean_index[(level, "pose_init")]
            after = mean_index[(level, "bev_refined")]
            chamfer = ci_index[(level, "chamfer_l1_m")]
            accuracy = ci_index[(level, "accuracy_mean_m")]
            completeness = ci_index[(level, "completeness_mean_m")]
            recall = ci_index[(level, "recall_0p5m")]
            fscore = ci_index[(level, "fscore_1p0m")]
            table_rows.append(
                {
                    "variant": variant,
                    "level": level,
                    "protocol_role": "primary" if level in primary_levels else "predeclared_extension",
                    "perturbation": PERTURBATION[level],
                    "scenes": chamfer["scenes"],
                    "total_scenes": chamfer["total_scenes"],
                    "runs": chamfer["runs"],
                    "total_runs": chamfer["total_runs"],
                    "accepted_runs": chamfer["accepted_runs"],
                    "fallback_runs": chamfer["fallback_runs"],
                    "accepted_rate": chamfer["accepted_rate"],
                    "pose_recovery_runs": chamfer["pose_recovery_runs"],
                    "pose_rmse_before_m": chamfer["pose_rmse_before_m"],
                    "pose_rmse_after_m": chamfer["pose_rmse_after_m"],
                    "chamfer_before_m": before["chamfer_l1_m"],
                    "chamfer_after_m": after["chamfer_l1_m"],
                    "delta_chamfer_m": chamfer["mean_delta"],
                    "delta_chamfer_ci95_low_m": chamfer["mean_ci95_low"],
                    "delta_chamfer_ci95_high_m": chamfer["mean_ci95_high"],
                    "chamfer_improved_scenes": chamfer["improved_scenes"],
                    "delta_accuracy_mean_m": accuracy["mean_delta"],
                    "delta_accuracy_ci95_low_m": accuracy["mean_ci95_low"],
                    "delta_accuracy_ci95_high_m": accuracy["mean_ci95_high"],
                    "delta_completeness_mean_m": completeness["mean_delta"],
                    "delta_completeness_ci95_low_m": completeness["mean_ci95_low"],
                    "delta_completeness_ci95_high_m": completeness["mean_ci95_high"],
                    "recall_0p5_before": before["recall_0p5m"],
                    "recall_0p5_after": after["recall_0p5m"],
                    "delta_recall_0p5": recall["mean_delta"],
                    "delta_recall_0p5_ci95_low": recall["mean_ci95_low"],
                    "delta_recall_0p5_ci95_high": recall["mean_ci95_high"],
                    "fscore_1p0_before": before["fscore_1p0m"],
                    "fscore_1p0_after": after["fscore_1p0m"],
                    "delta_fscore_1p0": fscore["mean_delta"],
                    "delta_fscore_1p0_ci95_low": fscore["mean_ci95_low"],
                    "delta_fscore_1p0_ci95_high": fscore["mean_ci95_high"],
                }
            )
        return table_rows

    write_csv(args.table, build_variant_table("non_ground_static"))
    all_static_table = args.table.with_name(f"{args.table.stem}_all_static{args.table.suffix}")
    write_csv(all_static_table, build_variant_table("all_static"))
    qa = {
        "config": str(args.config),
        "manifest_rows": len(manifest),
        "metric_rows": len(metrics),
        "delta_rows": len(deltas),
        "reference_scenes": len(references),
        "level_counts": dict(actual_counts),
        "fallback_runs": len(fallback_ids),
        "invalid_metric_rows": sum(row.get("evaluation_valid", "true").lower() != "true" for row in metrics),
        "invalid_delta_rows": sum(row.get("evaluation_valid", "true").lower() != "true" for row in deltas),
        "invalid_scene_variants": sorted(
            {
                f"{row['scene']}:{row['variant']}:{row.get('invalid_reason', '')}"
                for row in metrics
                if row.get("evaluation_valid", "true").lower() != "true"
            }
        ),
        "non_ground_summary": str(args.table),
        "all_static_summary": str(all_static_table),
        "all_populated_numeric_values_finite": True,
        "all_fallbacks_are_identity": True,
        "statistical_unit": "scene after averaging deterministic seeds",
        "lidar_role": "evaluation only",
        "gt_pose_role": "method input anchor; pose metrics are controlled recovery diagnostics only",
    }
    args.qa.parent.mkdir(parents=True, exist_ok=True)
    args.qa.write_text(json.dumps(qa, indent=2), encoding="utf-8")
    print(args.table)
    print(args.qa)


if __name__ == "__main__":
    main()
