#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import open3d as o3d


ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "tools"
sys.path.insert(0, str(TOOLS))

from upright_sim3 import fit_upright_sim3


RUNNER_PATH = Path(__file__).with_name("run_nuscenes_vggt_backbone_mappano3d.py")
RUNNER_SPEC = importlib.util.spec_from_file_location("vggt_backbone_runner", RUNNER_PATH)
RUNNER = importlib.util.module_from_spec(RUNNER_SPEC)
assert RUNNER_SPEC.loader is not None
RUNNER_SPEC.loader.exec_module(RUNNER)

POST = RUNNER.POST

GLOBAL_REFINER_PATH = Path(__file__).with_name("run_nuscenes_global_vggt_refine.py")
GLOBAL_REFINER_SPEC = importlib.util.spec_from_file_location(
    "vggt_backbone_global_refiner_perturbation", GLOBAL_REFINER_PATH
)
GLOBAL_REFINER = importlib.util.module_from_spec(GLOBAL_REFINER_SPEC)
assert GLOBAL_REFINER_SPEC.loader is not None
GLOBAL_REFINER_SPEC.loader.exec_module(GLOBAL_REFINER)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise RuntimeError("No perturbation rows were produced")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_cloud(path: Path) -> tuple[np.ndarray, np.ndarray | None]:
    cloud = o3d.io.read_point_cloud(str(path))
    points = np.asarray(cloud.points, dtype=np.float64)
    colors = np.asarray(cloud.colors, dtype=np.float64) if cloud.has_colors() else None
    return points, colors


def perturbation(seed: int, seed_count: int, yaw_deg: float, translation_m: float) -> tuple[float, float, float]:
    yaw = yaw_deg if seed % 2 == 0 else -yaw_deg
    angle = 2.0 * math.pi * seed / seed_count
    return yaw, translation_m * math.cos(angle), translation_m * math.sin(angle)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Controlled noisy-anchor study for shared-VGGT MapPano3D chunks."
    )
    parser.add_argument("--scene", nargs="+", required=True)
    parser.add_argument("--clean-root", type=Path, required=True)
    parser.add_argument("--vggt-root", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--dataroot", type=Path, required=True)
    parser.add_argument("--version-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--level-name", default="rigid_mild")
    parser.add_argument("--perturb-yaw-deg", type=float, default=2.0)
    parser.add_argument("--perturb-translation-m", type=float, default=0.5)
    parser.add_argument("--map-resolution-m", type=float, default=0.1)
    parser.add_argument("--score-corridor-m", type=float, default=15.0)
    parser.add_argument("--point-dilate-px", type=int, default=2)
    parser.add_argument("--max-road-points", type=int, default=12000)
    parser.add_argument("--max-backbone-points", type=int, default=1500)
    parser.add_argument("--max-input-anchor-rmse", type=float, default=1.5)
    parser.add_argument("--anchor-slack-m", type=float, default=4.5)
    parser.add_argument("--anchor-weight", type=float, default=0.05)
    parser.add_argument("--skeleton-weight", type=float, default=1.0)
    parser.add_argument("--yaw-bound-deg", type=float, default=8.0)
    parser.add_argument("--translation-bound-m", type=float, default=4.0)
    parser.add_argument("--max-refine-translation-m", type=float, default=1.5)
    parser.add_argument("--min-objective-improvement", type=float, default=0.01)
    parser.add_argument("--voxel-m", type=float, default=0.1)
    parser.add_argument("--intersection-margin-m", type=float, default=0.0)
    parser.add_argument("--refine-all-chunks", action="store_true")
    args = parser.parse_args()

    manifest_rows: list[dict[str, object]] = []
    for scene in args.scene:
        input_dir = args.input_root / scene
        target_poses = np.load(input_dir / "camera_conditions.npz")["poses"].astype(np.float64)
        predicted_poses = RUNNER.read_poses(
            args.vggt_root / scene / "VGGT" / "camera_poses.txt"
        )
        ranges = RUNNER.chunk_ranges(len(target_poses), 20, 10)
        if len(predicted_poses) != len(target_poses):
            raise ValueError(f"Pose count mismatch for {scene}")

        clean_scene = args.clean_root / scene
        chunks: list[dict[str, object]] = []
        for chunk_index, (start, end) in enumerate(ranges):
            chunk_dir = clean_scene / "chunks" / f"chunk_{chunk_index:03d}"
            points, colors = load_cloud(chunk_dir / "pose_init_static.ply")
            road, _ = load_cloud(chunk_dir / "pose_init_road.ply")
            alignment = fit_upright_sim3(
                predicted_poses[start:end], target_poses[start:end], target_forward_axis=2
            )
            chunks.append(
                {
                    "start": start,
                    "end": end,
                    "points": points,
                    "colors": colors,
                    "road": road,
                    "cameras": np.asarray(alignment["aligned_centers"], dtype=np.float64),
                }
            )

        _, log_record, map_record = POST.scene_map_record(args.version_dir, scene)
        full_mask = POST.load_map_mask(args.dataroot / map_record["filename"], {})
        intersection_polygons = GLOBAL_REFINER.load_intersection_polygons(
            args.dataroot / "maps" / "expansion" / f"{log_record['location']}.json"
        )

        for seed in range(args.seeds):
            noise_yaw, noise_tx, noise_ty = perturbation(
                seed,
                args.seeds,
                args.perturb_yaw_deg,
                args.perturb_translation_m,
            )
            noise_delta = {
                "yaw": noise_yaw,
                "scale": 1.0,
                "tx": noise_tx,
                "ty": noise_ty,
            }
            pose_points: list[np.ndarray] = []
            pose_colors: list[np.ndarray] = []
            refined_points: list[np.ndarray] = []
            refined_colors: list[np.ndarray] = []
            chunk_rows: list[dict[str, object]] = []

            for chunk_index, chunk in enumerate(chunks):
                start = int(chunk["start"])
                end = int(chunk["end"])
                nominal_target = target_poses[start:end]
                noisy_target = POST.perturb_target_poses(
                    nominal_target,
                    noise_yaw,
                    1.0,
                    (noise_tx, noise_ty),
                )
                noisy_points = POST.apply_bev_to_points(
                    np.asarray(chunk["points"]), nominal_target[:, :3, 3], noise_delta
                )
                noisy_road = POST.apply_bev_to_points(
                    np.asarray(chunk["road"]), nominal_target[:, :3, 3], noise_delta
                )
                noisy_cameras = POST.apply_bev_to_points(
                    np.asarray(chunk["cameras"]), nominal_target[:, :3, 3], noise_delta
                )
                refined, _, metrics = RUNNER.refine_chunk(
                    noisy_points,
                    noisy_road,
                    noisy_cameras,
                    noisy_target,
                    full_mask,
                    args,
                )
                has_crossing_support = GLOBAL_REFINER.crossing_supported(
                    nominal_target[:, :2, 3],
                    intersection_polygons,
                    args.intersection_margin_m,
                )
                if not has_crossing_support and not args.refine_all_chunks:
                    refined, _, metrics = GLOBAL_REFINER.reject_non_crossing(
                        noisy_points,
                        noisy_road,
                        metrics,
                    )
                score_improvement = float(
                    metrics["score_before"]["route_fit_score"]
                    - metrics["score_after"]["route_fit_score"]
                )
                accepted = bool(metrics["refine"]["accepted"]) and score_improvement > 0.0
                if not accepted:
                    refined = noisy_points
                refined_cameras = (
                    POST.apply_bev_to_points(
                        noisy_cameras,
                        noisy_target[:, :3, 3],
                        metrics["refine"],
                    )
                    if accepted
                    else noisy_cameras
                )
                pose_error = np.linalg.norm(
                    noisy_cameras - nominal_target[:, :3, 3], axis=1
                )
                refined_error = np.linalg.norm(
                    refined_cameras - nominal_target[:, :3, 3], axis=1
                )
                pose_points.append(noisy_points)
                refined_points.append(refined)
                if chunk["colors"] is not None:
                    pose_colors.append(np.asarray(chunk["colors"]))
                    refined_colors.append(np.asarray(chunk["colors"]))
                chunk_rows.append(
                    {
                        "chunk": chunk_index,
                        "crossing_supported": has_crossing_support,
                        "accepted": accepted,
                        "score_improvement": score_improvement,
                        "pose_init_map_score": float(
                            metrics["score_before"]["route_fit_score"]
                        ),
                        "refined_map_score": float(
                            metrics["score_after"]["route_fit_score"]
                        )
                        if accepted
                        else float(metrics["score_before"]["route_fit_score"]),
                        "pose_nominal_anchor_rmse_m": float(
                            np.sqrt(np.mean(pose_error * pose_error))
                        ),
                        "refined_nominal_anchor_rmse_m": float(
                            np.sqrt(np.mean(refined_error * refined_error))
                        ),
                    }
                )

            run_id = f"{args.level_name}_seed{seed:02d}_{scene}"
            run_dir = args.output_root / "runs" / run_id
            run_dir.mkdir(parents=True, exist_ok=True)
            pose_cloud = run_dir / "registered_pose_init_cloud.ply"
            refined_cloud = run_dir / "registered_refined_cloud.ply"
            RUNNER.write_cloud(
                pose_cloud,
                np.concatenate(pose_points),
                np.concatenate(pose_colors) if pose_colors else None,
                args.voxel_m,
            )
            RUNNER.write_cloud(
                refined_cloud,
                np.concatenate(refined_points),
                np.concatenate(refined_colors) if refined_colors else None,
                args.voxel_m,
            )
            report = {
                "run_id": run_id,
                "scene": scene,
                "level": args.level_name,
                "seed": seed,
                "perturb_yaw_deg": noise_yaw,
                "perturb_tx_m": noise_tx,
                "perturb_ty_m": noise_ty,
                "perturb_scale": 1.0,
                "accepted_chunks": sum(bool(row["accepted"]) for row in chunk_rows),
                "crossing_supported_chunks": sum(
                    bool(row["crossing_supported"]) for row in chunk_rows
                ),
                "total_chunks": len(chunk_rows),
                "pose_init_nominal_anchor_rmse_m": float(
                    np.mean([row["pose_nominal_anchor_rmse_m"] for row in chunk_rows])
                ),
                "refined_nominal_anchor_rmse_m": float(
                    np.mean([row["refined_nominal_anchor_rmse_m"] for row in chunk_rows])
                ),
                "pose_init_map_score": float(
                    np.mean([row["pose_init_map_score"] for row in chunk_rows])
                ),
                "refined_map_score": float(
                    np.mean([row["refined_map_score"] for row in chunk_rows])
                ),
                "chunks": chunk_rows,
            }
            (run_dir / "metrics.json").write_text(
                json.dumps(report, indent=2), encoding="utf-8"
            )
            manifest_rows.append(
                {
                    **{key: value for key, value in report.items() if key != "chunks"},
                    "refine_accepted": report["accepted_chunks"] > 0,
                    "sim3_identifiable": True,
                    "pose_init_cloud": str(pose_cloud),
                    "refined_cloud": str(refined_cloud),
                }
            )
            print(
                f"{run_id}: {report['accepted_chunks']}/{report['total_chunks']} "
                f"chunks, anchors {report['pose_init_nominal_anchor_rmse_m']:.3f}->"
                f"{report['refined_nominal_anchor_rmse_m']:.3f} m"
            )

    write_csv(args.output_root / "perturbation_manifest.csv", manifest_rows)
    print(args.output_root / "perturbation_manifest.csv")


if __name__ == "__main__":
    main()
