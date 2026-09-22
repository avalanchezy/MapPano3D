#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d


ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "tools"
sys.path.insert(0, str(TOOLS))

from upright_sim3 import fit_upright_sim3


RUNNER_PATH = Path(__file__).with_name("run_nuscenes_vggt_backbone_mappano3d.py")
SPEC = importlib.util.spec_from_file_location("vggt_backbone_runner_global", RUNNER_PATH)
RUNNER = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(RUNNER)
POST = RUNNER.POST


def load_cloud(path: Path) -> tuple[np.ndarray, np.ndarray | None]:
    cloud = o3d.io.read_point_cloud(str(path))
    points = np.asarray(cloud.points, dtype=np.float64)
    colors = np.asarray(cloud.colors, dtype=np.float64) if cloud.has_colors() else None
    return points, colors


def invert_sim3(points: np.ndarray, alignment: dict[str, object]) -> np.ndarray:
    scale = float(alignment["scale"])
    rotation = np.asarray(alignment["rotation"], dtype=np.float64)
    translation = np.asarray(alignment["translation"], dtype=np.float64)
    return ((points - translation) / scale) @ rotation


def load_intersection_polygons(map_path: Path) -> list[np.ndarray]:
    records = json.loads(map_path.read_text(encoding="utf-8"))
    nodes = {
        row["token"]: (float(row["x"]), float(row["y"]))
        for row in records["node"]
    }
    polygons = {row["token"]: row for row in records["polygon"] if row["token"]}
    output: list[np.ndarray] = []
    for segment in records["road_segment"]:
        if not bool(segment.get("is_intersection", False)):
            continue
        polygon = polygons.get(segment.get("polygon_token", ""))
        if polygon is None:
            continue
        exterior = np.asarray(
            [nodes[token] for token in polygon["exterior_node_tokens"]],
            dtype=np.float32,
        )
        if len(exterior) >= 3:
            output.append(exterior)
    return output


def crossing_supported(
    camera_xy: np.ndarray,
    polygons: list[np.ndarray],
    margin_m: float,
) -> bool:
    for polygon in polygons:
        low = polygon.min(axis=0) - margin_m
        high = polygon.max(axis=0) + margin_m
        candidates = camera_xy[
            np.all((camera_xy >= low) & (camera_xy <= high), axis=1)
        ]
        for point in candidates:
            distance = cv2.pointPolygonTest(
                polygon,
                (float(point[0]), float(point[1])),
                measureDist=True,
            )
            if distance >= -margin_m:
                return True
    return False


def extract_ground_proxy_points(
    points: np.ndarray,
    camera_centers: np.ndarray,
    corridor_m: float,
    max_points: int,
    seed: int,
) -> tuple[np.ndarray, dict[str, float | int]]:
    """Extract an automatic dominant-height slab as non-semantic road evidence."""
    finite = np.isfinite(points).all(axis=1)
    camera_xy = camera_centers[:, :2]
    low = camera_xy.min(axis=0) - 140.0
    high = camera_xy.max(axis=0) + 140.0
    roi = finite & np.all((points[:, :2] >= low) & (points[:, :2] <= high), axis=1)
    planar_distance_sq = np.min(
        np.sum((points[:, None, :2] - camera_xy[None, :, :]) ** 2, axis=2),
        axis=1,
    )
    roi &= planar_distance_sq <= corridor_m * corridor_m

    camera_height = float(np.median(camera_centers[:, 2]))
    plausible_ground = roi & (points[:, 2] >= camera_height - 3.0) & (
        points[:, 2] <= camera_height + 0.25
    )
    vertical = points[plausible_ground, 2]
    if len(vertical) < 500:
        vertical = points[roi, 2]
    if len(vertical) == 0:
        raise RuntimeError("Cannot extract a ground proxy from an empty finite cloud")

    lo, hi = np.percentile(vertical, [2.0, 98.0])
    if hi <= lo:
        dominant = float(np.median(vertical))
        band = 0.5
    else:
        bin_width = max(0.05, min(0.25, float(hi - lo) / 80.0))
        bins = np.arange(lo, hi + bin_width, bin_width)
        histogram, edges = np.histogram(vertical, bins=bins)
        index = int(np.argmax(histogram))
        dominant = float((edges[index] + edges[index + 1]) * 0.5)
        band = max(0.25, min(0.75, 2.5 * bin_width))

    keep = roi & (np.abs(points[:, 2] - dominant) <= band)
    if int(np.count_nonzero(keep)) < 2000:
        band = max(band, 1.0)
        keep = roi & (np.abs(points[:, 2] - dominant) <= band)
    indices = np.flatnonzero(keep)
    if len(indices) == 0:
        raise RuntimeError("Dominant-height extraction produced no ground-proxy points")
    if max_points > 0 and len(indices) > max_points:
        rng = np.random.default_rng(seed)
        indices = rng.choice(indices, max_points, replace=False)
    return points[indices], {
        "dominant_z_m": dominant,
        "ground_band_m": float(band),
        "ground_points": int(len(indices)),
        "camera_corridor_m": float(corridor_m),
    }


def reject_non_crossing(
    registered: np.ndarray,
    road: np.ndarray,
    metrics: dict[str, object],
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    refine = dict(metrics["refine"])
    refine.update(
        {
            "yaw": 0.0,
            "scale": 1.0,
            "tx": 0.0,
            "ty": 0.0,
            "accepted": False,
            "fallback_reason": "not_crossing_supported",
        }
    )
    metrics["refine"] = refine
    metrics["score_after"] = metrics["score_before"]
    metrics["refined_anchor_rmse_m"] = metrics["anchor_rmse_m"]
    return registered.copy(), road.copy(), metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply only MapPano3D BEV refinement to the standard global VGGT-Long placement."
    )
    parser.add_argument("--scene", nargs="+", required=True)
    parser.add_argument("--chunk-anchor-root", type=Path, required=True)
    parser.add_argument("--vggt-root", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--dataroot", type=Path, required=True)
    parser.add_argument("--version-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--chunk-size", type=int, default=40)
    parser.add_argument("--overlap", type=int, default=20)
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
    parser.add_argument(
        "--road-evidence",
        choices=("semantic", "ground_proxy"),
        default="semantic",
        help="Evidence used to optimize BEV transforms; evaluation always uses semantic roads.",
    )
    args = parser.parse_args()

    for scene in args.scene:
        target = np.load(args.input_root / scene / "camera_conditions.npz")["poses"].astype(np.float64)
        predicted = RUNNER.read_poses(args.vggt_root / scene / "VGGT" / "camera_poses.txt")
        ranges = RUNNER.chunk_ranges(len(target), args.chunk_size, args.overlap)
        global_alignment = fit_upright_sim3(predicted, target, target_forward_axis=2)
        global_cameras = np.asarray(global_alignment["aligned_centers"], dtype=np.float64)
        _, log_record, map_record = POST.scene_map_record(args.version_dir, scene)
        full_mask = POST.load_map_mask(args.dataroot / map_record["filename"], {})
        intersection_polygons = load_intersection_polygons(
            args.dataroot / "maps" / "expansion" / f"{log_record['location']}.json"
        )

        pose_points: list[np.ndarray] = []
        pose_colors: list[np.ndarray] = []
        refined_points: list[np.ndarray] = []
        refined_colors: list[np.ndarray] = []
        rows: list[dict[str, object]] = []
        for chunk_index, (start, end) in enumerate(ranges):
            chunk_dir = args.chunk_anchor_root / scene / "chunks" / f"chunk_{chunk_index:03d}"
            local_points, colors = load_cloud(chunk_dir / "pose_init_static.ply")
            local_road, _ = load_cloud(chunk_dir / "pose_init_road.ply")
            local_alignment = fit_upright_sim3(
                predicted[start:end], target[start:end], target_forward_axis=2
            )
            source_points = invert_sim3(local_points, local_alignment)
            source_road = invert_sim3(local_road, local_alignment)
            registered = RUNNER.apply_sim3(source_points, global_alignment)
            registered_road = RUNNER.apply_sim3(source_road, global_alignment)
            evaluation_road = registered_road
            if len(evaluation_road) > args.max_road_points:
                rng = np.random.default_rng(0)
                evaluation_road = evaluation_road[
                    rng.choice(len(evaluation_road), args.max_road_points, replace=False)
                ]
            refine_target = target[start:end].copy()
            refine_target[:, :3, 3] = global_cameras[start:end]
            if args.road_evidence == "ground_proxy":
                optimization_road, evidence_info = extract_ground_proxy_points(
                    registered,
                    global_cameras[start:end],
                    args.score_corridor_m,
                    args.max_road_points,
                    2000 + chunk_index,
                )
            else:
                optimization_road = evaluation_road
                evidence_info = {
                    "road_evidence": "semantic",
                    "evidence_points": int(len(optimization_road)),
                }
            has_crossing_support = crossing_supported(
                target[start:end, :2, 3],
                intersection_polygons,
                args.intersection_margin_m,
            )
            refined, _, metrics = RUNNER.refine_chunk(
                registered,
                optimization_road,
                global_cameras[start:end],
                refine_target,
                full_mask,
                args,
            )
            if not has_crossing_support and not args.refine_all_chunks:
                refined, _, metrics = reject_non_crossing(
                    registered,
                    optimization_road,
                    metrics,
                )
            optimization_before = metrics["score_before"]
            optimization_after = metrics["score_after"]
            refined_semantic_road = POST.apply_bev_to_points(
                evaluation_road,
                refine_target[:, :3, 3],
                metrics["refine"],
            )
            map_crop = POST.make_map_crop(
                full_mask,
                refine_target[:, :2, 3],
                args.map_resolution_m,
                margin_m=140.0,
            )
            target_road = POST.local_route_mask(
                map_crop,
                refine_target[:, :2, 3],
                args.score_corridor_m,
            )
            target_center = POST.skeletonize_mask(target_road)
            metrics["optimization_score_before"] = optimization_before
            metrics["optimization_score_after"] = optimization_after
            metrics["score_before"] = POST.route_score_metrics(
                evaluation_road[:, :2],
                map_crop,
                target_road,
                target_center,
                args.point_dilate_px,
            )
            metrics["score_after"] = POST.route_score_metrics(
                refined_semantic_road[:, :2],
                map_crop,
                target_road,
                target_center,
                args.point_dilate_px,
            )
            metrics["evidence"] = evidence_info
            pose_points.append(registered)
            refined_points.append(refined)
            if colors is not None:
                pose_colors.append(colors)
                refined_colors.append(colors)
            rows.append(
                {
                    "chunk": chunk_index,
                    "start": start,
                    "end": end,
                    "crossing_supported": has_crossing_support,
                    **metrics,
                }
            )

        output_dir = args.output_root / scene
        output_dir.mkdir(parents=True, exist_ok=True)
        merged_pose = np.concatenate(pose_points)
        merged_refined = np.concatenate(refined_points)
        merged_pose_colors = np.concatenate(pose_colors) if pose_colors else None
        merged_refined_colors = np.concatenate(refined_colors) if refined_colors else None
        counts = {
            "vggt_long": RUNNER.write_cloud(
                output_dir / "vggt_long_registered_static_cloud.ply",
                merged_pose,
                merged_pose_colors,
                args.voxel_m,
            ),
            "mappano3d_without_bev": RUNNER.write_cloud(
                output_dir / "registered_pose_init_cloud.ply",
                merged_pose,
                merged_pose_colors,
                args.voxel_m,
            ),
            "mappano3d": RUNNER.write_cloud(
                output_dir / "registered_refined_cloud.ply",
                merged_refined,
                merged_refined_colors,
                args.voxel_m,
            ),
        }
        report = {
            "scene": scene,
            "backbone": "VGGT",
            "placement_mode": "standard VGGT-Long global placement + local BEV refinement",
            "optimization_road_evidence": args.road_evidence,
            "evaluation_road_evidence": "semantic",
            "chunk_size_views": args.chunk_size,
            "overlap_views": args.overlap,
            "chunks": len(rows),
            "crossing_supported_chunks": sum(
                bool(row["crossing_supported"]) for row in rows
            ),
            "accepted_chunks": sum(bool(row["refine"]["accepted"]) for row in rows),
            "counts": counts,
            "global_vggt_alignment": {
                "identifiable": bool(global_alignment["identifiable"]),
                "scale": float(global_alignment["scale"]),
                "yaw_deg": float(global_alignment["yaw_deg"]),
            },
            "chunk_metrics": rows,
        }
        (output_dir / "metrics.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        print(
            f"{scene}: {report['accepted_chunks']}/{report['chunks']} chunks, "
            f"{counts['mappano3d']} points"
        )


if __name__ == "__main__":
    main()
