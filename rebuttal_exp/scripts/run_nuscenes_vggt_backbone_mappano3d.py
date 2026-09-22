#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "tools"
sys.path.insert(0, str(TOOLS))

from upright_sim3 import fit_upright_sim3


POSTPROCESS_PATH = TOOLS / "run_nuscenes_mappano3d_long_postprocess.py"
SPEC = importlib.util.spec_from_file_location("nuscenes_map_postprocess", POSTPROCESS_PATH)
POST = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(POST)


EXCLUDED_STATIC_CLASSES = {
    "sky",
    "person",
    "rider",
    "car",
    "truck",
    "bus",
    "train",
    "motorcycle",
    "bicycle",
}


def read_poses(path: Path) -> np.ndarray:
    values = np.loadtxt(path, dtype=np.float64)
    if values.ndim == 1:
        values = values[None]
    if values.shape[1] != 16:
        raise ValueError(f"Expected flattened 4x4 poses in {path}, got {values.shape}")
    return values.reshape(-1, 4, 4)


def read_intrinsics(path: Path) -> np.ndarray:
    values = np.loadtxt(path, dtype=np.float64)
    if values.ndim == 1:
        values = values[None]
    if values.shape[1] != 4:
        raise ValueError(f"Expected fx fy cx cy in {path}, got {values.shape}")
    matrices = np.zeros((len(values), 3, 3), dtype=np.float64)
    matrices[:, 0, 0] = values[:, 0]
    matrices[:, 1, 1] = values[:, 1]
    matrices[:, 0, 2] = values[:, 2]
    matrices[:, 1, 2] = values[:, 3]
    matrices[:, 2, 2] = 1.0
    return matrices


def chunk_ranges(image_count: int, chunk_size: int, overlap: int) -> list[tuple[int, int]]:
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")
    if image_count <= chunk_size:
        return [(0, image_count)]
    step = chunk_size - overlap
    count = (image_count - overlap + step - 1) // step
    return [(i * step, min(i * step + chunk_size, image_count)) for i in range(count)]


def numbered_chunk_clouds(pcd_dir: Path) -> list[Path]:
    paths = [
        path
        for path in pcd_dir.glob("*_pcd.ply")
        if path.stem.removesuffix("_pcd").isdigit()
    ]
    return sorted(paths, key=lambda path: int(path.stem.removesuffix("_pcd")))


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def class_ids(classes_path: Path) -> tuple[set[int], set[int]]:
    record = json.loads(classes_path.read_text(encoding="utf-8"))
    road_ids = {
        int(item["id"])
        for item in record["classes"]
        if item["name"].strip().lower() == "road"
    }
    excluded_ids = {
        int(item["id"])
        for item in record["classes"]
        if item["name"].strip().lower() in EXCLUDED_STATIC_CLASSES
    }
    if not road_ids:
        raise RuntimeError(f"No road class in {classes_path}")
    return road_ids, excluded_ids


def semantic_subsets(
    points: np.ndarray,
    poses: np.ndarray,
    intrinsics: np.ndarray,
    masks: list[Path],
    road_ids: set[int],
    excluded_ids: set[int],
    min_vote_ratio: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    visible_votes = np.zeros(len(points), dtype=np.uint16)
    road_votes = np.zeros(len(points), dtype=np.uint16)
    excluded_votes = np.zeros(len(points), dtype=np.uint16)
    homogeneous = np.c_[points, np.ones(len(points), dtype=np.float64)]

    for pose, intrinsic, mask_path in zip(poses, intrinsics, masks):
        labels = np.asarray(Image.open(mask_path))
        width = max(1, int(round(2.0 * intrinsic[0, 2])))
        height = max(1, int(round(2.0 * intrinsic[1, 2])))
        if labels.shape[:2] != (height, width):
            labels = cv2.resize(labels, (width, height), interpolation=cv2.INTER_NEAREST)

        camera = (np.linalg.inv(pose) @ homogeneous.T).T[:, :3]
        depth = camera[:, 2]
        safe_depth = np.maximum(depth, 1e-9)
        u = np.rint(intrinsic[0, 0] * camera[:, 0] / safe_depth + intrinsic[0, 2]).astype(np.int64)
        v = np.rint(intrinsic[1, 1] * camera[:, 1] / safe_depth + intrinsic[1, 2]).astype(np.int64)
        valid = (depth > 1e-6) & (u >= 0) & (u < width) & (v >= 0) & (v < height)
        indices = np.flatnonzero(valid)
        if not len(indices):
            continue
        pixels = v[indices] * width + u[indices]
        order = np.argsort(depth[indices])
        _, first = np.unique(pixels[order], return_index=True)
        visible = indices[order[first]]
        visible_labels = labels[v[visible], u[visible]].astype(np.int64)
        visible_votes[visible] += 1
        road_votes[visible[np.isin(visible_labels, list(road_ids))]] += 1
        excluded_votes[visible[np.isin(visible_labels, list(excluded_ids))]] += 1

    road_ratio = np.divide(
        road_votes,
        visible_votes,
        out=np.zeros(len(points), dtype=np.float64),
        where=visible_votes > 0,
    )
    excluded_ratio = np.divide(
        excluded_votes,
        visible_votes,
        out=np.zeros(len(points), dtype=np.float64),
        where=visible_votes > 0,
    )
    road_keep = (visible_votes > 0) & (road_ratio >= min_vote_ratio)
    static_keep = (visible_votes == 0) | (excluded_ratio < min_vote_ratio)
    stats = {
        "points": int(len(points)),
        "visible_points": int(np.count_nonzero(visible_votes)),
        "road_points": int(np.count_nonzero(road_keep)),
        "static_points": int(np.count_nonzero(static_keep)),
        "excluded_points": int(np.count_nonzero(~static_keep)),
    }
    return road_keep, static_keep, stats


def apply_sim3(points: np.ndarray, alignment: dict[str, object]) -> np.ndarray:
    scale = float(alignment["scale"])
    rotation = np.asarray(alignment["rotation"], dtype=np.float64)
    translation = np.asarray(alignment["translation"], dtype=np.float64)
    return scale * (points @ rotation.T) + translation


def write_cloud(path: Path, points: np.ndarray, colors: np.ndarray | None, voxel_m: float) -> int:
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    if colors is not None and len(colors) == len(points):
        cloud.colors = o3d.utility.Vector3dVector(colors)
    if voxel_m > 0:
        cloud = cloud.voxel_down_sample(voxel_m)
    path.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(path), cloud, write_ascii=False)
    return len(cloud.points)


def refine_chunk(
    registered: np.ndarray,
    road: np.ndarray,
    aligned_cameras: np.ndarray,
    target: np.ndarray,
    full_mask: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    map_crop = POST.make_map_crop(full_mask, target[:, :2, 3], args.map_resolution_m, margin_m=140.0)
    target_road = POST.local_route_mask(map_crop, target[:, :2, 3], args.score_corridor_m)
    target_center = POST.skeletonize_mask(target_road)
    local_crop = POST.restrict_crop_to_mask(map_crop, target_road)
    center_crop = POST.restrict_crop_to_mask(map_crop, target_center)

    if len(road) > args.max_road_points:
        rng = np.random.default_rng(0)
        road = road[rng.choice(len(road), args.max_road_points, replace=False)]
    backbone = POST.road_backbone_xy(road[:, :2], local_crop, args.point_dilate_px)
    backbone_points_raw = len(backbone)
    if len(backbone) > args.max_backbone_points:
        rng = np.random.default_rng(0)
        backbone = backbone[
            rng.choice(len(backbone), args.max_backbone_points, replace=False)
        ]
    anchor_error = np.linalg.norm(aligned_cameras - target[:, :3, 3], axis=1)
    anchor_rmse = float(np.sqrt(np.mean(anchor_error * anchor_error)))
    before = POST.route_score_metrics(
        road[:, :2], map_crop, target_road, target_center, args.point_dilate_px
    )
    delta, _ = POST.refine_bev(
        road[:, :2],
        backbone,
        aligned_cameras[:, :2],
        target[:, :2, 3],
        local_crop,
        center_crop,
        input_anchor_rmse=anchor_rmse,
        max_input_anchor_rmse=args.max_input_anchor_rmse,
        max_backbone_points=args.max_backbone_points,
        anchor_slack_m=args.anchor_slack_m,
        anchor_weight=args.anchor_weight,
        yaw_bound_deg=args.yaw_bound_deg,
        translation_bound_m=args.translation_bound_m,
        max_refine_translation_m=args.max_refine_translation_m,
        scale_bound_frac=0.0,
        allow_scale=False,
        skeleton_weight=args.skeleton_weight,
        min_objective_improvement=args.min_objective_improvement,
    )
    optimizer_delta = dict(delta)
    refined = POST.apply_bev_to_points(registered, target[:, :3, 3], delta)
    refined_road = POST.apply_bev_to_points(road, target[:, :3, 3], delta)
    after = POST.route_score_metrics(
        refined_road[:, :2], map_crop, target_road, target_center, args.point_dilate_px
    )
    if bool(delta["accepted"]) and (
        float(after["route_fit_score"]) >= float(before["route_fit_score"])
    ):
        delta = {
            **delta,
            "yaw": 0.0,
            "scale": 1.0,
            "tx": 0.0,
            "ty": 0.0,
            "accepted": False,
            "fallback_reason": "route_score_not_improved",
        }
        refined = registered.copy()
        refined_road = road.copy()
        after = before
    refined_cameras = POST.apply_bev_to_points(
        aligned_cameras, target[:, :3, 3], delta
    )
    refined_anchor_error = np.linalg.norm(
        refined_cameras - target[:, :3, 3], axis=1
    )
    refined_anchor_rmse = float(
        np.sqrt(np.mean(refined_anchor_error * refined_anchor_error))
    )
    metrics: dict[str, object] = {
        "anchor_rmse_m": anchor_rmse,
        "refined_anchor_rmse_m": refined_anchor_rmse,
        "road_points_used": int(len(road)),
        "road_backbone_points_raw": int(backbone_points_raw),
        "road_backbone_points_used": int(len(backbone)),
        "optimizer_refine": optimizer_delta,
        "refine": delta,
        "score_before": before,
        "score_after": after,
    }
    return refined, refined_road, metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply MapPano3D chunk placement and BEV refinement to a VGGT backbone on nuScenes."
    )
    parser.add_argument("--scene", required=True)
    parser.add_argument("--vggt-dir", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--semantic-root", type=Path, required=True)
    parser.add_argument("--dataroot", type=Path, required=True)
    parser.add_argument("--version-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--placement-mode",
        choices=("chunk_anchors", "global_vggt"),
        default="chunk_anchors",
    )
    parser.add_argument("--chunk-size", type=int, default=40)
    parser.add_argument("--overlap", type=int, default=20)
    parser.add_argument("--semantic-vote-ratio", type=float, default=0.5)
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
    args = parser.parse_args()

    manifest = read_manifest(args.input_dir / "manifest.csv")
    conditions = np.load(args.input_dir / "camera_conditions.npz")
    target_poses = conditions["poses"].astype(np.float64)
    predicted_poses = read_poses(args.vggt_dir / "camera_poses.txt")
    intrinsics = read_intrinsics(args.vggt_dir / "intrinsic.txt")
    if not (len(manifest) == len(target_poses) == len(predicted_poses) == len(intrinsics)):
        raise ValueError("Manifest, target pose, predicted pose, and intrinsic counts must match")

    global_alignment = fit_upright_sim3(
        predicted_poses, target_poses, target_forward_axis=2
    )
    global_aligned_cameras = np.asarray(
        global_alignment["aligned_centers"], dtype=np.float64
    )

    road_ids, excluded_ids = class_ids(args.semantic_root / "classes.json")
    semantic_dir = args.semantic_root / "semantic_class_id" / args.scene
    semantic_paths = [semantic_dir / f"{Path(row['image']).stem}_sem.png" for row in manifest]
    missing = [path for path in semantic_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} semantic masks; first: {missing[0]}")

    ranges = chunk_ranges(len(manifest), args.chunk_size, args.overlap)
    pcd_paths = numbered_chunk_clouds(args.vggt_dir / "pcd")
    if len(pcd_paths) != len(ranges):
        raise ValueError(f"Expected {len(ranges)} chunk clouds, found {len(pcd_paths)}")

    _, _, map_record = POST.scene_map_record(args.version_dir, args.scene)
    full_mask = POST.load_map_mask(args.dataroot / map_record["filename"], {})

    baseline_points: list[np.ndarray] = []
    baseline_colors: list[np.ndarray] = []
    pose_points: list[np.ndarray] = []
    pose_colors: list[np.ndarray] = []
    refined_points: list[np.ndarray] = []
    refined_colors: list[np.ndarray] = []
    pose_roads: list[np.ndarray] = []
    refined_roads: list[np.ndarray] = []
    chunk_metrics: list[dict[str, object]] = []

    chunks_dir = args.output_dir / "chunks"
    for chunk_index, ((start, end), pcd_path) in enumerate(zip(ranges, pcd_paths)):
        cloud = o3d.io.read_point_cloud(str(pcd_path))
        points = np.asarray(cloud.points, dtype=np.float64)
        colors = np.asarray(cloud.colors, dtype=np.float64) if cloud.has_colors() else None
        road_keep, static_keep, semantic_stats = semantic_subsets(
            points,
            predicted_poses[start:end],
            intrinsics[start:end],
            semantic_paths[start:end],
            road_ids,
            excluded_ids,
            args.semantic_vote_ratio,
        )
        static_points = points[static_keep]
        static_colors = colors[static_keep] if colors is not None else None
        road_points = points[road_keep]
        baseline_points.append(static_points)
        if static_colors is not None:
            baseline_colors.append(static_colors)

        if args.placement_mode == "chunk_anchors":
            alignment = fit_upright_sim3(
                predicted_poses[start:end],
                target_poses[start:end],
                target_forward_axis=2,
            )
            refine_target = target_poses[start:end]
        else:
            alignment = global_alignment
            refine_target = target_poses[start:end].copy()
            refine_target[:, :3, 3] = global_aligned_cameras[start:end]
        registered = apply_sim3(static_points, alignment)
        registered_road = apply_sim3(road_points, alignment)
        aligned_cameras = (
            np.asarray(alignment["aligned_centers"], dtype=np.float64)
            if args.placement_mode == "chunk_anchors"
            else global_aligned_cameras[start:end]
        )
        refined, refined_road, refine_metrics = refine_chunk(
            registered,
            registered_road,
            aligned_cameras,
            refine_target,
            full_mask,
            args,
        )

        pose_points.append(registered)
        refined_points.append(refined)
        pose_roads.append(registered_road)
        refined_roads.append(refined_road)
        if static_colors is not None:
            pose_colors.append(static_colors)
            refined_colors.append(static_colors)

        chunk_dir = chunks_dir / f"chunk_{chunk_index:03d}"
        write_cloud(chunk_dir / "pose_init_static.ply", registered, static_colors, 0.0)
        write_cloud(chunk_dir / "refined_static.ply", refined, static_colors, 0.0)
        write_cloud(chunk_dir / "pose_init_road.ply", registered_road, None, 0.0)
        write_cloud(chunk_dir / "refined_road.ply", refined_road, None, 0.0)
        record: dict[str, object] = {
            "chunk": chunk_index,
            "start": start,
            "end": end,
            "images": end - start,
            "source_cloud": str(pcd_path),
            "semantic": semantic_stats,
            "sim3_scale": float(alignment["scale"]),
            "sim3_yaw_deg": float(alignment["yaw_deg"]),
            "sim3_up_error_deg": float(alignment["up_error_deg"]),
            **refine_metrics,
        }
        chunk_metrics.append(record)
        (chunk_dir / "metrics.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
        print(
            f"[{args.scene}] chunk {chunk_index + 1}/{len(ranges)} "
            f"static={len(static_points)} road={len(road_points)} "
            f"accepted={refine_metrics['refine']['accepted']}"
        )

    baseline_local = np.concatenate(baseline_points)
    baseline_color = np.concatenate(baseline_colors) if baseline_colors else None
    baseline_registered = apply_sim3(baseline_local, global_alignment)
    global_anchor_error = np.linalg.norm(
        np.asarray(global_alignment["aligned_centers"], dtype=np.float64)
        - target_poses[:, :3, 3],
        axis=1,
    )
    pose_merged = np.concatenate(pose_points)
    refined_merged = np.concatenate(refined_points)
    pose_color = np.concatenate(pose_colors) if pose_colors else None
    refined_color = np.concatenate(refined_colors) if refined_colors else None

    args.output_dir.mkdir(parents=True, exist_ok=True)
    counts = {
        "vggt_long": write_cloud(
            args.output_dir / "vggt_long_registered_static_cloud.ply",
            baseline_registered,
            baseline_color,
            args.voxel_m,
        ),
        "mappano3d_without_bev": write_cloud(
            args.output_dir / "registered_pose_init_cloud.ply",
            pose_merged,
            pose_color,
            args.voxel_m,
        ),
        "mappano3d": write_cloud(
            args.output_dir / "registered_refined_cloud.ply",
            refined_merged,
            refined_color,
            args.voxel_m,
        ),
        "pose_init_road": write_cloud(
            args.output_dir / "registered_pose_init_road.ply",
            np.concatenate(pose_roads),
            None,
            args.voxel_m,
        ),
        "refined_road": write_cloud(
            args.output_dir / "registered_refined_road.ply",
            np.concatenate(refined_roads),
            None,
            args.voxel_m,
        ),
    }
    summary = {
        "scene": args.scene,
        "backbone": "VGGT",
        "placement_mode": args.placement_mode,
        "input_protocol": "six native perspective views per timestamp",
        "chunk_size_views": args.chunk_size,
        "overlap_views": args.overlap,
        "chunks": len(ranges),
        "refine_accepted_chunks": int(sum(bool(row["refine"]["accepted"]) for row in chunk_metrics)),
        "counts": counts,
        "global_vggt_alignment": {
            "scale": float(global_alignment["scale"]),
            "yaw_deg": float(global_alignment["yaw_deg"]),
            "up_error_deg": float(global_alignment["up_error_deg"]),
            "alignment_mode": str(global_alignment["alignment_mode"]),
            "identifiable": bool(global_alignment["identifiable"]),
            "source_planar_variance": float(global_alignment["source_planar_variance"]),
            "camera_anchor_rmse_m": float(
                np.sqrt(np.mean(global_anchor_error * global_anchor_error))
            ),
            "camera_anchor_p90_m": float(np.percentile(global_anchor_error, 90)),
        },
        "chunk_metrics": chunk_metrics,
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(args.output_dir / "registered_refined_cloud.ply")


if __name__ == "__main__":
    main()
