#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import open3d as o3d
from PIL import Image


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
    poses = []
    for line in path.read_text(encoding="utf-8").splitlines():
        values = np.fromstring(line, sep=" ")
        if values.size == 16:
            poses.append(values.reshape(4, 4))
    if not poses:
        raise RuntimeError(f"No poses found in {path}")
    return np.stack(poses)


def project(points_camera: np.ndarray, width: int, height: int, vertical_sign: int):
    x, y, z = points_camera.T
    radius = np.linalg.norm(points_camera, axis=1)
    valid = np.isfinite(radius) & (radius > 1e-6)
    yaw = np.arctan2(x, z)
    pitch = np.arctan2(y, np.sqrt(x * x + z * z))
    u = ((yaw / (2.0 * math.pi) + 0.5) * width) % width
    v = (0.5 + vertical_sign * pitch / math.pi) * height
    valid &= (v >= 0) & (v < height)
    return np.rint(u).astype(np.int64) % width, np.clip(np.rint(v), 0, height - 1).astype(np.int64), radius, valid


def visible_indices(u: np.ndarray, v: np.ndarray, radius: np.ndarray, valid: np.ndarray, width: int) -> np.ndarray:
    indices = np.flatnonzero(valid)
    if len(indices) == 0:
        return indices
    pixel = v[indices] * width + u[indices]
    order = np.argsort(radius[indices])
    _, first = np.unique(pixel[order], return_index=True)
    return indices[order[first]]


def class_records(classes_path: Path) -> list[dict]:
    record = json.loads(classes_path.read_text(encoding="utf-8"))
    classes = list(record["classes"])
    if not classes:
        raise RuntimeError(f"No semantic classes found in {classes_path}")
    return classes


def road_class_ids(classes: list[dict], classes_path: Path) -> set[int]:
    ids = {int(item["id"]) for item in classes if item["name"].strip().lower() == "road"}
    if not ids:
        raise RuntimeError(f"Road class not found in {classes_path}")
    return ids


def write_cloud(path: Path, points: np.ndarray, colors: np.ndarray | None, keep: np.ndarray) -> None:
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points[keep])
    if colors is not None and len(colors) == len(points):
        cloud.colors = o3d.utility.Vector3dVector(colors[keep])
    o3d.io.write_point_cloud(str(path), cloud, write_ascii=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Project a PanoVGGT cloud into semantic panoramas and extract road points.")
    parser.add_argument("--cloud", type=Path, required=True)
    parser.add_argument("--poses", type=Path, required=True)
    parser.add_argument("--semantic-dir", type=Path, required=True)
    parser.add_argument("--classes", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-road-vote-ratio", type=float, default=0.5)
    parser.add_argument("--min-excluded-vote-ratio", type=float, default=0.5)
    parser.add_argument("--min-visible-votes", type=int, default=1)
    parser.add_argument(
        "--vertical-sign",
        type=int,
        choices=(-1, 1),
        default=1,
        help="Fixed ERP convention. PanoVGGT camera y points downward, so the default is +1.",
    )
    args = parser.parse_args()

    cloud = o3d.io.read_point_cloud(str(args.cloud))
    points = np.asarray(cloud.points, dtype=np.float64)
    colors = np.asarray(cloud.colors, dtype=np.float64) if cloud.has_colors() else None
    poses = read_poses(args.poses)
    semantic_paths = sorted(args.semantic_dir.glob("*_sem.png"))
    if not semantic_paths:
        raise RuntimeError("No matching poses and semantic masks")
    if len(poses) != len(semantic_paths):
        raise RuntimeError(
            f"Pose/mask count mismatch: {len(poses)} poses versus "
            f"{len(semantic_paths)} semantic masks. Refusing order-based truncation."
        )
    count = len(poses)
    classes = class_records(args.classes)
    road_ids = road_class_ids(classes, args.classes)
    class_names = {int(item["id"]): item["name"].strip().lower() for item in classes}
    class_count = max(class_names) + 1
    excluded_ids = {class_id for class_id, name in class_names.items() if name in EXCLUDED_STATIC_CLASSES}
    visible_votes = np.zeros(len(points), dtype=np.uint16)
    road_votes = np.zeros(len(points), dtype=np.uint16)
    semantic_votes = np.zeros((len(points), class_count), dtype=np.uint16)
    frame_stats = []

    homogeneous = np.c_[points, np.ones(len(points), dtype=np.float64)]
    for index in range(count):
        labels = np.asarray(Image.open(semantic_paths[index]))
        height, width = labels.shape[:2]
        camera_points = (np.linalg.inv(poses[index]) @ homogeneous.T).T[:, :3]
        u, v, radius, valid = project(camera_points, width, height, args.vertical_sign)
        visible = visible_indices(u, v, radius, valid, width)
        visible_labels = labels[v[visible], u[visible]].astype(np.int64)
        known = (visible_labels >= 0) & (visible_labels < class_count)
        if np.any(known):
            np.add.at(semantic_votes, (visible[known], visible_labels[known]), 1)
        road = np.isin(visible_labels, list(road_ids))
        road_ratio = float(np.mean(road)) if len(road) else 0.0
        visible_votes[visible] += 1
        road_votes[visible[road]] += 1
        frame_stats.append(
            {
                "frame": semantic_paths[index].stem,
                "visible_points": int(len(visible)),
                "road_points": int(np.sum(road)),
                "road_ratio": road_ratio,
                "vertical_sign": args.vertical_sign,
            }
        )

    ratio = np.divide(road_votes, visible_votes, out=np.zeros(len(points), dtype=np.float64), where=visible_votes > 0)
    keep = (visible_votes >= args.min_visible_votes) & (ratio >= args.min_road_vote_ratio)
    dominant_class = np.argmax(semantic_votes, axis=1)
    dominant_votes = semantic_votes[np.arange(len(points)), dominant_class]
    dominant_ratio = np.divide(
        dominant_votes,
        visible_votes,
        out=np.zeros(len(points), dtype=np.float64),
        where=visible_votes > 0,
    )
    confidently_excluded = np.isin(dominant_class, list(excluded_ids)) & (
        dominant_ratio >= args.min_excluded_vote_ratio
    )
    static_keep = ~confidently_excluded
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_cloud(args.output_dir / "semantic_road_points.ply", points, colors, keep)
    write_cloud(args.output_dir / "semantic_static_points.ply", points, colors, static_keep)
    excluded_counts = {
        class_names.get(class_id, str(class_id)): int(np.sum(confidently_excluded & (dominant_class == class_id)))
        for class_id in sorted(excluded_ids)
    }
    stats = {
        "input_points": int(len(points)),
        "semantic_road_points": int(np.sum(keep)),
        "semantic_static_points": int(np.sum(static_keep)),
        "semantic_excluded_points": int(np.sum(confidently_excluded)),
        "excluded_counts": excluded_counts,
        "visible_points": int(np.sum(visible_votes > 0)),
        "frames": count,
        "road_class_ids": sorted(road_ids),
        "min_road_vote_ratio": args.min_road_vote_ratio,
        "min_excluded_vote_ratio": args.min_excluded_vote_ratio,
        "min_visible_votes": args.min_visible_votes,
        "vertical_sign": args.vertical_sign,
        "frame_matching": "strict sorted one-to-one; equal counts required",
        "frame_stats": frame_stats,
    }
    (args.output_dir / "semantic_road_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(args.output_dir / "semantic_static_points.ply")


if __name__ == "__main__":
    main()
