#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys

import cv2
import numpy as np
import open3d as o3d
from PIL import Image
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent))
from upright_sim3 import fit_upright_sim3

Image.MAX_IMAGE_PIXELS = None


VIEW_MODES = ("heading_fov60", "sweep_fov60", "panorama_stitch_e2")
METHODS = ("VGGT", "VGGT-Omega", "Pi3X", "Pi3X_camera", "PanoVGGT")


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def index_by_token(rows: list[dict]) -> dict[str, dict]:
    return {row["token"]: row for row in rows}


def read_poses(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    poses = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            vals = np.fromstring(line.strip(), sep=" ")
            if vals.size == 16:
                poses.append(vals.reshape(4, 4))
    if not poses:
        return None
    return np.stack(poses).astype(np.float64)


def umeyama_transform(source: np.ndarray, target: np.ndarray) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    src_mean = source.mean(axis=0)
    tgt_mean = target.mean(axis=0)
    src_centered = source - src_mean
    tgt_centered = target - tgt_mean
    cov = (tgt_centered.T @ src_centered) / len(source)
    u, singular, vt = np.linalg.svd(cov)
    sign = np.ones(3, dtype=np.float64)
    if np.linalg.det(u @ vt) < 0:
        sign[-1] = -1.0
    rot = u @ np.diag(sign) @ vt
    src_var = np.mean(np.sum(src_centered * src_centered, axis=1))
    scale = float(np.sum(singular * sign) / max(src_var, 1e-12))
    trans = tgt_mean - scale * (rot @ src_mean)
    aligned = scale * (source @ rot.T) + trans
    return scale, rot, trans, aligned


def apply_sim3(points: np.ndarray, scale: float, rot: np.ndarray, trans: np.ndarray) -> np.ndarray:
    return scale * (points @ rot.T) + trans


def path_from_done(value: str) -> Path:
    return Path(value)


def load_method_target(sequence_dir: Path) -> np.ndarray:
    conditions = sequence_dir / "camera_conditions.npz"
    if not conditions.exists():
        raise FileNotFoundError(conditions)
    return np.load(conditions)["poses"].astype(np.float64)


def scene_map_record(version_dir: Path, scene_name: str) -> tuple[dict, dict, dict]:
    scenes = load_json(version_dir / "scene.json")
    logs = index_by_token(load_json(version_dir / "log.json"))
    maps = load_json(version_dir / "map.json")
    scene = next(row for row in scenes if row["name"] == scene_name)
    log = logs[scene["log_token"]]
    map_record = next(row for row in maps if scene["log_token"] in row["log_tokens"])
    return scene, log, map_record


def load_map_mask(map_path: Path, cache: dict[str, np.ndarray]) -> np.ndarray:
    key = str(map_path)
    if key not in cache:
        cache[key] = np.asarray(Image.open(map_path).convert("L")) > 0
    return cache[key]


def make_map_crop(mask: np.ndarray, target_xy: np.ndarray, resolution: float, margin_m: float) -> dict[str, object]:
    height, width = mask.shape
    px = target_xy[:, 0] / resolution
    py = height - target_xy[:, 1] / resolution
    margin = int(round(margin_m / resolution))
    left = max(0, int(np.floor(px.min())) - margin)
    right = min(width, int(np.ceil(px.max())) + margin)
    top = max(0, int(np.floor(py.min())) - margin)
    bottom = min(height, int(np.ceil(py.max())) + margin)
    crop = mask[top:bottom, left:right]
    # cv2.distanceTransform measures distance to the nearest zero pixel, so road
    # pixels are zero and non-road pixels carry nearest-road distance.
    outside = (~crop).astype(np.uint8)
    distance = cv2.distanceTransform(outside, cv2.DIST_L2, 3).astype(np.float32) * resolution
    return {
        "mask": crop,
        "distance": distance,
        "left": left,
        "top": top,
        "full_height": height,
        "resolution": resolution,
    }


def xy_to_crop_px(xy: np.ndarray, crop: dict[str, object]) -> np.ndarray:
    if len(xy) == 0:
        return np.zeros((0, 2), dtype=np.float64)
    resolution = float(crop["resolution"])
    full_height = int(crop["full_height"])
    left = int(crop["left"])
    top = int(crop["top"])
    px = xy[:, 0] / resolution - left
    py = full_height - xy[:, 1] / resolution - top
    return np.stack([px, py], axis=1)


def crop_px_to_xy(xy_px: np.ndarray, crop: dict[str, object]) -> np.ndarray:
    if len(xy_px) == 0:
        return np.zeros((0, 2), dtype=np.float64)
    resolution = float(crop["resolution"])
    x = (xy_px[:, 0] + int(crop["left"])) * resolution
    y = (int(crop["full_height"]) - xy_px[:, 1] - int(crop["top"])) * resolution
    return np.stack([x, y], axis=1)


def rasterize_points(xy_px: np.ndarray, shape: tuple[int, int], radius_px: int) -> np.ndarray:
    height, width = shape
    mask = np.zeros((height, width), dtype=bool)
    if len(xy_px) == 0:
        return mask
    xi = np.rint(xy_px[:, 0]).astype(np.int64)
    yi = np.rint(xy_px[:, 1]).astype(np.int64)
    valid = (xi >= 0) & (xi < width) & (yi >= 0) & (yi < height)
    mask[yi[valid], xi[valid]] = True
    if radius_px > 0 and np.any(mask):
        mask = ndimage.binary_dilation(mask, iterations=radius_px)
    return mask


def local_route_mask(crop: dict[str, object], target_xy: np.ndarray, corridor_m: float) -> np.ndarray:
    road = crop["mask"].astype(bool)
    corridor_px = max(1, int(round(corridor_m / float(crop["resolution"]))))
    target_px = xy_to_crop_px(target_xy, crop)
    path = np.zeros(road.shape, dtype=np.uint8)
    if len(target_px) == 1:
        x, y = np.rint(target_px[0]).astype(int)
        if 0 <= x < road.shape[1] and 0 <= y < road.shape[0]:
            cv2.circle(path, (x, y), corridor_px, 1, thickness=-1)
    else:
        pts = np.rint(target_px).astype(np.int32)
        for p0, p1 in zip(pts[:-1], pts[1:]):
            cv2.line(path, tuple(p0), tuple(p1), 1, thickness=corridor_px * 2 + 1)
    target = road & (path > 0)
    if int(np.count_nonzero(target)) < 100:
        target = road
    return target


def restrict_crop_to_mask(crop: dict[str, object], target_mask: np.ndarray) -> dict[str, object]:
    restricted = dict(crop)
    restricted["mask"] = target_mask.astype(bool)
    outside = (~restricted["mask"]).astype(np.uint8)
    restricted["distance"] = (
        cv2.distanceTransform(outside, cv2.DIST_L2, 3).astype(np.float32) * float(crop["resolution"])
    )
    return restricted


def skeletonize_mask(mask: np.ndarray) -> np.ndarray:
    if not np.any(mask):
        return mask.copy()
    try:
        from skimage.morphology import skeletonize  # type: ignore

        return skeletonize(mask)
    except Exception:
        dist = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 3)
        local_max = dist == cv2.dilate(dist, np.ones((5, 5), dtype=np.uint8))
        return mask & local_max


def road_backbone_xy(ground_xy: np.ndarray, crop: dict[str, object], radius_px: int) -> np.ndarray:
    if len(ground_xy) == 0:
        return np.zeros((0, 2), dtype=np.float64)
    occupancy = rasterize_points(xy_to_crop_px(ground_xy, crop), crop["mask"].shape, radius_px)
    occupancy = ndimage.binary_closing(occupancy, iterations=2)
    backbone = skeletonize_mask(occupancy)
    py, px = np.nonzero(backbone)
    return crop_px_to_xy(np.stack([px, py], axis=1).astype(np.float64), crop)


def sample_field(field: np.ndarray, xy_px: np.ndarray, invalid_value: float = 1e3) -> np.ndarray:
    if len(xy_px) == 0:
        return np.zeros((0,), dtype=np.float64)
    h, w = field.shape
    xi = np.rint(xy_px[:, 0]).astype(np.int64)
    yi = np.rint(xy_px[:, 1]).astype(np.int64)
    valid = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
    out = np.full((len(xy_px),), invalid_value, dtype=np.float64)
    out[valid] = field[yi[valid], xi[valid]]
    return out


def distance_field_metrics(xy: np.ndarray, crop: dict[str, object]) -> tuple[float, float]:
    if len(xy) == 0:
        return 1e3, 1e3
    distances = sample_field(crop["distance"], xy_to_crop_px(xy, crop))
    return float(np.mean(distances)), float(np.percentile(distances, 90))


def route_score_metrics(
    xy: np.ndarray,
    crop: dict[str, object],
    target_road: np.ndarray,
    target_center: np.ndarray,
    point_dilate_px: int,
) -> dict[str, float | int]:
    xy_px = xy_to_crop_px(xy[np.isfinite(xy).all(axis=1)] if len(xy) else xy, crop)
    road_dt = ndimage.distance_transform_edt(~target_road)
    center_dt = ndimage.distance_transform_edt(~target_center) if np.any(target_center) else road_dt
    point_to_road = sample_field(road_dt, xy_px)
    point_to_center = sample_field(center_dt, xy_px)

    point_mask = rasterize_points(xy_px, target_road.shape, point_dilate_px)
    point_dt = ndimage.distance_transform_edt(~point_mask)
    ty, tx = np.nonzero(target_center if np.any(target_center) else target_road)
    target_to_point = point_dt[ty, tx] if len(tx) else np.array([1e3], dtype=np.float64)
    union = point_mask | target_road
    inter = point_mask & target_road

    outside_ratio = float(np.mean(point_to_road > 0.5)) if len(point_to_road) else 1.0
    outside_mean = float(np.mean(point_to_road)) if len(point_to_road) else 1e3
    outside_rmse = float(np.sqrt(np.mean(point_to_road**2))) if len(point_to_road) else 1e3
    outside_p90 = float(np.percentile(point_to_road, 90)) if len(point_to_road) else 1e3
    center_p50 = float(np.percentile(point_to_center, 50)) if len(point_to_center) else 1e3
    center_rmse = float(np.sqrt(np.mean(point_to_center**2))) if len(point_to_center) else 1e3
    center_p90 = float(np.percentile(point_to_center, 90)) if len(point_to_center) else 1e3
    coverage_3 = float(np.mean(target_to_point <= 3.0)) if len(target_to_point) else 0.0
    coverage_5 = float(np.mean(target_to_point <= 5.0)) if len(target_to_point) else 0.0
    target_to_point_mean = float(np.mean(target_to_point)) if len(target_to_point) else 1e3
    target_to_point_rmse = float(np.sqrt(np.mean(target_to_point**2))) if len(target_to_point) else 1e3
    target_to_point_p90 = float(np.percentile(target_to_point, 90)) if len(target_to_point) else 1e3
    iou = float(np.count_nonzero(inter) / max(np.count_nonzero(union), 1))
    fit_score = (
        outside_mean
        + 0.20 * outside_p90
        + 0.08 * center_p90
        + 0.10 * target_to_point_p90
        + 8.0 * outside_ratio
        + 4.0 * (1.0 - coverage_5)
    )
    return {
        "route_fit_score": float(fit_score),
        "road_points": int(len(xy_px)),
        "target_road_pixels": int(np.count_nonzero(target_road)),
        "target_center_pixels": int(np.count_nonzero(target_center)),
        "outside_ratio": outside_ratio,
        "outside_mean_px": outside_mean,
        "outside_rmse_px": outside_rmse,
        "outside_p90_px": outside_p90,
        "center_p50_px": center_p50,
        "center_rmse_px": center_rmse,
        "center_p90_px": center_p90,
        "target_to_point_mean_px": target_to_point_mean,
        "target_to_point_rmse_px": target_to_point_rmse,
        "target_to_point_p90_px": target_to_point_p90,
        "target_center_coverage_3px": coverage_3,
        "target_center_coverage_5px": coverage_5,
        "projection_iou": iou,
    }


def sample_ground_proxy(points: np.ndarray, max_points: int, rng: np.random.Generator) -> np.ndarray:
    finite = points[np.isfinite(points).all(axis=1)]
    if len(finite) == 0:
        return finite
    z = finite[:, 2]
    lo = np.percentile(z, 2)
    hi = np.percentile(z, 25)
    slab = finite[(z >= lo) & (z <= hi)]
    if len(slab) < 200:
        lo = np.percentile(z, 0)
        hi = np.percentile(z, 35)
        slab = finite[(z >= lo) & (z <= hi)]
    if len(slab) > max_points:
        idx = rng.choice(len(slab), size=max_points, replace=False)
        slab = slab[idx]
    return slab


def road_metrics(xy: np.ndarray, crop: dict[str, object]) -> dict[str, float]:
    if len(xy) == 0:
        return {
            "road_valid": 0,
            "road_outside": math.nan,
            "road_mean": math.nan,
            "road_p90": math.nan,
            "road_p95": math.nan,
        }
    mask = crop["mask"]
    distance = crop["distance"]
    resolution = float(crop["resolution"])
    full_height = int(crop["full_height"])
    left = int(crop["left"])
    top = int(crop["top"])
    h, w = mask.shape
    px = np.rint(xy[:, 0] / resolution).astype(np.int64) - left
    py = np.rint(full_height - xy[:, 1] / resolution).astype(np.int64) - top
    valid = (px >= 0) & (px < w) & (py >= 0) & (py < h)
    if not np.any(valid):
        return {
            "road_valid": 0,
            "road_outside": math.nan,
            "road_mean": math.nan,
            "road_p90": math.nan,
            "road_p95": math.nan,
        }
    px = px[valid]
    py = py[valid]
    d = distance[py, px]
    on_road = mask[py, px]
    return {
        "road_valid": int(len(px)),
        "road_outside": float(1.0 - np.mean(on_road)),
        "road_mean": float(np.mean(d)),
        "road_p90": float(np.percentile(d, 90)),
        "road_p95": float(np.percentile(d, 95)),
    }


def bev_delta(points_xy: np.ndarray, pivot: np.ndarray, yaw_deg: float, scale: float, trans_xy: tuple[float, float]) -> np.ndarray:
    yaw = math.radians(yaw_deg)
    c = math.cos(yaw)
    s = math.sin(yaw)
    rot = np.array([[c, -s], [s, c]], dtype=np.float64)
    return pivot + scale * ((points_xy - pivot) @ rot.T) + np.asarray(trans_xy, dtype=np.float64)


def perturb_target_poses(
    target: np.ndarray,
    yaw_deg: float,
    scale: float,
    trans_xy: tuple[float, float],
) -> np.ndarray:
    out = target.copy()
    pivot_xy = target[:, :2, 3].mean(axis=0)
    out[:, :2, 3] = bev_delta(target[:, :2, 3], pivot_xy, yaw_deg, scale, trans_xy)
    pivot_z = float(np.mean(target[:, 2, 3]))
    out[:, 2, 3] = pivot_z + scale * (target[:, 2, 3] - pivot_z)
    return out


def anchor_error(camera_xy: np.ndarray, target_xy: np.ndarray, pivot: np.ndarray, yaw_deg: float, scale: float, trans_xy: tuple[float, float]) -> np.ndarray:
    moved = bev_delta(camera_xy, pivot, yaw_deg, scale, trans_xy)
    return np.linalg.norm(moved - target_xy, axis=1)


def refine_bev(
    ground_xy: np.ndarray,
    backbone_xy: np.ndarray,
    camera_xy: np.ndarray,
    target_xy: np.ndarray,
    map_crop: dict[str, object],
    center_crop: dict[str, object],
    *,
    input_anchor_rmse: float,
    max_input_anchor_rmse: float,
    max_backbone_points: int,
    anchor_slack_m: float,
    anchor_weight: float,
    yaw_bound_deg: float,
    translation_bound_m: float,
    max_refine_translation_m: float,
    scale_bound_frac: float,
    allow_scale: bool,
    skeleton_weight: float,
    min_objective_improvement: float,
) -> tuple[dict[str, float], np.ndarray]:
    identity = {
        "yaw": 0.0,
        "scale": 1.0,
        "tx": 0.0,
        "ty": 0.0,
        "objective": math.nan,
        "base_objective": math.nan,
        "objective_improvement": 0.0,
        "accepted": False,
        "fallback_reason": "",
    }
    if len(ground_xy) == 0 or len(camera_xy) == 0:
        identity["fallback_reason"] = "missing_road_or_camera_evidence"
        return identity, camera_xy
    if len(backbone_xy) > max_backbone_points:
        identity["fallback_reason"] = "road_backbone_too_complex"
        return identity, camera_xy
    if input_anchor_rmse > max_input_anchor_rmse:
        identity["fallback_reason"] = "nonrigid_camera_anchor_residual"
        return identity, camera_xy
    pivot = target_xy.mean(axis=0)
    base_anchor = np.linalg.norm(camera_xy - target_xy, axis=1)
    base_rmse = float(np.sqrt(np.mean(base_anchor * base_anchor)))
    base_p90 = float(np.percentile(base_anchor, 90))
    max_rmse = base_rmse + anchor_slack_m
    max_p90 = base_p90 + anchor_slack_m
    base_metrics = road_metrics(ground_xy, map_crop)
    base_center_mean, base_center_p90 = distance_field_metrics(backbone_xy, center_crop)
    base_objective = (
        base_metrics["road_mean"]
        + 0.20 * base_metrics["road_p90"]
        + 3.0 * base_metrics["road_outside"]
        + skeleton_weight * (base_center_mean + 0.20 * base_center_p90)
        + anchor_weight * base_rmse
    )
    best = {
        **identity,
        "objective": float(base_objective),
        "base_objective": float(base_objective),
        "fallback_reason": "no_objective_improvement",
    }

    coarse_yaw_step = 2.0 if yaw_bound_deg > 2.0 else max(yaw_bound_deg, 0.5)
    coarse_trans_step = 1.0 if translation_bound_m > 1.0 else max(translation_bound_m, 0.25)
    coarse_yaw = np.arange(-yaw_bound_deg, yaw_bound_deg + 1e-9, coarse_yaw_step)
    coarse_trans = np.arange(-translation_bound_m, translation_bound_m + 1e-9, coarse_trans_step)
    if allow_scale and scale_bound_frac > 0:
        coarse_scale = np.linspace(1.0 - scale_bound_frac, 1.0 + scale_bound_frac, 5)
    else:
        coarse_scale = np.array([1.0])
    stages = [
        (coarse_yaw, coarse_trans, coarse_scale, False),
        (np.arange(-1.0, 1.01, 0.5), np.arange(-0.5, 0.51, 0.25), np.array([0.99, 1.0, 1.01]), True),
    ]
    center_yaw = 0.0
    center_tx = 0.0
    center_ty = 0.0
    center_scale = 1.0
    for yaw_offsets, trans_offsets, scales, relative_scale in stages:
        for yaw in center_yaw + yaw_offsets:
            stage_scales = center_scale * scales if relative_scale and allow_scale else scales
            for sc in stage_scales:
                if not allow_scale:
                    sc = 1.0
                for dx in center_tx + trans_offsets:
                    for dy in center_ty + trans_offsets:
                        if math.hypot(float(dx), float(dy)) > max_refine_translation_m + 1e-9:
                            continue
                        err = anchor_error(camera_xy, target_xy, pivot, yaw, sc, (dx, dy))
                        rmse = float(np.sqrt(np.mean(err * err)))
                        if rmse > max_rmse or float(np.percentile(err, 90)) > max_p90:
                            continue
                        moved_xy = bev_delta(ground_xy, pivot, yaw, sc, (dx, dy))
                        metrics = road_metrics(moved_xy, map_crop)
                        if metrics["road_valid"] == 0 or math.isnan(metrics["road_mean"]):
                            continue
                        moved_backbone = bev_delta(backbone_xy, pivot, yaw, sc, (dx, dy))
                        center_mean, center_p90 = distance_field_metrics(moved_backbone, center_crop)
                        objective = (
                            metrics["road_mean"]
                            + 0.20 * metrics["road_p90"]
                            + 3.0 * metrics["road_outside"]
                            + skeleton_weight * (center_mean + 0.20 * center_p90)
                            + anchor_weight * rmse
                            + 0.02 * abs(float(yaw))
                            + 0.03 * math.hypot(float(dx), float(dy))
                            + 4.0 * abs(math.log(max(float(sc), 1e-6)))
                        )
                        if objective < best["objective"]:
                            best = {
                                "yaw": float(yaw),
                                "scale": float(sc),
                                "tx": float(dx),
                                "ty": float(dy),
                                "objective": float(objective),
                            }
        center_yaw = best["yaw"]
        center_scale = best["scale"]
        center_tx = best["tx"]
        center_ty = best["ty"]
    improvement = float(base_objective - best["objective"])
    refine_translation = math.hypot(float(best["tx"]), float(best["ty"]))
    if refine_translation > max_refine_translation_m:
        best = {
            **identity,
            "objective": float(base_objective),
            "base_objective": float(base_objective),
            "objective_improvement": improvement,
            "fallback_reason": "translation_update_exceeds_limit",
        }
    elif improvement < min_objective_improvement:
        best = {
            **identity,
            "objective": float(base_objective),
            "base_objective": float(base_objective),
            "objective_improvement": improvement,
            "fallback_reason": "insufficient_objective_improvement",
        }
    else:
        best["base_objective"] = float(base_objective)
        best["objective_improvement"] = improvement
        best["accepted"] = True
        best["fallback_reason"] = ""
    moved_camera = bev_delta(camera_xy, pivot, best["yaw"], best["scale"], (best["tx"], best["ty"]))
    return best, moved_camera


def apply_bev_to_points(points: np.ndarray, target_xyz: np.ndarray, delta: dict[str, float]) -> np.ndarray:
    out = points.copy()
    pivot = target_xyz[:, :2].mean(axis=0)
    out[:, :2] = bev_delta(out[:, :2], pivot, delta["yaw"], delta["scale"], (delta["tx"], delta["ty"]))
    pivot_z = float(np.mean(target_xyz[:, 2]))
    out[:, 2] = pivot_z + delta["scale"] * (out[:, 2] - pivot_z)
    return out


def summarize_errors(err: np.ndarray) -> dict[str, float]:
    return {
        "anchor_rmse": float(np.sqrt(np.mean(err * err))),
        "anchor_median": float(np.median(err)),
        "anchor_p90": float(np.percentile(err, 90)),
    }


def process_one(
    *,
    dataroot: Path,
    version_dir: Path,
    long_root: Path,
    out_root: Path,
    view_mode: str,
    scene_name: str,
    method: str,
    resolution: float,
    max_ground_points: int,
    score_corridor_m: float,
    point_dilate_px: int,
    save_clouds: bool,
    map_cache: dict[str, np.ndarray],
    semantic_road_root: Path | None,
    semantic_static_root: Path | None,
    perturb_yaw_deg: float,
    perturb_scale: float,
    perturb_tx_m: float,
    perturb_ty_m: float,
    max_input_anchor_rmse: float,
    max_backbone_points: int,
    anchor_slack_m: float,
    anchor_weight: float,
    yaw_bound_deg: float,
    translation_bound_m: float,
    max_refine_translation_m: float,
    scale_bound_frac: float,
    allow_refine_scale: bool,
    skeleton_weight: float,
    min_objective_improvement: float,
) -> dict[str, object] | None:
    final_dir = long_root / view_mode / scene_name / method
    done_path = final_dir / "RUN_DONE.json"
    if not done_path.exists():
        return None
    done = load_json(done_path)
    pose_path = path_from_done(done.get("camera_poses", str(final_dir / "camera_poses.txt")))
    point_path = path_from_done(done["point_cloud"])
    sequence_dir = path_from_done(done["sequence_dir"])
    poses = read_poses(pose_path)
    if poses is None:
        return None
    nominal_target = load_method_target(sequence_dir)
    n = min(len(poses), len(nominal_target))
    poses = poses[:n]
    nominal_target = nominal_target[:n]
    target = perturb_target_poses(
        nominal_target,
        perturb_yaw_deg,
        perturb_scale,
        (perturb_tx_m, perturb_ty_m),
    )
    alignment = fit_upright_sim3(
        poses,
        target,
        target_forward_axis=0 if view_mode == "panorama_stitch_e2" else 2,
    )
    scale = float(alignment["scale"])
    rot = np.asarray(alignment["rotation"], dtype=np.float64)
    trans = np.asarray(alignment["translation"], dtype=np.float64)
    aligned_cameras = np.asarray(alignment["aligned_centers"], dtype=np.float64)
    long_err = np.linalg.norm(aligned_cameras - target[:, :3, 3], axis=1)
    input_anchor_rmse = float(np.sqrt(np.mean(long_err * long_err)))

    scene, log, map_record = scene_map_record(version_dir, scene_name)
    full_mask = load_map_mask(dataroot / map_record["filename"], map_cache)

    semantic_static_path = None
    if semantic_static_root is not None:
        semantic_static_path = semantic_static_root / scene_name / method / "semantic_static_points.ply"
    if semantic_static_path is not None and semantic_static_path.exists():
        pcd = o3d.io.read_point_cloud(str(semantic_static_path))
        cloud_evidence_source = "semantic_static_points"
    else:
        pcd = o3d.io.read_point_cloud(str(point_path))
        cloud_evidence_source = "unfiltered_reconstruction"
    points = np.asarray(pcd.points, dtype=np.float64)
    colors = np.asarray(pcd.colors, dtype=np.float64) if pcd.has_colors() else None
    registered = apply_sim3(points, scale, rot, trans)
    rng = np.random.default_rng(0)
    semantic_road_path = None
    if semantic_road_root is not None:
        semantic_road_path = semantic_road_root / scene_name / method / "semantic_road_points.ply"
    if semantic_road_path is not None and semantic_road_path.exists():
        road_cloud = o3d.io.read_point_cloud(str(semantic_road_path))
        road_local = np.asarray(road_cloud.points, dtype=np.float64)
        ground = apply_sim3(road_local, scale, rot, trans)
        if len(ground) > max_ground_points:
            ground = ground[rng.choice(len(ground), max_ground_points, replace=False)]
        road_evidence_source = "semantic_road_points"
    else:
        ground = sample_ground_proxy(registered, max_ground_points, rng)
        road_evidence_source = "ground_slab_proxy"
    map_crop = make_map_crop(full_mask, target[:, :2, 3], resolution, margin_m=140.0)
    target_road = local_route_mask(map_crop, target[:, :2, 3], score_corridor_m)
    target_center = skeletonize_mask(target_road)
    local_map_crop = restrict_crop_to_mask(map_crop, target_road)
    center_map_crop = restrict_crop_to_mask(map_crop, target_center)
    backbone_xy = road_backbone_xy(
        ground[:, :2] if len(ground) else ground,
        local_map_crop,
        point_dilate_px,
    )
    before_road = road_metrics(ground[:, :2] if len(ground) else ground, local_map_crop)
    before_score = route_score_metrics(
        ground[:, :2] if len(ground) else ground,
        map_crop,
        target_road,
        target_center,
        point_dilate_px,
    )

    delta, refined_camera_xy = refine_bev(
        ground[:, :2] if len(ground) else ground,
        backbone_xy,
        aligned_cameras[:, :2],
        target[:, :2, 3],
        local_map_crop,
        center_map_crop,
        input_anchor_rmse=input_anchor_rmse,
        max_input_anchor_rmse=max_input_anchor_rmse,
        max_backbone_points=max_backbone_points,
        anchor_slack_m=anchor_slack_m,
        anchor_weight=anchor_weight,
        yaw_bound_deg=yaw_bound_deg,
        translation_bound_m=translation_bound_m,
        max_refine_translation_m=max_refine_translation_m,
        scale_bound_frac=scale_bound_frac,
        allow_scale=allow_refine_scale,
        skeleton_weight=skeleton_weight,
        min_objective_improvement=min_objective_improvement,
    )
    refined_points = apply_bev_to_points(registered, target[:, :3, 3], delta)
    refined_ground = apply_bev_to_points(ground, target[:, :3, 3], delta) if len(ground) else ground
    refined_camera_xyz = apply_bev_to_points(aligned_cameras, target[:, :3, 3], delta)
    refined_camera_xy = refined_camera_xyz[:, :2]
    refined_err_3d = np.linalg.norm(refined_camera_xyz - target[:, :3, 3], axis=1)
    pose_init_nominal_err = np.linalg.norm(aligned_cameras - nominal_target[:, :3, 3], axis=1)
    refined_nominal_err = np.linalg.norm(refined_camera_xyz - nominal_target[:, :3, 3], axis=1)
    after_road = road_metrics(refined_ground[:, :2] if len(refined_ground) else refined_ground, local_map_crop)
    after_score = route_score_metrics(
        refined_ground[:, :2] if len(refined_ground) else refined_ground,
        map_crop,
        target_road,
        target_center,
        point_dilate_px,
    )

    out_dir = out_root / view_mode / scene_name / method
    out_dir.mkdir(parents=True, exist_ok=True)
    if save_clouds:
        pose_init_pcd = o3d.geometry.PointCloud()
        pose_init_pcd.points = o3d.utility.Vector3dVector(registered)
        refined_pcd = o3d.geometry.PointCloud()
        refined_pcd.points = o3d.utility.Vector3dVector(refined_points)
        if colors is not None and len(colors) == len(refined_points):
            pose_init_pcd.colors = o3d.utility.Vector3dVector(colors)
            refined_pcd.colors = o3d.utility.Vector3dVector(colors)
        o3d.io.write_point_cloud(str(out_dir / "registered_pose_init_cloud.ply"), pose_init_pcd, write_ascii=False)
        o3d.io.write_point_cloud(str(out_dir / "registered_refined_cloud.ply"), refined_pcd, write_ascii=False)
    np.savetxt(out_dir / "aligned_camera_centers.txt", aligned_cameras, fmt="%.8f")
    np.savetxt(out_dir / "refined_camera_centers.txt", refined_camera_xyz, fmt="%.8f")
    np.savetxt(out_dir / "input_anchor_camera_centers.txt", target[:, :3, 3], fmt="%.8f")
    np.savetxt(out_dir / "nominal_camera_centers.txt", nominal_target[:, :3, 3], fmt="%.8f")

    row: dict[str, object] = {
        "scene": scene_name,
        "location": log["location"],
        "view_mode": view_mode,
        "method": method,
        "image_count": int(n),
        "point_count": int(len(points)),
        "ground_proxy_count": int(len(ground)),
        "road_backbone_count": int(len(backbone_xy)),
        "road_evidence_source": road_evidence_source,
        "cloud_evidence_source": cloud_evidence_source,
        "map": map_record["filename"],
        "sim3_scale": float(scale),
        "sim3_alignment": alignment["alignment_mode"],
        "sim3_identifiable": alignment["identifiable"],
        "sim3_target_path_m": alignment["target_path_m"],
        "sim3_source_planar_variance": alignment["source_planar_variance"],
        "sim3_yaw_deg": alignment["yaw_deg"],
        "sim3_up_error_deg": alignment["up_error_deg"],
        "input_perturb_yaw_deg": float(perturb_yaw_deg),
        "input_perturb_scale": float(perturb_scale),
        "input_perturb_tx_m": float(perturb_tx_m),
        "input_perturb_ty_m": float(perturb_ty_m),
        "refine_yaw_deg": delta["yaw"],
        "refine_scale": delta["scale"],
        "refine_tx_m": delta["tx"],
        "refine_ty_m": delta["ty"],
        "refine_objective": delta["objective"],
        "refine_base_objective": delta["base_objective"],
        "refine_objective_improvement": delta["objective_improvement"],
        "refine_accepted": delta["accepted"],
        "refine_fallback_reason": delta["fallback_reason"],
    }
    row.update({f"long_{key}": value for key, value in summarize_errors(long_err).items()})
    row.update({f"mappano_{key}": value for key, value in summarize_errors(refined_err_3d).items()})
    row.update({f"pose_init_nominal_{key}": value for key, value in summarize_errors(pose_init_nominal_err).items()})
    row.update({f"mappano_nominal_{key}": value for key, value in summarize_errors(refined_nominal_err).items()})
    row.update({f"pose_init_{key}": value for key, value in before_road.items()})
    row.update({f"mappano_{key}": value for key, value in after_road.items()})
    row.update({f"pose_init_score_{key}": value for key, value in before_score.items()})
    row.update({f"mappano_score_{key}": value for key, value in after_score.items()})
    (out_dir / "metrics.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
    return row


def aggregate(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault((str(row["view_mode"]), str(row["method"])), []).append(row)
    out = []
    numeric_keys = [
        "long_anchor_rmse",
        "long_anchor_p90",
        "mappano_anchor_rmse",
        "mappano_anchor_p90",
        "pose_init_road_outside",
        "pose_init_road_mean",
        "pose_init_road_p90",
        "mappano_road_outside",
        "mappano_road_mean",
        "mappano_road_p90",
        "pose_init_score_route_fit_score",
        "pose_init_score_outside_ratio",
        "pose_init_score_outside_p90_px",
        "pose_init_score_center_p90_px",
        "pose_init_score_target_to_point_p90_px",
        "pose_init_score_target_center_coverage_5px",
        "pose_init_score_projection_iou",
        "mappano_score_route_fit_score",
        "mappano_score_outside_ratio",
        "mappano_score_outside_p90_px",
        "mappano_score_center_p90_px",
        "mappano_score_target_to_point_p90_px",
        "mappano_score_target_center_coverage_5px",
        "mappano_score_projection_iou",
    ]
    for (view_mode, method), items in sorted(grouped.items()):
        row: dict[str, object] = {"view_mode": view_mode, "method": method, "scene_count": len(items)}
        for key in numeric_keys:
            vals = np.asarray([float(item[key]) for item in items if not math.isnan(float(item[key]))], dtype=np.float64)
            row[f"mean_{key}"] = float(np.mean(vals)) if len(vals) else math.nan
        out.append(row)
    return out


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="MapPano3D-style nuScenes postprocess for Long baseline outputs.")
    parser.add_argument("--dataroot", type=Path, default=Path("data/nuscenes"))
    parser.add_argument("--version-dir", type=Path, default=None)
    parser.add_argument("--long-root", type=Path, default=Path("outputs/nuscenes/raw"))
    parser.add_argument("--out-root", type=Path, default=Path("outputs/nuscenes/refined"))
    parser.add_argument("--scene", nargs="*", default=None)
    parser.add_argument("--view-mode", nargs="+", choices=VIEW_MODES, default=list(VIEW_MODES))
    parser.add_argument("--method", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--resolution", type=float, default=0.1)
    parser.add_argument("--max-ground-points", type=int, default=12000)
    parser.add_argument("--score-corridor-m", type=float, default=35.0)
    parser.add_argument("--point-dilate-px", type=int, default=2)
    parser.add_argument("--semantic-road-root", type=Path, default=None)
    parser.add_argument("--semantic-static-root", type=Path, default=None)
    parser.add_argument("--perturb-yaw-deg", type=float, default=0.0)
    parser.add_argument("--perturb-scale", type=float, default=1.0)
    parser.add_argument("--perturb-tx-m", type=float, default=0.0)
    parser.add_argument("--perturb-ty-m", type=float, default=0.0)
    parser.add_argument("--max-input-anchor-rmse", type=float, default=1.5)
    parser.add_argument("--max-backbone-points", type=int, default=1500)
    parser.add_argument("--anchor-slack-m", type=float, default=4.5)
    parser.add_argument("--anchor-weight", type=float, default=0.15)
    parser.add_argument("--yaw-bound-deg", type=float, default=8.0)
    parser.add_argument("--translation-bound-m", type=float, default=4.0)
    parser.add_argument("--max-refine-translation-m", type=float, default=1.5)
    parser.add_argument("--scale-bound-frac", type=float, default=0.05)
    parser.add_argument("--fixed-scale", action="store_true")
    parser.add_argument("--skeleton-weight", type=float, default=0.35)
    parser.add_argument("--min-objective-improvement", type=float, default=0.10)
    parser.add_argument("--no-save-clouds", action="store_true")
    args = parser.parse_args()

    version_dir = args.version_dir or (args.dataroot / "v1.0-mini")
    scenes = args.scene
    if scenes is None:
        scenes = sorted(path.name for path in (args.long_root / args.view_mode[0]).iterdir() if path.is_dir())
    args.out_root.mkdir(parents=True, exist_ok=True)

    rows = []
    map_cache: dict[str, np.ndarray] = {}
    for scene_name in scenes:
        for view_mode in args.view_mode:
            for method in args.method:
                row = process_one(
                    dataroot=args.dataroot,
                    version_dir=version_dir,
                    long_root=args.long_root,
                    out_root=args.out_root,
                    view_mode=view_mode,
                    scene_name=scene_name,
                    method=method,
                    resolution=args.resolution,
                    max_ground_points=args.max_ground_points,
                    score_corridor_m=args.score_corridor_m,
                    point_dilate_px=args.point_dilate_px,
                    save_clouds=not args.no_save_clouds,
                    map_cache=map_cache,
                    semantic_road_root=args.semantic_road_root,
                    semantic_static_root=args.semantic_static_root,
                    perturb_yaw_deg=args.perturb_yaw_deg,
                    perturb_scale=args.perturb_scale,
                    perturb_tx_m=args.perturb_tx_m,
                    perturb_ty_m=args.perturb_ty_m,
                    max_input_anchor_rmse=args.max_input_anchor_rmse,
                    max_backbone_points=args.max_backbone_points,
                    anchor_slack_m=args.anchor_slack_m,
                    anchor_weight=args.anchor_weight,
                    yaw_bound_deg=args.yaw_bound_deg,
                    translation_bound_m=args.translation_bound_m,
                    max_refine_translation_m=args.max_refine_translation_m,
                    scale_bound_frac=args.scale_bound_frac,
                    allow_refine_scale=not args.fixed_scale,
                    skeleton_weight=args.skeleton_weight,
                    min_objective_improvement=args.min_objective_improvement,
                )
                if row is not None:
                    rows.append(row)
                    print(
                        f"[done] {view_mode}/{scene_name}/{method}: "
                        f"Long RMSE={row['long_anchor_rmse']:.3f}, "
                        f"MapPano score={row['mappano_score_route_fit_score']:.3f}"
                    )
    write_csv(args.out_root / "metrics_all.csv", rows)
    summary = aggregate(rows)
    write_csv(args.out_root / "metrics_summary.csv", summary)
    print(args.out_root / "metrics_summary.csv")


if __name__ == "__main__":
    main()
