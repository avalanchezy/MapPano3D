#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import html
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import open3d as o3d
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage


ROOT = Path(__file__).resolve().parents[1]


@dataclass
class ChunkRecord:
    chunk_id: str
    chunk_type: str
    node_id: str
    from_node_id: str
    to_node_id: str
    routes: str
    num_frames: int
    map_x: float
    map_y: float
    graph_length_px: float | None


@dataclass
class CandidateResult:
    mode: str
    yaw_deg: float
    dx: float
    dz: float
    scale: float
    score: float
    road_outside_mean: float
    road_outside_ratio: float
    road_dist_p50: float
    road_dist_p90: float
    skel_dist_mean: float
    skel_dist_p90: float
    sat_score: float
    reg: float
    node_anchor_error: float


def read_csv_dicts(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def group_rows(rows: list[dict[str, str]], key: str) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(row.get(key, ""), []).append(row)
    return grouped


def parse_float(value: str, default: float | None = None) -> float | None:
    value = (value or "").strip()
    if not value:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def read_chunk_records(path: Path) -> list[ChunkRecord]:
    records = []
    for row in read_csv_dicts(path):
        records.append(
            ChunkRecord(
                chunk_id=row["chunk_id"],
                chunk_type=row.get("chunk_type", ""),
                node_id=row.get("node_id", ""),
                from_node_id=row.get("from_node_id", ""),
                to_node_id=row.get("to_node_id", ""),
                routes=row.get("routes", ""),
                num_frames=int(float(row.get("num_frames") or 0)),
                map_x=float(row.get("map_x_px") or 0.0),
                map_y=float(row.get("map_y_px") or 0.0),
                graph_length_px=parse_float(row.get("graph_length_px", ""), None),
            )
        )
    return records


def read_graph(graph_root: Path) -> tuple[dict[str, tuple[float, float]], list[dict[str, str]]]:
    nodes = {}
    for row in read_csv_dicts(graph_root / "nodes.csv"):
        nodes[row["node_id"]] = (float(row["x_px"]), float(row["y_px"]))
    edges = read_csv_dicts(graph_root / "edges.csv")
    return nodes, edges


def load_metrics(path: Path) -> dict[str, dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    chunks = data.get("chunks", data if isinstance(data, list) else [])
    return {chunk["chunk_id"]: chunk for chunk in chunks}


def read_point_cloud(path: Path, max_points: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    pcd = o3d.io.read_point_cloud(str(path))
    points = np.asarray(pcd.points, dtype=np.float64)
    colors = np.asarray(pcd.colors, dtype=np.float64)
    if colors.shape[0] != points.shape[0]:
        colors = np.full((points.shape[0], 3), 0.75, dtype=np.float64)
    if max_points > 0 and len(points) > max_points:
        rng = np.random.default_rng(seed)
        keep = rng.choice(len(points), size=max_points, replace=False)
        points = points[keep]
        colors = colors[keep]
    return points, np.clip(colors, 0.0, 1.0)


def apply_sim3(points: np.ndarray, sim3: dict) -> np.ndarray:
    scale = float(sim3["scale"])
    rot = np.asarray(sim3["rotation"], dtype=np.float64)
    trans = np.asarray(sim3["translation"], dtype=np.float64)
    return scale * (points @ rot.T) + trans


def apply_base_registration(
    points: np.ndarray,
    metric: dict,
    pivot: tuple[float, float],
) -> np.ndarray:
    """Apply the same base registration used by registered_chunks."""
    out = apply_sim3(points, metric["sim3"])
    junction = metric.get("junction_refine")
    if junction and junction.get("junction_accepted"):
        out = transform_xz(
            out,
            float(junction["delta_yaw_deg"]),
            float(junction["delta_tx_px"]),
            float(junction["delta_tz_px"]),
            float(junction["delta_scale"]),
            pivot,
        )
    return out


def extract_ground_proxy_points(
    points: np.ndarray,
    map_size: tuple[int, int],
    max_points: int,
    seed: int,
) -> tuple[np.ndarray, dict[str, float | int]]:
    """Extract the automatic dominant-height slab used as baseline road evidence."""
    width, height = map_size
    finite = np.isfinite(points).all(axis=1)
    roi = (
        finite
        & (points[:, 0] >= -120.0)
        & (points[:, 0] <= width + 120.0)
        & (points[:, 2] >= -120.0)
        & (points[:, 2] <= height + 120.0)
    )
    vertical = points[roi, 1]
    if len(vertical) < 500:
        vertical = points[finite, 1]
    if len(vertical) == 0:
        raise RuntimeError("Cannot extract a ground proxy from an empty finite cloud")

    lo, hi = np.percentile(vertical, [2.0, 98.0])
    if hi <= lo:
        dominant = float(np.median(vertical))
        band = 8.0
    else:
        bin_width = max(1.5, min(5.0, float(hi - lo) / 80.0))
        bins = np.arange(lo, hi + bin_width, bin_width)
        hist, edges = np.histogram(vertical, bins=bins)
        index = int(np.argmax(hist))
        dominant = float((edges[index] + edges[index + 1]) * 0.5)
        band = max(5.0, min(12.0, 2.5 * bin_width))

    keep = roi & (np.abs(points[:, 1] - dominant) <= band)
    if int(np.count_nonzero(keep)) < 2000:
        band = max(band, 18.0)
        keep = roi & (np.abs(points[:, 1] - dominant) <= band)
    indices = np.flatnonzero(keep)
    if len(indices) == 0:
        raise RuntimeError("Dominant-height extraction produced no ground-proxy points")
    if max_points > 0 and len(indices) > max_points:
        rng = np.random.default_rng(seed)
        indices = rng.choice(indices, max_points, replace=False)
    return points[indices], {
        "dominant_y": dominant,
        "ground_band": float(band),
        "ground_points": int(len(indices)),
    }


def transform_xz(
    points: np.ndarray,
    yaw_deg: float,
    dx: float,
    dz: float,
    scale: float,
    pivot: tuple[float, float],
) -> np.ndarray:
    if len(points) == 0:
        return points.copy()
    yaw = math.radians(yaw_deg)
    ca, sa = math.cos(yaw), math.sin(yaw)
    out = points.copy()
    x = points[:, 0] - pivot[0]
    z = points[:, 2] - pivot[1]
    out[:, 0] = pivot[0] + scale * (ca * x - sa * z) + dx
    out[:, 2] = pivot[1] + scale * (sa * x + ca * z) + dz
    return out


def transform_xz_array(
    xy: np.ndarray,
    yaw_deg: float,
    dx: float,
    dz: float,
    scale: float,
    pivot: tuple[float, float],
) -> np.ndarray:
    if len(xy) == 0:
        return xy.copy()
    yaw = math.radians(yaw_deg)
    ca, sa = math.cos(yaw), math.sin(yaw)
    x = xy[:, 0] - pivot[0]
    y = xy[:, 1] - pivot[1]
    out = np.empty_like(xy, dtype=np.float64)
    out[:, 0] = pivot[0] + scale * (ca * x - sa * y) + dx
    out[:, 1] = pivot[1] + scale * (sa * x + ca * y) + dz
    return out


def draw_graph_masks(
    size: tuple[int, int],
    nodes: dict[str, tuple[float, float]],
    edges: list[dict[str, str]],
    half_width_px: float,
) -> tuple[np.ndarray, np.ndarray]:
    width, height = size
    corridor = Image.new("L", (width, height), 0)
    center = Image.new("L", (width, height), 0)
    cd = ImageDraw.Draw(corridor)
    ld = ImageDraw.Draw(center)
    corridor_width = max(3, int(round(2.0 * half_width_px)))
    center_width = 3
    for edge in edges:
        a = nodes.get(edge["from_node"])
        b = nodes.get(edge["to_node"])
        if not a or not b:
            continue
        cd.line([a, b], fill=255, width=corridor_width)
        ld.line([a, b], fill=255, width=center_width)
    return np.asarray(corridor) > 0, np.asarray(center) > 0


def make_distance_fields(
    map_size: tuple[int, int],
    nodes: dict[str, tuple[float, float]],
    edges: list[dict[str, str]],
    half_width_px: float,
) -> dict[str, np.ndarray]:
    corridor, center = draw_graph_masks(map_size, nodes, edges, half_width_px)
    return {
        "corridor": corridor,
        "center": center,
        "outside_dt": ndimage.distance_transform_edt(~corridor),
        "center_dt": ndimage.distance_transform_edt(~center),
    }


def fields_from_masks(corridor: np.ndarray, center: np.ndarray) -> dict[str, np.ndarray]:
    corridor = corridor.astype(bool)
    center = center.astype(bool)
    return {
        "corridor": corridor,
        "center": center,
        "outside_dt": ndimage.distance_transform_edt(~corridor),
        "center_dt": ndimage.distance_transform_edt(~center),
    }


def load_mask_distance_fields(
    map_size: tuple[int, int],
    road_mask_path: Path,
    road_centerline_path: Path | None,
    road_skeleton_path: Path | None,
    center_mode: str,
) -> dict[str, np.ndarray]:
    width, height = map_size
    road_img = Image.open(road_mask_path).convert("L")
    if road_img.size != map_size:
        road_img = road_img.resize(map_size, Image.Resampling.NEAREST)
    corridor = np.asarray(road_img) > 0

    centers = []
    if road_centerline_path is not None and road_centerline_path.exists():
        center_img = Image.open(road_centerline_path).convert("L")
        if center_img.size != map_size:
            center_img = center_img.resize(map_size, Image.Resampling.NEAREST)
        centers.append(("centerline", np.asarray(center_img) > 0))
    if road_skeleton_path is not None and road_skeleton_path.exists():
        skel_img = Image.open(road_skeleton_path).convert("L")
        if skel_img.size != map_size:
            skel_img = skel_img.resize(map_size, Image.Resampling.NEAREST)
        centers.append(("skeleton", np.asarray(skel_img) > 0))

    if center_mode == "centerline":
        center = next((mask for name, mask in centers if name == "centerline"), None)
    elif center_mode == "skeleton":
        center = next((mask for name, mask in centers if name == "skeleton"), None)
    elif center_mode == "union":
        center = None
        for _, mask in centers:
            center = mask.copy() if center is None else (center | mask)
    else:
        raise ValueError(f"unknown center_mode: {center_mode}")
    if center is None:
        center = skeletonize_road_points(np.column_stack(np.nonzero(corridor))[:, ::-1], (width, height))
        raster = np.zeros((height, width), dtype=bool)
        if len(center):
            xi = np.rint(center[:, 0]).astype(np.int64)
            yi = np.rint(center[:, 1]).astype(np.int64)
            valid = (xi >= 0) & (xi < width) & (yi >= 0) & (yi < height)
            raster[yi[valid], xi[valid]] = True
        center = raster

    return fields_from_masks(corridor, center)


def skeleton_mask(mask: np.ndarray) -> np.ndarray:
    if not np.any(mask):
        return mask.copy()
    try:
        from skimage.morphology import skeletonize  # type: ignore

        return skeletonize(mask)
    except Exception:
        dist = ndimage.distance_transform_edt(mask)
        local_max = dist == ndimage.maximum_filter(dist, size=5)
        return mask & local_max


def build_interval_mask(
    map_size: tuple[int, int],
    chunk: ChunkRecord,
    nodes: dict[str, tuple[float, float]],
    edges: list[dict[str, str]],
    item_rows: list[dict[str, str]],
    mode: str,
    interval_buffer_px: float,
    junction_radius_px: float,
) -> np.ndarray:
    width, height = map_size
    image = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(image)
    line_width = max(3, int(round(interval_buffer_px * 2.0 + 1.0)))

    def circle(x: float, y: float, r: float) -> None:
        draw.ellipse((x - r, y - r, x + r, y + r), fill=255)

    def line(points: list[tuple[float, float]]) -> None:
        if len(points) >= 2:
            draw.line(points, fill=255, width=line_width, joint="curve")
        for x, y in points:
            circle(x, y, max(2.0, interval_buffer_px * 0.6))

    drew = False
    if mode == "chunk_slam" and item_rows:
        by_route: dict[str, list[tuple[int, float, float]]] = {}
        for row in item_rows:
            try:
                order = int(float(row.get("item_order", "0")))
                x = float(row["map_x_px"])
                y = float(row["map_y_px"])
            except Exception:
                continue
            by_route.setdefault(row.get("route", ""), []).append((order, x, y))
        for vals in by_route.values():
            vals.sort(key=lambda item: item[0])
            pts = [(x, y) for _, x, y in vals]
            if pts:
                line(pts)
                drew = True

    if not drew:
        if chunk.chunk_type == "crossing_node" and chunk.node_id:
            node = nodes.get(chunk.node_id, (chunk.map_x, chunk.map_y))
            circle(node[0], node[1], junction_radius_px)
            for edge in edges:
                if edge.get("from_node") == chunk.node_id or edge.get("to_node") == chunk.node_id:
                    a = nodes.get(edge["from_node"])
                    b = nodes.get(edge["to_node"])
                    if a and b:
                        line([a, b])
                        drew = True
        elif chunk.from_node_id and chunk.to_node_id:
            a = nodes.get(chunk.from_node_id)
            b = nodes.get(chunk.to_node_id)
            if a and b:
                line([a, b])
                drew = True

    circle(chunk.map_x, chunk.map_y, max(interval_buffer_px, junction_radius_px if chunk.chunk_type == "crossing_node" else interval_buffer_px))
    return np.asarray(image) > 0


def localize_red_fields(
    global_fields: dict[str, np.ndarray],
    interval_mask: np.ndarray,
    min_road_pixels: int,
) -> tuple[dict[str, np.ndarray], dict[str, float | str]]:
    red_road = global_fields["corridor"] & interval_mask
    if int(np.count_nonzero(red_road)) < min_road_pixels:
        expanded = ndimage.binary_dilation(interval_mask, iterations=8)
        red_road = global_fields["corridor"] & expanded
        reason = "expanded_interval"
    else:
        reason = "ok"

    red_center = global_fields["center"] & ndimage.binary_dilation(red_road, iterations=3)
    if int(np.count_nonzero(red_center)) < 12 and np.any(red_road):
        red_center = skeleton_mask(red_road)
        reason = f"{reason}; skeletonized_center"
    if int(np.count_nonzero(red_road)) < min_road_pixels or int(np.count_nonzero(red_center)) < 8:
        # Keep the target local even when sparse. A bad local target should score badly,
        # not silently fall back to all roads in the map.
        reason = f"{reason}; sparse_local_target"
    return fields_from_masks(red_road, red_center), {
        "interval_road_pixels": float(np.count_nonzero(red_road)),
        "interval_center_pixels": float(np.count_nonzero(red_center)),
        "interval_reason": reason,
    }


def sample_field(field: np.ndarray, xy: np.ndarray, invalid_value: float = 1e3) -> np.ndarray:
    if len(xy) == 0:
        return np.zeros((0,), dtype=np.float64)
    h, w = field.shape
    xi = np.rint(xy[:, 0]).astype(np.int64)
    yi = np.rint(xy[:, 1]).astype(np.int64)
    valid = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
    out = np.full((len(xy),), invalid_value, dtype=np.float64)
    out[valid] = field[yi[valid], xi[valid]]
    return out


def road_stats(points_xz: np.ndarray, fields: dict[str, np.ndarray], half_width_px: float) -> dict[str, float]:
    if len(points_xz) == 0:
        return {
            "outside_mean": 1e3,
            "outside_ratio": 1.0,
            "dist_p50": 1e3,
            "dist_p90": 1e3,
        }
    outside = sample_field(fields["outside_dt"], points_xz)
    center = sample_field(fields["center_dt"], points_xz)
    return {
        "outside_mean": float(np.mean(outside)),
        "outside_ratio": float(np.mean(outside > 1e-6)),
        "dist_p50": float(np.percentile(center, 50)),
        "dist_p90": float(np.percentile(center, 90)),
    }


def skeletonize_road_points(
    points_xz: np.ndarray,
    map_size: tuple[int, int],
    crop_margin: int = 24,
) -> np.ndarray:
    if len(points_xz) < 32:
        return points_xz.copy()
    width, height = map_size
    min_x = max(0, int(math.floor(float(np.min(points_xz[:, 0])))) - crop_margin)
    max_x = min(width - 1, int(math.ceil(float(np.max(points_xz[:, 0])))) + crop_margin)
    min_y = max(0, int(math.floor(float(np.min(points_xz[:, 1])))) - crop_margin)
    max_y = min(height - 1, int(math.ceil(float(np.max(points_xz[:, 1])))) + crop_margin)
    if max_x <= min_x or max_y <= min_y:
        return points_xz.copy()

    crop_w = max_x - min_x + 1
    crop_h = max_y - min_y + 1
    mask = np.zeros((crop_h, crop_w), dtype=bool)
    xi = np.rint(points_xz[:, 0]).astype(np.int64) - min_x
    yi = np.rint(points_xz[:, 1]).astype(np.int64) - min_y
    valid = (xi >= 0) & (xi < crop_w) & (yi >= 0) & (yi < crop_h)
    mask[yi[valid], xi[valid]] = True
    mask = ndimage.binary_dilation(mask, iterations=3)
    mask = ndimage.binary_closing(mask, iterations=2)

    try:
        from skimage.morphology import skeletonize  # type: ignore

        skel = skeletonize(mask)
    except Exception:
        dist = ndimage.distance_transform_edt(mask)
        local_max = dist == ndimage.maximum_filter(dist, size=5)
        skel = mask & local_max & (dist >= max(1.5, float(np.percentile(dist[mask], 45))))

    yy, xx = np.nonzero(skel)
    if len(xx) < 8:
        return points_xz.copy()
    skel_xy = np.column_stack([xx + min_x, yy + min_y]).astype(np.float64)
    if len(skel_xy) > 1200:
        rng = np.random.default_rng(1234)
        skel_xy = skel_xy[rng.choice(len(skel_xy), 1200, replace=False)]
    return skel_xy


def graph_local_bbox(
    chunk: ChunkRecord,
    nodes: dict[str, tuple[float, float]],
    edges: list[dict[str, str]],
    pad: float,
) -> tuple[float, float, float, float]:
    pts = [(chunk.map_x, chunk.map_y)]
    if chunk.node_id and chunk.node_id in nodes:
        pts.append(nodes[chunk.node_id])
    if chunk.from_node_id and chunk.from_node_id in nodes:
        pts.append(nodes[chunk.from_node_id])
    if chunk.to_node_id and chunk.to_node_id in nodes:
        pts.append(nodes[chunk.to_node_id])
    for edge in edges:
        if chunk.node_id and (edge["from_node"] == chunk.node_id or edge["to_node"] == chunk.node_id):
            pts.append(nodes[edge["from_node"]])
            pts.append(nodes[edge["to_node"]])
        if chunk.from_node_id and chunk.to_node_id:
            if {edge["from_node"], edge["to_node"]} == {chunk.from_node_id, chunk.to_node_id}:
                pts.append(nodes[edge["from_node"]])
                pts.append(nodes[edge["to_node"]])
    arr = np.asarray(pts, dtype=np.float64)
    return (
        float(np.min(arr[:, 0]) - pad),
        float(np.min(arr[:, 1]) - pad),
        float(np.max(arr[:, 0]) + pad),
        float(np.max(arr[:, 1]) + pad),
    )


def union_bbox(
    arrays: list[np.ndarray],
    map_size: tuple[int, int],
    extra: tuple[float, float, float, float] | None,
    pad: float,
) -> tuple[int, int, int, int]:
    width, height = map_size
    xs = []
    ys = []
    for arr in arrays:
        if len(arr) == 0:
            continue
        xs.extend(arr[:, 0].tolist())
        ys.extend(arr[:, 1].tolist())
    if extra is not None:
        xs.extend([extra[0], extra[2]])
        ys.extend([extra[1], extra[3]])
    if not xs:
        return 0, 0, width - 1, height - 1
    x0 = max(0, int(math.floor(min(xs) - pad)))
    y0 = max(0, int(math.floor(min(ys) - pad)))
    x1 = min(width - 1, int(math.ceil(max(xs) + pad)))
    y1 = min(height - 1, int(math.ceil(max(ys) + pad)))
    if x1 <= x0:
        x1 = min(width - 1, x0 + 64)
    if y1 <= y0:
        y1 = min(height - 1, y0 + 64)
    return x0, y0, x1, y1


def build_map_gradient(map_img_low: Image.Image) -> np.ndarray:
    gray = np.asarray(map_img_low.convert("L"), dtype=np.float64) / 255.0
    gx = ndimage.sobel(gray, axis=1)
    gy = ndimage.sobel(gray, axis=0)
    grad = np.hypot(gx, gy)
    if float(np.max(grad)) > 0:
        grad /= float(np.max(grad))
    return grad


def candidate_score(
    mode: str,
    road_xz: np.ndarray,
    skel_xz: np.ndarray,
    full_xz: np.ndarray,
    fields: dict[str, np.ndarray],
    map_grad: np.ndarray,
    half_width_px: float,
    yaw_deg: float,
    dx: float,
    dz: float,
    scale: float,
    pivot: tuple[float, float],
    node_anchor_weight: float,
) -> CandidateResult:
    road_t = transform_xz_array(road_xz, yaw_deg, dx, dz, scale, pivot)
    skel_t = transform_xz_array(skel_xz, yaw_deg, dx, dz, scale, pivot)
    full_t = transform_xz_array(full_xz, yaw_deg, dx, dz, scale, pivot)

    road = road_stats(road_t, fields, half_width_px)
    skel_dist = sample_field(fields["center_dt"], skel_t)
    if len(skel_dist) == 0:
        skel_mean = 1e3
        skel_p90 = 1e3
    else:
        skel_mean = float(np.mean(skel_dist))
        skel_p90 = float(np.percentile(skel_dist, 90))
    sat = sample_field(map_grad, full_t, invalid_value=0.0)
    sat_score = float(1.0 - np.mean(sat)) if len(sat) else 1.0
    reg = (
        0.018 * math.hypot(dx, dz)
        + 0.045 * abs(yaw_deg)
        + 22.0 * abs(math.log(max(scale, 1e-6)))
    )
    node_anchor_error = math.hypot(dx, dz)
    # Node/SLAM anchors provide the initial pose and local search window.
    # The final 2D refinement score should be driven by the road mask/skeleton.
    node_anchor_term = 0.0

    corridor_term = road["outside_mean"] + 4.0 * road["outside_ratio"] + 0.05 * road["dist_p90"]
    skeleton_term = skel_mean + 0.20 * skel_p90
    satellite_term = 2.0 * sat_score
    if mode == "corridor":
        score = corridor_term + reg + node_anchor_term
    elif mode == "skeleton":
        score = skeleton_term + 0.6 * reg + node_anchor_term
    elif mode == "satellite":
        score = satellite_term + 0.35 * corridor_term + reg + node_anchor_term
    elif mode == "hybrid":
        score = 0.62 * corridor_term + 0.38 * skeleton_term + 0.20 * satellite_term + reg + node_anchor_term
    else:
        score = 0.62 * corridor_term + 0.38 * skeleton_term + reg + node_anchor_term

    return CandidateResult(
        mode=mode,
        yaw_deg=float(yaw_deg),
        dx=float(dx),
        dz=float(dz),
        scale=float(scale),
        score=float(score),
        road_outside_mean=road["outside_mean"],
        road_outside_ratio=road["outside_ratio"],
        road_dist_p50=road["dist_p50"],
        road_dist_p90=road["dist_p90"],
        skel_dist_mean=skel_mean,
        skel_dist_p90=skel_p90,
        sat_score=sat_score,
        reg=float(reg),
        node_anchor_error=float(node_anchor_error),
    )


def search_mode(
    mode: str,
    road_xz: np.ndarray,
    skel_xz: np.ndarray,
    full_xz: np.ndarray,
    fields: dict[str, np.ndarray],
    map_grad: np.ndarray,
    half_width_px: float,
    pivot: tuple[float, float],
    yaw_values: list[float],
    shift_values: list[float],
    scale_values: list[float],
    fine_yaw_offsets: list[float],
    fine_shift_offsets: list[float],
    fine_scale_multipliers: list[float],
    node_anchor_weight: float,
) -> CandidateResult:
    best: CandidateResult | None = None
    for yaw in yaw_values:
        for scale in scale_values:
            for dx in shift_values:
                for dz in shift_values:
                    result = candidate_score(
                        mode,
                        road_xz,
                        skel_xz,
                        full_xz,
                        fields,
                        map_grad,
                        half_width_px,
                        yaw,
                        dx,
                        dz,
                        scale,
                        pivot,
                        node_anchor_weight,
                    )
                    if best is None or result.score < best.score:
                        best = result
    assert best is not None

    fine_yaw = [best.yaw_deg + v for v in fine_yaw_offsets]
    fine_shift = fine_shift_offsets
    fine_scale = [best.scale * s for s in fine_scale_multipliers]
    for yaw in fine_yaw:
        for scale in fine_scale:
            for ddx in fine_shift:
                for ddz in fine_shift:
                    result = candidate_score(
                        mode,
                        road_xz,
                        skel_xz,
                        full_xz,
                        fields,
                        map_grad,
                        half_width_px,
                        yaw,
                        best.dx + ddx,
                        best.dz + ddz,
                        scale,
                        pivot,
                        node_anchor_weight,
                    )
                    if result.score < best.score:
                        best = result
    return best


def draw_graph(
    draw: ImageDraw.ImageDraw,
    nodes: dict[str, tuple[float, float]],
    edges: list[dict[str, str]],
    crop_low: tuple[int, int, int, int],
    scale_xy: tuple[float, float],
    half_width_px: float,
) -> None:
    sx, sy = scale_xy
    x0, y0, _, _ = crop_low
    for edge in edges:
        a = nodes.get(edge["from_node"])
        b = nodes.get(edge["to_node"])
        if not a or not b:
            continue
        line = [
            ((a[0] - x0) * sx, (a[1] - y0) * sy),
            ((b[0] - x0) * sx, (b[1] - y0) * sy),
        ]
        width = max(2, int(round(0.55 * half_width_px * (sx + sy) * 0.5)))
        draw.line(line, fill=(0, 96, 190, 80), width=width)
    for node_id, xy in nodes.items():
        px = (xy[0] - x0) * sx
        py = (xy[1] - y0) * sy
        r = 3
        draw.ellipse((px - r, py - r, px + r, py + r), fill=(255, 255, 255, 200), outline=(0, 96, 190, 135))
        draw.text((px + 4, py - 5), node_id, fill=(0, 75, 150, 150), font=ImageFont.load_default())


def blend_points(
    image: Image.Image,
    points_xz: np.ndarray,
    colors: np.ndarray,
    crop_low: tuple[int, int, int, int],
    scale_xy: tuple[float, float],
    alpha: float,
    radius: int = 1,
) -> None:
    if len(points_xz) == 0:
        return
    sx, sy = scale_xy
    x0, y0, _, _ = crop_low
    arr = np.asarray(image).copy()
    h, w = arr.shape[:2]
    xi = np.rint((points_xz[:, 0] - x0) * sx).astype(np.int64)
    yi = np.rint((points_xz[:, 1] - y0) * sy).astype(np.int64)
    valid = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
    xi = xi[valid]
    yi = yi[valid]
    cols = np.clip(colors[valid] * 255.0, 0, 255).astype(np.uint8)
    if radius <= 1:
        base = arr[yi, xi, :3].astype(np.float32)
        arr[yi, xi, :3] = (base * (1.0 - alpha) + cols.astype(np.float32) * alpha).astype(np.uint8)
    else:
        for x, y, color in zip(xi, yi, cols):
            y0p, y1p = max(0, y - radius), min(h, y + radius + 1)
            x0p, x1p = max(0, x - radius), min(w, x + radius + 1)
            base = arr[y0p:y1p, x0p:x1p, :3].astype(np.float32)
            arr[y0p:y1p, x0p:x1p, :3] = (
                base * (1.0 - alpha) + color.astype(np.float32) * alpha
            ).astype(np.uint8)
    image.paste(Image.fromarray(arr))


def draw_points(
    image: Image.Image,
    points_xz: np.ndarray,
    crop_low: tuple[int, int, int, int],
    scale_xy: tuple[float, float],
    color: tuple[int, int, int, int],
    radius: int,
    max_points: int,
    seed: int,
) -> None:
    if len(points_xz) == 0:
        return
    pts = points_xz
    if len(pts) > max_points:
        rng = np.random.default_rng(seed)
        pts = pts[rng.choice(len(pts), max_points, replace=False)]
    sx, sy = scale_xy
    x0, y0, _, _ = crop_low
    draw = ImageDraw.Draw(image, "RGBA")
    for x, y in pts:
        px = (x - x0) * sx
        py = (y - y0) * sy
        draw.ellipse((px - radius, py - radius, px + radius, py + radius), fill=color)


def make_panel(
    mode: str,
    result: CandidateResult,
    background_high: Image.Image,
    crop_low: tuple[int, int, int, int],
    scale_xy: tuple[float, float],
    nodes: dict[str, tuple[float, float]],
    edges: list[dict[str, str]],
    half_width_px: float,
    full_points: np.ndarray,
    full_colors: np.ndarray,
    road_xz: np.ndarray,
    skel_xz: np.ndarray,
    pivot: tuple[float, float],
    title: str,
    seed: int,
) -> Image.Image:
    sx, sy = scale_xy
    x0, y0, x1, y1 = crop_low
    crop_high = (
        int(round(x0 * sx)),
        int(round(y0 * sy)),
        int(round((x1 + 1) * sx)),
        int(round((y1 + 1) * sy)),
    )
    panel = background_high.crop(crop_high).convert("RGBA")
    draw_graph(ImageDraw.Draw(panel, "RGBA"), nodes, edges, crop_low, scale_xy, half_width_px)

    full_t = transform_xz(full_points, result.yaw_deg, result.dx, result.dz, result.scale, pivot)
    road_t = transform_xz_array(road_xz, result.yaw_deg, result.dx, result.dz, result.scale, pivot)
    skel_t = transform_xz_array(skel_xz, result.yaw_deg, result.dx, result.dz, result.scale, pivot)
    blend_points(panel, full_t[:, [0, 2]], full_colors, crop_low, scale_xy, alpha=0.58, radius=1)
    draw_points(panel, road_t, crop_low, scale_xy, (255, 48, 34, 92), radius=1, max_points=6000, seed=seed)
    draw_points(panel, skel_t, crop_low, scale_xy, (255, 0, 210, 215), radius=2, max_points=1800, seed=seed + 11)

    draw = ImageDraw.Draw(panel, "RGBA")
    font = ImageFont.load_default()
    box_h = 72
    draw.rectangle((0, 0, panel.size[0], box_h), fill=(255, 255, 255, 222))
    draw.text((10, 8), title, fill=(0, 0, 0, 255), font=font)
    draw.text(
        (10, 28),
        f"yaw {result.yaw_deg:+.1f} deg  dx {result.dx:+.1f}  dz {result.dz:+.1f}  scale {result.scale:.3f}",
        fill=(25, 35, 50, 255),
        font=font,
    )
    draw.text(
        (10, 48),
        f"outside {result.road_outside_ratio:.2f}  road p90 {result.road_dist_p90:.1f}  skel p90 {result.skel_dist_p90:.1f}",
        fill=(25, 35, 50, 255),
        font=font,
    )
    return panel.convert("RGB")


def paste_grid(panels: list[Image.Image], labels: list[str]) -> Image.Image:
    if not panels:
        return Image.new("RGB", (1, 1), "white")
    w = max(p.size[0] for p in panels)
    h = max(p.size[1] for p in panels)
    cols = 2
    rows = math.ceil(len(panels) / cols)
    out = Image.new("RGB", (cols * w, rows * h), (240, 240, 240))
    for idx, panel in enumerate(panels):
        x = (idx % cols) * w
        y = (idx // cols) * h
        out.paste(panel, (x, y))
    return out


def render_global_overlay(
    out_path: Path,
    background_high: Image.Image,
    map_size: tuple[int, int],
    nodes: dict[str, tuple[float, float]],
    edges: list[dict[str, str]],
    half_width_px: float,
    chunks_for_global: list[tuple[np.ndarray, np.ndarray, CandidateResult, tuple[float, float]]],
    max_points_per_chunk: int,
) -> None:
    low_w, low_h = map_size
    sx = background_high.size[0] / low_w
    sy = background_high.size[1] / low_h
    canvas = background_high.copy().convert("RGBA")
    draw_graph(ImageDraw.Draw(canvas, "RGBA"), nodes, edges, (0, 0, low_w - 1, low_h - 1), (sx, sy), half_width_px)
    for idx, (points, colors, result, pivot) in enumerate(chunks_for_global):
        pts = points
        cols = colors
        if len(pts) > max_points_per_chunk:
            rng = np.random.default_rng(9000 + idx)
            keep = rng.choice(len(pts), max_points_per_chunk, replace=False)
            pts = pts[keep]
            cols = cols[keep]
        pts_t = transform_xz(pts, result.yaw_deg, result.dx, result.dz, result.scale, pivot)
        blend_points(
            canvas,
            pts_t[:, [0, 2]],
            cols,
            (0, 0, low_w - 1, low_h - 1),
            (sx, sy),
            alpha=0.38,
            radius=1,
        )
    canvas.convert("RGB").save(out_path)


def write_index(
    out_dir: Path,
    rows: list[dict[str, str]],
    crossing_first: list[dict[str, str]],
) -> None:
    def link(path: str) -> str:
        return html.escape(path).replace("\\", "/")

    table_rows = []
    for row in rows:
        table_rows.append(
            "<tr>"
            f"<td>{html.escape(row['chunk_id'])}</td>"
            f"<td>{html.escape(row['chunk_type'])}</td>"
            f"<td>{html.escape(row['best_mode'])}</td>"
            f"<td>{row['base_road_outside_ratio']}</td>"
            f"<td>{row['hybrid_road_outside_ratio']}</td>"
            f"<td>{row['hybrid_skel_p90']}</td>"
            f"<td>{row['hybrid_yaw_deg']}</td>"
            f"<td>{row['hybrid_dx']}</td>"
            f"<td>{row['hybrid_dz']}</td>"
            f"<td><a href='{link(row['comparison_png'])}'>comparison</a></td>"
            "</tr>"
        )

    cards = []
    for row in crossing_first[:18]:
        cards.append(
            "<figure>"
            f"<a href='{link(row['comparison_png'])}'><img src='{link(row['comparison_png'])}'></a>"
            f"<figcaption>{html.escape(row['chunk_id'])} / {html.escape(row['chunk_type'])}</figcaption>"
            "</figure>"
        )

    html_text = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Map-unit BEV to 2D map refinement</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 24px; color: #172033; }}
h1 {{ font-size: 24px; margin-bottom: 8px; }}
.note {{ max-width: 960px; line-height: 1.45; color: #3b4658; }}
.globals img {{ width: min(100%, 980px); border: 1px solid #ccd3df; margin: 8px 0 18px; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(420px, 1fr)); gap: 14px; }}
figure {{ margin: 0; border: 1px solid #ccd3df; padding: 8px; background: #fff; }}
figure img {{ width: 100%; display: block; }}
figcaption {{ margin-top: 6px; font-size: 13px; }}
table {{ border-collapse: collapse; font-size: 12px; margin-top: 22px; }}
td, th {{ border: 1px solid #d7dde8; padding: 5px 7px; }}
th {{ background: #eef3fb; }}
</style>
</head>
<body>
<h1>Map-unit BEV to 2D map refinement</h1>
<p class="note">
Each panel overlays the fine 2D map segmentation, chunk BEV colors, road-only points, and extracted road skeleton.
Thin blue nodes/lines only indicate the SLAM/graph interval used to choose the local target; scoring is computed against the red fine road segmentation inside that interval. Lower outside ratio and skeleton distance are better.
</p>
<div class="globals">
<h2>Global overlays</h2>
<p><a href="global_base_overlay.png">base</a> / <a href="global_hybrid_overlay.png">hybrid</a></p>
<img src="global_hybrid_overlay.png">
</div>
<h2>Crossing and chunk comparisons</h2>
<div class="grid">
{''.join(cards)}
</div>
<h2>Metrics</h2>
<table>
<thead>
<tr><th>chunk</th><th>type</th><th>best</th><th>base outside</th><th>hybrid outside</th><th>hybrid skel p90</th><th>yaw</th><th>dx</th><th>dz</th><th>image</th></tr>
</thead>
<tbody>
{''.join(table_rows)}
</tbody>
</table>
</body>
</html>
"""
    (out_dir / "index.html").write_text(html_text, encoding="utf-8")


def fmt(value: float) -> str:
    if not math.isfinite(value):
        return ""
    return f"{value:.6g}"


def parse_float_list(text: str) -> list[float]:
    vals = []
    for part in text.split(","):
        part = part.strip()
        if part:
            vals.append(float(part))
    if not vals:
        raise ValueError("empty float list")
    return vals


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunk-root", type=Path, required=True)
    parser.add_argument("--base-recon", type=Path, required=True)
    parser.add_argument("--full-recon", type=Path, required=True)
    parser.add_argument("--graph-root", type=Path, required=True)
    parser.add_argument("--map", type=Path, required=True, help="Map image on the input coordinate grid.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--road-mask", type=Path, default=None)
    parser.add_argument("--road-centerline", type=Path, default=None)
    parser.add_argument("--road-skeleton", type=Path, default=None)
    parser.add_argument(
        "--road-center-mode",
        choices=("centerline", "skeleton", "union"),
        default="centerline",
    )
    parser.add_argument("--road-half-width-px", type=float, default=7.0)
    parser.add_argument("--max-road-points", type=int, default=45000)
    parser.add_argument("--max-full-points", type=int, default=140000)
    parser.add_argument("--max-score-full-points", type=int, default=12000)
    parser.add_argument(
        "--road-evidence",
        choices=("semantic", "ground_proxy"),
        default="semantic",
        help="Evidence used by refinement; the default preserves the original semantic-road behavior.",
    )
    parser.add_argument("--max-chunks", type=int, default=0)
    parser.add_argument("--crossing-node-anchor-weight", type=float, default=0.35)
    parser.add_argument("--straight-node-anchor-weight", type=float, default=0.10)
    parser.add_argument("--yaw-values", default="-5,-3,-1.5,0,1.5,3,5")
    parser.add_argument("--shift-values", default="-10,-6,-3,0,3,6,10")
    parser.add_argument("--scale-values", default="0.985,0.995,1.0,1.005,1.015")
    parser.add_argument("--fine-yaw-offsets", default="-1,-0.5,0,0.5,1")
    parser.add_argument("--fine-shift-offsets", default="-2,-1,0,1,2")
    parser.add_argument("--fine-scale-multipliers", default="0.995,1.0,1.005")
    parser.add_argument("--freeze-straight", action="store_true")
    parser.add_argument("--target-interval", choices=("global", "chunk_slam", "graph"), default="chunk_slam")
    parser.add_argument("--interval-buffer-px", type=float, default=34.0)
    parser.add_argument("--junction-interval-radius-px", type=float, default=76.0)
    parser.add_argument("--min-local-road-pixels", type=int, default=160)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    chunk_out = args.out / "chunks"
    chunk_out.mkdir(parents=True, exist_ok=True)

    current_map = Image.open(args.map).convert("RGB")
    background_high = current_map
    map_size = current_map.size
    scale_xy = (background_high.size[0] / map_size[0], background_high.size[1] / map_size[1])
    nodes, edges = read_graph(args.graph_root)
    metrics = load_metrics(args.base_recon / "registration_metrics.json")
    chunks = read_chunk_records(args.chunk_root / "chunk_manifest.csv")
    chunk_items_path = args.chunk_root / "chunk_items.csv"
    chunk_items = group_rows(read_csv_dicts(chunk_items_path), "chunk_id") if chunk_items_path.exists() else {}
    if args.max_chunks > 0:
        chunks = chunks[: args.max_chunks]

    if args.road_mask is not None:
        global_fields = load_mask_distance_fields(
            map_size,
            args.road_mask,
            args.road_centerline,
            args.road_skeleton,
            args.road_center_mode,
        )
    else:
        global_fields = make_distance_fields(map_size, nodes, edges, args.road_half_width_px)
    map_grad = build_map_gradient(current_map)

    yaw_values = parse_float_list(args.yaw_values)
    shift_values = parse_float_list(args.shift_values)
    scale_values = parse_float_list(args.scale_values)
    fine_yaw_offsets = parse_float_list(args.fine_yaw_offsets)
    fine_shift_offsets = parse_float_list(args.fine_shift_offsets)
    fine_scale_multipliers = parse_float_list(args.fine_scale_multipliers)

    summary_rows: list[dict[str, str]] = []
    json_rows: list[dict] = []
    base_global = []
    hybrid_global = []

    for idx, chunk in enumerate(chunks):
        metric = metrics.get(chunk.chunk_id)
        if not metric or "sim3" not in metric:
            print(f"skip {chunk.chunk_id}: no sim3")
            continue
        full_ply = args.full_recon / "registered_chunks" / f"{chunk.chunk_id}_map.ply"
        road_ply = args.base_recon / "road_clouds" / f"{chunk.chunk_id}.ply"
        if not full_ply.exists() or (args.road_evidence == "semantic" and not road_ply.exists()):
            print(f"skip {chunk.chunk_id}: missing ply")
            continue

        full_points, full_colors = read_point_cloud(full_ply, args.max_full_points, 1000 + idx)
        pivot = (chunk.map_x, chunk.map_y)
        evidence_info: dict[str, float | int | str]
        if args.road_evidence == "semantic":
            road_local, _ = read_point_cloud(road_ply, args.max_road_points, 2000 + idx)
            road_map = apply_base_registration(road_local, metric, pivot)
            evidence_info = {
                "road_evidence": "semantic",
                "road_reference_transform": "sim3_plus_accepted_junction_delta",
                "evidence_points": int(len(road_map)),
            }
        else:
            road_map, ground_info = extract_ground_proxy_points(
                full_points,
                map_size,
                args.max_road_points,
                2000 + idx,
            )
            evidence_info = {
                "road_evidence": "ground_proxy",
                "road_reference_transform": "registered_full_cloud_map_frame",
                "evidence_points": int(len(road_map)),
                **ground_info,
            }

        if len(road_map) > args.max_road_points:
            rng = np.random.default_rng(3000 + idx)
            road_map = road_map[rng.choice(len(road_map), args.max_road_points, replace=False)]

        road_xz = road_map[:, [0, 2]]
        skel_xz = skeletonize_road_points(road_xz, map_size)
        score_full = full_points[:, [0, 2]]
        if len(score_full) > args.max_score_full_points:
            rng = np.random.default_rng(4000 + idx)
            score_full = score_full[rng.choice(len(score_full), args.max_score_full_points, replace=False)]

        fields = global_fields
        interval_info: dict[str, float | str] = {
            "interval_road_pixels": float(np.count_nonzero(global_fields["corridor"])),
            "interval_center_pixels": float(np.count_nonzero(global_fields["center"])),
            "interval_reason": "global",
        }
        if args.target_interval != "global":
            interval_mask = build_interval_mask(
                map_size,
                chunk,
                nodes,
                edges,
                chunk_items.get(chunk.chunk_id, []),
                args.target_interval,
                args.interval_buffer_px,
                args.junction_interval_radius_px,
            )
            fields, interval_info = localize_red_fields(global_fields, interval_mask, args.min_local_road_pixels)

        node_anchor_weight = (
            args.crossing_node_anchor_weight
            if chunk.chunk_type == "crossing_node"
            else args.straight_node_anchor_weight
        )
        base = candidate_score(
            "base",
            road_xz,
            skel_xz,
            score_full,
            fields,
            map_grad,
            args.road_half_width_px,
            0.0,
            0.0,
            0.0,
            1.0,
            pivot,
            node_anchor_weight,
        )
        if args.freeze_straight and chunk.chunk_type == "straight_edge":
            corridor = CandidateResult(**{**base.__dict__, "mode": "corridor"})
            skeleton = CandidateResult(**{**base.__dict__, "mode": "skeleton"})
            hybrid = CandidateResult(**{**base.__dict__, "mode": "hybrid"})
        else:
            corridor = search_mode(
                "corridor",
                road_xz,
                skel_xz,
                score_full,
                fields,
                map_grad,
                args.road_half_width_px,
                pivot,
                yaw_values,
                shift_values,
                scale_values,
                fine_yaw_offsets,
                fine_shift_offsets,
                fine_scale_multipliers,
                node_anchor_weight,
            )
            skeleton = search_mode(
                "skeleton",
                road_xz,
                skel_xz,
                score_full,
                fields,
                map_grad,
                args.road_half_width_px,
                pivot,
                yaw_values,
                shift_values,
                scale_values,
                fine_yaw_offsets,
                fine_shift_offsets,
                fine_scale_multipliers,
                node_anchor_weight,
            )
            hybrid = search_mode(
                "hybrid",
                road_xz,
                skel_xz,
                score_full,
                fields,
                map_grad,
                args.road_half_width_px,
                pivot,
                yaw_values,
                shift_values,
                scale_values,
                fine_yaw_offsets,
                fine_shift_offsets,
                fine_scale_multipliers,
                node_anchor_weight,
            )
        results = {"base": base, "corridor": corridor, "skeleton": skeleton, "hybrid": hybrid}
        best_mode = min(("corridor", "skeleton", "hybrid"), key=lambda name: results[name].score)

        full_xz = full_points[:, [0, 2]]
        graph_bbox = graph_local_bbox(chunk, nodes, edges, pad=72.0)
        transformed_arrays = [
            transform_xz_array(road_xz, r.yaw_deg, r.dx, r.dz, r.scale, pivot)
            for r in results.values()
        ]
        transformed_arrays += [
            transform_xz(full_points, r.yaw_deg, r.dx, r.dz, r.scale, pivot)[:, [0, 2]]
            for r in results.values()
        ]
        crop = union_bbox(transformed_arrays, map_size, graph_bbox, pad=28.0)

        panels = [
            make_panel(
                mode,
                result,
                background_high,
                crop,
                scale_xy,
                nodes,
                edges,
                args.road_half_width_px,
                full_points,
                full_colors,
                road_xz,
                skel_xz,
                pivot,
                f"{chunk.chunk_id} / {mode}",
                seed=5000 + idx,
            )
            for mode, result in results.items()
        ]
        comp = paste_grid(panels, list(results.keys()))
        chunk_dir = chunk_out / chunk.chunk_id
        chunk_dir.mkdir(parents=True, exist_ok=True)
        comp_path = chunk_dir / "comparison.png"
        comp.save(comp_path)

        base_global.append((full_points, full_colors, base, pivot))
        hybrid_global.append((full_points, full_colors, hybrid, pivot))

        row = {
            "chunk_id": chunk.chunk_id,
            "chunk_type": chunk.chunk_type,
            "routes": chunk.routes,
            "num_frames": str(chunk.num_frames),
            "road_evidence": args.road_evidence,
            "evidence_points": str(evidence_info["evidence_points"]),
            "best_mode": best_mode,
            "comparison_png": str(comp_path.relative_to(args.out)).replace("\\", "/"),
            "base_road_outside_ratio": fmt(base.road_outside_ratio),
            "base_road_p90": fmt(base.road_dist_p90),
            "corridor_score": fmt(corridor.score),
            "corridor_road_outside_ratio": fmt(corridor.road_outside_ratio),
            "corridor_skel_p90": fmt(corridor.skel_dist_p90),
            "corridor_yaw_deg": fmt(corridor.yaw_deg),
            "corridor_dx": fmt(corridor.dx),
            "corridor_dz": fmt(corridor.dz),
            "corridor_scale": fmt(corridor.scale),
            "skeleton_score": fmt(skeleton.score),
            "skeleton_road_outside_ratio": fmt(skeleton.road_outside_ratio),
            "skeleton_skel_p90": fmt(skeleton.skel_dist_p90),
            "skeleton_yaw_deg": fmt(skeleton.yaw_deg),
            "skeleton_dx": fmt(skeleton.dx),
            "skeleton_dz": fmt(skeleton.dz),
            "skeleton_scale": fmt(skeleton.scale),
            "hybrid_score": fmt(hybrid.score),
            "hybrid_road_outside_ratio": fmt(hybrid.road_outside_ratio),
            "hybrid_road_p90": fmt(hybrid.road_dist_p90),
            "hybrid_skel_p90": fmt(hybrid.skel_dist_p90),
            "hybrid_sat_score": fmt(hybrid.sat_score),
            "hybrid_node_anchor_error": fmt(hybrid.node_anchor_error),
            "hybrid_yaw_deg": fmt(hybrid.yaw_deg),
            "hybrid_dx": fmt(hybrid.dx),
            "hybrid_dz": fmt(hybrid.dz),
            "hybrid_scale": fmt(hybrid.scale),
            "target_interval": args.target_interval,
            "interval_road_pixels": fmt(float(interval_info["interval_road_pixels"])),
            "interval_center_pixels": fmt(float(interval_info["interval_center_pixels"])),
            "interval_reason": str(interval_info["interval_reason"]),
        }
        summary_rows.append(row)
        json_rows.append(
            {
                "chunk": chunk.__dict__,
                "evidence": evidence_info,
                "best_mode": best_mode,
                "results": {name: result.__dict__ for name, result in results.items()},
                "comparison_png": row["comparison_png"],
            }
        )
        print(
            f"[{idx + 1:02d}/{len(chunks):02d}] {chunk.chunk_id}: "
            f"base outside={base.road_outside_ratio:.3f}, "
            f"hybrid outside={hybrid.road_outside_ratio:.3f}, "
            f"hybrid skel_p90={hybrid.skel_dist_p90:.2f}, "
            f"delta=({hybrid.yaw_deg:+.1f},{hybrid.dx:+.1f},{hybrid.dz:+.1f},{hybrid.scale:.3f})"
        )

    if summary_rows:
        fieldnames = list(summary_rows[0].keys())
        with (args.out / "chunk_refine_summary.csv").open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summary_rows)
        (args.out / "chunk_refine_metrics.json").write_text(
            json.dumps(
                {
                    "map": str(args.map),
                    "map_size_low": map_size,
                    "map_size_high": background_high.size,
                    "road_half_width_px": args.road_half_width_px,
                    "road_reference_transform": (
                        "sim3_plus_accepted_junction_delta"
                        if args.road_evidence == "semantic"
                        else "registered_full_cloud_map_frame"
                    ),
                    "run_config": {
                        "chunk_root": str(args.chunk_root),
                        "base_recon": str(args.base_recon),
                        "full_recon": str(args.full_recon),
                        "graph_root": str(args.graph_root),
                        "road_mask": str(args.road_mask) if args.road_mask else None,
                        "road_centerline": str(args.road_centerline) if args.road_centerline else None,
                        "road_skeleton": str(args.road_skeleton) if args.road_skeleton else None,
                        "road_center_mode": args.road_center_mode,
                        "road_evidence": args.road_evidence,
                        "yaw_values": yaw_values,
                        "shift_values": shift_values,
                        "scale_values": scale_values,
                        "fine_yaw_offsets": fine_yaw_offsets,
                        "fine_shift_offsets": fine_shift_offsets,
                        "fine_scale_multipliers": fine_scale_multipliers,
                        "freeze_straight": args.freeze_straight,
                        "target_interval": args.target_interval,
                        "interval_buffer_px": args.interval_buffer_px,
                        "junction_interval_radius_px": args.junction_interval_radius_px,
                        "min_local_road_pixels": args.min_local_road_pixels,
                    },
                    "chunks": json_rows,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        crossing_first = sorted(
            summary_rows,
            key=lambda row: (0 if row["chunk_type"] == "crossing_node" else 1, row["chunk_id"]),
        )
        write_index(args.out, summary_rows, crossing_first)
        render_global_overlay(
            args.out / "global_base_overlay.png",
            background_high,
            map_size,
            nodes,
            edges,
            args.road_half_width_px,
            base_global,
            max_points_per_chunk=18000,
        )
        render_global_overlay(
            args.out / "global_hybrid_overlay.png",
            background_high,
            map_size,
            nodes,
            edges,
            args.road_half_width_px,
            hybrid_global,
            max_points_per_chunk=18000,
        )
    print(f"done: {args.out}")


if __name__ == "__main__":
    main()
