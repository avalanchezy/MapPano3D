#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import open3d as o3d
from PIL import Image, ImageDraw
from scipy import ndimage


ROOT = Path(__file__).resolve().parents[1]


def read_csv_dicts(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: float | int | str | None) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if not math.isfinite(float(value)):
        return ""
    return f"{float(value):.6g}"


def group_rows(rows: list[dict[str, str]], key: str) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(row.get(key, ""), []).append(row)
    return grouped


def load_point_cloud(path: Path, max_points: int, seed: int) -> np.ndarray:
    pcd = o3d.io.read_point_cloud(str(path))
    points = np.asarray(pcd.points, dtype=np.float64)
    points = points[np.isfinite(points).all(axis=1)]
    if max_points > 0 and len(points) > max_points:
        rng = np.random.default_rng(seed)
        points = points[rng.choice(len(points), max_points, replace=False)]
    return points


def apply_sim3(points: np.ndarray, sim3: dict) -> np.ndarray:
    scale = float(sim3["scale"])
    rot = np.asarray(sim3["rotation"], dtype=np.float64)
    trans = np.asarray(sim3["translation"], dtype=np.float64)
    return scale * (points @ rot.T) + trans


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


def delta_params_from_junction(rec: dict) -> tuple[float, float, float, float] | None:
    info = rec.get("junction_refine") or {}
    if not rec.get("junction_accepted"):
        return None
    keys = ("delta_yaw_deg", "delta_tx_px", "delta_tz_px", "delta_scale")
    if not all(k in info and info[k] is not None for k in keys):
        return None
    return (
        float(info["delta_yaw_deg"]),
        float(info["delta_tx_px"]),
        float(info["delta_tz_px"]),
        float(info["delta_scale"]),
    )


def load_bev_refine_selection(refine_dir: Path | None) -> dict[str, dict]:
    if refine_dir is None:
        return {}
    metrics_path = refine_dir / "chunk_refine_metrics.json"
    selected_path = refine_dir / "selected_chunk_summary.csv"
    if not metrics_path.exists() or not selected_path.exists():
        return {}
    data = json.loads(metrics_path.read_text(encoding="utf-8"))
    selected = {row["chunk_id"]: row["selected_mode"] for row in read_csv_dicts(selected_path)}
    out = {}
    for item in data.get("chunks", []):
        chunk_id = item["chunk"]["chunk_id"]
        mode = selected.get(chunk_id)
        if not mode:
            continue
        result = item["results"][mode]
        out[chunk_id] = {
            "mode": mode,
            "yaw_deg": float(result["yaw_deg"]),
            "dx": float(result["dx"]),
            "dz": float(result["dz"]),
            "scale": float(result["scale"]),
        }
    return out


def load_mapseg(map_seg: Path, map_size: tuple[int, int], center_mode: str) -> dict[str, np.ndarray]:
    road_img = Image.open(map_seg / "road_mask_binary.png").convert("L")
    if road_img.size != map_size:
        road_img = road_img.resize(map_size, Image.Resampling.NEAREST)
    road = np.asarray(road_img) > 0

    centers = []
    for name, filename in (
        ("centerline", "road_centerline_pixel.png"),
        ("skeleton", "road_skeleton_pixel.png"),
    ):
        path = map_seg / filename
        if path.exists():
            img = Image.open(path).convert("L")
            if img.size != map_size:
                img = img.resize(map_size, Image.Resampling.NEAREST)
            centers.append((name, np.asarray(img) > 0))
    if center_mode == "centerline":
        center = next((mask for name, mask in centers if name == "centerline"), None)
    elif center_mode == "skeleton":
        center = next((mask for name, mask in centers if name == "skeleton"), None)
    elif center_mode == "union":
        center = None
        for _, mask in centers:
            center = mask.copy() if center is None else (center | mask)
    else:
        raise ValueError(center_mode)
    if center is None:
        center = skeletonize_mask(road)
    return {"road": road, "center": center}


def skeletonize_mask(mask: np.ndarray) -> np.ndarray:
    if not np.any(mask):
        return mask.copy()
    try:
        from skimage.morphology import skeletonize  # type: ignore

        return skeletonize(mask)
    except Exception:
        dist = ndimage.distance_transform_edt(mask)
        return mask & (dist == ndimage.maximum_filter(dist, size=5))


def build_interval_mask(
    item_rows: list[dict[str, str]],
    map_size: tuple[int, int],
    fallback_center: tuple[float, float],
    interval_buffer_px: float,
    junction_radius_px: float,
    chunk_type: str,
) -> np.ndarray:
    width, height = map_size
    img = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(img)
    line_width = max(3, int(round(interval_buffer_px * 2.0 + 1.0)))

    by_route: dict[str, list[tuple[int, float, float]]] = {}
    for row in item_rows:
        try:
            order = int(float(row.get("item_order", "0")))
            x = float(row["map_x_px"])
            y = float(row["map_y_px"])
        except Exception:
            continue
        by_route.setdefault(row.get("route", ""), []).append((order, x, y))

    drew = False
    for vals in by_route.values():
        vals.sort(key=lambda item: item[0])
        pts = [(x, y) for _, x, y in vals]
        if len(pts) >= 2:
            draw.line(pts, fill=255, width=line_width, joint="curve")
            drew = True
        for x, y in pts:
            r = max(2.0, interval_buffer_px * 0.6)
            draw.ellipse((x - r, y - r, x + r, y + r), fill=255)
            drew = True
    if not drew:
        x, y = fallback_center
        r = junction_radius_px if chunk_type == "crossing_node" else interval_buffer_px
        draw.ellipse((x - r, y - r, x + r, y + r), fill=255)
    return np.asarray(img) > 0


def local_target_masks(
    mapseg: dict[str, np.ndarray],
    interval: np.ndarray,
    min_target_pixels: int,
) -> tuple[np.ndarray, np.ndarray, str]:
    road = mapseg["road"] & interval
    reason = "ok"
    if int(np.count_nonzero(road)) < min_target_pixels:
        road = mapseg["road"] & ndimage.binary_dilation(interval, iterations=8)
        reason = "expanded_interval"
    center = mapseg["center"] & ndimage.binary_dilation(road, iterations=3)
    if int(np.count_nonzero(center)) < 8 and np.any(road):
        center = skeletonize_mask(road)
        reason += "; skeletonized_center"
    if int(np.count_nonzero(road)) < min_target_pixels:
        reason += "; sparse_target"
    return road, center, reason


def rasterize_points(
    xy: np.ndarray,
    map_size: tuple[int, int],
    radius_px: int,
) -> np.ndarray:
    width, height = map_size
    mask = np.zeros((height, width), dtype=bool)
    if len(xy) == 0:
        return mask
    xi = np.rint(xy[:, 0]).astype(np.int64)
    yi = np.rint(xy[:, 1]).astype(np.int64)
    valid = (xi >= 0) & (xi < width) & (yi >= 0) & (yi < height)
    mask[yi[valid], xi[valid]] = True
    if radius_px > 0 and np.any(mask):
        mask = ndimage.binary_dilation(mask, iterations=radius_px)
    return mask


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


def evaluate_xy(
    xy: np.ndarray,
    target_road: np.ndarray,
    target_center: np.ndarray,
    map_size: tuple[int, int],
    point_dilate_px: int,
) -> dict[str, float | int]:
    xy = xy[np.isfinite(xy).all(axis=1)] if len(xy) else xy
    road_dt = ndimage.distance_transform_edt(~target_road)
    center_dt = ndimage.distance_transform_edt(~target_center) if np.any(target_center) else road_dt
    point_to_road = sample_field(road_dt, xy)
    point_to_center = sample_field(center_dt, xy)

    point_mask = rasterize_points(xy, map_size, point_dilate_px)
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
        "road_points": int(len(xy)),
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
        "route_fit_score": float(fit_score),
    }


def weighted_mean(rows: list[dict], key: str, weight_key: str) -> float:
    total = 0.0
    weight = 0.0
    for row in rows:
        v = row.get(key)
        w = row.get(weight_key, 0)
        if v is None or v == "":
            continue
        total += float(v) * float(w)
        weight += float(w)
    return total / weight if weight else float("nan")


def weighted_rmse(rows: list[dict], key: str, weight_key: str) -> float:
    total = 0.0
    weight = 0.0
    for row in rows:
        v = row.get(key)
        w = row.get(weight_key, 0)
        if v is None or v == "":
            continue
        total += float(v) ** 2 * float(w)
        weight += float(w)
    return math.sqrt(total / weight) if weight else float("nan")


def write_html(out_dir: Path, rows: list[dict], summary: dict) -> None:
    body = []
    for row in sorted(rows, key=lambda item: float(item["route_fit_score"]), reverse=True):
        body.append(
            "<tr>"
            f"<td>{row['chunk_id']}</td>"
            f"<td>{row['chunk_type']}</td>"
            f"<td>{row['method_label']}</td>"
            f"<td>{row['route_fit_score']}</td>"
            f"<td>{row['outside_ratio']}</td>"
            f"<td>{row['outside_rmse_px']}</td>"
            f"<td>{row['outside_p90_px']}</td>"
            f"<td>{row['center_rmse_px']}</td>"
            f"<td>{row['center_p90_px']}</td>"
            f"<td>{row['target_center_coverage_5px']}</td>"
            f"<td>{row['projection_iou']}</td>"
            f"<td>{row['target_reason']}</td>"
            "</tr>"
        )
    out_dir.joinpath("index.html").write_text(
        f"""<!doctype html>
<meta charset="utf-8">
<title>MapSeg Projection Fit</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 24px; color: #172033; }}
.stats {{ display: flex; gap: 12px; flex-wrap: wrap; margin: 18px 0; }}
.stat {{ border: 1px solid #d4dce8; padding: 10px 12px; min-width: 170px; }}
.label {{ color: #64748b; font-size: 12px; }}
.value {{ font-weight: 700; font-size: 20px; }}
table {{ border-collapse: collapse; font-size: 12px; }}
td, th {{ border: 1px solid #d7dde8; padding: 5px 7px; }}
th {{ background: #eef3fb; }}
</style>
<h1>MapSeg Projection Fit</h1>
<p>Road/ground 3D projection is evaluated against the red fine road segmentation. SLAM/chunk trajectory only selects the local interval.</p>
<p><a href="projection_fit_summary.csv">CSV</a> / <a href="projection_fit_metrics.json">JSON</a></p>
<section class="stats">
<div class="stat"><div class="label">Method</div><div class="value">{summary['method_name']}</div></div>
<div class="stat"><div class="label">Chunks</div><div class="value">{summary['chunks']}</div></div>
<div class="stat"><div class="label">Mean Score</div><div class="value">{summary['weighted_route_fit_score']:.2f}</div></div>
<div class="stat"><div class="label">Road RMSE px</div><div class="value">{summary['weighted_outside_rmse_px']:.2f}</div></div>
<div class="stat"><div class="label">Center RMSE px</div><div class="value">{summary['weighted_center_rmse_px']:.2f}</div></div>
<div class="stat"><div class="label">Outside Ratio</div><div class="value">{summary['weighted_outside_ratio']:.3f}</div></div>
<div class="stat"><div class="label">Coverage 5px</div><div class="value">{summary['weighted_target_center_coverage_5px']:.3f}</div></div>
</section>
<table>
<thead><tr><th>chunk</th><th>type</th><th>method</th><th>score</th><th>outside</th><th>road RMSE</th><th>outside p90</th><th>center RMSE</th><th>center p90</th><th>coverage 5px</th><th>IoU</th><th>target</th></tr></thead>
<tbody>{''.join(body)}</tbody>
</table>
""",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate 3D road projection fit to fine 2D road segmentation.")
    parser.add_argument("--method-name", default="method")
    parser.add_argument("--recon-dir", type=Path, default=None)
    parser.add_argument("--chunk-dir", type=Path, default=None)
    parser.add_argument(
        "--global-road-ply",
        type=Path,
        default=None,
        help="Optional already map-registered road/ground PLY for methods without PanoVGGT chunk outputs.",
    )
    parser.add_argument("--map-seg", type=Path, required=True)
    parser.add_argument("--base-map", type=Path, required=True)
    parser.add_argument("--bev-refine-dir", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--center-mode", choices=("centerline", "skeleton", "union"), default="union")
    parser.add_argument("--target-interval", choices=("chunk_slam", "global"), default="chunk_slam")
    parser.add_argument("--interval-buffer-px", type=float, default=34.0)
    parser.add_argument("--junction-interval-radius-px", type=float, default=76.0)
    parser.add_argument("--min-target-pixels", type=int, default=160)
    parser.add_argument("--max-road-points", type=int, default=80_000)
    parser.add_argument("--point-dilate-px", type=int, default=2)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    map_size = Image.open(args.base_map).size
    mapseg = load_mapseg(args.map_seg, map_size, args.center_mode)

    if args.global_road_ply is not None:
        road_points = load_point_cloud(args.global_road_ply, args.max_road_points if args.max_road_points > 0 else 0, 17)
        road_xy = road_points[:, [0, 2]]
        stats = evaluate_xy(road_xy, mapseg["road"], mapseg["center"], map_size, args.point_dilate_px)
        row = {
            "method_name": args.method_name,
            "chunk_id": "global",
            "chunk_type": "global_road_ply",
            "routes": "global",
            "method_label": "global_road_ply",
            "target_interval": "global",
            "target_reason": "global_road_ply",
        }
        row.update({key: fmt(value) for key, value in stats.items()})
        numeric = dict(row)
        for key, value in stats.items():
            numeric[key] = float(value)
        summary = {
            "method_name": args.method_name,
            "recon_dir": "",
            "chunk_dir": "",
            "global_road_ply": str(args.global_road_ply),
            "bev_refine_dir": "",
            "map_seg": str(args.map_seg),
            "chunks": 1,
            "weighted_route_fit_score": float(stats["route_fit_score"]),
            "weighted_outside_ratio": float(stats["outside_ratio"]),
            "weighted_outside_mean_px": float(stats["outside_mean_px"]),
            "weighted_outside_rmse_px": float(stats["outside_rmse_px"]),
            "weighted_outside_p90_px": float(stats["outside_p90_px"]),
            "weighted_center_rmse_px": float(stats["center_rmse_px"]),
            "weighted_center_p90_px": float(stats["center_p90_px"]),
            "weighted_target_to_point_rmse_px": float(stats["target_to_point_rmse_px"]),
            "weighted_target_center_coverage_3px": float(stats["target_center_coverage_3px"]),
            "weighted_target_center_coverage_5px": float(stats["target_center_coverage_5px"]),
            "weighted_projection_iou": float(stats["projection_iou"]),
        }
        write_csv(args.out_dir / "projection_fit_summary.csv", [row])
        (args.out_dir / "projection_fit_metrics.json").write_text(
            json.dumps({"summary": summary, "chunks": [numeric]}, indent=2),
            encoding="utf-8",
        )
        write_html(args.out_dir, [row], summary)
        print(f"[done] {args.out_dir / 'index.html'}")
        print(json.dumps(summary, indent=2))
        return

    if args.recon_dir is None or args.chunk_dir is None:
        raise ValueError("Either --global-road-ply or both --recon-dir/--chunk-dir are required.")

    metrics = json.loads((args.recon_dir / "registration_metrics.json").read_text(encoding="utf-8"))
    records = metrics.get("chunks", [])
    manifest = {row["chunk_id"]: row for row in read_csv_dicts(args.chunk_dir / "chunk_manifest.csv")}
    items = group_rows(read_csv_dicts(args.chunk_dir / "chunk_items.csv"), "chunk_id")
    bev_delta = load_bev_refine_selection(args.bev_refine_dir)

    rows = []
    for idx, rec in enumerate(records):
        chunk_id = rec["chunk_id"]
        chunk = manifest.get(chunk_id, {})
        road_ply = args.recon_dir / "road_clouds" / f"{chunk_id}.ply"
        if not road_ply.exists():
            print(f"skip {chunk_id}: missing road-only ply")
            continue
        road_local = load_point_cloud(road_ply, args.max_road_points, 1000 + idx)
        road_map = apply_sim3(road_local, rec["sim3"])
        pivot = (
            float(chunk.get("map_x_px") or rec.get("map_x_px") or 0.0),
            float(chunk.get("map_y_px") or rec.get("map_y_px") or 0.0),
        )
        junction_delta = delta_params_from_junction(rec)
        if junction_delta is not None:
            yaw, dx, dz, scale = junction_delta
            road_xy = transform_xz_array(road_map[:, [0, 2]], yaw, dx, dz, scale, pivot)
        else:
            road_xy = road_map[:, [0, 2]]
        selected = bev_delta.get(chunk_id)
        method_label = rec.get("registration_method", "")
        if selected is not None:
            road_xy = transform_xz_array(
                road_xy,
                selected["yaw_deg"],
                selected["dx"],
                selected["dz"],
                selected["scale"],
                pivot,
            )
            method_label += f"+bev_{selected['mode']}"

        if args.target_interval == "global":
            target_road = mapseg["road"]
            target_center = mapseg["center"]
            reason = "global"
        else:
            interval = build_interval_mask(
                items.get(chunk_id, []),
                map_size,
                pivot,
                args.interval_buffer_px,
                args.junction_interval_radius_px,
                str(rec.get("chunk_type", "")),
            )
            target_road, target_center, reason = local_target_masks(mapseg, interval, args.min_target_pixels)

        stats = evaluate_xy(road_xy, target_road, target_center, map_size, args.point_dilate_px)
        row = {
            "method_name": args.method_name,
            "chunk_id": chunk_id,
            "chunk_type": rec.get("chunk_type", ""),
            "routes": rec.get("routes", ""),
            "method_label": method_label,
            "target_interval": args.target_interval,
            "target_reason": reason,
        }
        row.update({key: fmt(value) for key, value in stats.items()})
        rows.append(row)
        print(
            f"[{idx + 1:02d}/{len(records):02d}] {chunk_id}: "
            f"score={row['route_fit_score']} outside={row['outside_ratio']} cov5={row['target_center_coverage_5px']}"
        )

    if not rows:
        raise RuntimeError("no evaluable chunks")
    numeric_rows = []
    for row in rows:
        converted = dict(row)
        for key in (
            "road_points",
            "target_road_pixels",
            "target_center_pixels",
            "outside_ratio",
            "outside_mean_px",
            "outside_rmse_px",
            "outside_p90_px",
            "center_p50_px",
            "center_rmse_px",
            "center_p90_px",
            "target_to_point_mean_px",
            "target_to_point_rmse_px",
            "target_to_point_p90_px",
            "target_center_coverage_3px",
            "target_center_coverage_5px",
            "projection_iou",
            "route_fit_score",
        ):
            converted[key] = float(row[key]) if row.get(key, "") != "" else float("nan")
        numeric_rows.append(converted)
    summary = {
        "method_name": args.method_name,
        "recon_dir": str(args.recon_dir),
        "chunk_dir": str(args.chunk_dir),
        "bev_refine_dir": str(args.bev_refine_dir) if args.bev_refine_dir else "",
        "map_seg": str(args.map_seg),
        "chunks": len(rows),
        "weighted_route_fit_score": weighted_mean(numeric_rows, "route_fit_score", "road_points"),
        "weighted_outside_ratio": weighted_mean(numeric_rows, "outside_ratio", "road_points"),
        "weighted_outside_mean_px": weighted_mean(numeric_rows, "outside_mean_px", "road_points"),
        "weighted_outside_rmse_px": weighted_rmse(numeric_rows, "outside_rmse_px", "road_points"),
        "weighted_outside_p90_px": weighted_mean(numeric_rows, "outside_p90_px", "road_points"),
        "weighted_center_rmse_px": weighted_rmse(numeric_rows, "center_rmse_px", "road_points"),
        "weighted_center_p90_px": weighted_mean(numeric_rows, "center_p90_px", "road_points"),
        "weighted_target_to_point_rmse_px": weighted_rmse(numeric_rows, "target_to_point_rmse_px", "target_center_pixels"),
        "weighted_target_center_coverage_3px": weighted_mean(numeric_rows, "target_center_coverage_3px", "target_center_pixels"),
        "weighted_target_center_coverage_5px": weighted_mean(numeric_rows, "target_center_coverage_5px", "target_center_pixels"),
        "weighted_projection_iou": weighted_mean(numeric_rows, "projection_iou", "target_road_pixels"),
    }
    write_csv(args.out_dir / "projection_fit_summary.csv", rows)
    (args.out_dir / "projection_fit_metrics.json").write_text(
        json.dumps({"summary": summary, "chunks": numeric_rows}, indent=2),
        encoding="utf-8",
    )
    write_html(args.out_dir, rows, summary)
    print(f"[done] {args.out_dir / 'index.html'}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
