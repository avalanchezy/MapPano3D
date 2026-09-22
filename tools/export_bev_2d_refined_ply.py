#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import open3d as o3d




def read_selected(path: Path) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as fh:
        return {row["chunk_id"]: row["selected_mode"] for row in csv.DictReader(fh)}


def transform_xz(points: np.ndarray, result: dict, pivot: tuple[float, float]) -> np.ndarray:
    yaw = math.radians(float(result["yaw_deg"]))
    dx = float(result["dx"])
    dz = float(result["dz"])
    scale = float(result["scale"])
    ca, sa = math.cos(yaw), math.sin(yaw)
    out = points.copy()
    x = points[:, 0] - pivot[0]
    z = points[:, 2] - pivot[1]
    out[:, 0] = pivot[0] + scale * (ca * x - sa * z) + dx
    out[:, 2] = pivot[1] + scale * (sa * x + ca * z) + dz
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--refine-dir", type=Path, required=True)
    parser.add_argument("--full-recon", type=Path, required=True)
    parser.add_argument("--out-name", default="merged_selected_2d_refined_map.ply")
    parser.add_argument("--write-chunks", action="store_true")
    args = parser.parse_args()

    selected = read_selected(args.refine_dir / "selected_chunk_summary.csv")
    data = json.loads((args.refine_dir / "chunk_refine_metrics.json").read_text(encoding="utf-8"))
    chunk_out = args.refine_dir / "registered_chunks_selected_2d_refined"
    if args.write_chunks:
        chunk_out.mkdir(parents=True, exist_ok=True)

    all_points = []
    all_colors = []
    written = 0
    for item in data["chunks"]:
        chunk = item["chunk"]
        chunk_id = chunk["chunk_id"]
        mode = selected.get(chunk_id)
        if not mode:
            continue
        result = item["results"][mode]
        pivot = (float(chunk["map_x"]), float(chunk["map_y"]))
        src = args.full_recon / "registered_chunks" / f"{chunk_id}_map.ply"
        if not src.exists():
            print(f"skip missing {src}")
            continue
        pcd = o3d.io.read_point_cloud(str(src))
        points = np.asarray(pcd.points, dtype=np.float64)
        colors = np.asarray(pcd.colors, dtype=np.float64)
        if colors.shape[0] != points.shape[0]:
            colors = np.full((points.shape[0], 3), 0.75, dtype=np.float64)
        refined = transform_xz(points, result, pivot)
        all_points.append(refined)
        all_colors.append(np.clip(colors, 0.0, 1.0))

        if args.write_chunks:
            out_pcd = o3d.geometry.PointCloud()
            out_pcd.points = o3d.utility.Vector3dVector(refined)
            out_pcd.colors = o3d.utility.Vector3dVector(np.clip(colors, 0.0, 1.0))
            o3d.io.write_point_cloud(str(chunk_out / f"{chunk_id}_selected_2d_refined_map.ply"), out_pcd)
        written += 1
        print(f"[{written:02d}] {chunk_id}: selected={mode}")

    if not all_points:
        raise RuntimeError("No chunks exported")
    merged = o3d.geometry.PointCloud()
    merged.points = o3d.utility.Vector3dVector(np.vstack(all_points))
    merged.colors = o3d.utility.Vector3dVector(np.vstack(all_colors))
    out_path = args.refine_dir / args.out_name
    o3d.io.write_point_cloud(str(out_path), merged)
    print(out_path)
    print(f"chunks={written} points={len(merged.points)}")


if __name__ == "__main__":
    main()
