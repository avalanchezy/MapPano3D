#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


SCENES = (
    "scene-0061",
    "scene-0103",
    "scene-0553",
    "scene-0655",
    "scene-0757",
    "scene-0796",
    "scene-0916",
    "scene-1077",
    "scene-1094",
    "scene-1100",
)


def physical_ego_path_m(input_dir: Path) -> float:
    with (input_dir / "manifest.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    poses = np.load(input_dir / "camera_conditions.npz")["poses"]
    if len(rows) != len(poses):
        raise ValueError(f"Manifest/pose mismatch in {input_dir}")

    grouped: dict[int, list[np.ndarray]] = {}
    for row, pose in zip(rows, poses):
        grouped.setdefault(int(row["sample_index"]), []).append(pose[:3, 3])
    centers = np.asarray(
        [np.mean(grouped[index], axis=0) for index in sorted(grouped)],
        dtype=np.float64,
    )
    if len(centers) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(centers, axis=0), axis=1)))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the shared-VGGT nuScenes reconstruction comparison manifest."
    )
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scene", nargs="+", choices=SCENES, default=list(SCENES))
    parser.add_argument("--mappano-manifest", type=Path, default=None)
    parser.add_argument("--min-target-path-m", type=float, default=5.0)
    parser.add_argument(
        "--refined-cloud-name",
        default="registered_refined_cloud.ply",
    )
    args = parser.parse_args()

    mappano_rows: dict[str, dict[str, str]] = {}
    if args.mappano_manifest is not None:
        with args.mappano_manifest.open(newline="", encoding="utf-8") as handle:
            mappano_rows = {
                row["scene"]: row for row in csv.DictReader(handle)
            }

    rows: list[dict[str, object]] = []
    for scene in args.scene:
        run_dir = args.run_root / scene
        report_path = run_dir / "metrics.json"
        if not report_path.exists():
            raise FileNotFoundError(report_path)
        report = json.loads(report_path.read_text(encoding="utf-8"))
        path_m = physical_ego_path_m(args.input_root / scene)
        identifiable = (
            bool(report["global_vggt_alignment"]["identifiable"])
            and path_m >= args.min_target_path_m
        )
        common = {
            "scene": scene,
            "alignment_identifiable": identifiable,
            "target_path_m": path_m,
            "input_protocol": (
                "six native cameras; shared "
                f"{report['chunk_size_views']}-image VGGT chunks with "
                f"{report['overlap_views']}-image overlap"
            ),
            "chunk_size_views": int(report["chunk_size_views"]),
            "overlap_views": int(report["overlap_views"]),
            "backbone": "VGGT",
        }
        mappano = mappano_rows.get(scene)
        refined_cloud = (
            Path(mappano["refined_cloud"])
            if mappano is not None
            else run_dir / args.refined_cloud_name
        )
        rows.extend(
            [
                {
                    **common,
                    "method": "VGGT-Long",
                    "cloud_path": str(run_dir / "vggt_long_registered_static_cloud.ply"),
                },
                {
                    **common,
                    "method": "MapPano3D",
                    "cloud_path": str(refined_cloud),
                },
            ]
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(args.output)


if __name__ == "__main__":
    main()
