#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import importlib.util
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
POSTPROCESS_PATH = ROOT / "tools" / "run_nuscenes_mappano3d_long_postprocess.py"
SPEC = importlib.util.spec_from_file_location("nuscenes_map_postprocess", POSTPROCESS_PATH)
POST = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(POST)


LEVELS = {
    "clean": (0.0, 0.0, 0.0),
    "rigid_mild": (2.0, 0.5, 0.0),
    "rigid_medium": (5.0, 1.0, 0.0),
    "rigid_severe": (10.0, 2.0, 0.0),
    "scale_mild": (0.0, 0.0, 0.02),
    "scale_medium": (0.0, 0.0, 0.05),
    "scale_severe": (0.0, 0.0, 0.10),
}


def perturbation(level: str, seed: int, seed_count: int) -> tuple[float, float, float, float]:
    yaw_magnitude, translation_magnitude, scale_fraction = LEVELS[level]
    yaw_sign = 1.0 if seed % 2 == 0 else -1.0
    scale_sign = 1.0 if (seed // 2) % 2 == 0 else -1.0
    angle = 2.0 * math.pi * seed / max(seed_count, 1)
    return (
        yaw_sign * yaw_magnitude,
        translation_magnitude * math.cos(angle),
        translation_magnitude * math.sin(angle),
        1.0 + scale_sign * scale_fraction,
    )


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise RuntimeError("No perturbation-study rows were produced")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run deterministic nuScenes map-refinement perturbations without LiDAR access.")
    parser.add_argument("--dataroot", type=Path, required=True)
    parser.add_argument("--version-dir", type=Path, required=True)
    parser.add_argument("--long-root", type=Path, required=True)
    parser.add_argument("--semantic-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--scene", nargs="+", required=True)
    parser.add_argument("--level", nargs="+", choices=sorted(LEVELS), required=True)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--score-corridor-m", type=float, default=15.0)
    parser.add_argument("--max-input-anchor-rmse", type=float, default=1.5)
    parser.add_argument("--max-backbone-points", type=int, default=1500)
    parser.add_argument("--anchor-slack-m", type=float, default=4.5)
    parser.add_argument("--anchor-weight", type=float, default=0.05)
    parser.add_argument("--skeleton-weight", type=float, default=1.0)
    parser.add_argument("--max-refine-translation-m", type=float, default=1.5)
    parser.add_argument("--min-objective-improvement", type=float, default=0.01)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    rows = []
    map_cache: dict[str, object] = {}
    for level in args.level:
        seed_values = [0] if level == "clean" else list(range(args.seeds))
        for seed in seed_values:
            yaw, tx, ty, scale = perturbation(level, seed, args.seeds)
            for scene in args.scene:
                run_id = f"{level}_seed{seed:02d}_{scene}"
                run_root = args.out_root / "runs" / run_id
                run_dir = run_root / "panorama_stitch_e2" / scene / "PanoVGGT"
                metrics_path = run_dir / "metrics.json"
                pose_cloud = run_dir / "registered_pose_init_cloud.ply"
                refined_cloud = run_dir / "registered_refined_cloud.ply"
                if args.resume and metrics_path.exists() and pose_cloud.exists() and refined_cloud.exists():
                    row = POST.load_json(metrics_path)
                else:
                    row = POST.process_one(
                        dataroot=args.dataroot,
                        version_dir=args.version_dir,
                        long_root=args.long_root,
                        out_root=run_root,
                        view_mode="panorama_stitch_e2",
                        scene_name=scene,
                        method="PanoVGGT",
                        resolution=0.1,
                        max_ground_points=12000,
                        score_corridor_m=args.score_corridor_m,
                        point_dilate_px=2,
                        save_clouds=True,
                        map_cache=map_cache,
                        semantic_road_root=args.semantic_root,
                        semantic_static_root=args.semantic_root,
                        perturb_yaw_deg=yaw,
                        perturb_scale=scale,
                        perturb_tx_m=tx,
                        perturb_ty_m=ty,
                        max_input_anchor_rmse=args.max_input_anchor_rmse,
                        max_backbone_points=args.max_backbone_points,
                        anchor_slack_m=args.anchor_slack_m,
                        anchor_weight=args.anchor_weight,
                        yaw_bound_deg=max(8.0, abs(yaw) + 2.0),
                        translation_bound_m=max(4.0, math.hypot(tx, ty) + 2.0),
                        max_refine_translation_m=args.max_refine_translation_m,
                        scale_bound_frac=max(0.05, abs(scale - 1.0) + 0.02),
                        allow_refine_scale=level.startswith("scale_"),
                        skeleton_weight=args.skeleton_weight,
                        min_objective_improvement=args.min_objective_improvement,
                    )
                    if row is None:
                        raise RuntimeError(f"Missing PanoVGGT reconstruction for {scene}")
                rows.append(
                    {
                        "run_id": run_id,
                        "scene": scene,
                        "level": level,
                        "seed": seed,
                        "perturb_yaw_deg": yaw,
                        "perturb_tx_m": tx,
                        "perturb_ty_m": ty,
                        "perturb_scale": scale,
                        "refine_accepted": row["refine_accepted"],
                        "refine_yaw_deg": row["refine_yaw_deg"],
                        "refine_tx_m": row["refine_tx_m"],
                        "refine_ty_m": row["refine_ty_m"],
                        "refine_scale": row["refine_scale"],
                        "refine_objective_improvement": row["refine_objective_improvement"],
                        "refine_fallback_reason": row["refine_fallback_reason"],
                        "pose_init_input_anchor_rmse": row["long_anchor_rmse"],
                        "mappano_input_anchor_rmse": row["mappano_anchor_rmse"],
                        "sim3_alignment": row["sim3_alignment"],
                        "sim3_identifiable": row["sim3_identifiable"],
                        "sim3_target_path_m": row["sim3_target_path_m"],
                        "sim3_up_error_deg": row["sim3_up_error_deg"],
                        "pose_init_nominal_anchor_rmse": row["pose_init_nominal_anchor_rmse"],
                        "mappano_nominal_anchor_rmse": row["mappano_nominal_anchor_rmse"],
                        "pose_init_map_score": row["pose_init_score_route_fit_score"],
                        "mappano_map_score": row["mappano_score_route_fit_score"],
                        "pose_init_cloud": str(pose_cloud),
                        "refined_cloud": str(refined_cloud),
                    }
                )
                print(f"[done] {run_id}: accepted={row['refine_accepted']}")

    manifest = args.out_root / "perturbation_manifest.csv"
    write_csv(manifest, rows)
    print(manifest)


if __name__ == "__main__":
    main()
