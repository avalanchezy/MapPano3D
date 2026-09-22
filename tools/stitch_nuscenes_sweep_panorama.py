#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image


CAMERA_SWEEP = [
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_FRONT_LEFT",
]


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def index_by_token(rows: list[dict]) -> dict[str, dict]:
    return {row["token"]: row for row in rows}


def quat_to_rot(q: list[float]) -> np.ndarray:
    w, x, y, z = [float(v) for v in q]
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def transform(translation: list[float], rotation: list[float]) -> np.ndarray:
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = quat_to_rot(rotation)
    out[:3, 3] = np.asarray(translation, dtype=np.float64)
    return out


def scene_samples(scene: dict, samples_by_token: dict[str, dict]) -> list[dict]:
    out = []
    token = scene["first_sample_token"]
    while token:
        sample = samples_by_token[token]
        out.append(sample)
        token = sample.get("next", "")
    return out


def index_key_cameras(
    sample_data: list[dict],
    calibrated_by_token: dict[str, dict],
    sensors_by_token: dict[str, dict],
) -> dict[tuple[str, str], dict]:
    out = {}
    for row in sample_data:
        if not row.get("is_key_frame", False):
            continue
        calib = calibrated_by_token[row["calibrated_sensor_token"]]
        sensor = sensors_by_token[calib["sensor_token"]]
        channel = sensor["channel"]
        if channel.startswith("CAM_"):
            out[(row["sample_token"], channel)] = row
    return out


def make_dirs(width: int, height: int, vmin_deg: float, vmax_deg: float) -> np.ndarray:
    yaw = np.linspace(-np.pi, np.pi, width, endpoint=False, dtype=np.float64)
    elev = np.deg2rad(np.linspace(vmax_deg, vmin_deg, height, dtype=np.float64))
    yy, ee = np.meshgrid(yaw, elev)
    dirs = np.stack(
        [
            np.cos(ee) * np.cos(yy),
            np.cos(ee) * np.sin(yy),
            np.sin(ee),
        ],
        axis=-1,
    )
    return dirs.reshape(-1, 3)


def bilinear_sample(image: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    x0 = np.floor(u).astype(np.int64)
    y0 = np.floor(v).astype(np.int64)
    x1 = np.clip(x0 + 1, 0, w - 1)
    y1 = np.clip(y0 + 1, 0, h - 1)
    x0 = np.clip(x0, 0, w - 1)
    y0 = np.clip(y0, 0, h - 1)
    wx = (u - x0)[:, None]
    wy = (v - y0)[:, None]
    top = image[y0, x0] * (1.0 - wx) + image[y0, x1] * wx
    bot = image[y1, x0] * (1.0 - wx) + image[y1, x1] * wx
    return (top * (1.0 - wy) + bot * wy).clip(0, 255).astype(np.uint8)


def stitch_sample(
    *,
    dataroot: Path,
    sample: dict,
    camera_index: dict[tuple[str, str], dict],
    calibrated_by_token: dict[str, dict],
    ego_by_token: dict[str, dict],
    dirs_ego: np.ndarray,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray]:
    n = len(dirs_ego)
    pano = np.zeros((n, 3), dtype=np.uint8)
    label = np.full(n, -1, dtype=np.int16)
    best = np.full(n, -np.inf, dtype=np.float64)

    for channel_idx, channel in enumerate(CAMERA_SWEEP):
        row = camera_index[(sample["token"], channel)]
        calib = calibrated_by_token[row["calibrated_sensor_token"]]
        image = np.asarray(Image.open(dataroot / row["filename"]).convert("RGB"))
        k = np.asarray(calib["camera_intrinsic"], dtype=np.float64)
        t_ego_cam = transform(calib["translation"], calib["rotation"])
        dirs_cam = dirs_ego @ t_ego_cam[:3, :3]
        z = dirs_cam[:, 2]
        valid = z > 1e-6
        u = k[0, 0] * dirs_cam[:, 0] / np.maximum(z, 1e-6) + k[0, 2]
        v = k[1, 1] * dirs_cam[:, 1] / np.maximum(z, 1e-6) + k[1, 2]
        valid &= (u >= 0) & (u < image.shape[1] - 1) & (v >= 0) & (v < image.shape[0] - 1)
        score = z / np.linalg.norm(dirs_cam, axis=1)
        take = valid & (score > best)
        if np.any(take):
            pano[take] = bilinear_sample(image, u[take], v[take])
            label[take] = channel_idx
            best[take] = score[take]

    return pano.reshape(height, width, 3), label.reshape(height, width)


def main() -> None:
    parser = argparse.ArgumentParser(description="Stitch nuScenes six-camera key-frame sweeps into 360 panoramas.")
    parser.add_argument("--dataroot", type=Path, default=Path("data/nuscenes"))
    parser.add_argument("--version-dir", type=Path, default=None)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--width", type=int, default=2048)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--vmin-deg", type=float, default=-30.0)
    parser.add_argument("--vmax-deg", type=float, default=25.0)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    version_dir = args.version_dir or (args.dataroot / "v1.0-mini")
    scenes = load_json(version_dir / "scene.json")
    samples_by_token = index_by_token(load_json(version_dir / "sample.json"))
    sample_data = load_json(version_dir / "sample_data.json")
    calibrated_by_token = index_by_token(load_json(version_dir / "calibrated_sensor.json"))
    ego_by_token = index_by_token(load_json(version_dir / "ego_pose.json"))
    sensors_by_token = index_by_token(load_json(version_dir / "sensor.json"))
    camera_index = index_key_cameras(sample_data, calibrated_by_token, sensors_by_token)
    scene = next(row for row in scenes if row["name"] == args.scene)
    samples = scene_samples(scene, samples_by_token)
    if args.limit is not None:
        samples = samples[: args.limit]

    image_dir = args.out_dir / "images"
    label_dir = args.out_dir / "camera_labels"
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    dirs_ego = make_dirs(args.width, args.height, args.vmin_deg, args.vmax_deg)
    rows = []
    poses = []
    first_timestamp = samples[0]["timestamp"]
    for idx, sample in enumerate(samples):
        pano, labels = stitch_sample(
            dataroot=args.dataroot,
            sample=sample,
            camera_index=camera_index,
            calibrated_by_token=calibrated_by_token,
            ego_by_token=ego_by_token,
            dirs_ego=dirs_ego,
            width=args.width,
            height=args.height,
        )
        image_name = f"{idx:06d}.jpg"
        Image.fromarray(pano).save(image_dir / image_name, quality=95)
        Image.fromarray(((labels + 1) * 35).clip(0, 255).astype(np.uint8)).save(label_dir / f"{idx:06d}.png")
        front_data = camera_index[(sample["token"], "CAM_FRONT")]
        poses.append(transform(**{key: ego_by_token[front_data["ego_pose_token"]][key] for key in ("translation", "rotation")}))
        rows.append(
            {
                "sequence_index": idx,
                "image": image_name,
                "scene": args.scene,
                "sample_index": idx,
                "sample_token": sample["token"],
                "timestamp": sample["timestamp"],
                "time_sec": f"{(sample['timestamp'] - first_timestamp) / 1e6:.6f}",
                "view_mode": "panorama_stitch",
                "view_name": "sweep_panorama",
            }
        )
    with (args.out_dir / "manifest.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "dataset": "nuScenes",
        "scene": args.scene,
        "view_mode": "panorama_stitch",
        "channels": CAMERA_SWEEP,
        "width": args.width,
        "height": args.height,
        "vmin_deg": args.vmin_deg,
        "vmax_deg": args.vmax_deg,
        "image_count": len(rows),
        "note": "Calibration-based infinite-depth cylindrical panorama; near objects can show seams/parallax.",
    }
    with (args.out_dir / "sequence.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    np.savez_compressed(args.out_dir / "camera_conditions.npz", poses=np.stack(poses))
    print(f"Stitched {len(rows)} panorama(s) under {args.out_dir}")


if __name__ == "__main__":
    main()
