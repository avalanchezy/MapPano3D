#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import numpy as np


CAMERA_SWEEP = [
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_FRONT_LEFT",
]
VIEW_MODES = {
    "heading_fov60": ["CAM_FRONT"],
    "sweep_fov60": CAMERA_SWEEP,
}


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def index_by_token(rows: list[dict]) -> dict[str, dict]:
    return {row["token"]: row for row in rows}


def index_sample_data_by_sample_channel(
    rows: list[dict],
    calibrated_by_token: dict[str, dict],
    sensors_by_token: dict[str, dict],
) -> dict[tuple[str, str], dict]:
    out = {}
    for row in rows:
        if not row.get("is_key_frame", False):
            continue
        calib = calibrated_by_token[row["calibrated_sensor_token"]]
        sensor = sensors_by_token[calib["sensor_token"]]
        channel = sensor["channel"]
        if channel.startswith("CAM_"):
            out[(row["sample_token"], channel)] = row
    return out


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


def scene_samples(scene: dict, samples: dict[str, dict]) -> list[dict]:
    out = []
    token = scene["first_sample_token"]
    while token:
        sample = samples[token]
        out.append(sample)
        token = sample.get("next", "")
    return out


def copy_or_link(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    if mode == "symlink":
        dst.symlink_to(src)
    elif mode == "hardlink":
        try:
            dst.hardlink_to(src)
        except OSError:
            shutil.copy2(src, dst)
    else:
        shutil.copy2(src, dst)


def build_sequence(
    *,
    dataroot: Path,
    out_root: Path,
    scene: dict,
    samples_by_token: dict[str, dict],
    sample_data_by_sample_channel: dict[tuple[str, str], dict],
    calibrated_by_token: dict[str, dict],
    ego_by_token: dict[str, dict],
    view_mode: str,
    copy_mode: str,
) -> None:
    channels = VIEW_MODES[view_mode]
    rows = []
    intrinsics = []
    poses = []
    image_names = []
    samples = scene_samples(scene, samples_by_token)
    first_timestamp = samples[0]["timestamp"]
    seq_dir = out_root / view_mode / scene["name"]
    image_dir = seq_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    sequence_index = 0
    for sample_index, sample in enumerate(samples):
        for channel_index, channel in enumerate(channels):
            sd = sample_data_by_sample_channel.get((sample["token"], channel))
            if sd is None:
                raise KeyError(f"Missing {channel} for sample {sample['token']}")
            if not sd.get("is_key_frame", False):
                raise ValueError(f"Expected key-frame sample_data for {channel}: {sd['token']}")
            src = dataroot / sd["filename"]
            if not src.exists():
                raise FileNotFoundError(src)

            stem = f"{sequence_index:06d}_{sample_index:04d}_{channel}"
            dst_name = f"{stem}{src.suffix.lower()}"
            dst = image_dir / dst_name
            copy_or_link(src, dst, copy_mode)

            calib = calibrated_by_token[sd["calibrated_sensor_token"]]
            ego = ego_by_token[sd["ego_pose_token"]]
            pose = transform(ego["translation"], ego["rotation"]) @ transform(
                calib["translation"], calib["rotation"]
            )
            k = np.asarray(calib["camera_intrinsic"], dtype=np.float32)

            image_names.append(dst_name)
            intrinsics.append(k)
            poses.append(pose.astype(np.float32))
            rows.append(
                {
                    "sequence_index": sequence_index,
                    "image": dst_name,
                    "scene": scene["name"],
                    "scene_token": scene["token"],
                    "sample_index": sample_index,
                    "sample_token": sample["token"],
                    "timestamp": sd["timestamp"],
                    "time_sec": f"{(sd['timestamp'] - first_timestamp) / 1e6:.6f}",
                    "view_mode": view_mode,
                    "view_name": channel,
                    "channel": channel,
                    "channel_order": channel_index,
                    "is_key_frame": int(bool(sd.get("is_key_frame", False))),
                    "ego_pose_token": sd["ego_pose_token"],
                    "calibrated_sensor_token": sd["calibrated_sensor_token"],
                    "source_path": str(src),
                    "filename": sd["filename"],
                }
            )
            sequence_index += 1

    with (seq_dir / "manifest.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    np.savez_compressed(
        seq_dir / "camera_conditions.npz",
        image_names=np.asarray(image_names),
        intrinsics=np.stack(intrinsics).astype(np.float32),
        poses=np.stack(poses).astype(np.float32),
    )

    summary = {
        "dataset": "nuScenes",
        "version": "v1.0-mini",
        "scene": scene["name"],
        "scene_token": scene["token"],
        "description": scene.get("description", ""),
        "view_mode": view_mode,
        "channels": channels,
        "sample_count": len(samples),
        "image_count": len(rows),
        "copy_mode": copy_mode,
        "note": "heading uses CAM_FRONT; sweep uses the six synchronized nuScenes key-frame cameras.",
    }
    with (seq_dir / "sequence.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build nuScenes-mini sequences for Long-style baselines.")
    parser.add_argument("--dataroot", type=Path, default=Path("data/nuscenes"))
    parser.add_argument("--version-dir", type=Path, default=None)
    parser.add_argument("--out-root", type=Path, default=Path("outputs/nuscenes/inputs"))
    parser.add_argument("--scene", nargs="*", default=None, help="Scene names to export; default: all scenes.")
    parser.add_argument("--view-mode", nargs="+", choices=sorted(VIEW_MODES), default=sorted(VIEW_MODES))
    parser.add_argument("--copy-mode", choices=["copy", "hardlink", "symlink"], default="hardlink")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    version_dir = args.version_dir or (args.dataroot / "v1.0-mini")
    scenes = load_json(version_dir / "scene.json")
    samples_by_token = index_by_token(load_json(version_dir / "sample.json"))
    sample_data = load_json(version_dir / "sample_data.json")
    calibrated_by_token = index_by_token(load_json(version_dir / "calibrated_sensor.json"))
    ego_by_token = index_by_token(load_json(version_dir / "ego_pose.json"))
    sensors_by_token = index_by_token(load_json(version_dir / "sensor.json"))
    sample_data_by_sample_channel = index_sample_data_by_sample_channel(
        sample_data, calibrated_by_token, sensors_by_token
    )
    selected = set(args.scene) if args.scene else None

    built = 0
    for scene in scenes:
        if selected is not None and scene["name"] not in selected:
            continue
        for view_mode in args.view_mode:
            build_sequence(
                dataroot=args.dataroot,
                out_root=args.out_root,
                scene=scene,
                samples_by_token=samples_by_token,
                sample_data_by_sample_channel=sample_data_by_sample_channel,
                calibrated_by_token=calibrated_by_token,
                ego_by_token=ego_by_token,
                view_mode=view_mode,
                copy_mode=args.copy_mode,
            )
            built += 1
    print(f"Built {built} sequence(s) under {args.out_root}")


if __name__ == "__main__":
    main()
