from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


RUNNER = load_module(
    "vggt_backbone_runner_test",
    ROOT / "rebuttal_exp" / "scripts" / "run_nuscenes_vggt_backbone_mappano3d.py",
)
BUILDER = load_module(
    "vggt_backbone_builder_test",
    ROOT / "rebuttal_exp" / "scripts" / "build_nuscenes_vggt_backbone_manifest.py",
)
GLOBAL_REFINER = load_module(
    "vggt_backbone_global_refiner_test",
    ROOT / "rebuttal_exp" / "scripts" / "run_nuscenes_global_vggt_refine.py",
)


def test_chunk_ranges_match_vggt_long_tail_policy() -> None:
    assert RUNNER.chunk_ranges(234, 20, 10)[-1] == (220, 234)
    assert len(RUNNER.chunk_ranges(234, 20, 10)) == 23
    assert RUNNER.chunk_ranges(240, 20, 10)[-1] == (220, 240)
    assert len(RUNNER.chunk_ranges(246, 20, 10)) == 24
    assert RUNNER.chunk_ranges(246, 20, 10)[-1] == (230, 246)


def test_chunk_ranges_for_rebuttal_40_frame_protocol() -> None:
    ranges = RUNNER.chunk_ranges(234, 40, 20)
    assert len(ranges) == 11
    assert ranges[0] == (0, 40)
    assert ranges[-1] == (200, 234)


def test_numbered_chunk_clouds_excludes_combined(tmp_path: Path) -> None:
    for name in ("10_pcd.ply", "2_pcd.ply", "combined_pcd.ply"):
        (tmp_path / name).touch()
    assert [path.name for path in RUNNER.numbered_chunk_clouds(tmp_path)] == [
        "2_pcd.ply",
        "10_pcd.ply",
    ]


def test_physical_path_groups_six_camera_centers(tmp_path: Path) -> None:
    rows = []
    poses = []
    offsets = np.array([-0.5, -0.3, -0.1, 0.1, 0.3, 0.5])
    for sample_index, ego_x in enumerate((0.0, 3.0)):
        for view_index, offset in enumerate(offsets):
            rows.append({"sample_index": sample_index, "image": f"{sample_index}_{view_index}.jpg"})
            pose = np.eye(4)
            pose[:3, 3] = [ego_x + offset, 0.0, 1.5]
            poses.append(pose)
    with (tmp_path / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_index", "image"])
        writer.writeheader()
        writer.writerows(rows)
    np.savez(tmp_path / "camera_conditions.npz", poses=np.asarray(poses))
    assert np.isclose(BUILDER.physical_ego_path_m(tmp_path), 3.0)


def test_sim3_inverse_recovers_source_points() -> None:
    angle = np.deg2rad(23.0)
    rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    alignment = {
        "scale": 2.7,
        "rotation": rotation,
        "translation": np.array([3.0, -4.0, 1.5]),
    }
    source = np.array([[1.0, 2.0, 3.0], [-2.0, 0.5, 4.0]])
    transformed = RUNNER.apply_sim3(source, alignment)
    recovered = GLOBAL_REFINER.invert_sim3(transformed, alignment)
    assert np.allclose(recovered, source)


def test_crossing_support_uses_polygon_margin() -> None:
    polygon = np.array(
        [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]],
        dtype=np.float32,
    )
    assert GLOBAL_REFINER.crossing_supported(
        np.array([[5.0, 5.0]]), [polygon], margin_m=0.0
    )
    assert GLOBAL_REFINER.crossing_supported(
        np.array([[12.0, 5.0]]), [polygon], margin_m=2.1
    )
    assert not GLOBAL_REFINER.crossing_supported(
        np.array([[13.0, 5.0]]), [polygon], margin_m=2.1
    )
