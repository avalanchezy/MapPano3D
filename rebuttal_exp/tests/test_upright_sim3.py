from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


WORKSPACE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WORKSPACE / "tools"))

from upright_sim3 import fit_upright_sim3


def _pose(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    pose = np.eye(4)
    pose[:3, :3] = rotation
    pose[:3, 3] = translation
    return pose


def test_recovers_upright_planar_similarity() -> None:
    angle = np.deg2rad(27.0)
    yaw = np.array(
        [[np.cos(angle), -np.sin(angle), 0.0], [np.sin(angle), np.cos(angle), 0.0], [0.0, 0.0, 1.0]]
    )
    tilt_angle = np.deg2rad(31.0)
    tilt = np.array(
        [[1.0, 0.0, 0.0], [0.0, np.cos(tilt_angle), -np.sin(tilt_angle)], [0.0, np.sin(tilt_angle), np.cos(tilt_angle)]]
    )
    rotation = yaw @ tilt
    scale = 3.2
    translation = np.array([12.0, -7.0, 1.5])
    target_centers = np.array(
        [[12.0, -7.0, 1.5], [15.0, -6.0, 1.5], [18.0, -3.0, 1.5], [21.0, 2.0, 1.5]]
    )
    source_centers = ((target_centers - translation) @ rotation) / scale
    camera_basis = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
    source_camera_rotation = rotation.T @ camera_basis
    source_poses = np.stack([_pose(source_camera_rotation, center) for center in source_centers])
    target_poses = np.stack([_pose(camera_basis, center) for center in target_centers])

    result = fit_upright_sim3(source_poses, target_poses, target_forward_axis=2)

    assert result["identifiable"]
    np.testing.assert_allclose(result["aligned_centers"], target_centers, atol=1e-8)
    np.testing.assert_allclose(result["scale"], scale, atol=1e-8)
    np.testing.assert_allclose(result["rotation"], rotation, atol=1e-8)


def test_stationary_sequence_is_marked_scale_unidentifiable() -> None:
    source = np.stack([_pose(np.eye(3), np.zeros(3)) for _ in range(4)])
    target = np.stack([_pose(np.eye(3), np.array([4.0, 5.0, 1.0])) for _ in range(4)])

    result = fit_upright_sim3(source, target, target_forward_axis=2)

    assert not result["identifiable"]
    assert result["scale"] == 1.0
    assert result["up_error_deg"] < 1e-8
