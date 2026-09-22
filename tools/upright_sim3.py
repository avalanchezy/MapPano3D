#!/usr/bin/env python3
from __future__ import annotations

import math

import numpy as np


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm < 1e-12:
        raise ValueError("Cannot normalize a zero vector")
    return vector / norm


def _rotation_between(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    source = _unit(np.asarray(source, dtype=np.float64))
    target = _unit(np.asarray(target, dtype=np.float64))
    cross = np.cross(source, target)
    cosine = float(np.clip(source @ target, -1.0, 1.0))
    sine = float(np.linalg.norm(cross))
    if sine < 1e-10:
        if cosine > 0.0:
            return np.eye(3, dtype=np.float64)
        helper = np.array([1.0, 0.0, 0.0]) if abs(source[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        axis = _unit(np.cross(source, helper))
        return 2.0 * np.outer(axis, axis) - np.eye(3, dtype=np.float64)
    skew = np.array(
        [
            [0.0, -cross[2], cross[1]],
            [cross[2], 0.0, -cross[0]],
            [-cross[1], cross[0], 0.0],
        ],
        dtype=np.float64,
    )
    return np.eye(3, dtype=np.float64) + skew + skew @ skew * ((1.0 - cosine) / (sine * sine))


def _orientation_yaw(
    source_poses: np.ndarray,
    target_poses: np.ndarray,
    upright_rotation: np.ndarray,
    target_forward_axis: int,
) -> float:
    source_forward = np.einsum(
        "ij,nj->ni", upright_rotation, source_poses[:, :3, 2]
    )[:, :2]
    target_forward = target_poses[:, :2, target_forward_axis]
    source_norm = np.linalg.norm(source_forward, axis=1)
    target_norm = np.linalg.norm(target_forward, axis=1)
    valid = (source_norm > 1e-8) & (target_norm > 1e-8)
    if not np.any(valid):
        return 0.0
    source_angle = np.arctan2(source_forward[valid, 1], source_forward[valid, 0])
    target_angle = np.arctan2(target_forward[valid, 1], target_forward[valid, 0])
    delta = target_angle - source_angle
    return float(math.atan2(np.mean(np.sin(delta)), np.mean(np.cos(delta))))


def fit_upright_sim3(
    source_poses: np.ndarray,
    target_poses: np.ndarray,
    *,
    target_forward_axis: int,
    min_target_path_m: float = 0.5,
) -> dict[str, object]:
    """Fit one camera-anchored Sim(3) while keeping the reconstruction upright.

    The predicted cameras use the OpenCV convention, so their local ``-y`` axis
    is up. Roll and pitch are fixed from that axis. The remaining yaw, scale,
    and translation are estimated from the horizontal camera trajectory. LiDAR
    and reconstructed scene points are deliberately not used.
    """
    source_poses = np.asarray(source_poses, dtype=np.float64)
    target_poses = np.asarray(target_poses, dtype=np.float64)
    if source_poses.shape != target_poses.shape or source_poses.ndim != 3 or source_poses.shape[1:] != (4, 4):
        raise ValueError(
            f"Expected matching Nx4x4 pose arrays, got {source_poses.shape} and {target_poses.shape}"
        )
    if target_forward_axis not in (0, 1, 2):
        raise ValueError(f"Invalid target forward axis: {target_forward_axis}")

    source_centers = source_poses[:, :3, 3]
    target_centers = target_poses[:, :3, 3]
    source_up = _unit(np.mean(-source_poses[:, :3, 1], axis=0))
    upright_rotation = _rotation_between(source_up, np.array([0.0, 0.0, 1.0]))
    upright_centers = source_centers @ upright_rotation.T

    source_xy = upright_centers[:, :2]
    target_xy = target_centers[:, :2]
    source_centered = source_xy - source_xy.mean(axis=0)
    target_centered = target_xy - target_xy.mean(axis=0)
    source_variance = float(np.mean(np.sum(source_centered * source_centered, axis=1)))
    target_path_m = float(np.sum(np.linalg.norm(np.diff(target_xy, axis=0), axis=1)))
    identifiable = target_path_m >= min_target_path_m and source_variance > 1e-10

    if identifiable:
        covariance = (target_centered.T @ source_centered) / len(source_centered)
        u, singular, vt = np.linalg.svd(covariance)
        sign = np.ones(2, dtype=np.float64)
        if np.linalg.det(u @ vt) < 0:
            sign[-1] = -1.0
        rotation_2d = u @ np.diag(sign) @ vt
        scale = float(np.sum(singular * sign) / source_variance)
        yaw_rad = float(math.atan2(rotation_2d[1, 0], rotation_2d[0, 0]))
        alignment_mode = "upright_planar_camera_trajectory"
    else:
        yaw_rad = _orientation_yaw(
            source_poses,
            target_poses,
            upright_rotation,
            target_forward_axis,
        )
        cosine, sine = math.cos(yaw_rad), math.sin(yaw_rad)
        rotation_2d = np.array([[cosine, -sine], [sine, cosine]], dtype=np.float64)
        scale = 1.0
        alignment_mode = "upright_orientation_only_scale_unidentifiable"

    yaw_rotation = np.eye(3, dtype=np.float64)
    yaw_rotation[:2, :2] = rotation_2d
    rotation = yaw_rotation @ upright_rotation
    translation = target_centers.mean(axis=0) - scale * (rotation @ source_centers.mean(axis=0))
    aligned_centers = scale * (source_centers @ rotation.T) + translation
    mapped_up = rotation @ source_up
    up_error_deg = float(np.degrees(np.arccos(np.clip(mapped_up[2], -1.0, 1.0))))

    return {
        "scale": scale,
        "rotation": rotation,
        "translation": translation,
        "aligned_centers": aligned_centers,
        "identifiable": bool(identifiable),
        "alignment_mode": alignment_mode,
        "target_path_m": target_path_m,
        "source_planar_variance": source_variance,
        "yaw_deg": float(np.degrees(yaw_rad)),
        "up_error_deg": up_error_deg,
    }
