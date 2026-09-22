from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


SCRIPT = Path(__file__).parents[1] / "scripts" / "evaluate_nuscenes_lidar.py"
SPEC = importlib.util.spec_from_file_location("nuscenes_lidar_eval", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_identity_geometry_metrics() -> None:
    points = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0], [-2.0, 1.0, 0.5]])
    metrics = MODULE.geometry_metrics(points, points.copy(), [0.5, 1.0])
    assert metrics["chamfer_l1_m"] == 0.0
    assert metrics["fscore_0p5m"] == 1.0
    assert metrics["vertical_abs_p90_m"] == 0.0


def test_cached_reference_tree_is_metric_equivalent() -> None:
    prediction = np.array([[0.0, 0.0, 0.0], [1.2, 2.1, 3.0], [-1.5, 0.8, 0.4]])
    reference = np.array([[0.1, 0.0, 0.0], [1.0, 2.0, 3.2], [-2.0, 1.0, 0.5]])
    direct = MODULE.geometry_metrics(prediction, reference, [0.5, 1.0])
    cached = MODULE.geometry_metrics(
        prediction,
        reference,
        [0.5, 1.0],
        reference_tree=cKDTree(reference),
    )
    assert direct == cached


def test_local_path_maps_wsl_drive_on_windows() -> None:
    path = MODULE.local_path("/mnt/c/example/cloud.ply")
    if os.name == "nt":
        assert path == Path("C:/example/cloud.ply")
    else:
        assert path == Path("/mnt/c/example/cloud.ply")


def test_dynamic_box_filter() -> None:
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.5, 0.0, 0.0],
            [0.0, 1.5, 0.0],
            [5.0, 0.0, 0.0],
        ]
    )
    annotations = [
        {
            "instance_token": "instance",
            "translation": [0.0, 0.0, 0.0],
            "rotation": [1.0, 0.0, 0.0, 0.0],
            "size": [2.0, 4.0, 2.0],
        }
    ]
    keep = MODULE.points_outside_boxes(
        points,
        annotations,
        {"instance": {"category_token": "category"}},
        {"category": {"name": "vehicle.car"}},
        0.0,
    )
    assert keep.tolist() == [False, False, True, True]
