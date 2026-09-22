from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).parents[2] / "tools" / "run_nuscenes_mappano3d_long_postprocess.py"
SPEC = importlib.util.spec_from_file_location("nuscenes_map_postprocess", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class TransformTests(unittest.TestCase):
    def test_perturb_target_identity(self) -> None:
        target = np.repeat(np.eye(4)[None], 3, axis=0)
        target[:, 0, 3] = [0.0, 1.0, 2.0]
        target[:, 1, 3] = [2.0, 2.0, 2.0]
        result = MODULE.perturb_target_poses(target, 0.0, 1.0, (0.0, 0.0))
        self.assertTrue(np.allclose(result, target))


    def test_perturb_target_translation_and_scale_about_centroid(self) -> None:
        target = np.repeat(np.eye(4)[None], 2, axis=0)
        target[:, :3, 3] = [[-1.0, 0.0, 1.0], [1.0, 0.0, 3.0]]
        result = MODULE.perturb_target_poses(target, 0.0, 2.0, (3.0, -2.0))
        self.assertTrue(np.allclose(result[:, :3, 3], [[1.0, -2.0, 0.0], [5.0, -2.0, 4.0]]))


    def test_apply_bev_scales_vertical_about_camera_height(self) -> None:
        points = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 3.0]])
        target = np.array([[0.0, 0.0, 2.0], [2.0, 0.0, 2.0]])
        delta = {"yaw": 0.0, "scale": 2.0, "tx": 0.0, "ty": 0.0}
        result = MODULE.apply_bev_to_points(points, target, delta)
        self.assertTrue(np.allclose(result[:, 2], [0.0, 4.0]))


if __name__ == "__main__":
    unittest.main()
