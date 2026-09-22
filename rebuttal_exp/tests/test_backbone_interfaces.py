import numpy as np
import pytest


@pytest.mark.parametrize("adapter_name", ["VGGTAdapter", "Pi3XAdapter", "PanoVGGTAdapter"])
def test_backbones_share_geometry_contract(adapter_name):
    from mappano3d import backbones

    images = [np.zeros((2, 3, 3)), np.ones((2, 3, 3))]
    frame_ids = ("frame_a", "frame_b")
    points = np.arange(36, dtype=float).reshape(2, 2, 3, 3)
    cameras = np.repeat(np.eye(4)[None], 2, axis=0)

    def predict_and_convert(received_images, received_ids):
        assert received_images is images
        assert received_ids == frame_ids
        return backbones.GeometryChunk(frame_ids, points, cameras, vertical_axis=1)

    adapter = getattr(backbones, adapter_name)(predict_and_convert)
    result = adapter.predict(images, frame_ids)
    np.testing.assert_array_equal(result.point_maps, points)
    np.testing.assert_array_equal(result.camera_to_world, cameras)
    assert result.vertical_axis == 1


def test_semantic_evidence_keeps_full_static_scene_separate():
    from mappano3d.backbones import GeometryChunk

    points = np.arange(18, dtype=float).reshape(1, 1, 6, 3)
    colors = points / 20
    labels = np.array([[[0, 2, 8, 10, 13, 0]]])
    valid = np.array([[[True, True, True, True, True, False]]])
    chunk = GeometryChunk(("frame",), points, np.eye(4)[None], 2, colors, valid)
    road, static = chunk.semantic_points(labels)
    np.testing.assert_array_equal(road.points, points.reshape(-1, 3)[:1])
    np.testing.assert_array_equal(static.points, points.reshape(-1, 3)[:3])
    np.testing.assert_array_equal(static.colors, colors.reshape(-1, 3)[:3])


def test_geometry_requires_image_point_association():
    from mappano3d.backbones import GeometryChunk

    with pytest.raises(ValueError, match="frame"):
        GeometryChunk(("a", "b"), np.zeros((1, 2, 2, 3)), np.eye(4)[None], 2)


def test_nonfinite_points_are_excluded_from_both_subsets():
    from mappano3d.backbones import GeometryChunk

    points = np.array([[[[1.0, 2.0, 3.0], [np.nan, 0.0, 0.0]]]])
    chunk = GeometryChunk(("a",), points, np.eye(4)[None], 2)
    road, static = chunk.semantic_points(np.zeros((1, 1, 2), dtype=int))
    assert len(road.points) == len(static.points) == 1
