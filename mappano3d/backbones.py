"""Shared output contract and model-specific integration hooks.

Callbacks run an upstream frozen predictor and convert its outputs; these
hooks do not download weights or assume an upstream model's tensor keys.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np


@dataclass
class PointCloud:
    points: np.ndarray
    colors: np.ndarray | None = None


@dataclass
class GeometryChunk:
    """Image-associated points and camera-to-world poses in one local frame.

    point_maps/colors: (N, H, W, 3); poses: (N, 4, 4).
    valid_mask/confidence: optional (N, H, W) arrays. vertical_axis identifies
    the upright adapter convention (1 for X/Z maps, 2 for nuScenes X/Y maps).
    The producer normalizes colors to [0, 1] and resolves camera conventions.
    """

    frame_ids: tuple[str, ...]
    point_maps: np.ndarray
    camera_to_world: np.ndarray
    vertical_axis: int
    colors: np.ndarray | None = None
    valid_mask: np.ndarray | None = None
    confidence: np.ndarray | None = None

    def __post_init__(self):
        shape = self.point_maps.shape
        if len(shape) != 4 or shape[-1] != 3 or shape[0] != len(self.frame_ids):
            raise ValueError("point_maps must have shape (frames, H, W, 3)")
        if self.camera_to_world.shape != (len(self.frame_ids), 4, 4):
            raise ValueError("camera_to_world must contain one 4x4 pose per frame")
        if self.vertical_axis not in (1, 2):
            raise ValueError("vertical_axis must be 1 (Y) or 2 (Z)")
        for name in ("valid_mask", "confidence"):
            value = getattr(self, name)
            if value is not None and value.shape != shape[:-1]:
                raise ValueError(f"{name} must match the point-map pixel grid")
        if self.colors is not None and self.colors.shape != shape:
            raise ValueError("colors must match point_maps")

    def semantic_points(self, labels, road_id=0, excluded_ids=(10, *range(11, 19))):
        """Return road evidence and the full static cloud on the same pixel grid."""
        if labels.shape != self.point_maps.shape[:-1]:
            raise ValueError("semantic labels must match the point-map pixel grid")
        valid = np.isfinite(self.point_maps).all(axis=-1)
        if self.valid_mask is not None:
            valid &= self.valid_mask.astype(bool)
        road = valid & (labels == road_id)
        static = valid & ~np.isin(labels, excluded_ids)

        def cloud(mask):
            colors = self.colors[mask] if self.colors is not None else None
            return PointCloud(self.point_maps[mask], colors)

        return cloud(road), cloud(static)


PredictAndConvert = Callable[[Sequence[np.ndarray], tuple[str, ...]], GeometryChunk]


class FeedForwardAdapter:
    """Bind a predictor/export converter to the shared geometry contract."""

    def __init__(self, predict_and_convert: PredictAndConvert):
        self.predict_and_convert = predict_and_convert

    def predict(self, images: Sequence[np.ndarray], frame_ids: tuple[str, ...]) -> GeometryChunk:
        if len(images) != len(frame_ids):
            raise ValueError("one frame identifier is required per input image")
        chunk = self.predict_and_convert(images, frame_ids)
        if tuple(chunk.frame_ids) != tuple(frame_ids):
            raise ValueError("the adapter must preserve input frame order")
        return chunk


class VGGTAdapter(FeedForwardAdapter):
    """VGGT integration hook; the full nuScenes workflow uses VGGT-Long exports."""

    name = "VGGT"


class Pi3XAdapter(FeedForwardAdapter):
    """Pi3X integration hook; supply a predictor-specific output converter."""

    name = "Pi3X"


class PanoVGGTAdapter(FeedForwardAdapter):
    """PanoVGGT integration hook for image-associated panoramic point maps."""

    name = "PanoVGGT"
