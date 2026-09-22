#!/usr/bin/env python3
"""Pixel-level road mask and skeleton extraction from aligned GSI raster layers.

Usage:
    python extract_road_skeleton.py \
        --photo map_gsi_photo_z18.png \
        --standard map_gsi_std_z18.png \
        --pale map_gsi_pale_z18.png \
        --output output_dir
"""

from pathlib import Path
import argparse
import csv
import json
import numpy as np
import cv2
from PIL import Image
from scipy.ndimage import convolve
from skimage.morphology import dilation, closing, disk, remove_small_holes, skeletonize

NEIGHBORS8 = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),            (0, 1),
    (1, -1),  (1, 0),   (1, 1),
]


def degree8(skeleton):
    kernel = np.ones((3, 3), dtype=np.uint8)
    return (
        convolve(skeleton.astype(np.uint8), kernel, mode="constant", cval=0)
        - skeleton.astype(np.uint8)
    )


def trace_from_endpoint(skeleton, degrees, start_yx):
    h, w = skeleton.shape
    path = [start_yx]
    previous = None
    current = start_yx

    for _ in range(h * w):
        y, x = current
        candidates = []
        for dy, dx in NEIGHBORS8:
            yy, xx = y + dy, x + dx
            if (
                0 <= yy < h
                and 0 <= xx < w
                and skeleton[yy, xx]
                and (previous is None or (yy, xx) != previous)
            ):
                candidates.append((yy, xx))

        if not candidates:
            break
        if current != start_yx and degrees[y, x] != 2:
            break
        if len(candidates) != 1:
            if current == start_yx:
                next_pixel = candidates[0]
            else:
                break
        else:
            next_pixel = candidates[0]

        previous, current = current, next_pixel
        path.append(current)
        if degrees[current] != 2:
            break

    return path


def prune_short_spurs(skeleton, max_length=8.0, border_margin=2, max_iterations=10):
    result = skeleton.copy()
    h, w = result.shape

    for _ in range(max_iterations):
        degrees = degree8(result)
        endpoints = np.argwhere(result & (degrees == 1))
        remove = np.zeros_like(result)

        for y, x in endpoints:
            y, x = int(y), int(x)
            if min(x, y, w - 1 - x, h - 1 - y) <= border_margin:
                continue

            path = trace_from_endpoint(result, degrees, (y, x))
            if len(path) <= 1:
                continue

            arr = np.asarray(path, dtype=np.int32)
            steps = np.diff(arr, axis=0)
            diagonal = np.all(np.abs(steps) == 1, axis=1)
            length = float(np.sum(np.where(diagonal, np.sqrt(2.0), 1.0)))

            if length <= max_length:
                for yy, xx in path[:-1]:
                    remove[yy, xx] = True

        if not remove.any():
            break
        result[remove] = False

    return result


def overlay_line(rgb, line, width=3, alpha=0.82):
    result = rgb.copy()
    visible = cv2.dilate(
        line.astype(np.uint8),
        np.ones((width, width), dtype=np.uint8),
        iterations=1,
    ) > 0
    red = np.asarray((255, 0, 0), dtype=np.float32)
    result[visible] = (
        result[visible].astype(np.float32) * (1.0 - alpha) + red * alpha
    ).astype(np.uint8)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--photo", required=True)
    parser.add_argument("--standard", required=True)
    parser.add_argument("--pale", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    photo = np.asarray(Image.open(args.photo).convert("RGB"))
    standard = np.asarray(Image.open(args.standard).convert("RGB"))
    pale = np.asarray(Image.open(args.pale).convert("RGB"))

    if not (photo.shape == standard.shape == pale.shape):
        raise ValueError("The three rasters must have identical dimensions.")

    channel_max = pale.max(axis=2)
    channel_min = pale.min(axis=2)
    channel_mean = pale.mean(axis=2)
    structure = ((channel_max - channel_min) <= 4) & (channel_mean < 245)

    barrier = cv2.dilate(
        structure.astype(np.uint8),
        np.ones((7, 7), dtype=np.uint8),
        iterations=1,
    ) > 0

    free = (~barrier).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(free, connectivity=4)
    if n <= 1:
        raise RuntimeError("No free-space component found.")

    road_id = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    road_core = labels == road_id

    road_mask = dilation(road_core, disk(3)) & (~structure)
    road_mask = remove_small_holes(road_mask, connectivity=2, area_threshold=1200)
    road_mask = closing(road_mask, disk(1))

    skeleton = prune_short_spurs(skeletonize(road_mask))
    centerline = prune_short_spurs(skeletonize(closing(road_mask, disk(10))))

    Image.fromarray((road_mask.astype(np.uint8) * 255), mode="L").save(
        output / "road_mask_binary.png"
    )
    Image.fromarray((skeleton.astype(np.uint8) * 255), mode="L").save(
        output / "road_skeleton_pixel.png"
    )
    Image.fromarray((centerline.astype(np.uint8) * 255), mode="L").save(
        output / "road_centerline_pixel.png"
    )

    Image.fromarray(overlay_line(photo, centerline)).save(
        output / "road_centerline_overlay_aerial.png"
    )
    Image.fromarray(overlay_line(pale, centerline)).save(
        output / "road_centerline_overlay_pale.png"
    )

    degrees = degree8(centerline)
    ys, xs = np.nonzero(centerline)
    with (output / "road_centerline_pixels.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.writer(f)
        writer.writerow(["x", "y", "degree8"])
        writer.writerows(
            (int(x), int(y), int(degrees[y, x]))
            for y, x in zip(ys, xs)
        )

    metadata = {
        "width": int(pale.shape[1]),
        "height": int(pale.shape[0]),
        "coordinate_origin": "top-left",
        "resized_or_resampled": False,
        "road_mask_pixels": int(road_mask.sum()),
        "skeleton_pixels": int(skeleton.sum()),
        "centerline_pixels": int(centerline.sum()),
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
