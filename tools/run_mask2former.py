#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys
from typing import Iterable

import cv2
import numpy as np
import torch
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def import_mask2former(mask2former_dir: Path):
    sys.path.insert(0, str(mask2former_dir))
    from detectron2.config import get_cfg
    from detectron2.data import MetadataCatalog
    from detectron2.engine.defaults import DefaultPredictor
    from detectron2.projects.deeplab import add_deeplab_config
    from detectron2.utils.visualizer import ColorMode, Visualizer
    from mask2former import add_maskformer2_config

    return get_cfg, MetadataCatalog, DefaultPredictor, add_deeplab_config, Visualizer, ColorMode, add_maskformer2_config


def setup_cfg(args):
    (
        get_cfg,
        MetadataCatalog,
        DefaultPredictor,
        add_deeplab_config,
        Visualizer,
        ColorMode,
        add_maskformer2_config,
    ) = import_mask2former(args.mask2former_dir)
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file(str(args.config))
    cfg.MODEL.WEIGHTS = str(args.weights)
    cfg.MODEL.DEVICE = args.device
    cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON = True
    cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON = False
    cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON = False
    if args.min_size_test:
        cfg.INPUT.MIN_SIZE_TEST = args.min_size_test
    if args.max_size_test:
        cfg.INPUT.MAX_SIZE_TEST = args.max_size_test
    cfg.freeze()
    return cfg, MetadataCatalog, DefaultPredictor, Visualizer, ColorMode


def iter_images(input_root: Path, routes: Iterable[str], frame_step: int) -> list[tuple[str, Path]]:
    items: list[tuple[str, Path]] = []
    for route in routes:
        route_dir = input_root / route
        if not route_dir.exists():
            raise FileNotFoundError(route_dir)
        if (route_dir / "images").is_dir():
            route_dir = route_dir / "images"
        frames = sorted(p for p in route_dir.iterdir() if p.suffix.lower() in IMG_EXTS)
        items.extend((route, p) for p in frames[::frame_step])
    return items


def palette_for_metadata(metadata, num_classes: int) -> np.ndarray:
    colors = getattr(metadata, "stuff_colors", None)
    if colors and len(colors) >= num_classes:
        return np.asarray(colors[:num_classes], dtype=np.uint8)
    rng = np.random.default_rng(7)
    return rng.integers(0, 255, size=(num_classes, 3), dtype=np.uint8)


def write_overlay(image_bgr: np.ndarray, sem: np.ndarray, palette_rgb: np.ndarray, out_path: Path, alpha: float) -> None:
    color_rgb = palette_rgb[np.clip(sem, 0, len(palette_rgb) - 1)]
    color_bgr = color_rgb[:, :, ::-1]
    overlay = cv2.addWeighted(image_bgr, 1.0 - alpha, color_bgr, alpha, 0)
    cv2.imwrite(str(out_path), overlay)


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch Mask2Former semantic segmentation for user-supplied image sequences.")
    parser.add_argument("--mask2former-dir", type=Path, default=Path("Mask2Former"))
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--routes", nargs="+", required=True, help="Sequence directory names under --input-root.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("Mask2Former/configs/cityscapes/semantic-segmentation/maskformer2_R50_bs16_90k.yaml"),
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path("Mask2Former/checkpoints/maskformer2_cityscapes_semantic_R50.pkl"),
    )
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--frame-step", type=int, default=1, help="Segment every Nth frame.")
    parser.add_argument("--min-size-test", type=int, default=1024)
    parser.add_argument("--max-size-test", type=int, default=2048)
    parser.add_argument("--overlay-alpha", type=float, default=0.55)
    parser.add_argument("--write-confidence", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    if args.frame_step < 1:
        raise ValueError("--frame-step must be >= 1")

    args.output_root.mkdir(parents=True, exist_ok=True)
    cfg, MetadataCatalog, DefaultPredictor, Visualizer, ColorMode = setup_cfg(args)
    predictor = DefaultPredictor(cfg)
    metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else "__unused")
    class_names = list(getattr(metadata, "stuff_classes", []) or [f"class_{i}" for i in range(cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES)])
    palette_rgb = palette_for_metadata(metadata, len(class_names))

    class_info = {
        "dataset": cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else None,
        "classes": [{"id": i, "name": name, "color_rgb": palette_rgb[i].tolist()} for i, name in enumerate(class_names)],
        "config": str(args.config),
        "weights": str(args.weights),
        "input_root": str(args.input_root),
        "routes": args.routes,
        "frame_step": args.frame_step,
        "min_size_test": cfg.INPUT.MIN_SIZE_TEST,
        "max_size_test": cfg.INPUT.MAX_SIZE_TEST,
    }
    (args.output_root / "classes.json").write_text(json.dumps(class_info, indent=2), encoding="utf-8")

    items = iter_images(args.input_root, args.routes, args.frame_step)
    if args.limit > 0:
        items = items[: args.limit]

    summary_rows = []
    with torch.no_grad():
        for route, img_path in tqdm(items, desc="segment"):
            image_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if image_bgr is None:
                print(f"[warn] cannot read {img_path}", file=sys.stderr)
                continue
            outputs = predictor(image_bgr)
            logits = outputs["sem_seg"].detach().cpu()
            sem = logits.argmax(dim=0).numpy().astype(np.uint8)

            route_mask_dir = args.output_root / "semantic_class_id" / route
            route_vis_dir = args.output_root / "overlay" / route
            route_mask_dir.mkdir(parents=True, exist_ok=True)
            route_vis_dir.mkdir(parents=True, exist_ok=True)
            stem = img_path.stem
            cv2.imwrite(str(route_mask_dir / f"{stem}_sem.png"), sem)
            write_overlay(image_bgr, sem, palette_rgb, route_vis_dir / f"{stem}_overlay.jpg", args.overlay_alpha)

            if args.write_confidence:
                conf = torch.softmax(logits, dim=0).amax(dim=0).numpy()
                route_conf_dir = args.output_root / "confidence" / route
                route_conf_dir.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(route_conf_dir / f"{stem}_conf.png"), np.clip(conf * 255, 0, 255).astype(np.uint8))

            counts = np.bincount(sem.reshape(-1), minlength=len(class_names))
            total = int(counts.sum())
            row = {
                "route": route,
                "image": img_path.name,
                "width": image_bgr.shape[1],
                "height": image_bgr.shape[0],
                "total_pixels": total,
            }
            for idx, name in enumerate(class_names):
                safe_name = name.replace(" ", "_").replace("/", "_")
                row[f"class_{idx}_{safe_name}_pixels"] = int(counts[idx])
                row[f"class_{idx}_{safe_name}_ratio"] = float(counts[idx] / max(total, 1))
            summary_rows.append(row)

    if summary_rows:
        csv_path = args.output_root / "frame_class_summary.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)

    print(f"[done] {args.output_root}")


if __name__ == "__main__":
    main()
