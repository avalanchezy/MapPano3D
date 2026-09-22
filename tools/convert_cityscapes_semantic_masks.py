#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

CITYSCAPES_CLASSES = [
    "road",
    "sidewalk",
    "building",
    "wall",
    "fence",
    "pole",
    "traffic light",
    "traffic sign",
    "vegetation",
    "terrain",
    "sky",
    "person",
    "rider",
    "car",
    "truck",
    "bus",
    "train",
    "motorcycle",
    "bicycle",
]

DYNAMIC_CLASSES = {
    "person",
    "rider",
    "car",
    "truck",
    "bus",
    "train",
    "motorcycle",
    "bicycle",
}

PROFILE_SPECS = {
    "road_only": {
        "description": "Keep only Cityscapes road pixels for road-guided pose refinement.",
        "keep": {"road"},
    },
    "road_sidewalk": {
        "description": "Keep road plus sidewalk pixels for road-guided pose refinement ablation.",
        "keep": {"road", "sidewalk"},
    },
    "no_sky": {
        "description": "Remove only semantic sky. This is the Mask2Former analogue of sky filtering.",
        "invalid": {"sky"},
    },
    "sky_dynamic": {
        "description": "Remove sky plus dynamic traffic objects and people.",
        "invalid": {"sky", *DYNAMIC_CLASSES},
    },
    "static_no_vegetation": {
        "description": "Remove sky, dynamic objects, vegetation, and terrain for a cleaner static map.",
        "invalid": {"sky", "vegetation", "terrain", *DYNAMIC_CLASSES},
    },
    "road_structure": {
        "description": "Keep road surface and hard structures, but drop sky, dynamic objects, and vegetation.",
        "keep": {
            "road",
            "sidewalk",
            "building",
            "wall",
            "fence",
            "pole",
            "traffic light",
            "traffic sign",
            "terrain",
        },
    },
    "structure": {
        "description": "Keep only hard vertical or map-stable structures.",
        "keep": {
            "building",
            "wall",
            "fence",
            "pole",
            "traffic light",
            "traffic sign",
        },
    },
}


def normalize_name(name: str) -> str:
    return name.strip().lower().replace("_", " ")


def load_class_names(seg_root: Path) -> list[str]:
    classes_json = seg_root / "classes.json"
    if not classes_json.exists():
        return CITYSCAPES_CLASSES
    data = json.loads(classes_json.read_text(encoding="utf-8"))
    classes = data.get("classes", [])
    names = [str(item.get("name", f"class_{idx}")) for idx, item in enumerate(classes)]
    return names or CITYSCAPES_CLASSES


def resolve_profile(profile: str, class_names: list[str]) -> dict:
    if profile not in PROFILE_SPECS:
        known = ", ".join(sorted(PROFILE_SPECS))
        raise ValueError(f"Unknown profile '{profile}'. Known profiles: {known}")
    spec = PROFILE_SPECS[profile]
    name_to_id = {normalize_name(name): idx for idx, name in enumerate(class_names)}

    resolved = {
        "description": spec["description"],
        "mode": "keep" if "keep" in spec else "invalid",
        "class_names": class_names,
    }
    key = "keep" if "keep" in spec else "invalid"
    wanted_names = {normalize_name(name) for name in spec[key]}
    missing = sorted(name for name in wanted_names if name not in name_to_id)
    if missing:
        raise ValueError(
            f"Profile '{profile}' refers to classes not present in metadata: {', '.join(missing)}"
        )
    resolved[f"{key}_ids"] = sorted(name_to_id[name] for name in wanted_names)
    resolved[f"{key}_names"] = sorted(wanted_names)
    return resolved


def semantic_to_image_stem(path: Path) -> str:
    stem = path.stem
    for suffix in ("_sem", "_semantic", "-sem", "-semantic"):
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def iter_semantic_masks(seg_root: Path, routes: list[str]) -> list[tuple[str, Path]]:
    items: list[tuple[str, Path]] = []
    semantic_root = seg_root / "semantic_class_id"
    for route in routes:
        route_dir = semantic_root / route
        if not route_dir.exists():
            raise FileNotFoundError(route_dir)
        masks = sorted(path for path in route_dir.iterdir() if path.suffix.lower() in IMG_EXTS)
        items.extend((route, path) for path in masks)
    return items


def build_valid_mask(sem: np.ndarray, resolved_profile: dict, dilate_invalid: int) -> np.ndarray:
    class_count = len(resolved_profile["class_names"])
    known = sem < class_count
    if resolved_profile["mode"] == "keep":
        keep_ids = np.asarray(resolved_profile["keep_ids"], dtype=sem.dtype)
        valid = np.isin(sem, keep_ids) & known
    else:
        invalid_ids = np.asarray(resolved_profile["invalid_ids"], dtype=sem.dtype)
        valid = (~np.isin(sem, invalid_ids)) & known

    if dilate_invalid > 0:
        radius = int(dilate_invalid)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
        invalid = cv2.dilate((~valid).astype(np.uint8), kernel, iterations=1)
        valid = invalid == 0
    return valid


def write_preview(
    image_path: Path,
    valid: np.ndarray,
    out_path: Path,
    alpha: float,
) -> bool:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        return False
    if image.shape[:2] != valid.shape:
        valid = cv2.resize(valid.astype(np.uint8), (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
    overlay = image.copy()
    overlay[~valid] = (0, 0, 180)
    preview = cv2.addWeighted(image, 1.0 - alpha, overlay, alpha, 0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), preview)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert Mask2Former Cityscapes semantic predictions into PanoVGGT valid masks."
    )
    parser.add_argument(
        "--seg-root",
        type=Path,
        required=True,
        help="Mask2Former output root containing semantic_class_id/ and classes.json.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        help="Output root for valid masks. Defaults to <seg-root>/valid_masks.",
    )
    parser.add_argument("--frame-root", type=Path, required=True)
    parser.add_argument("--routes", nargs="+", required=True)
    parser.add_argument(
        "--profiles",
        nargs="+",
        default=["all"],
        help=f"Profiles to write, or 'all'. Available: {', '.join(PROFILE_SPECS)}",
    )
    parser.add_argument(
        "--dilate-invalid",
        type=int,
        default=2,
        help="Dilate invalid regions by this many pixels to suppress boundary bleed.",
    )
    parser.add_argument("--write-preview", action="store_true")
    parser.add_argument("--preview-limit", type=int, default=24, help="Preview images per profile. 0 writes all.")
    parser.add_argument("--preview-alpha", type=float, default=0.45)
    args = parser.parse_args()

    if args.dilate_invalid < 0:
        raise ValueError("--dilate-invalid must be >= 0")
    output_root = args.output_root if args.output_root is not None else args.seg_root / "valid_masks"
    profiles = list(PROFILE_SPECS) if "all" in args.profiles else args.profiles
    class_names = load_class_names(args.seg_root)
    resolved_profiles = {profile: resolve_profile(profile, class_names) for profile in profiles}
    all_resolved_profiles = {
        profile: resolve_profile(profile, class_names)
        for profile in PROFILE_SPECS
    }

    items = iter_semantic_masks(args.seg_root, args.routes)
    if not items:
        raise FileNotFoundError(f"No semantic masks found under {args.seg_root / 'semantic_class_id'}")

    output_root.mkdir(parents=True, exist_ok=True)
    profile_info = {
        "seg_root": str(args.seg_root),
        "output_root": str(output_root),
        "dilate_invalid": args.dilate_invalid,
        "profiles": all_resolved_profiles,
    }
    (output_root / "profile_definitions.json").write_text(json.dumps(profile_info, indent=2), encoding="utf-8")

    summary_rows = []
    preview_counts = {profile: 0 for profile in profiles}
    for profile in profiles:
        resolved = resolved_profiles[profile]
        for route, sem_path in tqdm(items, desc=f"convert:{profile}"):
            sem = cv2.imread(str(sem_path), cv2.IMREAD_UNCHANGED)
            if sem is None:
                print(f"[warn] cannot read semantic mask: {sem_path}", file=sys.stderr)
                continue
            if sem.ndim == 3:
                sem = sem[:, :, 0]
            sem = sem.astype(np.uint16, copy=False)
            valid = build_valid_mask(sem, resolved, args.dilate_invalid)
            image_stem = semantic_to_image_stem(sem_path)

            route_out_dir = output_root / profile / route
            route_out_dir.mkdir(parents=True, exist_ok=True)
            out_mask = route_out_dir / f"{image_stem}.png"
            cv2.imwrite(str(out_mask), np.where(valid, 255, 0).astype(np.uint8))

            valid_pixels = int(valid.sum())
            total_pixels = int(valid.size)
            summary_rows.append(
                {
                    "profile": profile,
                    "route": route,
                    "semantic_mask": str(sem_path),
                    "valid_mask": str(out_mask),
                    "valid_pixels": valid_pixels,
                    "total_pixels": total_pixels,
                    "valid_ratio": valid_pixels / max(total_pixels, 1),
                }
            )

            if args.write_preview and (args.preview_limit == 0 or preview_counts[profile] < args.preview_limit):
                image_path = args.frame_root / route / f"{image_stem}.jpg"
                preview_path = output_root / "preview" / profile / route / f"{image_stem}_preview.jpg"
                if write_preview(image_path, valid, preview_path, args.preview_alpha):
                    preview_counts[profile] += 1

    if summary_rows:
        summary_path = output_root / "mask_summary.csv"
        if summary_path.exists():
            current_keys = {(row["profile"], row["route"]) for row in summary_rows}
            with summary_path.open(newline="", encoding="utf-8") as fh:
                existing_rows = [
                    row
                    for row in csv.DictReader(fh)
                    if (row.get("profile"), row.get("route")) not in current_keys
                ]
            summary_rows = existing_rows + summary_rows
        with summary_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)

    print(f"[done] valid masks: {output_root}")
    for profile in profiles:
        print(f"  {profile}: {output_root / profile}")


if __name__ == "__main__":
    main()
