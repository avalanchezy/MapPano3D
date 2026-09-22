# VGGT + nuScenes: Complete Workflow

This is the release's single end-to-end guide. Commands use Bash (Linux or
WSL), from the MapPano3D root unless stated otherwise. Keep separate upstream
environments for VGGT-Long and Mask2Former/Detectron2; the geometry/evaluation
environment uses `requirements.txt`. Switch to the named environment before
each inference step, retaining the shell variables below.

## 1. Dependencies and Data

Install this repository's requirements and tests as described in the README.
Clone the upstream geometry implementation and apply the export patch:

```bash
export ROOT="$PWD"
git clone https://github.com/DengKaiCQ/VGGT-Long.git VGGT-Long
git -C VGGT-Long checkout c160869d1d99c96bb227f414afb3bc68c29c9a76
git -C VGGT-Long apply ../patches/vggt-long-export.patch
```

Follow that pinned repository's installation and checkpoint instructions.
Its requirements omit two unconditional loop/export imports; install them
in the VGGT-Long environment and check imports before inference:

```bash
python -m pip install pypose trimesh
(cd "$ROOT/VGGT-Long" && python -c "import vggt_long")
```

For the released configuration, place VGGT `model.pt`, SALAD
`dino_salad.ckpt`, and `dinov2_vitb14_pretrain.pth` under `VGGT-Long/weights/`.
The configuration uses SALAD loop closure, not DBoW. Install Mask2Former
under `Mask2Former/` using its upstream environment and Cityscapes R50
semantic checkpoint; `run_mask2former.py --help` lists config/weight options.

Obtain nuScenes-mini and expansion maps from the dataset provider. The root
contains `v1.0-mini/`, `samples/`, and `maps/`, including expansion JSON maps
with intersection polygons. No MovieMap assets are needed.

```bash
export DATA="$ROOT/data/nuscenes"
export INPUT="$ROOT/outputs/nuscenes/inputs"
export SEMANTICS="$ROOT/outputs/nuscenes/semantics"
export RAW="$ROOT/outputs/nuscenes/raw"
export ANCHORS="$ROOT/outputs/nuscenes/anchors"
export REFINED="$ROOT/outputs/nuscenes/refined"
export REPORTS="$ROOT/outputs/nuscenes/reports"
SCENES=(scene-0061 scene-0103 scene-0553 scene-0655 scene-0757
        scene-0796 scene-0916 scene-1077 scene-1094 scene-1100)
```

## 2. Six-Camera Inputs and Semantics

The historical `sweep_fov60` directory name means **six native camera images**
in this nuScenes adapter: it neither stitches nor recrops them into panoramas.

```bash
python tools/build_nuscenes_long_sequences.py \
  --dataroot "$DATA" --version-dir "$DATA/v1.0-mini" \
  --out-root "$INPUT" --view-mode sweep_fov60 --copy-mode copy \
  --scene "${SCENES[@]}"

# In the Mask2Former/Detectron2 environment:
python tools/run_mask2former.py \
  --mask2former-dir "$ROOT/Mask2Former" \
  --input-root "$INPUT/sweep_fov60" --routes "${SCENES[@]}" \
  --output-root "$SEMANTICS"
```

Masks are `semantic_class_id/SCENE/IMAGE_STEM_sem.png`, with `classes.json`.
Road ID 0 supplies alignment evidence; sky ID 10 and dynamic IDs 11-18 are
excluded from static fusion. The complete static cloud retains other classes.

## 3. Shared VGGT Predictions

In the VGGT-Long environment:

```bash
cd "$ROOT/VGGT-Long"
for SCENE in "${SCENES[@]}"; do
  python vggt_long.py --image_dir "$INPUT/sweep_fov60/$SCENE/images" \
    --config "$ROOT/rebuttal_exp/configs/vggt_long_nuscenes_c40_o20.yaml" \
    --save_dir "$RAW/$SCENE/VGGT" || break
done
cd "$ROOT"
```

Each chunk contains **40 total camera images**, with **20-image overlap**,
not 40 six-camera timestamps. Both methods reuse these predictions, exported
chunk clouds, camera poses, and semantic filtering.

## 4. Map Registration and Full-Scene Fusion

In the geometry/evaluation environment:

```bash
for SCENE in "${SCENES[@]}"; do
  python rebuttal_exp/scripts/run_nuscenes_vggt_backbone_mappano3d.py \
    --scene "$SCENE" --vggt-dir "$RAW/$SCENE/VGGT" \
    --input-dir "$INPUT/sweep_fov60/$SCENE" --semantic-root "$SEMANTICS" \
    --dataroot "$DATA" --version-dir "$DATA/v1.0-mini" \
    --output-dir "$ANCHORS/$SCENE" --chunk-size 40 --overlap 20 || break
done

python rebuttal_exp/scripts/run_nuscenes_global_vggt_refine.py \
  --scene "${SCENES[@]}" --chunk-anchor-root "$ANCHORS" --vggt-root "$RAW" \
  --input-root "$INPUT/sweep_fov60" --dataroot "$DATA" \
  --version-dir "$DATA/v1.0-mini" --output-root "$REFINED" \
  --chunk-size 40 --overlap 20 --road-evidence semantic
```

The first command prepares semantic chunk caches. The global refiner removes
their temporary per-chunk placement and applies one shared upright similarity
from the pre-refinement VGGT trajectory to the GT camera trajectory. This
alignment is fixed for both methods. Map intersections select supported
updates; no LiDAR or method-specific global re-fitting enters refinement.
The resulting files used for the matched comparison are:

- `vggt_long_registered_static_cloud.ply`: shared-placement VGGT-Long baseline.
- `registered_refined_cloud.ply`: MapPano3D full static reconstruction.
- `metrics.json`: per-chunk map metrics and acceptance decisions.

The bounded VGGT update fixes scale to one and preserves initialized height.
Both branches have the same static inputs and voxel settings; no extra
method-specific one-sided point deletion is used in this matched comparison.

## 5. LiDAR Evaluation and Ten-Scene Summary

```bash
python rebuttal_exp/scripts/build_nuscenes_vggt_backbone_manifest.py \
  --run-root "$REFINED" --input-root "$INPUT/sweep_fov60" \
  --scene "${SCENES[@]}" --min-target-path-m 0 \
  --output "$REPORTS/comparison_manifest.csv"

python rebuttal_exp/scripts/evaluate_nuscenes_method_comparison.py \
  --manifest "$REPORTS/comparison_manifest.csv" --dataroot "$DATA" \
  --version v1.0-mini --min-target-path-m 0 \
  --output-dir "$REPORTS/lidar"

python rebuttal_exp/scripts/summarize_nuscenes_vggt_backbone_results.py \
  --metrics "$REPORTS/lidar/per_scene_metrics.csv" \
  --manifest "$REPORTS/comparison_manifest.csv" --run-root "$REFINED" \
  --version-dir "$DATA/v1.0-mini" \
  --output-csv "$REPORTS/all10_detailed.csv" \
  --output-json "$REPORTS/all10_summary.json"
```

`--min-target-path-m 0` retains the two near-stationary scenes. The main
comparison uses `groups.all10` in the summary, not the moving-only diagnostic.
Check that it contains all ten scene IDs and that the per-scene CSV contains
both methods and both subsets for each scene (40 rows).

The evaluator uses a 35-m corridor, heights [-2.5, 6] m, 0.25-m voxels, and
reference dynamic-box removal. Relative height above 0.35 m defines the
non-ground subset. Accuracy and completeness are directed nearest-neighbor
distances; Chamfer-L1 is their **sum**. Precision/recall/F1 are reported at
0.5 and 1 m; the paper uses 1 m. Scene metrics are equally weighted, including
identity updates. No cloud is independently fitted to LiDAR.

Mini LiDAR evaluation informed acceptance-setting development. LiDAR is not
an input to the reconstruction, optimization, or per-update acceptance rule.
Map-fit metrics and LiDAR geometry metrics remain separate outputs.

## 6. Matched Road-Evidence Ablation

Reuse steps 1-3 and the same chunk caches. Run the global refiner with
`--road-evidence ground_proxy` and a different `--output-root`. Its evaluation
still uses the same semantic-road sample; only optimization evidence changes.
Compare initialization, ground-BEV, and road-BEV within this fixed protocol.

Pi3X and PanoVGGT are documented in the [interface reference](backbones.md),
not as additional end-to-end workflows. Unit tests exercise contracts and
geometry helpers; they do not run pretrained networks or reproduce experimental
scores. Full reproduction requires the actual nuScenes data, weights, and GPU.
