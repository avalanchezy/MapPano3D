# Reproduction

Run commands from the MapPano3D repository root unless stated otherwise.
Paths below are examples for your local data and upstream installations.
The geometry/evaluation requirements are separate from the upstream CUDA
environments used for model inference.

## nuScenes: Six-Camera Geometry

Obtain nuScenes-mini and the expansion maps from the dataset provider. The
data root must contain `v1.0-mini/`, `samples/`, and `maps/`, including expansion
JSON maps with intersection polygons. All ten mini scenes are used:
0061, 0103, 0553, 0655, 0757, 0796, 0916, 1077, 1094, 1100.

### Inputs and Semantics

The `sweep_fov60` directory name is retained from the experiment scripts. On
nuScenes this exports the **six native camera images** in timestamp order;
it does not stitch or recrop them into panoramas.

```bash
export ROOT="$PWD"
export DATA="$ROOT/data/nuscenes"
export INPUT="$ROOT/outputs/nuscenes/inputs"
export SEMANTICS="$ROOT/outputs/nuscenes/semantics"
export RAW="$ROOT/outputs/nuscenes/raw"
export ANCHORS="$ROOT/outputs/nuscenes/anchors"
export REFINED="$ROOT/outputs/nuscenes/refined"
export SCENE=scene-0796

python tools/build_nuscenes_long_sequences.py \
  --dataroot "$DATA" --version-dir "$DATA/v1.0-mini" \
  --out-root "$INPUT" --view-mode sweep_fov60 --copy-mode copy

# Run in the Mask2Former/Detectron2 environment.
python tools/run_mask2former.py \
  --mask2former-dir "$ROOT/Mask2Former" \
  --input-root "$INPUT/sweep_fov60" --routes "$SCENE" \
  --output-root "$SEMANTICS"
```

Use the default Cityscapes R50 checkpoint/config or provide `--config` and
`--weights`. Masks are saved as `semantic_class_id/SCENE/IMAGE_STEM_sem.png`,
with class definitions in `classes.json`. Run segmentation for every scene.

### Shared VGGT Predictions

Install VGGT-Long and apply the [export patch](backbones.md). Place upstream
weights in its `weights/` directory. The released YAML uses paths relative to
that repository; run inference there:

```bash
cd "$ROOT/VGGT-Long"
python vggt_long.py --image_dir "$INPUT/sweep_fov60/$SCENE/images" \
  --config "$ROOT/rebuttal_exp/configs/vggt_long_nuscenes_c40_o20.yaml" \
  --save_dir "$RAW/$SCENE/VGGT"
cd "$ROOT"
```

The configuration uses **40 total camera images per chunk**, **20-image
overlap**, and loop closure. Both methods reuse these predictions and static
semantic filtering.

### MapPano3D Registration and Full-Scene Fusion

```bash
python rebuttal_exp/scripts/run_nuscenes_vggt_backbone_mappano3d.py \
  --scene "$SCENE" --vggt-dir "$RAW/$SCENE/VGGT" \
  --input-dir "$INPUT/sweep_fov60/$SCENE" --semantic-root "$SEMANTICS" \
  --dataroot "$DATA" --version-dir "$DATA/v1.0-mini" \
  --output-dir "$ANCHORS/$SCENE" --chunk-size 40 --overlap 20

python rebuttal_exp/scripts/run_nuscenes_global_vggt_refine.py \
  --scene "$SCENE" --chunk-anchor-root "$ANCHORS" --vggt-root "$RAW" \
  --input-root "$INPUT/sweep_fov60" --dataroot "$DATA" \
  --version-dir "$DATA/v1.0-mini" --output-root "$REFINED" \
  --chunk-size 40 --overlap 20 --road-evidence semantic
```

The first command prepares semantic chunk caches. The second fixes a shared
global camera alignment for the actual VGGT-Long comparison and applies local
map refinement. Use the second command's output for the matched comparison:

- `vggt_long_registered_static_cloud.ply`: the registered VGGT-Long baseline.
- `registered_refined_cloud.ply`: MapPano3D full static reconstruction.
- `metrics.json`: per-chunk map metrics and accepted-update decisions.

### Independent LiDAR Evaluation

```bash
python rebuttal_exp/scripts/evaluate_nuscenes_lidar.py \
  --dataroot "$DATA" --version v1.0-mini --scene "$SCENE" \
  --pose-init-cloud "$REFINED/$SCENE/vggt_long_registered_static_cloud.ply" \
  --refined-cloud "$REFINED/$SCENE/registered_refined_cloud.ply" \
  --out-dir "$ROOT/outputs/nuscenes/lidar/$SCENE"
```

The evaluator uses a 35-m corridor, heights [-2.5, 6] m, 0.25-m voxels, and
dynamic-box filtering. A 0.35-m ground separation yields the non-ground
subset. Accuracy and completeness are directed nearest-neighbor distances;
Chamfer-L1 is their **sum**. Precision/recall/F1 use the stated threshold
(1 m in the main table). Average scenes equally, retaining all ten, including
identity updates. Do not fit a separate transform to LiDAR.

## User-Created 360-Degree Video

See [the custom-video interface](custom_360_inputs.md). Supply your own video,
map support, and camera anchors. No MovieMap metadata is included.

## Component and Robustness Studies

- For the nuScenes semantic-versus-ground ablation, keep predictions, anchors,
  and evaluation samples fixed and change `--road-evidence semantic` to
  `--road-evidence ground_proxy` in the global refiner. Compare against the same
  initialization cloud; semantic evidence remains the evaluation signal.
- `run_nuscenes_perturbation_study.py` runs the PanoVGGT anchor-stress study;
  `run_nuscenes_vggt_backbone_perturbations.py` provides the separate VGGT
  counterpart. Explicit scene and perturbation selections keep them distinct.

These scripts recompute experiments from locally available predictions and
data. No experimental results are generated by the unit tests. The tests
exercise upright alignment, shared-input chunk indexing, semantic projection
helpers, crossing support, and LiDAR evaluation geometry.
