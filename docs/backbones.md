# Feed-Forward Backbone Interface

MapPano3D's registration stage consumes geometry, not a particular neural
architecture. A local predictor supplies:

1. Colored 3D points or a point map for each image, with optional confidence.
2. Camera-to-world poses in the same local frame as those points.
3. The association between image pixels and points, or intrinsics and poses
   sufficient for semantic reprojection.
4. Frame identifiers linking predictions to camera anchors and semantic masks.

An adapter resolves the predictor's scale and coordinate convention, selects
road evidence and static scene points, and initializes the chunk. The same
bounded upright BEV registration then operates in map coordinates. Accepted
transforms propagate to the complete static cloud.

## Released Interfaces

- **Panoramic geometry:** use your own 360-degree video with PanoVGGT or another
  compatible predictor, then provide point clouds, camera anchors and map
  support through the [custom-video interface](custom_360_inputs.md).
  The PanoVGGT export patch is included; private dataset loaders are not.
- **VGGT:** `run_nuscenes_vggt_backbone_mappano3d.py` consumes numbered
  VGGT-Long chunk PLY files, `camera_poses.txt`, and `intrinsic.txt`.
  `run_nuscenes_global_vggt_refine.py` shares global placement with VGGT-Long
  and applies map-supported local updates to the same predictions.

The generic panoramic interface uses X/Z horizontally; the nuScenes adapter uses X/Y
horizontally and Z vertically. Resolve this convention before alignment.
Positive-scale upright updates preserve height in the nuScenes adapter.

Other predictors, including pi3, can expose the same geometry interface. This
release provides the interfaces above; adding another predictor
requires implementing its output conversion, not retraining MapPano3D.

## Upstream Snapshots and Export Patches

| Repository | Tested base commit | Patch |
| --- | --- | --- |
| PanoVGGT | `18575370eb6fdf447342a055701193d20cd859e6` | `patches/panovggt-inference.patch` |
| VGGT-Long | `c160869d1d99c96bb227f414afb3bc68c29c9a76` | `patches/vggt-long-export.patch` |
| PanoVGGT-Long | `0dfabf91ef4d6853aa4cad0aaf5011ba09a2e29b` | `patches/panovggt-long-export.patch` |

Use a separate clone for each backbone and follow its installation and weight
download instructions. For example, from the MapPano3D root:

```bash
git clone https://github.com/DengKaiCQ/VGGT-Long.git VGGT-Long
git -C VGGT-Long checkout c160869d1d99c96bb227f414afb3bc68c29c9a76
git -C VGGT-Long apply ../patches/vggt-long-export.patch
```

The VGGT patch retains per-chunk export information used by the shared-input
comparison. The PanoVGGT patch adds masked inference/export options used by
the panoramic adapter. These patches do not train or fine-tune the models.
