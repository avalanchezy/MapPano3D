# Feed-Forward Backbone Interfaces

MapPano3D separates neural geometry prediction from map registration. The
common API in `mappano3d/backbones.py` provides three named entry points:

| Entry point | Role | Released workflow |
| --- | --- | --- |
| `VGGTAdapter` | Perspective point-map integration hook | Complete VGGT-Long + nuScenes guide |
| `Pi3XAdapter` | Pi3X output-conversion hook | Shared `GeometryChunk` interface |
| `PanoVGGTAdapter` | Equirectangular point-map integration hook | Shared `GeometryChunk` interface and export patches |

Each hook takes a `predict_and_convert(images, frame_ids)` callback that
invokes an upstream frozen model or reads cached outputs, then returns
`GeometryChunk`. The callback owns model-specific tensor keys, camera
conversion, image resizing, and conversion to NumPy. This connects upstream
geometry to the same training-free registration and fusion stages.

## Shared Data Contract

| `GeometryChunk` field | Convention |
| --- | --- |
| `frame_ids` | Ordered tuple of frame identifiers |
| `point_maps` | `(N, H, W, 3)` points in one shared local world frame |
| `camera_to_world` | `(N, 4, 4)` poses in that same frame |
| `vertical_axis` | `1` for Y-up/XZ maps; `2` for Z-up/XY maps |
| `colors` | Optional `(N, H, W, 3)` RGB in `[0, 1]` |
| `valid_mask` | Optional `(N, H, W)` boolean geometry/confidence selection |
| `confidence` | Optional `(N, H, W)` scores; the converter chooses thresholds |

Points use the common chunk frame, not separate per-camera frames. Invert
world-to-camera matrices when needed. Point maps and semantic labels must
refer to the same resized/cropped pixel grid. Panorama associations follow
the equirectangular grid. `vertical_axis` describes the adapter convention;
it does not infer or rotate the ground plane.

`chunk.semantic_points(labels)` returns road evidence and the complete static
scene as two `PointCloud` objects. Defaults use Cityscapes road ID 0 and
exclude sky ID 10 and dynamic IDs 11-18; other label systems supply `road_id`
and `excluded_ids`. Buildings and vegetation remain in the static cloud.
Invalid/nonfinite points are omitted from both sets.

The map adapter aligns geometry to camera anchors and map coordinates, then
passes road evidence and static geometry separately to the refiner. Accepted
road-guided transforms align the complete static scene.

## Existing VGGT-Long Export Adapter

The complete guide uses the existing PLY export path rather than requiring
dense point maps in memory. `run_nuscenes_vggt_backbone_mappano3d.py` reads
numbered chunk PLY files, `camera_poses.txt`, and `intrinsic.txt`, and selects
semantics by reprojection. `run_nuscenes_global_vggt_refine.py` fixes one
shared global placement and refines the same chunk geometry locally.
The fixed-scale VGGT update changes X/Y and preserves Z.

## Upstream Export Patches

| Repository | Tested base commit | Patch |
| --- | --- | --- |
| PanoVGGT | `18575370eb6fdf447342a055701193d20cd859e6` | `patches/panovggt-inference.patch` |
| VGGT-Long | `c160869d1d99c96bb227f414afb3bc68c29c9a76` | `patches/vggt-long-export.patch` |
| PanoVGGT-Long | `0dfabf91ef4d6853aa4cad0aaf5011ba09a2e29b` | `patches/panovggt-long-export.patch` |

The patches preserve export information with neural weights frozen. Supply
a Pi3X output-conversion callback to connect its predictions to the shared
interface. The paper's experiments instantiate the framework with VGGT
and PanoVGGT.

Follow [VGGT + nuScenes](reproduction.md) for the complete workflow, or use
the [custom-data specification](custom_360_inputs.md) to prepare panoramic
map and anchor inputs.
