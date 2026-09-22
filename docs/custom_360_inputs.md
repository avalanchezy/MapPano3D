# Your Own 360-Degree Video

Record your own equirectangular 360-degree video. MovieMap videos, route
metadata, trajectories, calibration, map assets, and frame associations are
not included or downloaded by this interface. The published paper figure
is an illustration, not a sample dataset.

## Geometry and Semantics

1. Extract ordered frames from your video, retaining timestamps. For example,
   `ffmpeg -i your_video.mp4 -vf fps=2 inputs/sequence/frame_%06d.jpg` samples
   at 2 FPS; create the output directory first.
2. Obtain camera anchors from your own SLAM trajectory or route placement,
   and express them in the same coordinate frame as your road map.
3. Select fixed-40 windows around crossings and retain straight support
   windows. Associate every frame with its camera anchor and semantic mask.
4. Run a compatible frozen geometry predictor, such as PanoVGGT, on those
   windows. The upstream export patch supplies point maps and poses. Select
   road points using semantic labels and retain the full static point cloud
   separately. The framework does not assume that the final scene is road-only.
5. Align local predictions to your camera anchors and establish an upright
   vertical convention before map refinement.

`tools/run_mask2former.py` accepts `--input-root`, `--routes` (your sequence
directory names), and `--output-root`. The mask conversion tool accepts the
same sequence names and explicit frame/segmentation roots. No route IDs or
geographic coordinates are built in. Backbone output conversion is described
in [backbones.md](backbones.md).

## Map-Refinement Interface

The generic panoramic refiner takes the following files created from your
own inputs. These are format specifications, not supplied data:

| Argument | Required contents |
| --- | --- |
| `--map` | Map image on the working pixel grid |
| `--road-mask` | Binary road raster on that grid |
| `--road-skeleton` | Thinned road centerline on that grid |
| `--graph-root` | `nodes.csv` and `edges.csv` |
| `--chunk-root` | `chunk_manifest.csv`; optionally `chunk_items.csv` |
| `--base-recon` | `registration_metrics.json` and `road_clouds/CHUNK_ID.ply` |
| `--full-recon` | `registered_chunks/CHUNK_ID_map.ply` |

The panoramic map plane is X/Z; Y is height. Horizontal coordinates of
registered points, camera anchors, and graph nodes must agree with map pixels.
Metric nuScenes inputs use their separate adapter and coordinate convention.

- `nodes.csv`: `node_id,x_px,y_px`.
- `edges.csv`: `from_node,to_node`.
- `chunk_manifest.csv`: `chunk_id,chunk_type,node_id,from_node_id,to_node_id,
  routes,num_frames,map_x_px,map_y_px,graph_length_px`. Use `crossing_node`
  or `straight_edge` as the chunk type. The map coordinates give the pivot.
- `chunk_items.csv`: `chunk_id,route,item_order,map_x_px,map_y_px` describes
  frame-anchor positions for trajectory-based local support.
- `registration_metrics.json`: an object with a `chunks` list. Each entry
  has `chunk_id` and `sim3` with `scale`, a 3x3 `rotation`, and a length-3
  `translation`. The transform maps local road points to map coordinates.

The full static cloud must already have the corresponding initialization
applied. If an additional junction initialization is used, its accepted delta
must also be recorded in `junction_refine`; otherwise omit that field.

```bash
python tools/refine_map_unit_bev_to_2d_map.py \
  --map inputs/map.png --road-mask inputs/road_mask.png \
  --road-skeleton inputs/road_skeleton.png --graph-root inputs/graph \
  --chunk-root inputs/chunks --base-recon outputs/initialized \
  --full-recon outputs/static --out outputs/refined \
  --target-interval graph --freeze-straight
```

This entry point exposes the corridor, skeleton, and hybrid candidate results
in `chunk_refine_metrics.json`. Your adapter applies its acceptance checks and
writes `selected_chunk_summary.csv` with `chunk_id,selected_mode`; use `base`
for identity fallback. This selection is an algorithmic decision, not manual
selection against evaluation ground truth. The export tool then applies each
selected horizontal update to the full-resolution static cloud:

```bash
python tools/export_bev_2d_refined_ply.py \
  --refine-dir outputs/refined --full-recon outputs/static --write-chunks
```

Search ranges are expressed in map pixels and must match your raster scale.
The public interface contains no private experiment presets or data loaders.
