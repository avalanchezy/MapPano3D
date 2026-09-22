# Custom Panoramic Input Specification

Connect your own 360-degree recordings to MapPano3D by supplying local
geometry, camera anchors, and road-map support. The specifications below
describe the files consumed by the panoramic registration and fusion tools.

## Geometry

Use `PanoVGGTAdapter` or another predictor implementing the
[shared geometry contract](backbones.md). Local point maps, camera-to-world
poses, semantic labels, and frame IDs must agree. Retain road evidence and
full static geometry separately. Camera anchors can come from user-supplied
SLAM or route placement and must share the road map's frame.

## Registered Map Interface

The generic panoramic tools use X/Z horizontally and Y for height.
Horizontal point coordinates, anchors, and graph nodes use map pixels.
Metric nuScenes inputs use the separate XY/Z convention.

| Input | Required contents |
| --- | --- |
| Map image | Working raster grid |
| Road mask | Binary road raster on the same grid |
| Road skeleton | Thinned centerline on that grid |
| Graph directory | `nodes.csv` and `edges.csv` |
| Chunk directory | `chunk_manifest.csv`; optionally `chunk_items.csv` |
| Initialized road geometry | `registration_metrics.json` and `road_clouds/CHUNK_ID.ply` |
| Full static geometry | `registered_chunks/CHUNK_ID_map.ply` |

- `nodes.csv`: `node_id,x_px,y_px`.
- `edges.csv`: `from_node,to_node`.
- `chunk_manifest.csv`: `chunk_id,chunk_type,node_id,from_node_id,to_node_id,
  routes,num_frames,map_x_px,map_y_px,graph_length_px`. Types are
  `crossing_node` and `straight_edge`; map coordinates specify the pivot.
- `chunk_items.csv`: `chunk_id,route,item_order,map_x_px,map_y_px` associates
  frame-anchor positions with trajectory-based local support.
- `registration_metrics.json`: a `chunks` list with `chunk_id` and `sim3`
  containing `scale`, a 3x3 `rotation`, and length-3 `translation`.
  This initialization maps local road points to map coordinates.

Full static clouds already have the corresponding initialization applied.
An additional accepted junction delta is recorded in `junction_refine`.
The adapter's acceptance rule produces `selected_chunk_summary.csv` with
`chunk_id,selected_mode`; `base` denotes identity fallback. Selection uses
map/anchor evidence, not evaluation ground truth. `export_bev_2d_refined_ply.py`
applies selected horizontal updates to the full static cloud.

For a complete image-to-reconstruction example, follow the
[VGGT + nuScenes guide](reproduction.md).
