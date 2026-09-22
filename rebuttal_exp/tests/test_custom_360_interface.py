from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from refine_map_unit_bev_to_2d_map import read_chunk_records, read_graph, transform_xz


def test_user_graph_and_chunks(tmp_path):
    (tmp_path / "nodes.csv").write_text("node_id,x_px,y_px\na,10,20\nb,30,20\n")
    (tmp_path / "edges.csv").write_text("from_node,to_node\na,b\n")
    manifest = tmp_path / "chunk_manifest.csv"
    manifest.write_text(
        "chunk_id,chunk_type,node_id,num_frames,map_x_px,map_y_px\n"
        "my_chunk,crossing_node,a,40,10,20\n"
    )
    nodes, edges = read_graph(tmp_path)
    chunk = read_chunk_records(manifest)[0]
    assert nodes == {"a": (10.0, 20.0), "b": (30.0, 20.0)}
    assert edges == [{"from_node": "a", "to_node": "b"}]
    assert (chunk.chunk_id, chunk.num_frames, chunk.map_x) == ("my_chunk", 40, 10.0)


def test_horizontal_update_preserves_height():
    points = np.array([[1.0, 4.0, 2.0], [3.0, 8.0, 5.0]])
    updated = transform_xz(points, 5.0, 1.0, -2.0, 1.02, (0.0, 0.0))
    np.testing.assert_array_equal(updated[:, 1], points[:, 1])
