import gzip
import json
from pathlib import Path

import numpy as np
from PIL import Image


DEMO = Path(__file__).resolve().parents[2] / "docs" / "demo"


def test_demo_contains_only_display_data():
    scene = json.loads((DEMO / "scene.json").read_text())
    assert set(scene) == {"title", "positions", "colors", "count", "frames", "panoramas", "map"}
    assert scene["title"] == "MapPano3D"
    assert len(scene["frames"]) > 100
    for frame in scene["frames"]:
        assert set(frame) == {"position", "forward"}
        assert len(frame["position"]) == len(frame["forward"]) == 3
    for path in [scene["positions"], scene["colors"], scene["map"]["image"], *scene["panoramas"]["sheets"]]:
        assert Path(path).name == path
        assert (DEMO / path).is_file()


def test_point_cloud_and_panorama_dimensions():
    scene = json.loads((DEMO / "scene.json").read_text())
    xyz = np.frombuffer(gzip.decompress((DEMO / scene["positions"]).read_bytes()), dtype="<f4")
    rgb = gzip.decompress((DEMO / scene["colors"]).read_bytes())
    assert len(xyz) == 3 * scene["count"] == len(rgb)
    assert np.isfinite(xyz).all()
    atlas = scene["panoramas"]
    for sheet in atlas["sheets"]:
        with Image.open(DEMO / sheet) as image:
            assert image.size == (atlas["width"] * atlas["columns"], atlas["height"] * atlas["rows"])


def test_standalone_entrypoint():
    html = (DEMO / "index.html").read_text()
    assert 'id="cloud"' in html
    assert 'id="play"' in html
    assert 'id="timeline"' in html
    assert (DEMO / "viewer.js").is_file()
    assert (DEMO / "three.module.min.js").is_file()
