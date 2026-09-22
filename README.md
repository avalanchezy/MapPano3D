# MapPano3D

**Map-Guided Large-Scale Urban Reconstruction from Panoramic Videos**  
ACCV 2026

Yi Zhu, Kiyoharu Aizawa, Sebastien Valette, Satoshi Ikehata

MapPano3D is a **training-free, backbone-agnostic** framework for assembling
feed-forward local geometry into urban-scale reconstructions. Camera anchors
initialize placement, road topology selects local map support, and semantic
road evidence drives upright BEV registration. The accepted transform is
applied to the **full static scene**, not only to road points.

![MapPano3D framework](assets/pipeline.png)

The release provides a user-supplied panoramic geometry interface and
multi-camera perspective reconstruction with VGGT. Other
feed-forward predictors can be connected through their point maps, camera
poses, pixel correspondences, and coordinate conventions; see
[the backbone interface](docs/backbones.md).

## Getting Started

Use Python 3.10 or newer. Install the geometry and evaluation dependencies:

```bash
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt
python -m pytest
```

Backbone inference additionally requires the upstream model environment,
pretrained weights, and a suitable GPU. Mask2Former inference requires its
Detectron2 environment. Install those dependencies using their upstream
instructions; geometry post-processing can run on CPU.

**[Reproduction instructions](docs/reproduction.md)** cover the matched
six-camera nuScenes comparison, component ablations, and anchor perturbations.
For panoramic footage, record your own 360-degree video and follow the
**[custom-video input interface](docs/custom_360_inputs.md)**.

## Code Layout

| Directory | Contents |
| --- | --- |
| `tools/` | Semantic masks, road-mask extraction, generic BEV refinement, full-cloud export, map evaluation |
| `rebuttal_exp/scripts/` | VGGT adapter, nuScenes LiDAR evaluation and anchor perturbations |
| `rebuttal_exp/configs/` | Experiment configurations, including the 40-image/20-overlap VGGT setting |
| `patches/` | Export/inference changes for the pinned upstream backbone repositories |

MovieMap videos and metadata are not distributed. In particular, the release
does not include route manifests, trajectories, frame-to-map associations,
calibration, map graphs/masks, or private experiment configurations for the
Hokkaido and Kanazawa areas. The paper illustration is retained for presentation.
Create your own 360-degree recordings and associated map/anchor inputs; the
public interface does not download or reconstruct the private dataset metadata.
Model weights, reconstruction caches, and review material are also excluded.

## Evaluation

- **MovieMap (Hokkaido and Kanazawa):** the paper evaluates panoramic urban
  reconstruction in two areas. This release exposes the input interface for
  user-created video data, not those areas' metadata or experiment loaders.
- **nuScenes-mini:** all ten scenes; matched six-camera inputs and 40 total
  camera images per chunk, with 20-image overlap; all-static and non-ground
  LiDAR accuracy, completeness, Chamfer-L1, precision, recall, and F1.

LiDAR is an evaluation reference, not an input to local BEV optimization.
The nuScenes comparison shares camera-pose anchors and local VGGT predictions
between VGGT-Long and MapPano3D. Map-fit and LiDAR metrics measure different
properties and are reported separately.

## Citation

```bibtex
@inproceedings{zhu2026mappano3d,
  title={MapPano3D: Map-Guided Large-Scale Urban Reconstruction from Panoramic Videos},
  author={Zhu, Yi and Aizawa, Kiyoharu and Valette, Sebastien and Ikehata, Satoshi},
  booktitle={Asian Conference on Computer Vision (ACCV)},
  year={2026}
}
```

## License and Contact

Our original code is released under the [MIT License](LICENSE).
Upstream models, weights, datasets, and derived patches retain their respective
terms; see [third-party notices](THIRD_PARTY_NOTICES.md).

Corresponding author: Yi Zhu, `yi.zhu@creatis.insa-lyon.fr`.
