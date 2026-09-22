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

The code uses a shared geometry contract with **VGGT, Pi3X, and PanoVGGT
integration interfaces**. Registration consumes point maps, camera poses,
semantic associations, and coordinate conventions rather than a specific
neural architecture; see [the backbone interface](docs/backbones.md).

The complete reproduction workflow is **VGGT + nuScenes-mini**, including
all ten scenes and independent LiDAR evaluation. Pi3X and PanoVGGT expose
extension hooks and input specifications, not additional turnkey tutorials.
The paper's backbone experiments use VGGT and PanoVGGT; the Pi3X hook is an
integration interface, not a reported MapPano3D-Pi3X experiment.

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

**[VGGT + nuScenes reproduction](docs/reproduction.md)** is the single
end-to-end guide: data preparation, semantic masks, VGGT predictions,
map refinement, full-static fusion, and all-scene geometry evaluation.
For other predictors or your own 360-degree recordings, use the
[backbone contract](docs/backbones.md) and
[custom input specification](docs/custom_360_inputs.md).

## Code Layout

| Directory | Contents |
| --- | --- |
| `mappano3d/` | Shared geometry contract and VGGT/Pi3X/PanoVGGT integration hooks |
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
The nuScenes comparison uses one shared upright alignment from the
pre-refinement VGGT camera trajectory to the dataset's ground-truth camera
trajectory. Both methods reuse the same predictions and static filtering;
there is no method-specific global re-fitting. Mini evaluation informed the
development of acceptance settings. Map-fit and LiDAR metrics measure
different properties and are reported separately.

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
