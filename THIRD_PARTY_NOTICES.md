# Third-Party Notices

The root MIT license applies to original MapPano3D code. It does not relicense
the following projects, their weights, or data.

| Project | Source | Use |
| --- | --- | --- |
| PanoVGGT | https://github.com/YijingGuo-June/PanoVGGT | Native panoramic local geometry |
| VGGT-Long | https://github.com/DengKaiCQ/VGGT-Long | Perspective backbone and long-sequence baseline |
| PanoVGGT-Long adaptation | https://github.com/avalanchezy/PanoVGGT-Long | Panoramic long-sequence baseline |
| Mask2Former | https://github.com/facebookresearch/Mask2Former | Cityscapes semantic masks |
| Three.js | https://github.com/mrdoob/three.js | Interactive demo renderer and orbit controls (MIT) |
| Lucide | https://lucide.dev/ | Interactive demo control icons (ISC) |

`patches/panovggt-inference.patch` modifies PanoVGGT's MIT-licensed inference
and export code. The original copyright and license are preserved in
`third_party_licenses/PanoVGGT.txt`.

`patches/vggt-long-export.patch` and `patches/panovggt-long-export.patch`
contain changes to the respective upstream repositories. The licenses supplied
with those snapshots are retained in `third_party_licenses/VGGT.txt` and
`third_party_licenses/PanoVGGT-Long.txt`. These patches and the upstream code
remain subject to those terms, not the root MIT license. Install checkpoints
from their original providers and observe their separate conditions.

Three.js and Lucide license notices are included alongside the vendored
demo libraries in `docs/demo/three-LICENSE.txt` and `docs/demo/lucide-LICENSE.txt`.

Paper illustrations and demonstration assets contain research-result views
and dataset imagery. These visual assets retain their source attribution
and respective rights; the root MIT license applies to code. nuScenes data
and external map layers are obtained from their providers under their terms.
