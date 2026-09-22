# Third-Party Notices

The root MIT license applies to original MapPano3D code. It does not relicense
the following projects, their weights, or data.

| Project | Source | Use |
| --- | --- | --- |
| PanoVGGT | https://github.com/YijingGuo-June/PanoVGGT | Native panoramic local geometry |
| VGGT-Long | https://github.com/DengKaiCQ/VGGT-Long | Perspective backbone and long-sequence baseline |
| PanoVGGT-Long adaptation | https://github.com/avalanchezy/PanoVGGT-Long | Panoramic long-sequence baseline |
| Mask2Former | https://github.com/facebookresearch/Mask2Former | Cityscapes semantic masks |

`patches/panovggt-inference.patch` modifies PanoVGGT's MIT-licensed inference
and export code. The original copyright and license are preserved in
`third_party_licenses/PanoVGGT.txt`.

`patches/vggt-long-export.patch` and `patches/panovggt-long-export.patch`
contain changes to the respective upstream repositories. The licenses supplied
with those snapshots are retained in `third_party_licenses/VGGT.txt` and
`third_party_licenses/PanoVGGT-Long.txt`. These patches and the upstream code
remain subject to those terms, not the root MIT license. Install checkpoints
from their original providers and observe their separate conditions.

MovieMap videos and metadata, including the Hokkaido and Kanazawa route,
trajectory, calibration, map, and frame-association files, are not redistributed.
Users must create their own 360-degree video and corresponding inputs for the
panoramic interface. The framework illustration, including example data views,
is supplied only for documenting this research project, not as a dataset.
nuScenes data and external map layers must be obtained separately under their
providers' terms. The MIT license covers code, not dataset imagery or metadata.
