# Data sources and attribution

I keep image datasets, pretrained weights, and generated reconstructions outside Git. The repository's [rights notice](../LICENSE) does not replace upstream terms.

## ETH3D

Source: [ETH3D datasets](https://www.eth3d.net/datasets).

I use the high-resolution undistorted DSLR multi-view scenes `pipes`, `terrace`, `courtyard`, `relief`, and `relief_2`. The runner expects:

```text
data/eth3d/<scene>/
├── images/dslr_images_undistorted/*.JPG
└── dslr_calibration_undistorted/images.txt
```

Image basenames must match the calibration entries. The parser preserves the two-line COLMAP record structure, including empty observation lines. `relief` and `relief_2` depict the same physical interior but retain independent reference frames.

The main tables use the earlier record's metric ETH3D interpretation. Retaining the original calibration and input manifest is necessary to audit each future run.

## Mill 19 / Mega-NeRF

Source: [Mega-NeRF project and Mill 19 downloads](https://meganerf.cmusatyalab.org/). I use the `building-pixsfm` / `rubble-pixsfm` layout from my selected data preparation; archive layouts or revised distributions may differ.

```text
data/mill19/
├── building-pixsfm/train/
│   ├── rgbs/<frame>.jpg
│   └── metadata/<frame>.pt
└── rubble-pixsfm/train/
    ├── rgbs/<frame>.jpg
    └── metadata/<frame>.pt
```

Each tensor metadata file must contain a finite rigid 3×4 `c2w` matrix. I load tensor metadata with restricted `torch.load(..., weights_only=True)`; arbitrary pickled objects are not required.

I select consecutive frames from `train` to preserve overlap. A sparse novel-view-synthesis validation split is not interchangeable with an overlapping SfM sequence. The metadata poses are dataset-supplied references. They should not be described as independently surveyed truth without verifying their generation procedure.

The retained historical summary has no conversion from normalized coordinates to metres. I report `u` and keep the baseline-normalized interpretation separate from ETH3D metric errors.

## Models and software

| Resource | Role | Source |
| :--- | :--- | :--- |
| DINOv2 | Global descriptors for candidate retrieval | [Official repository](https://github.com/facebookresearch/dinov2), model identifier `facebook/dinov2-base` |
| ALIKED | Learned local features | [Official repository](https://github.com/Shiaoming/ALIKED); adapter from LightGlue |
| LightGlue | Learned local-feature matching | [Official repository](https://github.com/cvg/LightGlue) |
| COLMAP / pycolmap | SIFT baseline, geometric verification, incremental SfM | [Documentation](https://colmap.github.io/), [repository](https://github.com/colmap/colmap) |
| Mega-NeRF | Mill 19 dataset/reference metadata ecosystem | [Project](https://meganerf.cmusatyalab.org/) |

Check each upstream resource's code, model, and dataset terms at the version used. Those terms may differ even within one project. I do not vendor their weights or datasets in this repository.

## Local source material

I reviewed the earlier local implementation, SfM teaching guide, and project notes when reorganizing this study. The teaching diagrams use synthetic examples and are not experimental evidence. A separate Colab connection-tool checkout and machine-specific configuration are not part of this research repository. Manuscripts and their build outputs remain local and are excluded from commits.

The earlier pipeline header refers to a Kaggle IMC baseline reproduction without identifying the original notebook URL, author, or license. That lineage needs resolution before I make an independent-authorship claim or redistribute any unverified borrowed material. My rights notice preserves third-party terms and earlier MIT grants; it cannot settle missing provenance.
