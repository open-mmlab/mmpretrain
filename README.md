<div align="center">
  <img src="resources/mmcls-logo.png" width="600"/>
</div>

[![Build Status](https://github.com/open-mmlab/mmclassification/workflows/build/badge.svg)](https://github.com/open-mmlab/mmclassification/actions)
[![Documentation Status](https://readthedocs.org/projects/mmclassification/badge/?version=latest)](https://mmclassification.readthedocs.io/en/latest/?badge=latest)
[![codecov](https://codecov.io/gh/open-mmlab/mmclassification/branch/master/graph/badge.svg)](https://codecov.io/gh/open-mmlab/mmclassification)
[![license](https://img.shields.io/github/license/open-mmlab/mmclassification.svg)](https://github.com/open-mmlab/mmclassification/blob/master/LICENSE)

## Introduction

English | [简体中文](/README_zh-CN.md)

MMClassification is an open source image classification toolbox based on PyTorch. It is
a part of the [OpenMMLab](https://openmmlab.com/) project.

Documentation: https://mmclassification.readthedocs.io/en/latest/

![demo](https://user-images.githubusercontent.com/9102141/87268895-3e0d0780-c4fe-11ea-849e-6140b7e0d4de.gif)

### Major features

- Various backbones and pretrained models
- Bag of training tricks
- Large-scale training configs
- High efficiency and extensibility

## License

This project is released under the [Apache 2.0 license](LICENSE).

## Changelog

v0.15.0 was released in 31/8/2021.
Please refer to [changelog.md](docs/changelog.md) for details and release history.

## Benchmark and model zoo

Results and models are available in the [model zoo](docs/model_zoo.md).

Supported backbones:

- [x] ResNet
- [x] ResNeXt
- [x] SE-ResNet
- [x] SE-ResNeXt
- [x] RegNet
- [x] ShuffleNetV1
- [x] ShuffleNetV2
- [x] MobileNetV2
- [x] MobileNetV3
- [x] Swin-Transformer

## Installation

Please refer to [install.md](docs/install.md) for installation and dataset preparation.

## Getting Started

Please see [getting_started.md](docs/getting_started.md) for the basic usage of MMClassification. There are also tutorials for [finetuning models](docs/tutorials/finetune.md), [adding new dataset](docs/tutorials/new_dataset.md), [designing data pipeline](docs/tutorials/data_pipeline.md), and [adding new modules](docs/tutorials/new_modules.md).

## Citation

If you find this project useful in your research, please consider cite:

```BibTeX
@misc{2020mmclassification,
    title={OpenMMLab's Image Classification Toolbox and Benchmark},
    author={MMClassification Contributors},
    howpublished = {\url{https://github.com/open-mmlab/mmclassification}},
    year={2020}
}
```

## Contributing

We appreciate all contributions to improve MMClassification.
Please refer to [CONTRUBUTING.md](.github/CONTRIBUTING.md) for the contributing guideline.

## Acknowledgement

MMClassification is an open source project that is contributed by researchers and engineers from various colleges and companies. We appreciate all the contributors who implement their methods or add new features, as well as users who give valuable feedbacks.
We wish that the toolbox and benchmark could serve the growing research community by providing a flexible toolkit to reimplement existing methods and develop their own new classifiers.

## Projects in OpenMMLab

- [MMCV](https://github.com/open-mmlab/mmcv): OpenMMLab foundational library for computer vision.
- [MIM](https://github.com/open-mmlab/mim): MIM Installs OpenMMLab Packages.
- [MMClassification](https://github.com/open-mmlab/mmclassification): OpenMMLab image classification toolbox and benchmark.
- [MMDetection](https://github.com/open-mmlab/mmdetection): OpenMMLab detection toolbox and benchmark.
- [MMDetection3D](https://github.com/open-mmlab/mmdetection3d): OpenMMLab's next-generation platform for general 3D object detection.
- [MMSegmentation](https://github.com/open-mmlab/mmsegmentation): OpenMMLab semantic segmentation toolbox and benchmark.
- [MMAction2](https://github.com/open-mmlab/mmaction2): OpenMMLab's next-generation action understanding toolbox and benchmark.
- [MMTracking](https://github.com/open-mmlab/mmtracking): OpenMMLab video perception toolbox and benchmark.
- [MMPose](https://github.com/open-mmlab/mmpose): OpenMMLab pose estimation toolbox and benchmark.
- [MMEditing](https://github.com/open-mmlab/mmediting): OpenMMLab image and video editing toolbox.
- [MMOCR](https://github.com/open-mmlab/mmocr): OpenMMLab toolbox for text detection, recognition and understanding.
- [MMGeneration](https://github.com/open-mmlab/mmgeneration): OpenMMlab toolkit for generative models.
