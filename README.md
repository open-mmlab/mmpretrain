<div align="center">
  <img src="resources/mmcls-logo.png" width="600"/>
</div>

[![Build Status](https://github.com/open-mmlab/mmclassification/workflows/build/badge.svg)](https://github.com/open-mmlab/mmclassification/actions)
[![Documentation Status](https://readthedocs.org/projects/mmclassification/badge/?version=latest)](https://mmclassification.readthedocs.io/en/latest/?badge=latest)
[![codecov](https://codecov.io/gh/open-mmlab/mmclassification/branch/master/graph/badge.svg)](https://codecov.io/gh/open-mmlab/mmclassification)
[![license](https://img.shields.io/github/license/open-mmlab/mmclassification.svg)](https://github.com/open-mmlab/mmclassification/blob/master/LICENSE)
[![PyPI](https://badge.fury.io/py/mmcls.svg)](https://pypi.org/project/mmcls/)
[![Percentage of issues still open](https://isitmaintained.com/badge/open/open-mmlab/mmclassification.svg)](https://github.com/open-mmlab/mmclassification/issues)

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
- Powerful toolkits

## License

This project is released under the [Apache 2.0 license](LICENSE).

## Changelog

v0.19.0 was released in 31/12/2021.

Highlights of the new version:
- The **feature extraction** function has been enhanced. See [#593](https://github.com/open-mmlab/mmclassification/pull/593) for more details.
- Provide the high-acc **ResNet-50** training settings from [*ResNet strikes back*](https://arxiv.org/abs/2110.00476).
- Reproduce the training accuracy of **T2T-ViT** & **RegNetX**, and provide self-training checkpoints.
- Support **DeiT** & **Conformer** backbone and checkpoints.
- Provide a **CAM visualization** tool based on [pytorch-grad-cam](https://github.com/jacobgil/pytorch-grad-cam), and detailed [user guide](https://mmclassification.readthedocs.io/en/latest/tools/visualization.html#class-activation-map-visualization)!

v0.18.0 was released in 30/11/2021.

Highlights of the new version:
- Support **MLP-Mixer** backbone and provide pre-trained checkpoints.
- Add a tool to **visualize the learning rate curve** of the training phase. Welcome to use with the [tutorial](https://mmclassification.readthedocs.io/en/latest/tools/visualization.html#learning-rate-schedule-visualization)!

Please refer to [changelog.md](docs/en/changelog.md) for more details and other release history.

## Benchmark and model zoo

Results and models are available in the [model zoo](https://mmclassification.readthedocs.io/en/latest/model_zoo.html).

<details open>
<summary>Supported backbones</summary>

- [x] [VGG](https://github.com/open-mmlab/mmclassification/tree/master/configs/vgg)
- [x] [ResNet](https://github.com/open-mmlab/mmclassification/tree/master/configs/resnet)
- [x] [ResNeXt](https://github.com/open-mmlab/mmclassification/tree/master/configs/resnext)
- [x] [SE-ResNet](https://github.com/open-mmlab/mmclassification/tree/master/configs/seresnet)
- [x] [SE-ResNeXt](https://github.com/open-mmlab/mmclassification/tree/master/configs/seresnet)
- [x] [RegNet](https://github.com/open-mmlab/mmclassification/tree/master/configs/repvgg)
- [x] [ShuffleNetV1](https://github.com/open-mmlab/mmclassification/tree/master/configs/shufflenet_v1)
- [x] [ShuffleNetV2](https://github.com/open-mmlab/mmclassification/tree/master/configs/shufflenet_v2)
- [x] [MobileNetV2](https://github.com/open-mmlab/mmclassification/tree/master/configs/mobilenet_v2)
- [x] [MobileNetV3](https://github.com/open-mmlab/mmclassification/tree/master/configs/mobilenet_v3)
- [x] [Swin-Transformer](https://github.com/open-mmlab/mmclassification/tree/master/configs/swin_transformer)
- [x] [RepVGG](https://github.com/open-mmlab/mmclassification/tree/master/configs/repvgg)
- [x] [Vision-Transformer](https://github.com/open-mmlab/mmclassification/tree/master/configs/vision_transformer)
- [x] [Transformer-in-Transformer](https://github.com/open-mmlab/mmclassification/tree/master/configs/tnt)
- [x] [Res2Net](https://github.com/open-mmlab/mmclassification/tree/master/configs/res2net)
- [x] [MLP-Mixer](https://github.com/open-mmlab/mmclassification/tree/master/configs/mlp_mixer)
- [x] [DeiT](https://github.com/open-mmlab/mmclassification/tree/master/configs/deit)
- [x] [Conformer](https://github.com/open-mmlab/mmclassification/tree/master/configs/conformer)
- [x] [T2T-ViT](https://github.com/open-mmlab/mmclassification/tree/master/configs/t2t_vit)
- [x] [Twins](https://github.com/open-mmlab/mmclassification/tree/master/configs/twins)
- [x] [EfficientNet](https://github.com/open-mmlab/mmclassification/tree/master/configs/efficientnet)
- [x] [ConvNeXt](https://github.com/open-mmlab/mmclassification/tree/master/configs/convnext)
- [x] [HRNet](https://github.com/open-mmlab/mmclassification/tree/master/configs/hrnet)

</details>

## Installation

Please refer to [install.md](https://mmclassification.readthedocs.io/en/latest/install.html) for installation and dataset preparation.

## Getting Started
Please see [Getting Started](https://mmclassification.readthedocs.io/en/latest/getting_started.html) for the basic usage of MMClassification. There are also tutorials:

- [Learn about Configs](https://mmclassification.readthedocs.io/en/latest/tutorials/config.html)
- [Fine-tune Models](https://mmclassification.readthedocs.io/en/latest/tutorials/finetune.html)
- [Add New Dataset](https://mmclassification.readthedocs.io/en/latest/tutorials/new_dataset.html)
- [Customizie Data Pipeline](https://mmclassification.readthedocs.io/en/latest/tutorials/data_pipeline.html)
- [Add New Modules](https://mmclassification.readthedocs.io/en/latest/tutorials/new_modules.html)
- [Customizie Schedule](https://mmclassification.readthedocs.io/en/latest/tutorials/schedule.html)
- [Customizie Runtime Settings](https://mmclassification.readthedocs.io/en/latest/tutorials/runtime.html)

Colab tutorials are also provided:

- Learn about MMClassification **Python API**: [Preview the notebook](https://github.com/open-mmlab/mmclassification/blob/master/docs/en/tutorials/MMClassification_python.ipynb) or directly [run on Colab](https://colab.research.google.com/github/open-mmlab/mmclassification/blob/master/docs/en/tutorials/MMClassification_python.ipynb).
- Learn about MMClassification **CLI tools**: [Preview the notebook](https://github.com/open-mmlab/mmclassification/blob/master/docs/en/tutorials/MMClassification_tools.ipynb) or directly [run on Colab](https://colab.research.google.com/github/open-mmlab/mmclassification/blob/master/docs/en/tutorials/MMClassification_tools.ipynb).

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
Please refer to [CONTRUBUTING.md](https://mmclassification.readthedocs.io/en/latest/community/CONTRIBUTING.html) for the contributing guideline.

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
- [MMFlow](https://github.com/open-mmlab/mmflow) OpenMMLab optical flow toolbox and benchmark.
- [MMFewShot](https://github.com/open-mmlab/mmfewshot): OpenMMLab FewShot Learning Toolbox and Benchmark.
- [MMHuman3D](https://github.com/open-mmlab/mmhuman3d): OpenMMLab 3D Human Parametric Model Toolbox and Benchmark.
- [MMSelfSup](https://github.com/open-mmlab/mmselfsup): OpenMMLab self-supervised learning toolbox and benchmark.
- [MMRazor](https://github.com/open-mmlab/mmrazor): OpenMMLab Model Compression Toolbox and Benchmark.
- [MMDeploy](https://github.com/open-mmlab/mmdeploy): OpenMMLab Model Deployment Framework.
