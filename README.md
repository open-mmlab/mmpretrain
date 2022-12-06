<div align="center">

<img src="resources/mmcls-logo.png" width="600"/>
  <div>&nbsp;</div>
  <div align="center">
    <b><font size="5">OpenMMLab website</font></b>
    <sup>
      <a href="https://openmmlab.com">
        <i><font size="4">HOT</font></i>
      </a>
    </sup>
    &nbsp;&nbsp;&nbsp;&nbsp;
    <b><font size="5">OpenMMLab platform</font></b>
    <sup>
      <a href="https://platform.openmmlab.com">
        <i><font size="4">TRY IT OUT</font></i>
      </a>
    </sup>
  </div>
  <div>&nbsp;</div>

[![PyPI](https://img.shields.io/pypi/v/mmcls)](https://pypi.org/project/mmcls)
[![Docs](https://img.shields.io/badge/docs-latest-blue)](https://mmclassification.readthedocs.io/en/latest/)
[![Build Status](https://github.com/open-mmlab/mmclassification/workflows/build/badge.svg)](https://github.com/open-mmlab/mmclassification/actions)
[![codecov](https://codecov.io/gh/open-mmlab/mmclassification/branch/master/graph/badge.svg)](https://codecov.io/gh/open-mmlab/mmclassification)
[![license](https://img.shields.io/github/license/open-mmlab/mmclassification.svg)](https://github.com/open-mmlab/mmclassification/blob/master/LICENSE)
[![open issues](https://isitmaintained.com/badge/open/open-mmlab/mmclassification.svg)](https://github.com/open-mmlab/mmclassification/issues)
[![issue resolution](https://isitmaintained.com/badge/resolution/open-mmlab/mmclassification.svg)](https://github.com/open-mmlab/mmclassification/issues)

[📘 Documentation](https://mmclassification.readthedocs.io/en/latest/) |
[🛠️ Installation](https://mmclassification.readthedocs.io/en/latest/install.html) |
[👀 Model Zoo](https://mmclassification.readthedocs.io/en/latest/model_zoo.html) |
[🆕 Update News](https://mmclassification.readthedocs.io/en/latest/changelog.html) |
[🤔 Reporting Issues](https://github.com/open-mmlab/mmclassification/issues/new/choose)

:point_right: **MMClassification 1.0 branch is in trial, welcome every to [try it](https://github.com/open-mmlab/mmclassification/tree/1.x) and [discuss with us](https://github.com/open-mmlab/mmclassification/discussions)!** :point_left:

</div>

## Introduction

English | [简体中文](/README_zh-CN.md)

MMClassification is an open source image classification toolbox based on PyTorch. It is
a part of the [OpenMMLab](https://openmmlab.com/) project.

The master branch works with **PyTorch 1.5+**.

<div align="center">
  <img src="https://user-images.githubusercontent.com/9102141/87268895-3e0d0780-c4fe-11ea-849e-6140b7e0d4de.gif" width="70%"/>
</div>

### Major features

- Various backbones and pretrained models
- Bag of training tricks
- Large-scale training configs
- High efficiency and extensibility
- Powerful toolkits

## What's new

The MMClassification 1.0 has released! It's still unstable and in release candidate. If you want to try it, go
to [the 1.x branch](https://github.com/open-mmlab/mmclassification/tree/1.x) and discuss it with us in
[the discussion](https://github.com/open-mmlab/mmclassification/discussions).

v0.25.0 was released in 06/12/2022.
Highlights of the new version:

- Support MLU backend.
- Add `dist_train_arm.sh` for ARM device.

v0.24.1 was released in 31/10/2022.
Highlights of the new version:

- Support HUAWEI Ascend device.

v0.24.0 was released in 30/9/2022.
Highlights of the new version:

- Support **HorNet**, **EfficientFormerm**, **SwinTransformer V2** and **MViT** backbones.
- Support Standford Cars dataset.

Please refer to [changelog.md](docs/en/changelog.md) for more details and other release history.

## Installation

Below are quick steps for installation:

```shell
conda create -n open-mmlab python=3.8 pytorch=1.10 cudatoolkit=11.3 torchvision==0.11.0 -c pytorch -y
conda activate open-mmlab
pip3 install openmim
mim install mmcv-full
git clone https://github.com/open-mmlab/mmclassification.git
cd mmclassification
pip3 install -e .
```

Please refer to [install.md](https://mmclassification.readthedocs.io/en/latest/install.html) for more detailed installation and dataset preparation.

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

## Model zoo

Results and models are available in the [model zoo](https://mmclassification.readthedocs.io/en/latest/model_zoo.html).

<details open>
<summary>Supported backbones</summary>

- [x] [VGG](https://github.com/open-mmlab/mmclassification/tree/master/configs/vgg)
- [x] [ResNet](https://github.com/open-mmlab/mmclassification/tree/master/configs/resnet)
- [x] [ResNeXt](https://github.com/open-mmlab/mmclassification/tree/master/configs/resnext)
- [x] [SE-ResNet](https://github.com/open-mmlab/mmclassification/tree/master/configs/seresnet)
- [x] [SE-ResNeXt](https://github.com/open-mmlab/mmclassification/tree/master/configs/seresnet)
- [x] [RegNet](https://github.com/open-mmlab/mmclassification/tree/master/configs/regnet)
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
- [x] [VAN](https://github.com/open-mmlab/mmclassification/tree/master/configs/van)
- [x] [ConvMixer](https://github.com/open-mmlab/mmclassification/tree/master/configs/convmixer)
- [x] [CSPNet](https://github.com/open-mmlab/mmclassification/tree/master/configs/cspnet)
- [x] [PoolFormer](https://github.com/open-mmlab/mmclassification/tree/master/configs/poolformer)
- [x] [MViT](https://github.com/open-mmlab/mmclassification/tree/master/configs/mvit)
- [x] [EfficientFormer](https://github.com/open-mmlab/mmclassification/tree/master/configs/efficientformer)
- [x] [HorNet](https://github.com/open-mmlab/mmclassification/tree/master/configs/hornet)

</details>

## Contributing

We appreciate all contributions to improve MMClassification.
Please refer to [CONTRUBUTING.md](https://mmclassification.readthedocs.io/en/latest/community/CONTRIBUTING.html) for the contributing guideline.

## Acknowledgement

MMClassification is an open source project that is contributed by researchers and engineers from various colleges and companies. We appreciate all the contributors who implement their methods or add new features, as well as users who give valuable feedbacks.
We wish that the toolbox and benchmark could serve the growing research community by providing a flexible toolkit to reimplement existing methods and develop their own new classifiers.

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

## License

This project is released under the [Apache 2.0 license](LICENSE).

## Projects in OpenMMLab

- [MMCV](https://github.com/open-mmlab/mmcv): OpenMMLab foundational library for computer vision.
- [MIM](https://github.com/open-mmlab/mim): MIM installs OpenMMLab packages.
- [MMClassification](https://github.com/open-mmlab/mmclassification): OpenMMLab image classification toolbox and benchmark.
- [MMDetection](https://github.com/open-mmlab/mmdetection): OpenMMLab detection toolbox and benchmark.
- [MMDetection3D](https://github.com/open-mmlab/mmdetection3d): OpenMMLab's next-generation platform for general 3D object detection.
- [MMRotate](https://github.com/open-mmlab/mmrotate): OpenMMLab rotated object detection toolbox and benchmark.
- [MMSegmentation](https://github.com/open-mmlab/mmsegmentation): OpenMMLab semantic segmentation toolbox and benchmark.
- [MMOCR](https://github.com/open-mmlab/mmocr): OpenMMLab text detection, recognition, and understanding toolbox.
- [MMPose](https://github.com/open-mmlab/mmpose): OpenMMLab pose estimation toolbox and benchmark.
- [MMHuman3D](https://github.com/open-mmlab/mmhuman3d): OpenMMLab 3D human parametric model toolbox and benchmark.
- [MMSelfSup](https://github.com/open-mmlab/mmselfsup): OpenMMLab self-supervised learning toolbox and benchmark.
- [MMRazor](https://github.com/open-mmlab/mmrazor): OpenMMLab model compression toolbox and benchmark.
- [MMFewShot](https://github.com/open-mmlab/mmfewshot): OpenMMLab fewshot learning toolbox and benchmark.
- [MMAction2](https://github.com/open-mmlab/mmaction2): OpenMMLab's next-generation action understanding toolbox and benchmark.
- [MMTracking](https://github.com/open-mmlab/mmtracking): OpenMMLab video perception toolbox and benchmark.
- [MMFlow](https://github.com/open-mmlab/mmflow): OpenMMLab optical flow toolbox and benchmark.
- [MMEditing](https://github.com/open-mmlab/mmediting): OpenMMLab image and video editing toolbox.
- [MMGeneration](https://github.com/open-mmlab/mmgeneration): OpenMMLab image and video generative models toolbox.
- [MMDeploy](https://github.com/open-mmlab/mmdeploy): OpenMMLab model deployment framework.
