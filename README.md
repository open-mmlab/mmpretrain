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
[![Docs](https://img.shields.io/badge/docs-latest-blue)](https://mmclassification.readthedocs.io/en/1.x/)
[![Build Status](https://github.com/open-mmlab/mmclassification/workflows/build/badge.svg)](https://github.com/open-mmlab/mmclassification/actions)
[![codecov](https://codecov.io/gh/open-mmlab/mmclassification/branch/1.x/graph/badge.svg)](https://codecov.io/gh/open-mmlab/mmclassification)
[![license](https://img.shields.io/github/license/open-mmlab/mmclassification.svg)](https://github.com/open-mmlab/mmclassification/blob/1.x/LICENSE)
[![open issues](https://isitmaintained.com/badge/open/open-mmlab/mmclassification.svg)](https://github.com/open-mmlab/mmclassification/issues)
[![issue resolution](https://isitmaintained.com/badge/resolution/open-mmlab/mmclassification.svg)](https://github.com/open-mmlab/mmclassification/issues)

[📘 Documentation](https://mmclassification.readthedocs.io/en/1.x/) |
[🛠️ Installation](https://mmclassification.readthedocs.io/en/dev-1.x/get_started.html#installation) |
[👀 Model Zoo](https://mmclassification.readthedocs.io/en/1.x/modelzoo_statistics.html) |
[🆕 Update News](https://mmclassification.readthedocs.io/en/1.x/notes/changelog.html) |
[🤔 Reporting Issues](https://github.com/open-mmlab/mmclassification/issues/new/choose)

</div>

## Introduction

English | [简体中文](/README_zh-CN.md)

MMClassification is an open source image classification toolbox based on PyTorch. It is
a part of the [OpenMMLab](https://openmmlab.com/) project.

The `1.x` branch works with **PyTorch 1.6+**.

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

v1.0.0rc4 was released in 06/12/2022.

- Upgrade API to get pre-defined models of MMClassification. See [#1236](https://github.com/open-mmlab/mmclassification/pull/1236) for more details.
- Refactor BEiT backbone and support v1/v2 inference. See [#1144](https://github.com/open-mmlab/mmclassification/pull/1144).

v1.0.0rc3 was released in 21/11/2022.

- Add **Switch Recipe** Hook, Now we can modify training pipeline, mixup and loss settings during training, see [#1101](https://github.com/open-mmlab/mmclassification/pull/1101).
- Add **TIMM and HuggingFace** wrappers. Now you can train/use models in TIMM/HuggingFace directly, see [#1102](https://github.com/open-mmlab/mmclassification/pull/1102).
- Support **retrieval tasks**, see [#1055](https://github.com/open-mmlab/mmclassification/pull/1055).
- Reproduce **mobileone** training accuracy. See [#1191](https://github.com/open-mmlab/mmclassification/pull/1191)

This release introduced a brand new and flexible training & test engine, but it's still in progress. Welcome
to try according to [the documentation](https://mmclassification.readthedocs.io/en/1.x/).

And there are some BC-breaking changes. Please check [the migration tutorial](https://mmclassification.readthedocs.io/en/1.x/migration.html).

The release candidate will last until the end of 2022, and during the release candidate, we will develop on the `1.x` branch. And we will still maintain 0.x version still at least the end of 2023.

Please refer to [changelog.md](https://mmclassification.readthedocs.io/en/1.x/notes/changelog.html) for more details and other release history.

## Installation

Below are quick steps for installation:

```shell
conda create -n open-mmlab python=3.8 pytorch==1.10.1 torchvision==0.11.2 cudatoolkit=11.3 -c pytorch -y
conda activate open-mmlab
pip install openmim
git clone -b 1.x https://github.com/open-mmlab/mmclassification.git
cd mmclassification
mim install -e .
```

Please refer to [install.md](https://mmclassification.readthedocs.io/en/1.x/get_started.html) for more detailed installation and dataset preparation.

## User Guides

We provided a series of tutorials about the basic usage of MMClassification for new users:

- [Inference with existing models](https://mmclassification.readthedocs.io/en/1.x/user_guides/inference.html)
- [Prepare Dataset](https://mmclassification.readthedocs.io/en/1.x/user_guides/dataset_prepare.html)
- [Training and Test](https://mmclassification.readthedocs.io/en/1.x/user_guides/train_test.html)
- [Learn about Configs](https://mmclassification.readthedocs.io/en/1.x/user_guides/config.html)
- [Fine-tune Models](https://mmclassification.readthedocs.io/en/1.x/user_guides/finetune.html)
- [Analysis Tools](https://mmclassification.readthedocs.io/en/1.x/user_guides/analysis.html)
- [Visualization Tools](https://mmclassification.readthedocs.io/en/1.x/user_guides/visualization.html)
- [Other Useful Tools](https://mmclassification.readthedocs.io/en/1.x/user_guides/useful_tools.html)

## Model zoo

Results and models are available in the [model zoo](https://mmclassification.readthedocs.io/en/1.x/modelzoo_statistics.html).

<details open>
<summary>Supported backbones</summary>

- [x] [VGG](https://github.com/open-mmlab/mmclassification/tree/1.x/configs/vgg)
- [x] [ResNet](https://github.com/open-mmlab/mmclassification/tree/1.x/configs/resnet)
- [x] [ResNeXt](https://github.com/open-mmlab/mmclassification/tree/1.x/configs/resnext)
- [x] [SE-ResNet](https://github.com/open-mmlab/mmclassification/tree/1.x/configs/seresnet)
- [x] [SE-ResNeXt](https://github.com/open-mmlab/mmclassification/tree/1.x/configs/seresnet)
- [x] [RegNet](https://github.com/open-mmlab/mmclassification/tree/1.x/configs/regnet)
- [x] [ShuffleNetV1](https://github.com/open-mmlab/mmclassification/tree/1.x/configs/shufflenet_v1)
- [x] [ShuffleNetV2](https://github.com/open-mmlab/mmclassification/tree/1.x/configs/shufflenet_v2)
- [x] [MobileNetV2](https://github.com/open-mmlab/mmclassification/tree/1.x/configs/mobilenet_v2)
- [x] [MobileNetV3](https://github.com/open-mmlab/mmclassification/tree/1.x/configs/mobilenet_v3)
- [x] [Swin-Transformer](https://github.com/open-mmlab/mmclassification/tree/1.x/configs/swin_transformer)
- [x] [Swin-Transformer V2](https://github.com/open-mmlab/mmclassification/tree/1.x/configs/swin_transformer_v2)
- [x] [RepVGG](https://github.com/open-mmlab/mmclassification/tree/1.x/configs/repvgg)
- [x] [Vision-Transformer](https://github.com/open-mmlab/mmclassification/tree/1.x/configs/vision_transformer)
- [x] [Transformer-in-Transformer](https://github.com/open-mmlab/mmclassification/tree/1.x/configs/tnt)
- [x] [Res2Net](https://github.com/open-mmlab/mmclassification/tree/1.x/configs/res2net)
- [x] [MLP-Mixer](https://github.com/open-mmlab/mmclassification/tree/1.x/configs/mlp_mixer)
- [x] [DeiT](https://github.com/open-mmlab/mmclassification/tree/1.x/configs/deit)
- [x] [DeiT-3](https://github.com/open-mmlab/mmclassification/tree/1.x/configs/deit3)
- [x] [Conformer](https://github.com/open-mmlab/mmclassification/tree/1.x/configs/conformer)
- [x] [T2T-ViT](https://github.com/open-mmlab/mmclassification/tree/1.x/configs/t2t_vit)
- [x] [Twins](https://github.com/open-mmlab/mmclassification/tree/1.x/configs/twins)
- [x] [EfficientNet](https://github.com/open-mmlab/mmclassification/tree/1.x/configs/efficientnet)
- [x] [EdgeNeXt](https://github.com/open-mmlab/mmclassification/tree/1.x/configs/edgenext)
- [x] [ConvNeXt](https://github.com/open-mmlab/mmclassification/tree/1.x/configs/convnext)
- [x] [HRNet](https://github.com/open-mmlab/mmclassification/tree/1.x/configs/hrnet)
- [x] [VAN](https://github.com/open-mmlab/mmclassification/tree/1.x/configs/van)
- [x] [ConvMixer](https://github.com/open-mmlab/mmclassification/tree/1.x/configs/convmixer)
- [x] [CSPNet](https://github.com/open-mmlab/mmclassification/tree/1.x/configs/cspnet)
- [x] [PoolFormer](https://github.com/open-mmlab/mmclassification/tree/1.x/configs/poolformer)
- [x] [Inception V3](https://github.com/open-mmlab/mmclassification/tree/1.x/configs/inception_v3)
- [x] [MobileOne](https://github.com/open-mmlab/mmclassification/tree/1.x/configs/mobileone)
- [x] [EfficientFormer](https://github.com/open-mmlab/mmclassification/tree/1.x/configs/efficientformer)
- [x] [MViT](https://github.com/open-mmlab/mmclassification/tree/1.x/configs/mvit)
- [x] [HorNet](https://github.com/open-mmlab/mmclassification/tree/master/configs/hornet)
- [x] [MobileViT](https://github.com/open-mmlab/mmclassification/tree/1.x/configs/mobilevit)
- [x] [DaViT](https://github.com/open-mmlab/mmclassification/tree/1.x/configs/davit)
- [x] [RepLKNet](https://github.com/open-mmlab/mmclassification/tree/1.x/configs/replknet)
- [x] [BEiT](https://github.com/open-mmlab/mmclassification/tree/1.x/configs/beit) / [BEiT v2](https://github.com/open-mmlab/mmclassification/tree/1.x/configs/beitv2)
- [x] [EVA](https://github.com/open-mmlab/mmclassification/tree/1.x/configs/eva)

</details>

## Contributing

We appreciate all contributions to improve MMClassification.
Please refer to [CONTRUBUTING.md](https://mmclassification.readthedocs.io/en/1.x/notes/contribution_guide.html) for the contributing guideline.

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

- [MMEngine](https://github.com/open-mmlab/mmengine): OpenMMLab foundational library for training deep learning models.
- [MMCV](https://github.com/open-mmlab/mmcv): OpenMMLab foundational library for computer vision.
- [MIM](https://github.com/open-mmlab/mim): MIM installs OpenMMLab packages.
- [MMEval](https://github.com/open-mmlab/mmeval): A unified evaluation library for multiple machine learning libraries.
- [MMClassification](https://github.com/open-mmlab/mmclassification): OpenMMLab image classification toolbox and benchmark.
- [MMDetection](https://github.com/open-mmlab/mmdetection): OpenMMLab detection toolbox and benchmark.
- [MMDetection3D](https://github.com/open-mmlab/mmdetection3d): OpenMMLab's next-generation platform for general 3D object detection.
- [MMRotate](https://github.com/open-mmlab/mmrotate): OpenMMLab rotated object detection toolbox and benchmark.
- [MMYOLO](https://github.com/open-mmlab/mmyolo): OpenMMLab YOLO series toolbox and benchmark.
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
