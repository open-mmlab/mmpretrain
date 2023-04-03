<div align="center">

<img src="resources/mmpt-logo.png" width="600"/>
  <div>&nbsp;</div>
  <b><font size="7">Originates from MMClassification and MMSelfSup</font></b>
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

[![PyPI](https://img.shields.io/pypi/v/mmpretrain)](https://pypi.org/project/mmpretrain)
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

</div>

<div align="center">
  <a href="https://openmmlab.medium.com/" style="text-decoration:none;">
    <img src="https://user-images.githubusercontent.com/25839884/219255827-67c1a27f-f8c5-46a9-811d-5e57448c61d1.png" width="3%" alt="" /></a>
  <img src="https://user-images.githubusercontent.com/25839884/218346358-56cc8e2f-a2b8-487f-9088-32480cceabcf.png" width="3%" alt="" />
  <a href="https://discord.gg/raweFPmdzG" style="text-decoration:none;">
    <img src="https://user-images.githubusercontent.com/25839884/218347213-c080267f-cbb6-443e-8532-8e1ed9a58ea9.png" width="3%" alt="" /></a>
  <img src="https://user-images.githubusercontent.com/25839884/218346358-56cc8e2f-a2b8-487f-9088-32480cceabcf.png" width="3%" alt="" />
  <a href="https://twitter.com/OpenMMLab" style="text-decoration:none;">
    <img src="https://user-images.githubusercontent.com/25839884/218346637-d30c8a0f-3eba-4699-8131-512fb06d46db.png" width="3%" alt="" /></a>
  <img src="https://user-images.githubusercontent.com/25839884/218346358-56cc8e2f-a2b8-487f-9088-32480cceabcf.png" width="3%" alt="" />
  <a href="https://www.youtube.com/openmmlab" style="text-decoration:none;">
    <img src="https://user-images.githubusercontent.com/25839884/218346691-ceb2116a-465a-40af-8424-9f30d2348ca9.png" width="3%" alt="" /></a>
  <img src="https://user-images.githubusercontent.com/25839884/218346358-56cc8e2f-a2b8-487f-9088-32480cceabcf.png" width="3%" alt="" />
  <a href="https://space.bilibili.com/1293512903" style="text-decoration:none;">
    <img src="https://user-images.githubusercontent.com/25839884/219026751-d7d14cce-a7c9-4e82-9942-8375fca65b99.png" width="3%" alt="" /></a>
  <img src="https://user-images.githubusercontent.com/25839884/218346358-56cc8e2f-a2b8-487f-9088-32480cceabcf.png" width="3%" alt="" />
  <a href="https://www.zhihu.com/people/openmmlab" style="text-decoration:none;">
    <img src="https://user-images.githubusercontent.com/25839884/219026120-ba71e48b-6e94-4bd4-b4e9-b7d175b5e362.png" width="3%" alt="" /></a>
</div>

## Introduction

English | [简体中文](/README_zh-CN.md)

MMPreTrain is an open source visual pre-training toolbox based on PyTorch. It is a part of the [OpenMMLab](https://openmmlab.com/) project.

The `main` branch works with **PyTorch 1.8+**.

<div align="center">
  <img src="https://user-images.githubusercontent.com/9102141/87268895-3e0d0780-c4fe-11ea-849e-6140b7e0d4de.gif" width="70%"/>
</div>

### Major features

- Various backbones and pretrained models
- Bag of training strategies and tricks
- Large-scale training configs
- High efficiency and extensibility
- Powerful toolkits

## What's new

v1.0.0rc0 was released in 30/12/2022

- Integrated Self-supervised leanrning algorithms from **MMSelfSup**

This release introduced a brand new and flexible training & test engine, but it's still in progress. Welcome
to try according to [the documentation](https://mmclassification.readthedocs.io/en/1.x/).

And there are some BC-breaking changes. Please check [the migration tutorial](https://mmclassification.readthedocs.io/en/1.x/migration.html).

Please refer to [changelog](https://mmclassification.readthedocs.io/en/1.x/notes/changelog.html) for more details and other release history.

## Installation

Below are quick steps for installation:

```shell
conda create -n open-mmlab python=3.8 pytorch==1.10.1 torchvision==0.11.2 cudatoolkit=11.3 -c pytorch -y
conda activate open-mmlab
pip install openmim
git clone https://github.com/open-mmlab/mmpretrain.git
cd mmpretrain
mim install -e .
```

Please refer to [installation documentation](https://mmclassification.readthedocs.io/en/1.x/get_started.html) for more detailed installation and dataset preparation.

## User Guides

We provided a series of tutorials about the basic usage of MMPreTrain for new users:

- [Learn about Configs](https://mmclassification.readthedocs.io/en/1.x/user_guides/config.html)
- [Prepare Dataset](https://mmclassification.readthedocs.io/en/1.x/user_guides/dataset_prepare.html)
- [Inference with existing models](https://mmclassification.readthedocs.io/en/1.x/user_guides/inference.html)
- [Train](https://mmclassification.readthedocs.io/en/pretrain/user_guides/train.html)
- [Test](https://mmclassification.readthedocs.io/en/pretrain/user_guides/test.html)
- [Downstream tasks](https://mmclassification.readthedocs.io/en/pretrain/user_guides/downstream.html)

For more information, please refer to [our documentation](https://mmclassification.readthedocs.io/en/pretrain/).

## Model zoo

Results and models are available in the [model zoo](https://mmclassification.readthedocs.io/en/1.x/modelzoo_statistics.html).

<div align="center">
  <b>Architectures</b>
</div>
<table align="center">
  <tbody>
    <tr align="center" valign="bottom">
      <td>
        <b>Supported Backbones</b>
      </td>
      <td>
        <b>Self-supervised Learning</b>
      </td>
      <td>
        <b>Others</b>
      </td>
    </tr>
    <tr valign="top">
      <td>
        <ul>
        <li><a href="configs/vgg">VGG</a></li>
        <li><a href="configs/resnet">ResNet</a></li>
        <li><a href="configs/resnext">ResNeXt</a></li>
        <li><a href="configs/seresnet">SE-ResNet</a></li>
        <li><a href="configs/seresnet">SE-ResNeXt</a></li>
        <li><a href="configs/regnet">RegNet</a></li>
        <li><a href="configs/shufflenet_v1">ShuffleNet V1</a></li>
        <li><a href="configs/shufflenet_v2">ShuffleNet V2</a></li>
        <li><a href="configs/mobilenet_v2">MobileNet V2</a></li>
        <li><a href="configs/mobilenet_v3">MobileNet V3</a></li>
        <li><a href="configs/swin_transformer">Swin-Transformer</a></li>
        <li><a href="configs/swin_transformer_v2">Swin-Transformer V2</a></li>
        <li><a href="configs/repvgg">RepVGG</a></li>
        <li><a href="configs/vision_transformer">Vision-Transformer</a></li>
        <li><a href="configs/tnt">Transformer-in-Transformer</a></li>
        <li><a href="configs/res2net">Res2Net</a></li>
        <li><a href="configs/mlp_mixer">MLP-Mixer</a></li>
        <li><a href="configs/deit">DeiT</a></li>
        <li><a href="configs/deit3">DeiT-3</a></li>
        <li><a href="configs/conformer">Conformer</a></li>
        <li><a href="configs/t2t_vit">T2T-ViT</a></li>
        <li><a href="configs/twins">Twins</a></li>
        <li><a href="configs/efficientnet">EfficientNet</a></li>
        <li><a href="configs/edgenext">EdgeNeXt</a></li>
        <li><a href="configs/convnext">ConvNeXt</a></li>
        <li><a href="configs/hrnet">HRNet</a></li>
        <li><a href="configs/van">VAN</a></li>
        <li><a href="configs/convmixer">ConvMixer</a></li>
        <li><a href="configs/cspnet">CSPNet</a></li>
        <li><a href="configs/poolformer">PoolFormer</a></li>
        <li><a href="configs/inception_v3">Inception V3</a></li>
        <li><a href="configs/mobileone">MobileOne</a></li>
        <li><a href="configs/efficientformer">EfficientFormer</a></li>
        <li><a href="configs/mvit">MViT</a></li>
        <li><a href="configs/hornet">HorNet</a></li>
        <li><a href="configs/mobilevit">MobileViT</a></li>
        <li><a href="configs/davit">DaViT</a></li>
        <li><a href="configs/replknet">RepLKNet</a></li>
        <li><a href="configs/beit">BEiT</a></li>
        <li><a href="configs/mixmim">MixMIM</a></li>
        <li><a href="configs/efficientnet_v2">EfficientNet V2</a></li>
        </ul>
      </td>
      <td>
        <ul>
        <li><a href="configs/mocov2">MoCo V1 (CVPR'2020)</a></li>
        <li><a href="configs/simclr">SimCLR (ICML'2020)</a></li>
        <li><a href="configs/mocov2">MoCo V2 (arXiv'2020)</a></li>
        <li><a href="configs/byol">BYOL (NeurIPS'2020)</a></li>
        <li><a href="configs/swav">SwAV (NeurIPS'2020)</a></li>
        <li><a href="configs/densecl">DenseCL (CVPR'2021)</a></li>
        <li><a href="configs/simsiam">SimSiam (CVPR'2021)</a></li>
        <li><a href="configs/barlowtwins">Barlow Twins (ICML'2021)</a></li>
        <li><a href="configs/mocov3">MoCo V3 (ICCV'2021)</a></li>
        <li><a href="configs/beit">BEiT (ICLR'2022)</a></li>
        <li><a href="configs/mae">MAE (CVPR'2022)</a></li>
        <li><a href="configs/simmim">SimMIM (CVPR'2022)</a></li>
        <li><a href="configs/maskfeat">MaskFeat (CVPR'2022)</a></li>
        <li><a href="configs/cae">CAE (arXiv'2022)</a></li>
        <li><a href="configs/milan">MILAN (arXiv'2022)</a></li>
        <li><a href="configs/beitv2">BEiT V2 (arXiv'2022)</a></li>
        <li><a href="configs/eva">EVA (CVPR'2023)</a></li>
        <li><a href="configs/mixmim">MixMIM (arXiv'2022)</a></li>
        </ul>
      </td>
      <td>
      Image Retrieval Task:
        <ul>
        <li><a href="configs/arcface">ArcFace (CVPR'2019)</a></li>
        </ul>
      Training&Test Tips:
        <ul>
        <li><a href="https://arxiv.org/abs/1909.13719">RandAug</a></li>
        <li><a href="https://arxiv.org/abs/1805.09501">AutoAug</a></li>
        <li><a href="mmpretrain/datasets/samplers/repeat_aug.py">RepeatAugSamper</a></li>
        <li><a href="mmpretrain/models/tta/score_tta.py">TTA</a></li>
        <li>...</li>
        </ul>
      </td>
  </tbody>
</table>

## Contributing

We appreciate all contributions to improve MMPreTrain.
Please refer to [CONTRUBUTING](https://mmclassification.readthedocs.io/en/1.x/notes/contribution_guide.html) for the contributing guideline.

## Acknowledgement

MMPreTrain is an open source project that is contributed by researchers and engineers from various colleges and companies. We appreciate all the contributors who implement their methods or add new features, as well as users who give valuable feedbacks.
We wish that the toolbox and benchmark could serve the growing research community by providing a flexible toolkit to reimplement existing methods and supporting their own academic research.

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
- [MMPreTrain](https://github.com/open-mmlab/mmpretrain): OpenMMLab visual pre-training toolbox and benchmark.
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
