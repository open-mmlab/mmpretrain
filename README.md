<div align="center">

<img src="resources/mmpt-logo.png" width="600"/>
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

[![PyPI](https://img.shields.io/pypi/v/mmpretrain)](https://pypi.org/project/mmpretrain)
[![Docs](https://img.shields.io/badge/docs-latest-blue)](https://mmpretrain.readthedocs.io/en/latest/)
[![Build Status](https://github.com/open-mmlab/mmpretrain/workflows/build/badge.svg)](https://github.com/open-mmlab/mmpretrain/actions)
[![codecov](https://codecov.io/gh/open-mmlab/mmpretrain/branch/main/graph/badge.svg)](https://codecov.io/gh/open-mmlab/mmpretrain)
[![license](https://img.shields.io/github/license/open-mmlab/mmpretrain.svg)](https://github.com/open-mmlab/mmpretrain/blob/main/LICENSE)
[![open issues](https://isitmaintained.com/badge/open/open-mmlab/mmpretrain.svg)](https://github.com/open-mmlab/mmpretrain/issues)
[![issue resolution](https://isitmaintained.com/badge/resolution/open-mmlab/mmpretrain.svg)](https://github.com/open-mmlab/mmpretrain/issues)

[📘 Documentation](https://mmpretrain.readthedocs.io/en/latest/) |
[🛠️ Installation](https://mmpretrain.readthedocs.io/en/latest/get_started.html#installation) |
[👀 Model Zoo](https://mmpretrain.readthedocs.io/en/latest/modelzoo_statistics.html) |
[🆕 Update News](https://mmpretrain.readthedocs.io/en/latest/notes/changelog.html) |
[🤔 Reporting Issues](https://github.com/open-mmlab/mmpretrain/issues/new/choose)

<img src="https://user-images.githubusercontent.com/36138628/230307505-4727ad0a-7d71-4069-939d-b499c7e272b7.png" width="400"/>

English | [简体中文](/README_zh-CN.md)

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

MMPreTrain is an open source pre-training toolbox based on PyTorch. It is a part of the [OpenMMLab](https://openmmlab.com/) project.

The `main` branch works with **PyTorch 1.8+**.

### Major features

- Various backbones and pretrained models
- Rich training strategies (supervised learning, self-supervised learning, multi-modality learning etc.)
- Bag of training tricks
- Large-scale training configs
- High efficiency and extensibility
- Powerful toolkits for model analysis and experiments
- Various out-of-box inference tasks.
  - Image Classification
  - Image Caption
  - Visual Question Answering
  - Visual Grounding
  - Retrieval (Image-To-Image, Text-To-Image, Image-To-Text)

https://github.com/open-mmlab/mmpretrain/assets/26739999/e4dcd3a2-f895-4d1b-a351-fbc74a04e904

## What's new

🌟 v1.2.0 was released in 04/01/2023

- Support LLaVA 1.5.
- Implement of RAM with a gradio interface.

🌟 v1.1.0 was released in 12/10/2023

- Support Mini-GPT4 training and provide a Chinese model (based on Baichuan-7B)
- Support zero-shot classification based on CLIP.

🌟 v1.0.0 was released in 04/07/2023

- Support inference of more **multi-modal** algorithms, such as [**LLaVA**](./configs/llava/), [**MiniGPT-4**](./configs/minigpt4), [**Otter**](./configs/otter/), etc.
- Support around **10 multi-modal** datasets!
- Add [**iTPN**](./configs/itpn/), [**SparK**](./configs/spark/) self-supervised learning algorithms.
- Provide examples of [New Config](./mmpretrain/configs/) and [DeepSpeed/FSDP with FlexibleRunner](./configs/mae/benchmarks/). Here are the documentation links of [New Config](https://mmengine.readthedocs.io/en/latest/advanced_tutorials/config.html#a-pure-python-style-configuration-file-beta) and [DeepSpeed/FSDP with FlexibleRunner](https://mmengine.readthedocs.io/en/latest/api/generated/mmengine.runner.FlexibleRunner.html#mmengine.runner.FlexibleRunner).

🌟 Upgrade from MMClassification to MMPreTrain

- Integrated Self-supervised learning algorithms from **MMSelfSup**, such as **MAE**, **BEiT**, etc.
- Support **RIFormer**, a simple but effective vision backbone by removing token mixer.
- Refactor dataset pipeline visualization.
- Support **LeViT**, **XCiT**, **ViG**, **ConvNeXt-V2**, **EVA**, **RevViT**, **EfficientnetV2**, **CLIP**, **TinyViT** and **MixMIM** backbones.

This release introduced a brand new and flexible training & test engine, but it's still in progress. Welcome
to try according to [the documentation](https://mmpretrain.readthedocs.io/en/latest/).

And there are some BC-breaking changes. Please check [the migration tutorial](https://mmpretrain.readthedocs.io/en/latest/migration.html).

Please refer to [changelog](https://mmpretrain.readthedocs.io/en/latest/notes/changelog.html) for more details and other release history.

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

Please refer to [installation documentation](https://mmpretrain.readthedocs.io/en/latest/get_started.html) for more detailed installation and dataset preparation.

For multi-modality models support, please install the extra dependencies by:

```shell
mim install -e ".[multimodal]"
```

## User Guides

We provided a series of tutorials about the basic usage of MMPreTrain for new users:

- [Learn about Configs](https://mmpretrain.readthedocs.io/en/latest/user_guides/config.html)
- [Prepare Dataset](https://mmpretrain.readthedocs.io/en/latest/user_guides/dataset_prepare.html)
- [Inference with existing models](https://mmpretrain.readthedocs.io/en/latest/user_guides/inference.html)
- [Train](https://mmpretrain.readthedocs.io/en/latest/user_guides/train.html)
- [Test](https://mmpretrain.readthedocs.io/en/latest/user_guides/test.html)
- [Downstream tasks](https://mmpretrain.readthedocs.io/en/latest/user_guides/downstream.html)

For more information, please refer to [our documentation](https://mmpretrain.readthedocs.io/en/latest/).

## Model zoo

Results and models are available in the [model zoo](https://mmpretrain.readthedocs.io/en/latest/modelzoo_statistics.html).

<div align="center">
  <b>Overview</b>
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
        <b>Multi-Modality Algorithms</b>
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
        <li><a href="configs/revvit">RevViT</a></li>
        <li><a href="configs/convnext_v2">ConvNeXt V2</a></li>
        <li><a href="configs/vig">ViG</a></li>
        <li><a href="configs/xcit">XCiT</a></li>
        <li><a href="configs/levit">LeViT</a></li>
        <li><a href="configs/riformer">RIFormer</a></li>
        <li><a href="configs/glip">GLIP</a></li>
        <li><a href="configs/sam">ViT SAM</a></li>
        <li><a href="configs/eva02">EVA02</a></li>
        <li><a href="configs/dinov2">DINO V2</a></li>
        <li><a href="configs/hivit">HiViT</a></li>
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
        <li><a href="configs/itpn">iTPN (CVPR'2023)</a></li>
        <li><a href="configs/spark">SparK (ICLR'2023)</a></li>
        <li><a href="configs/mff">MFF (ICCV'2023)</a></li>
        </ul>
      </td>
      <td>
        <ul>
        <li><a href="configs/blip">BLIP (arxiv'2022)</a></li>
        <li><a href="configs/blip2">BLIP-2 (arxiv'2023)</a></li>
        <li><a href="configs/ofa">OFA (CoRR'2022)</a></li>
        <li><a href="configs/flamingo">Flamingo (NeurIPS'2022)</a></li>
        <li><a href="configs/chinese_clip">Chinese CLIP (arxiv'2022)</a></li>
        <li><a href="configs/minigpt4">MiniGPT-4 (arxiv'2023)</a></li>
        <li><a href="configs/llava">LLaVA (arxiv'2023)</a></li>
        <li><a href="configs/otter">Otter (arxiv'2023)</a></li>
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
        <li><a href="mmpretrain/datasets/samplers/repeat_aug.py">RepeatAugSampler</a></li>
        <li><a href="mmpretrain/models/tta/score_tta.py">TTA</a></li>
        <li>...</li>
        </ul>
      </td>
  </tbody>
</table>

## Contributing

We appreciate all contributions to improve MMPreTrain.
Please refer to [CONTRUBUTING](https://mmpretrain.readthedocs.io/en/latest/notes/contribution_guide.html) for the contributing guideline.

## Acknowledgement

MMPreTrain is an open source project that is contributed by researchers and engineers from various colleges and companies. We appreciate all the contributors who implement their methods or add new features, as well as users who give valuable feedbacks.
We wish that the toolbox and benchmark could serve the growing research community by providing a flexible toolkit to reimplement existing methods and supporting their own academic research.

## Citation

If you find this project useful in your research, please consider cite:

```BibTeX
@misc{2023mmpretrain,
    title={OpenMMLab's Pre-training Toolbox and Benchmark},
    author={MMPreTrain Contributors},
    howpublished = {\url{https://github.com/open-mmlab/mmpretrain}},
    year={2023}
}
```

## License

This project is released under the [Apache 2.0 license](LICENSE).

## Projects in OpenMMLab

- [MMEngine](https://github.com/open-mmlab/mmengine): OpenMMLab foundational library for training deep learning models.
- [MMCV](https://github.com/open-mmlab/mmcv): OpenMMLab foundational library for computer vision.
- [MIM](https://github.com/open-mmlab/mim): MIM installs OpenMMLab packages.
- [MMEval](https://github.com/open-mmlab/mmeval): A unified evaluation library for multiple machine learning libraries.
- [MMPreTrain](https://github.com/open-mmlab/mmpretrain): OpenMMLab pre-training toolbox and benchmark.
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
- [MMagic](https://github.com/open-mmlab/mmagic): Open**MM**Lab **A**dvanced, **G**enerative and **I**ntelligent **C**reation toolbox.
- [MMGeneration](https://github.com/open-mmlab/mmgeneration): OpenMMLab image and video generative models toolbox.
- [MMDeploy](https://github.com/open-mmlab/mmdeploy): OpenMMLab model deployment framework.
- [Playground](https://github.com/open-mmlab/playground): A central hub for gathering and showcasing amazing projects built upon OpenMMLab.
