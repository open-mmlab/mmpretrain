<div align="center">

<img src="resources/mmcls-logo.png" width="600"/>
  <div>&nbsp;</div>
  <div align="center">
    <b><font size="5">OpenMMLab 官网</font></b>
    <sup>
      <a href="https://openmmlab.com">
        <i><font size="4">HOT</font></i>
      </a>
    </sup>
    &nbsp;&nbsp;&nbsp;&nbsp;
    <b><font size="5">OpenMMLab 开放平台</font></b>
    <sup>
      <a href="https://platform.openmmlab.com">
        <i><font size="4">TRY IT OUT</font></i>
      </a>
    </sup>
  </div>
  <div>&nbsp;</div>

[![PyPI](https://img.shields.io/pypi/v/mmcls)](https://pypi.org/project/mmcls)
[![Docs](https://img.shields.io/badge/docs-latest-blue)](https://mmclassification.readthedocs.io/zh_CN/1.x/)
[![Build Status](https://github.com/open-mmlab/mmclassification/workflows/build/badge.svg)](https://github.com/open-mmlab/mmclassification/actions)
[![codecov](https://codecov.io/gh/open-mmlab/mmclassification/branch/1.x/graph/badge.svg)](https://codecov.io/gh/open-mmlab/mmclassification)
[![license](https://img.shields.io/github/license/open-mmlab/mmclassification.svg)](https://github.com/open-mmlab/mmclassification/blob/1.x/LICENSE)
[![open issues](https://isitmaintained.com/badge/open/open-mmlab/mmclassification.svg)](https://github.com/open-mmlab/mmclassification/issues)
[![issue resolution](https://isitmaintained.com/badge/resolution/open-mmlab/mmclassification.svg)](https://github.com/open-mmlab/mmclassification/issues)

[📘 中文文档](https://mmclassification.readthedocs.io/zh_CN/1.x/) |
[🛠️ 安装教程](https://mmclassification.readthedocs.io/zh_CN/1.x/get_started.html) |
[👀 模型库](https://mmclassification.readthedocs.io/zh_CN/1.x/modelzoo_statistics.html) |
[🆕 更新日志](https://mmclassification.readthedocs.io/en/1.x/notes/changelog.html) |
[🤔 报告问题](https://github.com/open-mmlab/mmclassification/issues/new/choose)

</div>

## Introduction

[English](/README.md) | 简体中文

MMClassification 是一款基于 PyTorch 的开源图像分类工具箱，是 [OpenMMLab](https://openmmlab.com/) 项目的成员之一

主分支代码目前支持 PyTorch 1.5 以上的版本。

<div align="center">
  <img src="https://user-images.githubusercontent.com/9102141/87268895-3e0d0780-c4fe-11ea-849e-6140b7e0d4de.gif" width="70%"/>
</div>

### 主要特性

- 支持多样的主干网络与预训练模型
- 支持配置多种训练技巧
- 大量的训练配置文件
- 高效率和高可扩展性
- 功能强大的工具箱

## 更新日志

2022/12/06 发布了 v1.0.0rc4 版本

- 更新了主要 API 接口，用以方便地获取 MMClassification 中预定义的模型。详见 [#1236](https://github.com/open-mmlab/mmclassification/pull/1236)。
- 重构 BEiT 主干网络结构，并支持 v1 和 v2 模型的推理。

2022/11/21 发布了 v1.0.0rc3 版本

- 添加了 **Switch Recipe Hook**，现在我们可以在训练过程中修改数据增强、Mixup设置、loss设置等
- 添加了 **TIMM 和 HuggingFace** 包装器，现在我们可以直接训练、使用 TIMM 和 HuggingFace 中的模型
- 支持了检索任务
- 复现了 **MobileOne** 训练精度

2022/8/31 发布了 v1.0.0rc0 版本

这个版本引入一个全新的，可扩展性强的训练和测试引擎，但目前仍在开发中。欢迎根据[文档](https://mmclassification.readthedocs.io/zh_CN/1.x/)进行试用。

同时，新版本中存在一些与旧版本不兼容的修改。请查看[迁移文档](https://mmclassification.readthedocs.io/zh_CN/1.x/migration.html)来详细了解这些变动。

新版本的公测将持续到 2022 年末，在此期间，我们将基于 `1.x` 分支进行更新，不会合入到 `master` 分支。另外，至少
到 2023 年末，我们会保持对 0.x 版本的维护。

发布历史和更新细节请参考 [更新日志](https://mmclassification.readthedocs.io/zh_CN/1.x/notes/changelog.html)

## 安装

以下是安装的简要步骤：

```shell
conda create -n open-mmlab python=3.8 pytorch==1.10.1 torchvision==0.11.2 cudatoolkit=11.3 -c pytorch -y
conda activate open-mmlab
pip3 install openmim
git clone -b 1.x https://github.com/open-mmlab/mmclassification.git
cd mmclassification
mim install -e .
```

更详细的步骤请参考 [安装指南](https://mmclassification.readthedocs.io/zh_CN/1.x/get_started.html) 进行安装。

## 基础教程

我们为新用户提供了一系列基础教程：

- [使用现有模型推理](https://mmclassification.readthedocs.io/zh_CN/1.x/user_guides/inference.html)
- [准备数据集](https://mmclassification.readthedocs.io/zh_CN/1.x/user_guides/dataset_prepare.html)
- [训练与测试](https://mmclassification.readthedocs.io/zh_CN/1.x/user_guides/train_test.html)
- [学习配置文件](https://mmclassification.readthedocs.io/zh_CN/1.x/user_guides/config.html)
- [如何微调模型](https://mmclassification.readthedocs.io/zh_CN/1.x/user_guides/finetune.html)
- [分析工具](https://mmclassification.readthedocs.io/zh_CN/1.x/user_guides/analysis.html)
- [可视化工具](https://mmclassification.readthedocs.io/zh_CN/1.x/user_guides/visualization.html)
- [其他工具](https://mmclassification.readthedocs.io/zh_CN/1.x/user_guides/useful_tools.html)

## 模型库

相关结果和模型可在 [model zoo](https://mmclassification.readthedocs.io/zh_CN/1.x/modelzoo_statistics.html) 中获得

<details open>
<summary>支持的主干网络</summary>

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

## 参与贡献

我们非常欢迎任何有助于提升 MMClassification 的贡献，请参考 [贡献指南](https://mmclassification.readthedocs.io/zh_CN/1.x/notes/contribution_guide.html) 来了解如何参与贡献。

## 致谢

MMClassification 是一款由不同学校和公司共同贡献的开源项目。我们感谢所有为项目提供算法复现和新功能支持的贡献者，以及提供宝贵反馈的用户。

我们希望该工具箱和基准测试可以为社区提供灵活的代码工具，供用户复现现有算法并开发自己的新模型，从而不断为开源社区提供贡献。

## 引用

如果你在研究中使用了本项目的代码或者性能基准，请参考如下 bibtex 引用 MMClassification。

```BibTeX
@misc{2020mmclassification,
    title={OpenMMLab's Image Classification Toolbox and Benchmark},
    author={MMClassification Contributors},
    howpublished = {\url{https://github.com/open-mmlab/mmclassification}},
    year={2020}
}
```

## 许可证

该项目开源自 [Apache 2.0 license](LICENSE).

## OpenMMLab 的其他项目

- [MMEngine](https://github.com/open-mmlab/mmengine): OpenMMLab 深度学习模型训练基础库
- [MMCV](https://github.com/open-mmlab/mmcv): OpenMMLab 计算机视觉基础库
- [MIM](https://github.com/open-mmlab/mim): MIM 是 OpenMMlab 项目、算法、模型的统一入口
- [MMEval](https://github.com/open-mmlab/mmeval): 统一开放的跨框架算法评测库
- [MMClassification](https://github.com/open-mmlab/mmclassification): OpenMMLab 图像分类工具箱
- [MMDetection](https://github.com/open-mmlab/mmdetection): OpenMMLab 目标检测工具箱
- [MMDetection3D](https://github.com/open-mmlab/mmdetection3d): OpenMMLab 新一代通用 3D 目标检测平台
- [MMRotate](https://github.com/open-mmlab/mmrotate): OpenMMLab 旋转框检测工具箱与测试基准
- [MMYOLO](https://github.com/open-mmlab/mmyolo): OpenMMLab YOLO 系列工具箱与测试基准
- [MMSegmentation](https://github.com/open-mmlab/mmsegmentation): OpenMMLab 语义分割工具箱
- [MMOCR](https://github.com/open-mmlab/mmocr): OpenMMLab 全流程文字检测识别理解工具包
- [MMPose](https://github.com/open-mmlab/mmpose): OpenMMLab 姿态估计工具箱
- [MMHuman3D](https://github.com/open-mmlab/mmhuman3d): OpenMMLab 人体参数化模型工具箱与测试基准
- [MMSelfSup](https://github.com/open-mmlab/mmselfsup): OpenMMLab 自监督学习工具箱与测试基准
- [MMRazor](https://github.com/open-mmlab/mmrazor): OpenMMLab 模型压缩工具箱与测试基准
- [MMFewShot](https://github.com/open-mmlab/mmfewshot): OpenMMLab 少样本学习工具箱与测试基准
- [MMAction2](https://github.com/open-mmlab/mmaction2): OpenMMLab 新一代视频理解工具箱
- [MMTracking](https://github.com/open-mmlab/mmtracking): OpenMMLab 一体化视频目标感知平台
- [MMFlow](https://github.com/open-mmlab/mmflow): OpenMMLab 光流估计工具箱与测试基准
- [MMEditing](https://github.com/open-mmlab/mmediting): OpenMMLab 图像视频编辑工具箱
- [MMGeneration](https://github.com/open-mmlab/mmgeneration): OpenMMLab 图片视频生成模型工具箱
- [MMDeploy](https://github.com/open-mmlab/mmdeploy): OpenMMLab 模型部署框架

## 欢迎加入 OpenMMLab 社区

扫描下方的二维码可关注 OpenMMLab 团队的 [知乎官方账号](https://www.zhihu.com/people/openmmlab)，加入 OpenMMLab 团队的 [官方交流 QQ 群](https://jq.qq.com/?_wv=1027&k=aCvMxdr3) 或联络 OpenMMLab 官方微信小助手

<div align="center">
<img src="https://github.com/open-mmlab/mmcv/raw/master/docs/en/_static/zhihu_qrcode.jpg" height="400" />  <img src="https://github.com/open-mmlab/mmcv/raw/master/docs/en/_static/qq_group_qrcode.jpg" height="400" /> <img src="https://github.com/open-mmlab/mmcv/raw/master/docs/en/_static/wechat_qrcode.jpg" height="400" />
</div>

我们会在 OpenMMLab 社区为大家

- 📢 分享 AI 框架的前沿核心技术
- 💻 解读 PyTorch 常用模块源码
- 📰 发布 OpenMMLab 的相关新闻
- 🚀 介绍 OpenMMLab 开发的前沿算法
- 🏃 获取更高效的问题答疑和意见反馈
- 🔥 提供与各行各业开发者充分交流的平台

干货满满 📘，等你来撩 💗，OpenMMLab 社区期待您的加入 👬
