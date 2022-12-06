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
[![Docs](https://img.shields.io/badge/docs-latest-blue)](https://mmclassification.readthedocs.io/zh_CN/latest/)
[![Build Status](https://github.com/open-mmlab/mmclassification/workflows/build/badge.svg)](https://github.com/open-mmlab/mmclassification/actions)
[![codecov](https://codecov.io/gh/open-mmlab/mmclassification/branch/master/graph/badge.svg)](https://codecov.io/gh/open-mmlab/mmclassification)
[![license](https://img.shields.io/github/license/open-mmlab/mmclassification.svg)](https://github.com/open-mmlab/mmclassification/blob/master/LICENSE)
[![open issues](https://isitmaintained.com/badge/open/open-mmlab/mmclassification.svg)](https://github.com/open-mmlab/mmclassification/issues)
[![issue resolution](https://isitmaintained.com/badge/resolution/open-mmlab/mmclassification.svg)](https://github.com/open-mmlab/mmclassification/issues)

[📘 中文文档](https://mmclassification.readthedocs.io/zh_CN/latest/) |
[🛠️ 安装教程](https://mmclassification.readthedocs.io/zh_CN/latest/install.html) |
[👀 模型库](https://mmclassification.readthedocs.io/zh_CN/latest/model_zoo.html) |
[🆕 更新日志](https://mmclassification.readthedocs.io/en/latest/changelog.html) |
[🤔 报告问题](https://github.com/open-mmlab/mmclassification/issues/new/choose)

:point_right: **MMClassification 1.0 版本即将正式发布，欢迎大家 [试用](https://github.com/open-mmlab/mmclassification/tree/1.x) 并 [参与讨论](https://github.com/open-mmlab/mmclassification/discussions)！** :point_left:

</div>

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

MMClassification 1.0 已经发布！目前仍在公测中，如果希望试用，请切换到 [1.x 分支](https://github.com/open-mmlab/mmclassification/tree/1.x)，并在[讨论版](https://github.com/open-mmlab/mmclassification/discussions) 参加开发讨论！

2022/12/06 发布了 v0.25.0 版本

- 支持 MLU 设备
- 添加了用于 ARM 设备训练的 `dist_train_arm.sh`

2022/10/31 发布了 v0.24.1 版本

- 支持了华为昇腾 NPU 设备。

2022/9/30 发布了 v0.24.0 版本

- 支持了 **HorNet**，**EfficientFormerm**，**SwinTransformer V2**，**MViT** 等主干网络。
- 支持了 Support Standford Cars 数据集。

2022/5/1 发布了 v0.23.0 版本

- 支持了 **DenseNet**，**VAN** 和 **PoolFormer** 三个网络，并提供了预训练模型。
- 支持在 IPU 上进行训练。
- 更新了 API 文档的样式，更方便查阅，[欢迎查阅](https://mmclassification.readthedocs.io/en/master/api/models.html)。

发布历史和更新细节请参考 [更新日志](docs/en/changelog.md)

## 安装

以下是安装的简要步骤：

```shell
conda create -n open-mmlab python=3.8 pytorch=1.10 cudatoolkit=11.3 torchvision==0.11.0 -c pytorch -y
conda activate open-mmlab
pip3 install openmim
mim install mmcv-full
git clone https://github.com/open-mmlab/mmclassification.git
cd mmclassification
pip3 install -e .
```

更详细的步骤请参考 [安装指南](https://mmclassification.readthedocs.io/zh_CN/latest/install.html) 进行安装。

## 基础教程

请参考 [基础教程](https://mmclassification.readthedocs.io/zh_CN/latest/getting_started.html) 来了解 MMClassification 的基本使用。MMClassification 也提供了其他更详细的教程：

- [如何编写配置文件](https://mmclassification.readthedocs.io/zh_CN/latest/tutorials/config.html)
- [如何微调模型](https://mmclassification.readthedocs.io/zh_CN/latest/tutorials/finetune.html)
- [如何增加新数据集](https://mmclassification.readthedocs.io/zh_CN/latest/tutorials/new_dataset.html)
- [如何设计数据处理流程](https://mmclassification.readthedocs.io/zh_CN/latest/tutorials/data_pipeline.html)
- [如何增加新模块](https://mmclassification.readthedocs.io/zh_CN/latest/tutorials/new_modules.html)
- [如何自定义优化策略](https://mmclassification.readthedocs.io/zh_CN/latest/tutorials/schedule.html)
- [如何自定义运行参数](https://mmclassification.readthedocs.io/zh_CN/latest/tutorials/runtime.html)

我们也提供了相应的中文 Colab 教程：

- 了解 MMClassification **Python API**：[预览 Notebook](https://github.com/open-mmlab/mmclassification/blob/master/docs/zh_CN/tutorials/MMClassification_python_cn.ipynb) 或者直接[在 Colab 上运行](https://colab.research.google.com/github/open-mmlab/mmclassification/blob/master/docs/zh_CN/tutorials/MMClassification_python_cn.ipynb)。
- 了解 MMClassification **命令行工具**：[预览 Notebook](https://github.com/open-mmlab/mmclassification/blob/master/docs/zh_CN/tutorials/MMClassification_tools_cn.ipynb) 或者直接[在 Colab 上运行](https://colab.research.google.com/github/open-mmlab/mmclassification/blob/master/docs/zh_CN/tutorials/MMClassification_tools_cn.ipynb)。

## 模型库

相关结果和模型可在 [model zoo](https://mmclassification.readthedocs.io/en/latest/model_zoo.html) 中获得

<details open>
<summary>支持的主干网络</summary>

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

## 参与贡献

我们非常欢迎任何有助于提升 MMClassification 的贡献，请参考 [贡献指南](https://mmclassification.readthedocs.io/zh_CN/latest/community/CONTRIBUTING.html) 来了解如何参与贡献。

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

- [MMCV](https://github.com/open-mmlab/mmcv): OpenMMLab 计算机视觉基础库
- [MIM](https://github.com/open-mmlab/mim): MIM 是 OpenMMlab 项目、算法、模型的统一入口
- [MMClassification](https://github.com/open-mmlab/mmclassification): OpenMMLab 图像分类工具箱
- [MMDetection](https://github.com/open-mmlab/mmdetection): OpenMMLab 目标检测工具箱
- [MMDetection3D](https://github.com/open-mmlab/mmdetection3d): OpenMMLab 新一代通用 3D 目标检测平台
- [MMRotate](https://github.com/open-mmlab/mmrotate): OpenMMLab 旋转框检测工具箱与测试基准
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
