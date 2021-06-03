<div align="center">
  <img src="resources/mmcls-logo.png" width="600"/>
</div>

[English](/README.md) | 简体中文

[![Build Status](https://github.com/open-mmlab/mmclassification/workflows/build/badge.svg)](https://github.com/open-mmlab/mmclassification/actions)
[![Documentation Status](https://readthedocs.org/projects/mmclassification/badge/?version=latest)](https://mmclassification.readthedocs.io/en/latest/?badge=latest)
[![codecov](https://codecov.io/gh/open-mmlab/mmclassification/branch/master/graph/badge.svg)](https://codecov.io/gh/open-mmlab/mmclassification)
[![license](https://img.shields.io/github/license/open-mmlab/mmclassification.svg)](https://github.com/open-mmlab/mmclassification/blob/master/LICENSE)

## Introduction

MMClassification 是一款基于 PyTorch 的开源图像分类工具箱，是 [OpenMMLab](https://openmmlab.com/) 项目的成员之一

参考文档：https://mmclassification.readthedocs.io/en/latest/

![demo](https://user-images.githubusercontent.com/9102141/87268895-3e0d0780-c4fe-11ea-849e-6140b7e0d4de.gif)

### 主要特性

- 支持多样的主干网络与预训练模型
- 支持配置多种训练技巧
- 大量的训练配置文件
- 高效率和高可扩展性

## 许可证

该项目开源自 [Apache 2.0 license](LICENSE).

## 更新日志

2021/6/3 发布了 v0.12.0 版本

发布历史和更新细节请参考 [更新日志](docs/changelog.md)

## 基准测试及模型库

相关结果和模型可在 [model zoo](docs/model_zoo.md) 中获得

支持的主干网络：

- [x] ResNet
- [x] ResNeXt
- [x] SE-ResNet
- [x] SE-ResNeXt
- [x] RegNet
- [x] ShuffleNetV1
- [x] ShuffleNetV2
- [x] MobileNetV2
- [x] MobileNetV3

## 安装

请参考 [安装指南](docs_zh-CN/install.md) 进行安装

## 基础教程

请参考 [基础教程](docs_zh-CN/getting_started.md) 来了解 MMClassification 的基本使用。其中还包含了 [如何微调模型](docs_zh-CN/tutorials/finetune.md), [如何增加新数据集](docs_zh-CN/tutorials/new_dataset.md), [如何设计数据处理流程](docs_zh-CN/tutorials/data_pipeline.md), 以及 [如何增加新模块](docs_zh-CN/tutorials/new_modules.md) 等指南。

## 参与贡献

我们非常欢迎任何有助于提升 MMClassification 的贡献，请参考 [贡献指南](.github/CONTRIBUTING.md) 来了解如何参与贡献。

## 致谢

MMClassification 是一款由不同学校和公司共同贡献的开源项目。我们感谢所有为项目提供算法复现和新功能支持的贡献者，以及提供宝贵反馈的用户。

我们希望该工具箱和基准测试可以为社区提供灵活的代码工具，供用户复现现有算法并开发自己的新模型，从而不断为开源社区提供贡献。

## OpenMMLab 的其他项目

- [MMCV](https://github.com/open-mmlab/mmcv): OpenMMLab 计算机视觉基础库
- [MMDetection](https://github.com/open-mmlab/mmdetection): OpenMMLab 检测工具箱与测试基准
- [MMDetection3D](https://github.com/open-mmlab/mmdetection3d): OpenMMLab 新一代通用 3D 目标检测平台
- [MMSegmentation](https://github.com/open-mmlab/mmsegmentation): OpenMMLab 语义分割工具箱与测试基准
- [MMAction2](https://github.com/open-mmlab/mmaction2): OpenMMLab 新一代视频理解工具箱与测试基准
- [MMTracking](https://github.com/open-mmlab/mmtracking): OpenMMLab 一体化视频目标感知平台
- [MMPose](https://github.com/open-mmlab/mmpose): OpenMMLab 姿态估计工具箱与测试基准
- [MMEditing](https://github.com/open-mmlab/mmediting): OpenMMLab 图像视频编辑工具箱
- [MMOCR](https://github.com/open-mmlab/mmocr): OpenMMLab 全流程文字检测识别理解工具包
- [MMGeneration](https://github.com/open-mmlab/mmgeneration): OpenMMLab 生成模型工具箱

## 欢迎加入 OpenMMLab 社区

扫描下方的二维码可关注 OpenMMLab 团队的 [知乎官方账号](https://www.zhihu.com/people/openmmlab)，加入 OpenMMLab 团队的 [官方交流 QQ 群](https://jq.qq.com/?_wv=1027&k=aCvMxdr3)

<div align="center">
<img src="/docs/imgs/zhihu_qrcode.jpg" height="400" />  <img src="/docs/imgs/qq_group_qrcode.jpg" height="400" />
</div>

我们会在 OpenMMLab 社区为大家

- 📢 分享 AI 框架的前沿核心技术
- 💻 解读 PyTorch 常用模块源码
- 📰 发布 OpenMMLab 的相关新闻
- 🚀 介绍 OpenMMLab 开发的前沿算法
- 🏃 获取更高效的问题答疑和意见反馈
- 🔥 提供与各行各业开发者充分交流的平台

干货满满 📘，等你来撩 💗，OpenMMLab 社区期待您的加入 👬
