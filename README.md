<div align="center">
  <img src="resources/mmcls-logo.png" width="600"/>
</div>

[![Build Status](https://github.com/open-mmlab/mmclassification/workflows/build/badge.svg)](https://github.com/open-mmlab/mmclassification/actions)
[![Documentation Status](https://readthedocs.org/projects/mmclassification/badge/?version=latest)](https://mmclassification.readthedocs.io/en/latest/?badge=latest)
[![codecov](https://codecov.io/gh/open-mmlab/mmclassification/branch/master/graph/badge.svg)](https://codecov.io/gh/open-mmlab/mmclassification)
[![license](https://img.shields.io/github/license/open-mmlab/mmclassification.svg)](https://github.com/open-mmlab/mmclassification/blob/master/LICENSE)

## Introduction

MMClassification is an open source image classification toolbox based on PyTorch. It is
a part of the [OpenMMLab](https://open-mmlab.github.io/) project.

Documentation: https://mmclassification.readthedocs.io/en/latest/

![demo](https://user-images.githubusercontent.com/9102141/87268895-3e0d0780-c4fe-11ea-849e-6140b7e0d4de.gif)

### Major features

- Various backbones and pretrained models
- Bag of training tricks
- Large-scale training configs
- High efficiency and extensibility

## License

This project is released under the [Apache 2.0 license](LICENSE).

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

## Installation

Please refer to [install.md](docs/install.md) for installation and dataset preparation.

## Getting Started

Please see [getting_started.md](docs/getting_started.md) for the basic usage of MMClassification. There are also tutorials for [finetuning models](docs/tutorials/finetune.md), [adding new dataset](docs/tutorials/new_dataset.md), [designing data pipeline](docs/tutorials/data_pipeline.md), and [adding new modules](docs/tutorials/new_modules.md).

## Contributing

We appreciate all contributions to improve MMClassification.
Please refer to [CONTRUBUTING.md](CONTRIBUTING.md) for the contributing guideline.

## Acknowledgement

MMClassification is an open source project that is contributed by researchers and engineers from various colleges and companies. We appreciate all the contributors who implement their methods or add new features, as well as users who give valuable feedbacks.
We wish that the toolbox and benchmark could serve the growing research community by providing a flexible toolkit to reimplement existing methods and develop their own new detectors.

Many thanks to Wenwei Zhang ([@ZwwWayne](https://github.com/ZwwWayne)), Jiarui Xu ([@xvjiarui](https://github.com/xvjiarui)), Xintao Wang ([@xinntao](https://github.com/xinntao)) and Zhizhong Li ([@innerlee](https://github.com/innerlee)) for their valuable advices and discussions.

## Citation

If you use this toolbox or benchmark in your research, please cite this project.

```
@misc{mmclassification,
  author =       {Yang, Lei and Li, Xiaojie and Lou, Zan and Yang, Mingmin and
                  Wang, Fei and Qian, Chen and Chen, Kai and Lin, Dahua},
  title =        {{MMClassification}},
  howpublished = {\url{https://github.com/open-mmlab/mmclassification}},
  year =         {2020}
}
```

## Contact

This repo is currently maintained by Lei Yang ([@yl-1993](http://github.com/yl-1993)), Xiaojie Li ([@xiaojieli0903](https://github.com/xiaojieli0903)) and Kai Chen ([@hellock](http://github.com/hellock)).
