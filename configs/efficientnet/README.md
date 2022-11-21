# EfficientNet

> [Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946v5)

<!-- [ALGORITHM] -->

## Introduction

EfficientNets are a family of image classification models, which achieve state-of-the-art accuracy, yet being an order-of-magnitude smaller and faster than previous models.

EfficientNets are based on AutoML and Compound Scaling. In particular, we first use [AutoML MNAS Mobile framework](https://ai.googleblog.com/2018/08/mnasnet-towards-automating-design-of.html) to develop a mobile-size baseline network, named as EfficientNet-B0; Then, we use the compound scaling method to scale up this baseline to obtain EfficientNet-B1 to B7.

<div align=center>
<img src="https://user-images.githubusercontent.com/26739999/150078232-d28c91fc-d0e8-43e3-9d20-b5162f0fb463.png" width="60%"/>
</div>

## Abstract

<details>

<summary>Click to show the detailed Abstract</summary>

<br>
Convolutional Neural Networks (ConvNets) are commonly developed at a fixed resource budget, and then scaled up for better accuracy if more resources are available. In this paper, we systematically study model scaling and identify that carefully balancing network depth, width, and resolution can lead to better performance. Based on this observation, we propose a new scaling method that uniformly scales all dimensions of depth/width/resolution using a simple yet highly effective compound coefficient. We demonstrate the effectiveness of this method on scaling up MobileNets and ResNet.   To go even further, we use neural architecture search to design a new baseline network and scale it up to obtain a family of models, called EfficientNets, which achieve much better accuracy and efficiency than previous ConvNets. In particular, our EfficientNet-B7 achieves state-of-the-art 84.3% top-1 accuracy on ImageNet, while being 8.4x smaller and 6.1x faster on inference than the best existing ConvNet. Our EfficientNets also transfer well and achieve state-of-the-art accuracy on CIFAR-100 (91.7%), Flowers (98.8%), and 3 other transfer learning datasets, with an order of magnitude fewer parameters.

</details>

## Results and models

### ImageNet-1k

In the result table, AA means trained with AutoAugment pre-processing, more details can be found in the [paper](https://arxiv.org/abs/1805.09501); AdvProp is a method to train with adversarial examples, more details can be found in the [paper](https://arxiv.org/abs/1911.09665); RA means trained with RandAugment pre-processing, more details can be found in the [paper](https://arxiv.org/abs/1909.13719); NoisyStudent means trained with extra JFT-300M unlabeled data, more details can be found in the [paper](https://arxiv.org/abs/1911.04252); L2-475 means the same L2 architecture with input image size 475.

Note: In MMClassification, we support training with AutoAugment, don't support AdvProp by now.

|                   Model                   | Params(M) | Flops(G) | Top-1 (%) | Top-5 (%) |                      Config                      |                                   Download                                   |
| :---------------------------------------: | :-------: | :------: | :-------: | :-------: | :----------------------------------------------: | :--------------------------------------------------------------------------: |
|             EfficientNet-B0\*             |   5.29    |   0.42   |   76.74   |   93.17   |    [config](./efficientnet-b0_8xb32_in1k.py)     | [model](https://download.openmmlab.com/mmclassification/v0/efficientnet/efficientnet-b0_3rdparty_8xb32_in1k_20220119-a7e2a0b1.pth) |
|          EfficientNet-B0 (AA)\*           |   5.29    |   0.42   |   77.26   |   93.41   |    [config](./efficientnet-b0_8xb32_in1k.py)     | [model](https://download.openmmlab.com/mmclassification/v0/efficientnet/efficientnet-b0_3rdparty_8xb32-aa_in1k_20220119-8d939117.pth) |
|     EfficientNet-B0 (AA + AdvProp)\*      |   5.29    |   0.42   |   77.53   |   93.61   | [config](./efficientnet-b0_8xb32-01norm_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/efficientnet/efficientnet-b0_3rdparty_8xb32-aa-advprop_in1k_20220119-26434485.pth) |
|   EfficientNet-B0 (RA + NoisyStudent)\*   |   5.29    |   0.42   |   77.63   |   94.00   |    [config](./efficientnet-b0_8xb32_in1k.py)     | [model](https://download.openmmlab.com/mmclassification/v0/efficientnet/efficientnet-b0_3rdparty-ra-noisystudent_in1k_20221103-75cd08d3.pth) |
|             EfficientNet-B1\*             |   7.79    |   0.74   |   78.68   |   94.28   |    [config](./efficientnet-b1_8xb32_in1k.py)     | [model](https://download.openmmlab.com/mmclassification/v0/efficientnet/efficientnet-b1_3rdparty_8xb32_in1k_20220119-002556d9.pth) |
|          EfficientNet-B1 (AA)\*           |   7.79    |   0.74   |   79.20   |   94.42   |    [config](./efficientnet-b1_8xb32_in1k.py)     | [model](https://download.openmmlab.com/mmclassification/v0/efficientnet/efficientnet-b1_3rdparty_8xb32-aa_in1k_20220119-619d8ae3.pth) |
|     EfficientNet-B1 (AA + AdvProp)\*      |   7.79    |   0.74   |   79.52   |   94.43   | [config](./efficientnet-b1_8xb32-01norm_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/efficientnet/efficientnet-b1_3rdparty_8xb32-aa-advprop_in1k_20220119-5715267d.pth) |
|   EfficientNet-B1 (RA + NoisyStudent)\*   |   7.79    |   0.74   |   81.44   |   95.83   |    [config](./efficientnet-b1_8xb32_in1k.py)     | [model](https://download.openmmlab.com/mmclassification/v0/efficientnet/efficientnet-b1_3rdparty-ra-noisystudent_in1k_20221103-756bcbc0.pth) |
|             EfficientNet-B2\*             |   9.11    |   1.07   |   79.64   |   94.80   |    [config](./efficientnet-b2_8xb32_in1k.py)     | [model](https://download.openmmlab.com/mmclassification/v0/efficientnet/efficientnet-b2_3rdparty_8xb32_in1k_20220119-ea374a30.pth) |
|          EfficientNet-B2 (AA)\*           |   9.11    |   1.07   |   80.21   |   94.96   |    [config](./efficientnet-b2_8xb32_in1k.py)     | [model](https://download.openmmlab.com/mmclassification/v0/efficientnet/efficientnet-b2_3rdparty_8xb32-aa_in1k_20220119-dd61e80b.pth) |
|     EfficientNet-B2 (AA + AdvProp)\*      |   9.11    |   1.07   |   80.45   |   95.07   | [config](./efficientnet-b2_8xb32-01norm_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/efficientnet/efficientnet-b2_3rdparty_8xb32-aa-advprop_in1k_20220119-1655338a.pth) |
|   EfficientNet-B2 (RA + NoisyStudent)\*   |   9.11    |   1.07   |   82.47   |   96.23   |    [config](./efficientnet-b2_8xb32_in1k.py)     | [model](https://download.openmmlab.com/mmclassification/v0/efficientnet/efficientnet-b1_3rdparty-ra-noisystudent_in1k_20221103-756bcbc0.pth) |
|             EfficientNet-B3\*             |   12.23   |   1.07   |   81.01   |   95.34   |    [config](./efficientnet-b3_8xb32_in1k.py)     | [model](https://download.openmmlab.com/mmclassification/v0/efficientnet/efficientnet-b3_3rdparty_8xb32_in1k_20220119-4b4d7487.pth) |
|          EfficientNet-B3 (AA)\*           |   12.23   |   1.07   |   81.58   |   95.67   |    [config](./efficientnet-b3_8xb32_in1k.py)     | [model](https://download.openmmlab.com/mmclassification/v0/efficientnet/efficientnet-b3_3rdparty_8xb32-aa_in1k_20220119-5b4887a0.pth) |
|     EfficientNet-B3 (AA + AdvProp)\*      |   12.23   |   1.07   |   81.81   |   95.69   | [config](./efficientnet-b3_8xb32-01norm_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/efficientnet/efficientnet-b3_3rdparty_8xb32-aa-advprop_in1k_20220119-53b41118.pth) |
|   EfficientNet-B3 (RA + NoisyStudent)\*   |   12.23   |   1.07   |   84.02   |   96.89   |    [config](./efficientnet-b3_8xb32_in1k.py)     | [model](https://download.openmmlab.com/mmclassification/v0/efficientnet/efficientnet-b3_3rdparty-ra-noisystudent_in1k_20221103-a4ab5fd6.pth) |
|             EfficientNet-B4\*             |   19.34   |   1.95   |   82.57   |   96.09   |    [config](./efficientnet-b4_8xb32_in1k.py)     | [model](https://download.openmmlab.com/mmclassification/v0/efficientnet/efficientnet-b4_3rdparty_8xb32_in1k_20220119-81fd4077.pth) |
|          EfficientNet-B4 (AA)\*           |   19.34   |   1.95   |   82.95   |   96.26   |    [config](./efficientnet-b4_8xb32_in1k.py)     | [model](https://download.openmmlab.com/mmclassification/v0/efficientnet/efficientnet-b4_3rdparty_8xb32-aa_in1k_20220119-45b8bd2b.pth) |
|     EfficientNet-B4 (AA + AdvProp)\*      |   19.34   |   1.95   |   83.25   |   96.44   | [config](./efficientnet-b4_8xb32-01norm_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/efficientnet/efficientnet-b4_3rdparty_8xb32-aa-advprop_in1k_20220119-38c2238c.pth) |
|   EfficientNet-B4 (RA + NoisyStudent)\*   |   19.34   |   1.95   |   85.25   |   97.52   |    [config](./efficientnet-b4_8xb32_in1k.py)     | [model](https://download.openmmlab.com/mmclassification/v0/efficientnet/efficientnet-b4_3rdparty-ra-noisystudent_in1k_20221103-16ba8a2d.pth) |
|             EfficientNet-B5\*             |   30.39   |   10.1   |   83.18   |   96.47   |    [config](./efficientnet-b5_8xb32_in1k.py)     | [model](https://download.openmmlab.com/mmclassification/v0/efficientnet/efficientnet-b5_3rdparty_8xb32_in1k_20220119-e9814430.pth) |
|          EfficientNet-B5 (AA)\*           |   30.39   |   10.1   |   83.82   |   96.76   |    [config](./efficientnet-b5_8xb32_in1k.py)     | [model](https://download.openmmlab.com/mmclassification/v0/efficientnet/efficientnet-b5_3rdparty_8xb32-aa_in1k_20220119-2cab8b78.pth) |
|     EfficientNet-B5 (AA + AdvProp)\*      |   30.39   |   10.1   |   84.21   |   96.98   | [config](./efficientnet-b5_8xb32-01norm_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/efficientnet/efficientnet-b5_3rdparty_8xb32-aa-advprop_in1k_20220119-f57a895a.pth) |
|   EfficientNet-B5 (RA + NoisyStudent)\*   |   30.39   |   10.1   |   86.08   |   97.75   |    [config](./efficientnet-b5_8xb32_in1k.py)     | [model](https://download.openmmlab.com/mmclassification/v0/efficientnet/efficientnet-b5_3rdparty-ra-noisystudent_in1k_20221103-111a185f.pth) |
|          EfficientNet-B6 (AA)\*           |   43.04   |   20.0   |   84.05   |   96.82   |    [config](./efficientnet-b6_8xb32_in1k.py)     | [model](https://download.openmmlab.com/mmclassification/v0/efficientnet/efficientnet-b6_3rdparty_8xb32-aa_in1k_20220119-45b03310.pth) |
|     EfficientNet-B6 (AA + AdvProp)\*      |   43.04   |   20.0   |   84.74   |   97.14   | [config](./efficientnet-b6_8xb32-01norm_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/efficientnet/efficientnet-b6_3rdparty_8xb32-aa-advprop_in1k_20220119-bfe3485e.pth) |
|   EfficientNet-B6 (RA + NoisyStudent)\*   |   43.04   |   20.0   |   86.47   |   97.87   |    [config](./efficientnet-b6_8xb32_in1k.py)     | [model](https://download.openmmlab.com/mmclassification/v0/efficientnet/efficientnet-b6_3rdparty-ra-noisystudent_in1k_20221103-7de7d2cc.pth) |
|          EfficientNet-B7 (AA)\*           |   66.35   |   39.3   |   84.38   |   96.88   |    [config](./efficientnet-b7_8xb32_in1k.py)     | [model](https://download.openmmlab.com/mmclassification/v0/efficientnet/efficientnet-b7_3rdparty_8xb32-aa_in1k_20220119-bf03951c.pth) |
|     EfficientNet-B7 (AA + AdvProp)\*      |   66.35   |   39.3   |   85.14   |   97.23   | [config](./efficientnet-b7_8xb32-01norm_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/efficientnet/efficientnet-b7_3rdparty_8xb32-aa-advprop_in1k_20220119-c6dbff10.pth) |
|   EfficientNet-B7 (RA + NoisyStudent)\*   |   66.35   |   65.0   |   86.83   |   98.08   |    [config](./efficientnet-b7_8xb32_in1k.py)     | [model](https://download.openmmlab.com/mmclassification/v0/efficientnet/efficientnet-b7_3rdparty-ra-noisystudent_in1k_20221103-a82894bc.pth) |
|     EfficientNet-B8 (AA + AdvProp)\*      |   87.41   |   65.0   |   85.38   |   97.28   | [config](./efficientnet-b8_8xb32-01norm_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/efficientnet/efficientnet-b8_3rdparty_8xb32-aa-advprop_in1k_20220119-297ce1b7.pth) |
| EfficientNet-L2-475 (RA + NoisyStudent)\* |  480.30   |  174.20  |   88.18   |   98.55   | [config](./efficientnet-l2_8xb32_in1k-475px.py)  | [model](https://download.openmmlab.com/mmclassification/v0/efficientnet/efficientnet-l2_3rdparty-ra-noisystudent_in1k-475px_20221103-5a0d8058.pth) |
|   EfficientNet-L2 (RA + NoisyStudent)\*   |  480.30   |  484.98  |   88.33   |   98.65   |  [config](./efficientnet-l2_8xb8_in1k-800px.py)  | [model](https://download.openmmlab.com/mmclassification/v0/efficientnet/efficientnet-l2_3rdparty-ra-noisystudent_in1k_20221103-be73be13.pth) |

*Models with * are converted from the [official repo](https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet). The config files of these models
are only for inference. We don't ensure these config files' training accuracy and welcome you to contribute your reproduction results.*

## How to use it?

<!-- [TABS-BEGIN] -->

**Predict image**

```python
>>> import torch
>>> from mmcls.apis import init_model, inference_model
>>>
>>> model = init_model('configs/efficientnet/efficientnet-b0_8xb32_in1k.py', "https://download.openmmlab.com/mmclassification/v0/efficientnet/efficientnet-b0_3rdparty_8xb32_in1k_20220119-a7e2a0b1.pth")
>>> predict = inference_model(model, 'demo/demo.JPEG')
>>> print(predict['pred_class'])
sea snake
>>> print(predict['pred_score'])
0.6968820691108704
```

**Use the model**

```python
>>> import torch
>>> from mmcls.apis import init_model
>>>
>>> model = init_model('configs/efficientnet/efficientnet-b0_8xb32_in1k.py', "https://download.openmmlab.com/mmclassification/v0/efficientnet/efficientnet-b0_3rdparty_8xb32_in1k_20220119-a7e2a0b1.pth")
>>> inputs = torch.rand(1, 3, 224, 224).to(model.data_preprocessor.device)
>>> # To get classification scores.
>>> out = model(inputs)
>>> print(out.shape)
torch.Size([1, 1000])
>>> # To extract features.
>>> outs = model.extract_feat(inputs)
>>> print(outs[0].shape)
torch.Size([1, 1280])
```

**Train/Test Command**

Place the ImageNet dataset to the `data/imagenet/` directory, or prepare datasets according to the [docs](https://mmclassification.readthedocs.io/en/1.x/user_guides/dataset_prepare.html#prepare-dataset).

Train:

```shell
python tools/train.py configs/efficientnet/efficientnet-b0_8xb32_in1k.py
```

Test:

```shell
python tools/test.py configs/efficientnet/efficientnet-b0_8xb32_in1k.py https://download.openmmlab.com/mmclassification/v0/efficientnet/efficientnet-b0_3rdparty_8xb32_in1k_20220119-a7e2a0b1.pth
```

<!-- [TABS-END] -->

For more configurable parameters, please refer to the [API](https://mmclassification.readthedocs.io/en/1.x/api/generated/mmcls.models.backbones.EfficientNet.html#mmcls.models.backbones.EfficientNet).

## Citation

```
@inproceedings{tan2019efficientnet,
  title={Efficientnet: Rethinking model scaling for convolutional neural networks},
  author={Tan, Mingxing and Le, Quoc},
  booktitle={International Conference on Machine Learning},
  pages={6105--6114},
  year={2019},
  organization={PMLR}
}
```
