# MobileOne

> [An Improved One millisecond Mobile Backbone](https://arxiv.org/abs/2206.04040)

<!-- [ALGORITHM] -->

## Introduction

Mobileone is proposed by apple and based on reparameterization. On the apple chips, the accuracy of the model is close to 0.76 on the ImageNet dataset when the latency is less than 1ms. Its main improvements based on [RepVGG](../repvgg) are fllowing:

- Reparameterization using Depthwise convolution and Pointwise convolution instead of normal convolution.
- Removal of the residual structure which is not friendly to access memory.

<div align=center>
<img src="https://user-images.githubusercontent.com/18586273/183552452-74657532-f461-48f7-9aa7-c23f006cdb07.png" width="40%"/>
</div>

## Abstract

<details>

<summary>Show the paper's abstract</summary>

<br>
Efficient neural network backbones for mobile devices are often optimized for metrics such as FLOPs or parameter count. However, these metrics may not correlate well with latency of the network when deployed on a mobile device. Therefore, we perform extensive analysis of different metrics by deploying several mobile-friendly networks on a mobile device. We identify and analyze architectural and optimization bottlenecks in recent efficient neural networks and provide ways to mitigate these bottlenecks. To this end, we design an efficient backbone MobileOne, with variants achieving an inference time under 1 ms on an iPhone12 with 75.9% top-1 accuracy on ImageNet. We show that MobileOne achieves state-of-the-art performance within the efficient architectures while being many times faster on mobile. Our best model obtains similar performance on ImageNet as MobileFormer while being 38x faster. Our model obtains 2.3% better top-1 accuracy on ImageNet than EfficientNet at similar latency. Furthermore, we show that our model generalizes to multiple tasks - image classification, object detection, and semantic segmentation with significant improvements in latency and accuracy as compared to existing efficient architectures when deployed on a mobile device.
</br>

</details>

## How to use

The checkpoints provided are all `training-time` models. Use the reparameterize tool or `switch_to_deploy` interface to switch them to more efficient `inference-time` architecture, which not only has fewer parameters but also less calculations.

<!-- [TABS-BEGIN] -->

**Predict image**

Use `classifier.backbone.switch_to_deploy()` interface to switch the MobileOne to a inference mode.

```python
>>> import torch
>>> from mmcls.apis import init_model, inference_model
>>>
>>> model = init_model('configs/mobileone/mobileone-s0_8xb32_in1k.py', 'https://download.openmmlab.com/mmclassification/v0/mobileone/mobileone-s0_8xb32_in1k_20221110-0bc94952.pth')
>>> predict = inference_model(model, 'demo/demo.JPEG')
>>> print(predict['pred_class'])
sea snake
>>> print(predict['pred_score'])
0.4539405107498169
>>>
>>> # switch to deploy mode
>>> model.backbone.switch_to_deploy()
>>> predict_deploy = inference_model(model, 'demo/demo.JPEG')
>>> print(predict_deploy['pred_class'])
sea snake
>>> print(predict_deploy['pred_score'])
0.4539395272731781
```

**Use the model**

```python
>>> import torch
>>> from mmcls.apis import init_model
>>>
>>> model = init_model('configs/mobileone/mobileone-s0_8xb32_in1k.py', 'https://download.openmmlab.com/mmclassification/v0/mobileone/mobileone-s0_8xb32_in1k_20221110-0bc94952.pth')
>>> inputs = torch.rand(1, 3, 224, 224).to(model.data_preprocessor.device)
>>> # To get classification scores.
>>> out = model(inputs)
>>> print(out.shape)
torch.Size([1, 1000])
>>> # To extract features.
>>> outs = model.extract_feat(inputs)
>>> print(outs[0].shape)
torch.Size([1, 768])
>>>
>>> # switch to deploy mode
>>> model.backbone.switch_to_deploy()
>>> out_deploy = model(inputs)
>>> print(out.shape)
torch.Size([1, 1000])
>>> assert torch.allclose(out, out_deploy) # pass without error
```

**Train/Test Command**

Place the ImageNet dataset to the `data/imagenet/` directory, or prepare datasets according to the [docs](https://mmclassification.readthedocs.io/en/1.x/user_guides/dataset_prepare.html#prepare-dataset).

Train:

```shell
python tools/train.py configs/mobileone/mobileone-s0_8xb32_in1k.py
```

Download Checkpoint:

```shell
wget https://download.openmmlab.com/mmclassification/v0/mobileone/mobileone-s0_8xb32_in1k_20221110-0bc94952.pth
```

Test use unfused model:

```shell
python tools/test.py configs/mobileone/mobileone-s0_8xb32_in1k.py mobileone-s0_8xb32_in1k_20221110-0bc94952.pth
```

Reparameterize checkpoint:

```shell
python ./tools/convert_models/reparameterize_model.py ./configs/mobileone/mobileone-s0_8xb32_in1k.py mobileone-s0_8xb32_in1k_20221110-0bc94952.pth mobileone_s0_deploy.pth
```

Test use fused model:

```shell
python tools/test.py configs/mobileone/deploy/mobileone-s0_deploy_8xb32_in1k.py mobileone_s0_deploy.pth
```

<!-- [TABS-END] -->

### Reparameterize Tool

Use provided tool to reparameterize the given model and save the checkpoint:

```bash
python tools/convert_models/reparameterize_model.py ${CFG_PATH} ${SRC_CKPT_PATH} ${TARGET_CKPT_PATH}
```

`${CFG_PATH}` is the config file path, `${SRC_CKPT_PATH}` is the source chenpoint file path, `${TARGET_CKPT_PATH}` is the target deploy weight file path.

For example:

```shell
wget https://download.openmmlab.com/mmclassification/v0/mobileone/mobileone-s0_8xb32_in1k_20221110-0bc94952.pth
python ./tools/convert_models/reparameterize_model.py ./configs/mobileone/mobileone-s0_8xb32_in1k.py mobileone-s0_8xb32_in1k_20221110-0bc94952.pth mobileone_s0_deploy.pth
```

To use reparameterized weights, the config file must switch to [**the deploy config files**](./deploy/).

```bash
python tools/test.py ${Deploy_CFG} ${Deploy_Checkpoint}
```

For example of using the reparameterized weights above:

```shell
python ./tools/test.py ./configs/mobileone/deploy/mobileone-s0_deploy_8xb32_in1k.py  mobileone_s0_deploy.pth
```

For more configurable parameters, please refer to the [API](https://mmclassification.readthedocs.io/en/1.x/api/generated/mmcls.models.backbones.MobileOne.html#mmcls.models.backbones.MobileOne).

## Results and models

### ImageNet-1k

|    Model     |            Params(M)            |            Flops(G)            | Top-1 (%) | Top-5 (%) |                        Config                         |                         Download                         |
| :----------: | :-----------------------------: | :----------------------------: | :-------: | :-------: | :---------------------------------------------------: | :------------------------------------------------------: |
| MobileOne-s0 |  5.29（train) \| 2.08 (deploy)  | 1.09 (train) \| 0.28 (deploy)  |   71.34   |   89.87   | [config (train)](./mobileone-s0_8xb32_in1k.py) \| [config (deploy)](./deploy/mobileone-s0_deploy_8xb32_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/mobileone/mobileone-s0_8xb32_in1k_20221110-0bc94952.pth) \| [log](https://download.openmmlab.com/mmclassification/v0/mobileone/mobileone-s0_8xb32_in1k_20221110-0bc94952.json) |
| MobileOne-s1 |  4.83 (train) \| 4.76 (deploy)  | 0.86 (train) \| 0.84 (deploy)  |   75.72   |   92.54   | [config (train)](./mobileone-s1_8xb32_in1k.py) \| [config (deploy)](./deploy/mobileone-s1_deploy_8xb32_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/mobileone/mobileone-s1_8xb32_in1k_20221110-ceeef467.pth)  \| [log](https://download.openmmlab.com/mmclassification/v0/mobileone/mobileone-s1_8xb32_in1k_20221110-ceeef467.json) |
| MobileOne-s2 |  7.88 (train) \| 7.88 (deploy)  | 1.34 (train)  \| 1.31 (deploy) |   77.37   |   93.34   | [config (train)](./mobileone-s2_8xb32_in1k.py) \|[config (deploy)](./deploy/mobileone-s2_deploy_8xb32_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/mobileone/mobileone-s2_8xb32_in1k_20221110-9c7ecb97.pth)  \| [log](https://download.openmmlab.com/mmclassification/v0/mobileone/mobileone-s2_8xb32_in1k_20221110-9c7ecb97.json) |
| MobileOne-s3 | 10.17 (train) \| 10.08 (deploy) | 1.95 (train)  \| 1.91 (deploy) |   78.06   |   93.83   | [config (train)](./mobileone-s3_8xb32_in1k.py) \|[config (deploy)](./deploy/mobileone-s3_deploy_8xb32_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/mobileone/mobileone-s3_8xb32_in1k_20221110-c95eb3bf.pth)  \| [log](https://download.openmmlab.com/mmclassification/v0/mobileone/mobileone-s3_8xb32_in1k_20221110-c95eb3bf.pth) |
| MobileOne-s4 | 14.95 (train) \| 14.84 (deploy) | 3.05 (train) \| 3.00 (deploy)  |   79.69   |   94.46   | [config (train)](./mobileone-s4_8xb32_in1k.py) \|[config (deploy)](./deploy/mobileone-s4_deploy_8xb32_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/mobileone/mobileone-s4_8xb32_in1k_20221110-28d888cb.pth) \| [log](https://download.openmmlab.com/mmclassification/v0/mobileone/mobileone-s4_8xb32_in1k_20221110-28d888cb.pth) |

## Citation

```bibtex
@article{mobileone2022,
  title={An Improved One millisecond Mobile Backbone},
  author={Vasu, Pavan Kumar Anasosalu and Gabriel, James and Zhu, Jeff and Tuzel, Oncel and Ranjan, Anurag},
  journal={arXiv preprint arXiv:2206.04040},
  year={2022}
}
```
