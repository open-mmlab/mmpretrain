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

## How to use it?

<!-- [TABS-BEGIN] -->

**Predict image**

```python
from mmpretrain import inference_model

predict = inference_model('mobileone-s0_8xb32_in1k', 'demo/bird.JPEG')
print(predict['pred_class'])
print(predict['pred_score'])
```

**Use the model**

```python
import torch
from mmpretrain import get_model

model = get_model('mobileone-s0_8xb32_in1k', pretrained=True)
inputs = torch.rand(1, 3, 224, 224)
out = model(inputs)
print(type(out))
# To extract features.
feats = model.extract_feat(inputs)
print(type(feats))
```

**Train/Test Command**

Prepare your dataset according to the [docs](https://mmpretrain.readthedocs.io/en/latest/user_guides/dataset_prepare.html#prepare-dataset).

Train:

```shell
python tools/train.py configs/mobileone/mobileone-s0_8xb32_in1k.py
```

Test:

```shell
python tools/test.py configs/mobileone/mobileone-s0_8xb32_in1k.py https://download.openmmlab.com/mmclassification/v0/mobileone/mobileone-s0_8xb32_in1k_20221110-0bc94952.pth
```

<!-- [TABS-END] -->

## Models and results

### Image Classification on ImageNet-1k

| Model                     |   Pretrain   | Params (M) | Flops (G) | Top-1 (%) | Top-5 (%) |                Config                |                                          Download                                          |
| :------------------------ | :----------: | :--------: | :-------: | :-------: | :-------: | :----------------------------------: | :----------------------------------------------------------------------------------------: |
| `mobileone-s0_8xb32_in1k` | From scratch |    2.08    |   0.27    |   71.34   |   89.87   | [config](mobileone-s0_8xb32_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/mobileone/mobileone-s0_8xb32_in1k_20221110-0bc94952.pth) \| [log](https://download.openmmlab.com/mmclassification/v0/mobileone/mobileone-s0_8xb32_in1k_20221110-0bc94952.json) |
| `mobileone-s1_8xb32_in1k` | From scratch |    4.76    |   0.82    |   75.72   |   92.54   | [config](mobileone-s1_8xb32_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/mobileone/mobileone-s1_8xb32_in1k_20221110-ceeef467.pth) \| [log](https://download.openmmlab.com/mmclassification/v0/mobileone/mobileone-s1_8xb32_in1k_20221110-ceeef467.json) |
| `mobileone-s2_8xb32_in1k` | From scratch |    7.81    |   1.30    |   77.37   |   93.34   | [config](mobileone-s2_8xb32_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/mobileone/mobileone-s2_8xb32_in1k_20221110-9c7ecb97.pth) \| [log](https://download.openmmlab.com/mmclassification/v0/mobileone/mobileone-s2_8xb32_in1k_20221110-9c7ecb97.json) |
| `mobileone-s3_8xb32_in1k` | From scratch |   10.08    |   1.89    |   78.06   |   93.83   | [config](mobileone-s3_8xb32_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/mobileone/mobileone-s3_8xb32_in1k_20221110-c95eb3bf.pth) \| [log](https://download.openmmlab.com/mmclassification/v0/mobileone/mobileone-s3_8xb32_in1k_20221110-c95eb3bf.json) |
| `mobileone-s4_8xb32_in1k` | From scratch |   14.84    |   2.98    |   79.69   |   94.46   | [config](mobileone-s4_8xb32_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/mobileone/mobileone-s4_8xb32_in1k_20221110-28d888cb.pth) \| [log](https://download.openmmlab.com/mmclassification/v0/mobileone/mobileone-s4_8xb32_in1k_20221110-28d888cb.json) |

## Citation

```bibtex
@article{mobileone2022,
  title={An Improved One millisecond Mobile Backbone},
  author={Vasu, Pavan Kumar Anasosalu and Gabriel, James and Zhu, Jeff and Tuzel, Oncel and Ranjan, Anurag},
  journal={arXiv preprint arXiv:2206.04040},
  year={2022}
}
```
