# Wide-ResNet

> [Wide Residual Networks](https://arxiv.org/abs/1605.07146)

<!-- [ALGORITHM] -->

## Abstract

Deep residual networks were shown to be able to scale up to thousands of layers and still have improving performance. However, each fraction of a percent of improved accuracy costs nearly doubling the number of layers, and so training very deep residual networks has a problem of diminishing feature reuse, which makes these networks very slow to train. To tackle these problems, in this paper we conduct a detailed experimental study on the architecture of ResNet blocks, based on which we propose a novel architecture where we decrease depth and increase width of residual networks. We call the resulting network structures wide residual networks (WRNs) and show that these are far superior over their commonly used thin and very deep counterparts. For example, we demonstrate that even a simple 16-layer-deep wide residual network outperforms in accuracy and efficiency all previous deep residual networks, including thousand-layer-deep networks, achieving new state-of-the-art results on CIFAR, SVHN, COCO, and significant improvements on ImageNet.

<div align=center>
<img src="https://user-images.githubusercontent.com/26739999/156701329-2c7ec7bc-23da-401b-86bf-dea8567ccee8.png" width="90%"/>
</div>

## How to use it?

<!-- [TABS-BEGIN] -->

**Predict image**

```python
from mmpretrain import inference_model

predict = inference_model('wide-resnet50_3rdparty_8xb32_in1k', 'demo/bird.JPEG')
print(predict['pred_class'])
print(predict['pred_score'])
```

**Use the model**

```python
import torch
from mmpretrain import get_model

model = get_model('wide-resnet50_3rdparty_8xb32_in1k', pretrained=True)
inputs = torch.rand(1, 3, 224, 224)
out = model(inputs)
print(type(out))
# To extract features.
feats = model.extract_feat(inputs)
print(type(feats))
```

**Test Command**

Prepare your dataset according to the [docs](https://mmpretrain.readthedocs.io/en/latest/user_guides/dataset_prepare.html#prepare-dataset).

Test:

```shell
python tools/test.py configs/wrn/wide-resnet50_8xb32_in1k.py https://download.openmmlab.com/mmclassification/v0/wrn/wide-resnet50_3rdparty_8xb32_in1k_20220304-66678344.pth
```

<!-- [TABS-END] -->

## Models and results

### Image Classification on ImageNet-1k

| Model                                      |   Pretrain   | Params (M) | Flops (G) | Top-1 (%) | Top-5 (%) |                   Config                   |                              Download                               |
| :----------------------------------------- | :----------: | :--------: | :-------: | :-------: | :-------: | :----------------------------------------: | :-----------------------------------------------------------------: |
| `wide-resnet50_3rdparty_8xb32_in1k`\*      | From scratch |   68.88    |   11.44   |   78.48   |   94.08   |   [config](wide-resnet50_8xb32_in1k.py)    | [model](https://download.openmmlab.com/mmclassification/v0/wrn/wide-resnet50_3rdparty_8xb32_in1k_20220304-66678344.pth) |
| `wide-resnet101_3rdparty_8xb32_in1k`\*     | From scratch |   126.89   |   22.81   |   78.84   |   94.28   |   [config](wide-resnet101_8xb32_in1k.py)   | [model](https://download.openmmlab.com/mmclassification/v0/wrn/wide-resnet101_3rdparty_8xb32_in1k_20220304-8d5f9d61.pth) |
| `wide-resnet50_3rdparty-timm_8xb32_in1k`\* | From scratch |   68.88    |   11.44   |   81.45   |   95.53   | [config](wide-resnet50_timm_8xb32_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/wrn/wide-resnet50_3rdparty-timm_8xb32_in1k_20220304-83ae4399.pth) |

*Models with * are converted from the [timm](https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/resnet.py). The config files of these models are only for inference. We haven't reproduce the training results.*

## Citation

```bibtex
@INPROCEEDINGS{Zagoruyko2016WRN,
    author = {Sergey Zagoruyko and Nikos Komodakis},
    title = {Wide Residual Networks},
    booktitle = {BMVC},
    year = {2016}}
```
