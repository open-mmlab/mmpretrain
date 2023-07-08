# DeiT III: Revenge of the ViT

> [DeiT III: Revenge of the ViT](https://arxiv.org/abs/2204.07118)

<!-- [ALGORITHM] -->

## Abstract

A Vision Transformer (ViT) is a simple neural architecture amenable to serve several computer vision tasks. It has limited built-in architectural priors, in contrast to more recent architectures that incorporate priors either about the input data or of specific tasks. Recent works show that ViTs benefit from self-supervised pre-training, in particular BerT-like pre-training like BeiT. In this paper, we revisit the supervised training of ViTs. Our procedure builds upon and simplifies a recipe introduced for training ResNet-50. It includes a new simple data-augmentation procedure with only 3 augmentations, closer to the practice in self-supervised learning. Our evaluations on Image classification (ImageNet-1k with and without pre-training on ImageNet-21k), transfer learning and semantic segmentation show that our procedure outperforms by a large margin previous fully supervised training recipes for ViT. It also reveals that the performance of our ViT trained with supervision is comparable to that of more recent architectures. Our results could serve as better baselines for recent self-supervised approaches demonstrated on ViT.

<div align=center>
<img src="https://user-images.githubusercontent.com/24734142/192964480-46726469-21d9-4e45-a06a-87c6a57c3367.png" width="90%"/>
</div>

## How to use it?

<!-- [TABS-BEGIN] -->

**Predict image**

```python
from mmpretrain import inference_model

predict = inference_model('deit3-small-p16_3rdparty_in1k', 'demo/bird.JPEG')
print(predict['pred_class'])
print(predict['pred_score'])
```

**Use the model**

```python
import torch
from mmpretrain import get_model

model = get_model('deit3-small-p16_3rdparty_in1k', pretrained=True)
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
python tools/test.py configs/deit3/deit3-small-p16_64xb64_in1k.py https://download.openmmlab.com/mmclassification/v0/deit3/deit3-small-p16_3rdparty_in1k_20221008-0f7c70cf.pth
```

<!-- [TABS-END] -->

## Models and results

### Image Classification on ImageNet-1k

| Model                                             |   Pretrain   | Params (M) | Flops (G) | Top-1 (%) | Top-5 (%) |                     Config                     |                         Download                         |
| :------------------------------------------------ | :----------: | :--------: | :-------: | :-------: | :-------: | :--------------------------------------------: | :------------------------------------------------------: |
| `deit3-small-p16_3rdparty_in1k`\*                 | From scratch |   22.06    |   4.61    |   81.35   |   95.31   |    [config](deit3-small-p16_64xb64_in1k.py)    | [model](https://download.openmmlab.com/mmclassification/v0/deit3/deit3-small-p16_3rdparty_in1k_20221008-0f7c70cf.pth) |
| `deit3-small-p16_3rdparty_in1k-384px`\*           | From scratch |   22.21    |   15.52   |   83.43   |   96.68   | [config](deit3-small-p16_64xb64_in1k-384px.py) | [model](https://download.openmmlab.com/mmclassification/v0/deit3/deit3-small-p16_3rdparty_in1k-384px_20221008-a2c1a0c7.pth) |
| `deit3-small-p16_in21k-pre_3rdparty_in1k`\*       | ImageNet-21k |   22.06    |   4.61    |   83.06   |   96.77   |    [config](deit3-small-p16_64xb64_in1k.py)    | [model](https://download.openmmlab.com/mmclassification/v0/deit3/deit3-small-p16_in21k-pre_3rdparty_in1k_20221009-dcd90827.pth) |
| `deit3-small-p16_in21k-pre_3rdparty_in1k-384px`\* | ImageNet-21k |   22.21    |   15.52   |   84.84   |   97.48   | [config](deit3-small-p16_64xb64_in1k-384px.py) | [model](https://download.openmmlab.com/mmclassification/v0/deit3/deit3-small-p16_in21k-pre_3rdparty_in1k-384px_20221009-de116dd7.pth) |
| `deit3-medium-p16_3rdparty_in1k`\*                | From scratch |   38.85    |   8.00    |   82.99   |   96.22   |   [config](deit3-medium-p16_64xb64_in1k.py)    | [model](https://download.openmmlab.com/mmclassification/v0/deit3/deit3-medium-p16_3rdparty_in1k_20221008-3b21284d.pth) |
| `deit3-medium-p16_in21k-pre_3rdparty_in1k`\*      | ImageNet-21k |   38.85    |   8.00    |   84.56   |   97.19   |   [config](deit3-medium-p16_64xb64_in1k.py)    | [model](https://download.openmmlab.com/mmclassification/v0/deit3/deit3-medium-p16_in21k-pre_3rdparty_in1k_20221009-472f11e2.pth) |
| `deit3-base-p16_3rdparty_in1k`\*                  | From scratch |   86.59    |   17.58   |   83.80   |   96.55   |    [config](deit3-base-p16_64xb64_in1k.py)     | [model](https://download.openmmlab.com/mmclassification/v0/deit3/deit3-base-p16_3rdparty_in1k_20221008-60b8c8bf.pth) |
| `deit3-base-p16_3rdparty_in1k-384px`\*            | From scratch |   86.88    |   55.54   |   85.08   |   97.25   | [config](deit3-base-p16_64xb32_in1k-384px.py)  | [model](https://download.openmmlab.com/mmclassification/v0/deit3/deit3-base-p16_3rdparty_in1k-384px_20221009-e19e36d4.pth) |
| `deit3-base-p16_in21k-pre_3rdparty_in1k`\*        | ImageNet-21k |   86.59    |   17.58   |   85.70   |   97.75   |    [config](deit3-base-p16_64xb64_in1k.py)     | [model](https://download.openmmlab.com/mmclassification/v0/deit3/deit3-base-p16_in21k-pre_3rdparty_in1k_20221009-87983ca1.pth) |
| `deit3-base-p16_in21k-pre_3rdparty_in1k-384px`\*  | ImageNet-21k |   86.88    |   55.54   |   86.73   |   98.11   | [config](deit3-base-p16_64xb32_in1k-384px.py)  | [model](https://download.openmmlab.com/mmclassification/v0/deit3/deit3-base-p16_in21k-pre_3rdparty_in1k-384px_20221009-5e4e37b9.pth) |
| `deit3-large-p16_3rdparty_in1k`\*                 | From scratch |   304.37   |   61.60   |   84.87   |   97.01   |    [config](deit3-large-p16_64xb64_in1k.py)    | [model](https://download.openmmlab.com/mmclassification/v0/deit3/deit3-large-p16_3rdparty_in1k_20221009-03b427ea.pth) |
| `deit3-large-p16_3rdparty_in1k-384px`\*           | From scratch |   304.76   |  191.21   |   85.82   |   97.60   | [config](deit3-large-p16_64xb16_in1k-384px.py) | [model](https://download.openmmlab.com/mmclassification/v0/deit3/deit3-large-p16_3rdparty_in1k-384px_20221009-4317ce62.pth) |
| `deit3-large-p16_in21k-pre_3rdparty_in1k`\*       | ImageNet-21k |   304.37   |   61.60   |   86.97   |   98.24   |    [config](deit3-large-p16_64xb64_in1k.py)    | [model](https://download.openmmlab.com/mmclassification/v0/deit3/deit3-large-p16_in21k-pre_3rdparty_in1k_20221009-d8d27084.pth) |
| `deit3-large-p16_in21k-pre_3rdparty_in1k-384px`\* | ImageNet-21k |   304.76   |  191.21   |   87.73   |   98.51   | [config](deit3-large-p16_64xb16_in1k-384px.py) | [model](https://download.openmmlab.com/mmclassification/v0/deit3/deit3-large-p16_in21k-pre_3rdparty_in1k-384px_20221009-75fea03f.pth) |
| `deit3-huge-p14_3rdparty_in1k`\*                  | From scratch |   632.13   |  167.40   |   85.21   |   97.36   |    [config](deit3-huge-p14_64xb32_in1k.py)     | [model](https://download.openmmlab.com/mmclassification/v0/deit3/deit3-huge-p14_3rdparty_in1k_20221009-e107bcb7.pth) |
| `deit3-huge-p14_in21k-pre_3rdparty_in1k`\*        | ImageNet-21k |   632.13   |  167.40   |   87.19   |   98.26   |    [config](deit3-huge-p14_64xb32_in1k.py)     | [model](https://download.openmmlab.com/mmclassification/v0/deit3/deit3-huge-p14_in21k-pre_3rdparty_in1k_20221009-19b8a535.pth) |

*Models with * are converted from the [official repo](https://github.com/facebookresearch/deit/blob/main/models_v2.py#L171). The config files of these models are only for inference. We haven't reproduce the training results.*

## Citation

```bibtex
@article{Touvron2022DeiTIR,
  title={DeiT III: Revenge of the ViT},
  author={Hugo Touvron and Matthieu Cord and Herve Jegou},
  journal={arXiv preprint arXiv:2204.07118},
  year={2022},
}
```
