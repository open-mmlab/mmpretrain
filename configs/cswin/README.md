# CSWin Transformer

> [CSWin Transformer: A General Vision Transformer Backbone with Cross-Shaped Windows](https://https://arxiv.org/pdf/2107.00652.pdf)

<!-- [ALGORITHM] -->

## Introduction

**CSWin Transformer** (the name `CSWin` stands for **C**ross-**S**haped **Win**dow) is introduced in [arxiv](https://arxiv.org/abs/2107.00652), which is a new general-purpose backbone for computer vision. It is a hierarchical Transformer and replaces the traditional full attention with our newly proposed cross-shaped window self-attention. The cross-shaped window self-attention mechanism computes self-attention in the horizontal and vertical stripes in parallel that from a cross-shaped window, with each stripe obtained by splitting the input feature into stripes of equal width. With CSWin, we could realize global attention with a limited computation cost.

CSWin Transformer achieves strong performance on ImageNet classification (87.5 on val with only 97G flops) and ADE20K semantic segmentation (`55.7 mIoU` on val), surpassing previous models by a large margin.

<div align=center>
<img src="https://github.com/microsoft/CSWin-Transformer/blob/main/teaser.png" width="90%"/>
</div>

## Abstract

<details>

<summary>Show the paper's abstract</summary>

<br>
We present CSWin Transformer, an efficient and effective Transformer-based backbone for general-purpose vision tasks. A challenging issue in Transformer design is that global self-attention is very expensive to compute whereas local self-attention often limits the field of interactions of each token. To address this issue, we develop the CrossShaped Window self-attention mechanism for computing self-attention in the horizontal and vertical stripes in parallel that form a cross-shaped window, with each stripe obtained by splitting the input feature into stripes of equal width. We provide a mathematical analysis of the effect of the stripe width and vary the stripe width for different layers of the Transformer network which achieves strong modeling capability while limiting the computation cost. We also introduce Locally-enhanced Positional Encoding (LePE), which handles the local positional information better than existing encoding schemes. LePE naturally supports arbitrary input resolutions, and is thus especially effective and friendly for downstream tasks. Incorporated with these designs and a hierarchical structure, CSWin Transformer demonstrates competitive performance on common vision tasks. Specifically, it achieves 85.4% Top-1 accuracy on ImageNet-1K without any extra training data or label, 53.9 box AP and 46.4 mask AP on the COCO detection task, and 52.2 mIOU on the ADE20K semantic segmentation task, surpassing previous state-of-the-art Swin Transformer backbone by +1.2, +2.0, +1.4, and +2.0 respectively under the similar FLOPs setting. By further pretraining on the larger dataset ImageNet-21K, we achieve 87.5% Top-1 accuracy on ImageNet-1K and high segmentation performance on ADE20K with 55.7 mIoU.
</br>

</details>

## How to use it?

<!-- [TABS-BEGIN] -->

**Predict image**

```python
>>> import torch
>>> from mmcls.apis import init_model, inference_model
>>>
>>> model = init_model('configs/cswin/cswin-tiny_b64_in1k.py', '')
>>> predict = inference_model(model, 'demo/demo.JPEG')
>>> print(predict['pred_class'])
sea snake
>>> print(predict['pred_score'])
0.8974384665489197
```

**Use the model**

```python
>>> import torch
>>> from mmcls.apis import init_model
>>>
>>> model = init_model('configs/cswin/cswin-base_b64_in1k.py', '')
>>> inputs = torch.rand(1, 3, 224, 224).to(model.data_preprocessor.device)
>>> # To get classification scores.
>>> out = model(inputs)
>>> print(out.shape)
torch.Size([1, 1000])
>>> # To extract features.
>>> outs = model.extract_feat(inputs)
>>> print(outs[0].shape)
torch.Size([1, 768])
```

**Train/Test Command**

Place the ImageNet dataset to the `data/imagenet/` directory, or prepare datasets according to the [docs](https://mmclassification.readthedocs.io/en/1.x/user_guides/dataset_prepare.html#prepare-dataset).

Train:

```shell
python tools/train.py configs/cswin/cswin-base_b64_in1k.py ""
```

Test:

```shell
python tools/test.py configs/cswin/cswin-base_b64_in1k.py ""
```

<!-- [TABS-END] -->

For more configurable parameters, please refer to the [API]("").

## Results and models

### ImageNet-1k

|   Model   |   Pretrain   | resolution | Params(M) | Flops(G) | Top-1 (%) | Top-5 (%) |                  Config                   |  Download   |
| :-------: | :----------: | :--------: | :-------: | :------: | :-------: | :-------: | :---------------------------------------: | :---------: |
| CSWin-T\* | From scratch |  224x224   |   22.32   |   4.34   |   82.66   |   96.27   |    [config](./cswin-tiny_b64_in1k.py)     | [model]("") |
| CSWin-S\* | From scratch |  224x224   |   34.64   |   6.83   |   83.58   |   96.55   |    [config](./cswin-small_b64_in1k.py)    | [model]("") |
| CSWin-B\* | From scratch |  224x224   |   77.38   |  14.99   |   84.11   |   96.92   |    [config](./cswin-base_b64_in1k.py)     | [model]("") |
| CSWin-B\* | From scratch |  384x384   |   77.38   |  33.19   |   85.52   |   97.53   | [config](./cswin-base_b64_in1k-384px.py)  | [model](")  |
| CSWin-L\* | From scratch |  224x224   |  173.26   |  34.04   |   86.35   |   97.98   |    [config](./cswin-large_b64_in1k.py)    | [model]("") |
| CSWin-L\* | From scratch |  384x384   |  173.26   |  102.06  |   87.49   |   98.36   | [config](./cswin-large_b64_in1k-384px.py) | [model]("") |

*Models with * are converted from the [official repo](https://github.com/microsoft/Swin-Transformer#main-results-on-imagenet-with-pretrained-models). The config files of these models are only for validation. We don't ensure these config files' training accuracy and welcome you to contribute your reproduction results.*

## Citation

```
@misc{dong2021cswin,
      title={CSWin Transformer: A General Vision Transformer Backbone with Cross-Shaped Windows},
        author={Xiaoyi Dong and Jianmin Bao and Dongdong Chen and Weiming Zhang and Nenghai Yu and Lu Yuan and Dong Chen and Baining Guo},
        year={2021},
        eprint={2107.00652},
        archivePrefix={arXiv},
        primaryClass={cs.CV}
}
```
