# Swin-Transformer

> [Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](https://arxiv.org/abs/2103.14030)

<!-- [ALGORITHM] -->

## Introduction

**Swin Transformer** (the name **Swin** stands for Shifted window) is initially described in [the paper](https://arxiv.org/pdf/2103.14030.pdf), which capably serves as a general-purpose backbone for computer vision. It is basically a hierarchical Transformer whose representation is computed with shifted windows. The shifted windowing scheme brings greater efficiency by limiting self-attention computation to non-overlapping local windows while also allowing for cross-window connection.

Swin Transformer achieves strong performance on COCO object detection (58.7 box AP and 51.1 mask AP on test-dev) and ADE20K semantic segmentation (53.5 mIoU on val), surpassing previous models by a large margin.

<div align=center>
<img src="https://user-images.githubusercontent.com/26739999/142576715-14668c6b-5cb8-4de8-ac51-419fae773c90.png" width="90%"/>
</div>

## Abstract

<details>

<summary>Show the paper's abstract</summary>

<br>
This paper presents a new vision Transformer, called Swin Transformer, that capably serves as a general-purpose backbone for computer vision. Challenges in adapting Transformer from language to vision arise from differences between the two domains, such as large variations in the scale of visual entities and the high resolution of pixels in images compared to words in text. To address these differences, we propose a hierarchical Transformer whose representation is computed with **Shifted windows**. The shifted windowing scheme brings greater efficiency by limiting self-attention computation to non-overlapping local windows while also allowing for cross-window connection. This hierarchical architecture has the flexibility to model at various scales and has linear computational complexity with respect to image size. These qualities of Swin Transformer make it compatible with a broad range of vision tasks, including image classification (87.3 top-1 accuracy on ImageNet-1K) and dense prediction tasks such as object detection (58.7 box AP and 51.1 mask AP on COCO test-dev) and semantic segmentation (53.5 mIoU on ADE20K val). Its performance surpasses the previous state-of-the-art by a large margin of +2.7 box AP and +2.6 mask AP on COCO, and +3.2 mIoU on ADE20K, demonstrating the potential of Transformer-based models as vision backbones. The hierarchical design and the shifted window approach also prove beneficial for all-MLP architectures.
</br>

</details>

## How to use it?

<!-- [TABS-BEGIN] -->

**Predict image**

```python
from mmpretrain import inference_model

predict = inference_model('swin-tiny_16xb64_in1k', 'demo/bird.JPEG')
print(predict['pred_class'])
print(predict['pred_score'])
```

**Use the model**

```python
import torch
from mmpretrain import get_model

model = get_model('swin-tiny_16xb64_in1k', pretrained=True)
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
python tools/train.py configs/swin_transformer/swin-tiny_16xb64_in1k.py
```

Test:

```shell
python tools/test.py configs/swin_transformer/swin-tiny_16xb64_in1k.py https://download.openmmlab.com/mmclassification/v0/swin-transformer/swin_tiny_224_b16x64_300e_imagenet_20210616_090925-66df6be6.pth
```

<!-- [TABS-END] -->

## Models and results

### Image Classification on ImageNet-1k

| Model                                      |   Pretrain   | Params (M) | Flops (G) | Top-1 (%) | Top-5 (%) |                  Config                   |                               Download                               |
| :----------------------------------------- | :----------: | :--------: | :-------: | :-------: | :-------: | :---------------------------------------: | :------------------------------------------------------------------: |
| `swin-tiny_16xb64_in1k`                    | From scratch |   28.29    |   4.36    |   81.18   |   95.61   |    [config](swin-tiny_16xb64_in1k.py)     | [model](https://download.openmmlab.com/mmclassification/v0/swin-transformer/swin_tiny_224_b16x64_300e_imagenet_20210616_090925-66df6be6.pth) \| [log](https://download.openmmlab.com/mmclassification/v0/swin-transformer/swin_tiny_224_b16x64_300e_imagenet_20210616_090925.json) |
| `swin-small_16xb64_in1k`                   | From scratch |   49.61    |   8.52    |   83.02   |   96.29   |    [config](swin-small_16xb64_in1k.py)    | [model](https://download.openmmlab.com/mmclassification/v0/swin-transformer/swin_small_224_b16x64_300e_imagenet_20210615_110219-7f9d988b.pth) \| [log](https://download.openmmlab.com/mmclassification/v0/swin-transformer/swin_small_224_b16x64_300e_imagenet_20210615_110219.json) |
| `swin-base_16xb64_in1k`                    | From scratch |   87.77    |   15.14   |   83.36   |   96.44   |    [config](swin-base_16xb64_in1k.py)     | [model](https://download.openmmlab.com/mmclassification/v0/swin-transformer/swin_base_224_b16x64_300e_imagenet_20210616_190742-93230b0d.pth) \| [log](https://download.openmmlab.com/mmclassification/v0/swin-transformer/swin_base_224_b16x64_300e_imagenet_20210616_190742.json) |
| `swin-tiny_3rdparty_in1k`\*                | From scratch |   28.29    |   4.36    |   81.18   |   95.52   |    [config](swin-tiny_16xb64_in1k.py)     | [model](https://download.openmmlab.com/mmclassification/v0/swin-transformer/convert/swin_tiny_patch4_window7_224-160bb0a5.pth) |
| `swin-small_3rdparty_in1k`\*               | From scratch |   49.61    |   8.52    |   83.21   |   96.25   |    [config](swin-small_16xb64_in1k.py)    | [model](https://download.openmmlab.com/mmclassification/v0/swin-transformer/convert/swin_small_patch4_window7_224-cc7a01c9.pth) |
| `swin-base_3rdparty_in1k`\*                | From scratch |   87.77    |   15.14   |   83.42   |   96.44   |    [config](swin-base_16xb64_in1k.py)     | [model](https://download.openmmlab.com/mmclassification/v0/swin-transformer/convert/swin_base_patch4_window7_224-4670dd19.pth) |
| `swin-base_3rdparty_in1k-384`\*            | From scratch |   87.90    |   44.49   |   84.49   |   96.95   | [config](swin-base_16xb64_in1k-384px.py)  | [model](https://download.openmmlab.com/mmclassification/v0/swin-transformer/convert/swin_base_patch4_window12_384-02c598a4.pth) |
| `swin-base_in21k-pre-3rdparty_in1k`\*      | From scratch |   87.77    |   15.14   |   85.16   |   97.50   |    [config](swin-base_16xb64_in1k.py)     | [model](https://download.openmmlab.com/mmclassification/v0/swin-transformer/convert/swin_base_patch4_window7_224_22kto1k-f967f799.pth) |
| `swin-base_in21k-pre-3rdparty_in1k-384`\*  | From scratch |   87.90    |   44.49   |   86.44   |   98.05   | [config](swin-base_16xb64_in1k-384px.py)  | [model](https://download.openmmlab.com/mmclassification/v0/swin-transformer/convert/swin_base_patch4_window12_384_22kto1k-d59b0d1d.pth) |
| `swin-large_in21k-pre-3rdparty_in1k`\*     | From scratch |   196.53   |   34.04   |   86.24   |   97.88   |    [config](swin-large_16xb64_in1k.py)    | [model](https://download.openmmlab.com/mmclassification/v0/swin-transformer/convert/swin_large_patch4_window7_224_22kto1k-5f0996db.pth) |
| `swin-large_in21k-pre-3rdparty_in1k-384`\* | From scratch |   196.74   |  100.04   |   87.25   |   98.25   | [config](swin-large_16xb64_in1k-384px.py) | [model](https://download.openmmlab.com/mmclassification/v0/swin-transformer/convert/swin_large_patch4_window12_384_22kto1k-0a40944b.pth) |

*Models with * are converted from the [official repo](https://github.com/microsoft/Swin-Transformer/blob/777f6c66604bb5579086c4447efe3620344d95a9/models/swin_transformer.py#L458). The config files of these models are only for inference. We haven't reproduce the training results.*

### Image Classification on CUB-200-2011

| Model                       |   Pretrain   | Params (M) | Flops (G) | Top-1 (%) |                 Config                 |                                            Download                                             |
| :-------------------------- | :----------: | :--------: | :-------: | :-------: | :------------------------------------: | :---------------------------------------------------------------------------------------------: |
| `swin-large_8xb8_cub-384px` | From scratch |   195.51   |  100.04   |   91.87   | [config](swin-large_8xb8_cub-384px.py) | [model](https://download.openmmlab.com/mmclassification/v0/swin-transformer/swin-large_8xb8_cub_384px_20220307-1bbaee6a.pth) \| [log](https://download.openmmlab.com/mmclassification/v0/swin-transformer/swin-large_8xb8_cub_384px_20220307-1bbaee6a.json) |

## Citation

```bibtex
@article{liu2021Swin,
  title={Swin Transformer: Hierarchical Vision Transformer using Shifted Windows},
  author={Liu, Ze and Lin, Yutong and Cao, Yue and Hu, Han and Wei, Yixuan and Zhang, Zheng and Lin, Stephen and Guo, Baining},
  journal={arXiv preprint arXiv:2103.14030},
  year={2021}
}
```
