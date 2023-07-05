# Swin-Transformer V2

> [Swin Transformer V2: Scaling Up Capacity and Resolution](https://arxiv.org/abs/2111.09883)

<!-- [ALGORITHM] -->

## Introduction

**Swin Transformer V2** is a work on the scale up visual model based on [Swin Transformer](https://github.com/open-mmlab/mmpretrain/tree/main/configs/swin_transformer). In the visual field, We can not increase the performance by just simply scaling up the visual model like NLP models. The possible reasons mentioned in the article are:

- Training instability when increasing the vision model
- Migrating the model trained at low resolution to a larger scale resolution task
- Too mush GPU memory

To solve it, The following method improvements are proposed in the paper:

- post normalization: layer normalization after self-attention layer and MLP block
- scaled cosine attention approach: use cosine similarity to calculate the relationship between token pairs
- log-spaced continuous position bias: redefine relative position encoding

<div align=center>
<img src="https://user-images.githubusercontent.com/42952108/180748696-ee7ed23d-7fee-4ccf-9eb5-f117db228a42.png" width="100%"/>
</div>

## Abstract

<details>

<summary>Show the detailed Abstract</summary>

<br>

Large-scale NLP models have been shown to significantly improve the performance on language tasks with no signs of saturation. They also demonstrate amazing few-shot capabilities like that of human beings. This paper aims to explore large-scale models in computer vision. We tackle three major issues in training and application of large vision models, including training instability, resolution gaps between pre-training and fine-tuning, and hunger on labelled data. Three main techniques are proposed: 1) a residual-post-norm method combined with cosine attention to improve training stability; 2) A log-spaced continuous position bias method to effectively transfer models pre-trained using low-resolution images to downstream tasks with high-resolution inputs; 3) A self-supervised pre-training method, SimMIM, to reduce the needs of vast labeled images. Through these techniques, this paper successfully trained a 3 billion-parameter Swin Transformer V2 model, which is the largest dense vision model to date, and makes it capable of training with images of up to 1,536×1,536 resolution. It set new performance records on 4 representative vision tasks, including ImageNet-V2 image classification, COCO object detection, ADE20K semantic segmentation, and Kinetics-400 video action classification. Also note our training is much more efficient than that in Google's billion-level visual models, which consumes 40 times less labelled data and 40 times less training time.

</br>

</details>

## How to use it?

<!-- [TABS-BEGIN] -->

**Predict image**

```python
from mmpretrain import inference_model

predict = inference_model('swinv2-tiny-w8_3rdparty_in1k-256px', 'demo/bird.JPEG')
print(predict['pred_class'])
print(predict['pred_score'])
```

**Use the model**

```python
import torch
from mmpretrain import get_model

model = get_model('swinv2-tiny-w8_3rdparty_in1k-256px', pretrained=True)
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
python tools/test.py configs/swin_transformer_v2/swinv2-tiny-w8_16xb64_in1k-256px.py https://download.openmmlab.com/mmclassification/v0/swin-v2/swinv2-tiny-w8_3rdparty_in1k-256px_20220803-e318968f.pth
```

<!-- [TABS-END] -->

## Models and results

### Pretrained models

| Model                                     | Params (M) | Flops (G) |                      Config                      |                                           Download                                           |
| :---------------------------------------- | :--------: | :-------: | :----------------------------------------------: | :------------------------------------------------------------------------------------------: |
| `swinv2-base-w12_3rdparty_in21k-192px`\*  |   87.92    |   8.51    | [config](swinv2-base-w12_8xb128_in21k-192px.py)  | [model](https://download.openmmlab.com/mmclassification/v0/swin-v2/pretrain/swinv2-base-w12_3rdparty_in21k-192px_20220803-f7dc9763.pth) |
| `swinv2-large-w12_3rdparty_in21k-192px`\* |   196.74   |   19.04   | [config](swinv2-large-w12_8xb128_in21k-192px.py) | [model](https://download.openmmlab.com/mmclassification/v0/swin-v2/pretrain/swinv2-large-w12_3rdparty_in21k-192px_20220803-d9073fee.pth) |

*Models with * are converted from the [official repo](https://github.com/microsoft/Swin-Transformer). The config files of these models are only for inference. We haven't reproduce the training results.*

### Image Classification on ImageNet-1k

| Model                                             |   Pretrain   | Params (M) | Flops (G) | Top-1 (%) | Top-5 (%) |                       Config                       |                       Download                       |
| :------------------------------------------------ | :----------: | :--------: | :-------: | :-------: | :-------: | :------------------------------------------------: | :--------------------------------------------------: |
| `swinv2-tiny-w8_3rdparty_in1k-256px`\*            | From scratch |   28.35    |   4.35    |   81.76   |   95.87   |   [config](swinv2-tiny-w8_16xb64_in1k-256px.py)    | [model](https://download.openmmlab.com/mmclassification/v0/swin-v2/swinv2-tiny-w8_3rdparty_in1k-256px_20220803-e318968f.pth) |
| `swinv2-tiny-w16_3rdparty_in1k-256px`\*           | From scratch |   28.35    |   4.40    |   82.81   |   96.23   |   [config](swinv2-tiny-w16_16xb64_in1k-256px.py)   | [model](https://download.openmmlab.com/mmclassification/v0/swin-v2/swinv2-tiny-w16_3rdparty_in1k-256px_20220803-9651cdd7.pth) |
| `swinv2-small-w8_3rdparty_in1k-256px`\*           | From scratch |   49.73    |   8.45    |   83.74   |   96.60   |   [config](swinv2-small-w8_16xb64_in1k-256px.py)   | [model](https://download.openmmlab.com/mmclassification/v0/swin-v2/swinv2-small-w8_3rdparty_in1k-256px_20220803-b01a4332.pth) |
| `swinv2-small-w16_3rdparty_in1k-256px`\*          | From scratch |   49.73    |   8.57    |   84.13   |   96.83   |  [config](swinv2-small-w16_16xb64_in1k-256px.py)   | [model](https://download.openmmlab.com/mmclassification/v0/swin-v2/swinv2-small-w16_3rdparty_in1k-256px_20220803-b707d206.pth) |
| `swinv2-base-w8_3rdparty_in1k-256px`\*            | From scratch |   87.92    |   14.99   |   84.20   |   96.86   |   [config](swinv2-base-w8_16xb64_in1k-256px.py)    | [model](https://download.openmmlab.com/mmclassification/v0/swin-v2/swinv2-base-w8_3rdparty_in1k-256px_20220803-8ff28f2b.pth) |
| `swinv2-base-w16_3rdparty_in1k-256px`\*           | From scratch |   87.92    |   15.14   |   84.60   |   97.05   |   [config](swinv2-base-w16_16xb64_in1k-256px.py)   | [model](https://download.openmmlab.com/mmclassification/v0/swin-v2/swinv2-base-w16_3rdparty_in1k-256px_20220803-5a1886b7.pth) |
| `swinv2-base-w16_in21k-pre_3rdparty_in1k-256px`\* | ImageNet-21k |   87.92    |   15.14   |   86.17   |   97.88   | [config](swinv2-base-w16_in21k-pre_16xb64_in1k-256px.py) | [model](https://download.openmmlab.com/mmclassification/v0/swin-v2/swinv2-base-w16_in21k-pre_3rdparty_in1k-256px_20220803-8d7aa8ad.pth) |
| `swinv2-base-w24_in21k-pre_3rdparty_in1k-384px`\* | ImageNet-21k |   87.92    |   34.07   |   87.14   |   98.23   | [config](swinv2-base-w24_in21k-pre_16xb64_in1k-384px.py) | [model](https://download.openmmlab.com/mmclassification/v0/swin-v2/swinv2-base-w24_in21k-pre_3rdparty_in1k-384px_20220803-44eb70f8.pth) |
| `swinv2-large-w16_in21k-pre_3rdparty_in1k-256px`\* | ImageNet-21k |   196.75   |   33.86   |   86.93   |   98.06   | [config](swinv2-large-w16_in21k-pre_16xb64_in1k-256px.py) | [model](https://download.openmmlab.com/mmclassification/v0/swin-v2/swinv2-large-w16_in21k-pre_3rdparty_in1k-256px_20220803-c40cbed7.pth) |
| `swinv2-large-w24_in21k-pre_3rdparty_in1k-384px`\* | ImageNet-21k |   196.75   |   76.20   |   87.59   |   98.27   | [config](swinv2-large-w24_in21k-pre_16xb64_in1k-384px.py) | [model](https://download.openmmlab.com/mmclassification/v0/swin-v2/swinv2-large-w24_in21k-pre_3rdparty_in1k-384px_20220803-3b36c165.pth) |

*Models with * are converted from the [official repo](https://github.com/microsoft/Swin-Transformer). The config files of these models are only for inference. We haven't reproduce the training results.*

## Citation

```bibtex
@article{https://doi.org/10.48550/arxiv.2111.09883,
  doi = {10.48550/ARXIV.2111.09883},
  url = {https://arxiv.org/abs/2111.09883},
  author = {Liu, Ze and Hu, Han and Lin, Yutong and Yao, Zhuliang and Xie, Zhenda and Wei, Yixuan and Ning, Jia and Cao, Yue and Zhang, Zheng and Dong, Li and Wei, Furu and Guo, Baining},
  keywords = {Computer Vision and Pattern Recognition (cs.CV), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {Swin Transformer V2: Scaling Up Capacity and Resolution},
  publisher = {arXiv},
  year = {2021},
  copyright = {Creative Commons Attribution 4.0 International}
}
```
