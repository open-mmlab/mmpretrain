# MaskFeat

> [Masked Feature Prediction for Self-Supervised Visual Pre-Training](https://arxiv.org/abs/2112.09133v1)

<!-- [ALGORITHM] -->

## Abstract

We present Masked Feature Prediction (MaskFeat) for self-supervised pre-training of video models. Our approach first randomly masks out a portion of the input sequence and then predicts the feature of the masked regions. We study five different types of features and find Histograms of Oriented Gradients (HOG), a hand-crafted feature descriptor, works particularly well in terms of both performance and efficiency. We observe that the local contrast normalization in HOG is essential for good results, which is in line with earlier work using HOG for visual recognition. Our approach can learn abundant visual knowledge and drive large-scale Transformer-based models. Without using extra model weights or supervision, MaskFeat pre-trained on unlabeled videos achieves unprecedented results of 86.7% with MViT-L on Kinetics-400, 88.3% on Kinetics-600, 80.4% on Kinetics-700, 38.8 mAP on AVA, and 75.0% on SSv2. MaskFeat further generalizes to image input, which can be interpreted as a video with a single frame and obtains competitive results on ImageNet.

<div align=center>
<img src="https://user-images.githubusercontent.com/48178838/190090285-428f07c0-0887-4ce8-b94f-f719cfd25622.png" width="60%"/>
</div>

## How to use it?

<!-- [TABS-BEGIN] -->

**Predict image**

```python
from mmpretrain import inference_model

predict = inference_model('vit-base-p16_maskfeat-pre_8xb256-coslr-100e_in1k', 'demo/bird.JPEG')
print(predict['pred_class'])
print(predict['pred_score'])
```

**Use the model**

```python
import torch
from mmpretrain import get_model

model = get_model('maskfeat_vit-base-p16_8xb256-amp-coslr-300e_in1k', pretrained=True)
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
python tools/train.py configs/maskfeat/maskfeat_vit-base-p16_8xb256-amp-coslr-300e_in1k.py
```

Test:

```shell
python tools/test.py configs/maskfeat/benchmarks/vit-base-p16_8xb256-coslr-100e_in1k.py https://download.openmmlab.com/mmselfsup/1.x/maskfeat/maskfeat_vit-base-p16_8xb256-amp-coslr-300e_in1k/vit-base-p16_ft-8xb256-coslr-100e_in1k/vit-base-p16_ft-8xb256-coslr-100e_in1k_20221028-5134431c.pth
```

<!-- [TABS-END] -->

## Models and results

### Pretrained models

| Model                                              | Params (M) | Flops (G) |                            Config                             |                                Download                                |
| :------------------------------------------------- | :--------: | :-------: | :-----------------------------------------------------------: | :--------------------------------------------------------------------: |
| `maskfeat_vit-base-p16_8xb256-amp-coslr-300e_in1k` |   85.88    |   17.58   | [config](maskfeat_vit-base-p16_8xb256-amp-coslr-300e_in1k.py) | [model](https://download.openmmlab.com/mmselfsup/1.x/maskfeat/maskfeat_vit-base-p16_8xb256-amp-coslr-300e_in1k/maskfeat_vit-base-p16_8xb256-amp-coslr-300e_in1k_20221101-6dfc8bf3.pth) \| [log](https://download.openmmlab.com/mmselfsup/1.x/maskfeat/maskfeat_vit-base-p16_8xb256-amp-coslr-300e_in1k/maskfeat_vit-base-p16_8xb256-amp-coslr-300e_in1k_20221101-6dfc8bf3.json) |

### Image Classification on ImageNet-1k

| Model                                     |                   Pretrain                   | Params (M) | Flops (G) | Top-1 (%) |                   Config                   |                   Download                    |
| :---------------------------------------- | :------------------------------------------: | :--------: | :-------: | :-------: | :----------------------------------------: | :-------------------------------------------: |
| `vit-base-p16_maskfeat-pre_8xb256-coslr-100e_in1k` | [MASKFEAT](https://download.openmmlab.com/mmselfsup/1.x/maskfeat/maskfeat_vit-base-p16_8xb256-amp-coslr-300e_in1k/maskfeat_vit-base-p16_8xb256-amp-coslr-300e_in1k_20221101-6dfc8bf3.pth) |   86.57    |   17.58   |   83.40   | [config](benchmarks/vit-base-p16_8xb256-coslr-100e_in1k.py) | [model](https://download.openmmlab.com/mmselfsup/1.x/maskfeat/maskfeat_vit-base-p16_8xb256-amp-coslr-300e_in1k/vit-base-p16_ft-8xb256-coslr-100e_in1k/vit-base-p16_ft-8xb256-coslr-100e_in1k_20221028-5134431c.pth) \| [log](https://download.openmmlab.com/mmselfsup/1.x/maskfeat/maskfeat_vit-base-p16_8xb256-amp-coslr-300e_in1k/vit-base-p16_ft-8xb256-coslr-100e_in1k/vit-base-p16_ft-8xb256-coslr-100e_in1k_20221028-5134431c.json) |

## Citation

```bibtex
@InProceedings{wei2022masked,
    author    = {Wei, Chen and Fan, Haoqi and Xie, Saining and Wu, Chao-Yuan and Yuille, Alan and Feichtenhofer, Christoph},
    title     = {Masked Feature Prediction for Self-Supervised Visual Pre-Training},
    booktitle = {CVPR},
    year      = {2022},
}
```
