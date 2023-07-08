# EVA

> [EVA: Exploring the Limits of Masked Visual Representation Learning at Scale](https://arxiv.org/abs/2211.07636)

<!-- [ALGORITHM] -->

## Abstract

We launch EVA, a vision-centric foundation model to explore the limits of visual representation at scale using only publicly accessible data. EVA is a vanilla ViT pre-trained to reconstruct the masked out image-text aligned vision features conditioned on visible image patches. Via this pretext task, we can efficiently scale up EVA to one billion parameters, and sets new records on a broad range of representative vision downstream tasks, such as image recognition, video action recognition, object detection, instance segmentation and semantic segmentation without heavy supervised training. Moreover, we observe quantitative changes in scaling EVA result in qualitative changes in transfer learning performance that are not present in other models. For instance, EVA takes a great leap in the challenging large vocabulary instance segmentation task: our model achieves almost the same state-of-the-art performance on LVISv1.0 dataset with over a thousand categories and COCO dataset with only eighty categories. Beyond a pure vision encoder, EVA can also serve as a vision-centric, multi-modal pivot to connect images and text. We find initializing the vision tower of a giant CLIP from EVA can greatly stabilize the training and outperform the training from scratch counterpart with much fewer samples and less compute, providing a new direction for scaling up and accelerating the costly training of multi-modal foundation models.

<div align=center>
<img src="https://user-images.githubusercontent.com/24734142/205410193-f1164e56-c117-4165-86f5-4cbfd797bc87.png" width="70%"/>
</div>

## How to use it?

<!-- [TABS-BEGIN] -->

**Predict image**

```python
from mmpretrain import inference_model

predict = inference_model('vit-base-p16_eva-mae-style-pre_8xb128-coslr-100e_in1k', 'demo/bird.JPEG')
print(predict['pred_class'])
print(predict['pred_score'])
```

**Use the model**

```python
import torch
from mmpretrain import get_model

model = get_model('eva-mae-style_vit-base-p16_16xb256-coslr-400e_in1k', pretrained=True)
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
python tools/train.py configs/eva/eva-mae-style_vit-base-p16_16xb256-coslr-400e_in1k.py
```

Test:

```shell
python tools/test.py configs/eva/benchmarks/vit-base-p16_8xb128-coslr-100e_in1k.py https://download.openmmlab.com/mmselfsup/1.x/eva/eva-mae-style_vit-base-p16_16xb256-coslr-400e_in1k/vit-base-p16_ft-8xb128-coslr-100e_in1k/vit-base-p16_ft-8xb128-coslr-100e_in1k_20221226-f61cf992.pth
```

<!-- [TABS-END] -->

## Models and results

### Pretrained models

| Model                                                | Params (M) | Flops (G) |                             Config                              |                              Download                              |
| :--------------------------------------------------- | :--------: | :-------: | :-------------------------------------------------------------: | :----------------------------------------------------------------: |
| `eva-mae-style_vit-base-p16_16xb256-coslr-400e_in1k` |   111.78   |   17.58   | [config](eva-mae-style_vit-base-p16_16xb256-coslr-400e_in1k.py) | [model](https://download.openmmlab.com/mmselfsup/1.x/eva/eva-mae-style_vit-base-p16_16xb256-coslr-400e_in1k/eva-mae-style_vit-base-p16_16xb256-coslr-400e_in1k_20221226-26d90f07.pth) \| [log](https://download.openmmlab.com/mmselfsup/1.x/eva/eva-mae-style_vit-base-p16_16xb256-coslr-400e_in1k/eva-mae-style_vit-base-p16_16xb256-coslr-400e_in1k_20221226-26d90f07.json) |
| `beit-l-p14_3rdparty-eva_in21k`\*                    |   303.18   |   81.08   |                 [config](eva-l-p14_headless.py)                 | [model](https://download.openmmlab.com/mmclassification/v0/eva/eva-l-p14_3rdparty-mim_in21k_20221213-3a5da50b.pth) |
| `beit-l-p14_eva-pre_3rdparty_in21k`\*                |   303.18   |   81.08   |                 [config](eva-l-p14_headless.py)                 | [model](https://download.openmmlab.com/mmclassification/v0/eva/eva-l-p14_mim-pre_3rdparty_in21k_20221213-8f194fa2.pth) |
| `beit-g-p16_3rdparty-eva_30m`\*                      |  1011.32   |  203.52   |                 [config](eva-g-p16_headless.py)                 | [model](https://download.openmmlab.com/mmclassification/v0/eva/eva-g-p16_3rdparty_30m_20221213-7bed23ee.pth) |
| `beit-g-p14_3rdparty-eva_30m`\*                      |  1011.60   |  267.17   |                 [config](eva-g-p14_headless.py)                 | [model](https://download.openmmlab.com/mmclassification/v0/eva/eva-g-p14_3rdparty_30m_20221213-3b7aca97.pth) |
| `beit-g-p14_eva-30m-pre_3rdparty_in21k`\*            |  1011.60   |  267.17   |                 [config](eva-g-p14_headless.py)                 | [model](https://download.openmmlab.com/mmclassification/v0/eva/eva-g-p14_30m-pre_3rdparty_in21k_20221213-d72285b7.pth) |

*Models with * are converted from the [official repo](https://github.com/baaivision/EVA). The config files of these models are only for inference. We haven't reproduce the training results.*

### Image Classification on ImageNet-1k

| Model                                   |                  Pretrain                  | Params (M) | Flops (G) | Top-1 (%) | Top-5 (%) |                  Config                  |                  Download                  |
| :-------------------------------------- | :----------------------------------------: | :--------: | :-------: | :-------: | :-------: | :--------------------------------------: | :----------------------------------------: |
| `vit-base-p16_eva-mae-style-pre_8xb128-coslr-100e_in1k` | [EVA MAE STYLE](https://download.openmmlab.com/mmselfsup/1.x/eva/eva-mae-style_vit-base-p16_16xb256-coslr-400e_in1k/eva-mae-style_vit-base-p16_16xb256-coslr-400e_in1k_20221226-26d90f07.pth) |   86.57    |   17.58   |   83.70   |    N/A    | [config](benchmarks/vit-base-p16_8xb128-coslr-100e_in1k.py) | [model](https://download.openmmlab.com/mmselfsup/1.x/eva/eva-mae-style_vit-base-p16_16xb256-coslr-400e_in1k/vit-base-p16_ft-8xb128-coslr-100e_in1k/vit-base-p16_ft-8xb128-coslr-100e_in1k_20221226-f61cf992.pth) \| [log](https://download.openmmlab.com/mmselfsup/1.x/eva/eva-mae-style_vit-base-p16_16xb256-coslr-400e_in1k/vit-base-p16_ft-8xb128-coslr-100e_in1k/vit-base-p16_ft-8xb128-coslr-100e_in1k_20221226-f61cf992.json) |
| `vit-base-p16_eva-mae-style-pre_8xb2048-linear-coslr-100e_in1k` | [EVA MAE STYLE](https://download.openmmlab.com/mmselfsup/1.x/eva/eva-mae-style_vit-base-p16_16xb256-coslr-400e_in1k/eva-mae-style_vit-base-p16_16xb256-coslr-400e_in1k_20221226-26d90f07.pth) |   86.57    |   17.58   |   69.00   |    N/A    | [config](benchmarks/vit-base-p16_8xb2048-linear-coslr-100e_in1k.py) | [model](https://download.openmmlab.com/mmselfsup/1.x/eva/eva-mae-style_vit-base-p16_16xb256-coslr-400e_in1k/vit-base-p16_linear-8xb2048-coslr-100e_in1k/vit-base-p16_linear-8xb2048-coslr-100e_in1k_20221226-ef51bf09.pth) \| [log](https://download.openmmlab.com/mmselfsup/1.x/eva/eva-mae-style_vit-base-p16_16xb256-coslr-400e_in1k/vit-base-p16_linear-8xb2048-coslr-100e_in1k/vit-base-p16_linear-8xb2048-coslr-100e_in1k_20221226-ef51bf09.json) |
| `beit-l-p14_eva-pre_3rdparty_in1k-196px`\* | [EVA](https://download.openmmlab.com/mmclassification/v0/eva/eva-l-p14_3rdparty-mim_in21k_20221213-3a5da50b.pth) |   304.14   |   61.57   |   87.94   |   98.5    | [config](eva-l-p14_8xb16_in1k-196px.py)  | [model](https://download.openmmlab.com/mmclassification/v0/eva/eva-l-p14_mim-pre_3rdparty_in1k-196px_20221214-2adf4d28.pth) |
| `beit-l-p14_eva-in21k-pre_3rdparty_in1k-196px`\* |              EVA ImageNet-21k              |   304.14   |   61.57   |   88.58   |   98.65   | [config](eva-l-p14_8xb16_in1k-196px.py)  | [model](https://download.openmmlab.com/mmclassification/v0/eva/eva-l-p14_mim-in21k-pre_3rdparty_in1k-196px_20221213-b730c7e7.pth) |
| `beit-l-p14_eva-pre_3rdparty_in1k-336px`\* | [EVA](https://download.openmmlab.com/mmclassification/v0/eva/eva-l-p14_3rdparty-mim_in21k_20221213-3a5da50b.pth) |   304.53   |  191.10   |   88.66   |   98.75   | [config](eva-l-p14_8xb16_in1k-336px.py)  | [model](https://download.openmmlab.com/mmclassification/v0/eva/eva-l-p14_mim-pre_3rdparty_in1k-336px_20221214-07785cfd.pth) |
| `beit-l-p14_eva-in21k-pre_3rdparty_in1k-336px`\* |              EVA ImageNet-21k              |   304.53   |  191.10   |   89.17   |   98.86   | [config](eva-l-p14_8xb16_in1k-336px.py)  | [model](https://download.openmmlab.com/mmclassification/v0/eva/eva-l-p14_mim-in21k-pre_3rdparty_in1k-336px_20221213-f25b7634.pth) |
| `beit-g-p14_eva-30m-in21k-pre_3rdparty_in1k-336px`\* | [EVA 30M ImageNet-21k](https://download.openmmlab.com/mmclassification/v0/eva/eva-g-p14_30m-pre_3rdparty_in21k_20221213-d72285b7.pth) |  1013.01   |  620.64   |   89.61   |   98.93   | [config](eva-g-p14_8xb16_in1k-336px.py)  | [model](https://download.openmmlab.com/mmclassification/v0/eva/eva-g-p14_30m-in21k-pre_3rdparty_in1k-336px_20221213-210f9071.pth) |
| `beit-g-p14_eva-30m-in21k-pre_3rdparty_in1k-560px`\* | [EVA 30M ImageNet-21k](https://download.openmmlab.com/mmclassification/v0/eva/eva-g-p14_30m-pre_3rdparty_in21k_20221213-d72285b7.pth) |  1014.45   |  1906.76  |   89.71   |   98.96   | [config](eva-g-p14_8xb16_in1k-560px.py)  | [model](https://download.openmmlab.com/mmclassification/v0/eva/eva-g-p14_30m-in21k-pre_3rdparty_in1k-560px_20221213-fa1c3652.pth) |

*Models with * are converted from the [official repo](https://github.com/baaivision/EVA). The config files of these models are only for inference. We haven't reproduce the training results.*

## Citation

```bibtex
@article{EVA,
  title={EVA: Exploring the Limits of Masked Visual Representation Learning at Scale},
  author={Fang, Yuxin and Wang, Wen and Xie, Binhui and Sun, Quan and Wu, Ledell and Wang, Xinggang and Huang, Tiejun and Wang, Xinlong and Cao, Yue},
  journal={arXiv preprint arXiv:2211.07636},
  year={2022}
}
```
