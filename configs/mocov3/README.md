# MoCoV3

> [An Empirical Study of Training Self-Supervised Vision Transformers](https://arxiv.org/abs/2104.02057)

<!-- [ALGORITHM] -->

## Abstract

This paper does not describe a novel method. Instead, it studies a straightforward, incremental, yet must-know baseline given the recent progress in computer vision: self-supervised learning for Vision Transformers (ViT). While the training recipes for standard convolutional networks have been highly mature and robust, the recipes for ViT are yet to be built, especially in the self-supervised scenarios where training becomes more challenging. In this work, we go back to basics and investigate the effects of several fundamental components for training self-supervised ViT. We observe that instability is a major issue that degrades accuracy, and it can be hidden by apparently good results. We reveal that these results are indeed partial failure, and they can be improved when training is made more stable. We benchmark ViT results in MoCo v3 and several other self-supervised frameworks, with ablations in various aspects. We discuss the currently positive evidence as well as challenges and open questions. We hope that this work will provide useful data points and experience for future research.

<div align=center>
<img  src="https://user-images.githubusercontent.com/36138628/151305362-e6e8ea35-b3b8-45f6-8819-634e67083218.png" width="500" />
</div>

## How to use it?

<!-- [TABS-BEGIN] -->

**Predict image**

```python
from mmpretrain import inference_model

predict = inference_model('resnet50_mocov3-100e-pre_8xb128-linear-coslr-90e_in1k', 'demo/bird.JPEG')
print(predict['pred_class'])
print(predict['pred_score'])
```

**Use the model**

```python
import torch
from mmpretrain import get_model

model = get_model('mocov3_resnet50_8xb512-amp-coslr-100e_in1k', pretrained=True)
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
python tools/train.py configs/mocov3/mocov3_resnet50_8xb512-amp-coslr-100e_in1k.py
```

Test:

```shell
python tools/test.py configs/mocov3/benchmarks/resnet50_8xb128-linear-coslr-90e_in1k.py https://download.openmmlab.com/mmselfsup/1.x/mocov3/mocov3_resnet50_8xb512-amp-coslr-100e_in1k/resnet50_linear-8xb128-coslr-90e_in1k/resnet50_linear-8xb128-coslr-90e_in1k_20220927-8f7d937e.pth
```

<!-- [TABS-END] -->

## Models and results

### Pretrained models

| Model                                              | Params (M) | Flops (G) |                            Config                             |                                Download                                |
| :------------------------------------------------- | :--------: | :-------: | :-----------------------------------------------------------: | :--------------------------------------------------------------------: |
| `mocov3_resnet50_8xb512-amp-coslr-100e_in1k`       |   68.01    |   4.11    |    [config](mocov3_resnet50_8xb512-amp-coslr-100e_in1k.py)    | [model](https://download.openmmlab.com/mmselfsup/1.x/mocov3/mocov3_resnet50_8xb512-amp-coslr-100e_in1k/mocov3_resnet50_8xb512-amp-coslr-100e_in1k_20220927-f1144efa.pth) \| [log](https://download.openmmlab.com/mmselfsup/1.x/mocov3/mocov3_resnet50_8xb512-amp-coslr-100e_in1k/mocov3_resnet50_8xb512-amp-coslr-100e_in1k_20220927-f1144efa.json) |
| `mocov3_resnet50_8xb512-amp-coslr-300e_in1k`       |   68.01    |   4.11    |    [config](mocov3_resnet50_8xb512-amp-coslr-300e_in1k.py)    | [model](https://download.openmmlab.com/mmselfsup/1.x/mocov3/mocov3_resnet50_8xb512-amp-coslr-300e_in1k/mocov3_resnet50_8xb512-amp-coslr-300e_in1k_20220927-1e4f3304.pth) \| [log](https://download.openmmlab.com/mmselfsup/1.x/mocov3/mocov3_resnet50_8xb512-amp-coslr-300e_in1k/mocov3_resnet50_8xb512-amp-coslr-300e_in1k_20220927-1e4f3304.json) |
| `mocov3_resnet50_8xb512-amp-coslr-800e_in1k`       |   68.01    |   4.11    |    [config](mocov3_resnet50_8xb512-amp-coslr-800e_in1k.py)    | [model](https://download.openmmlab.com/mmselfsup/1.x/mocov3/mocov3_resnet50_8xb512-amp-coslr-800e_in1k/mocov3_resnet50_8xb512-amp-coslr-800e_in1k_20220927-e043f51a.pth) \| [log](https://download.openmmlab.com/mmselfsup/1.x/mocov3/mocov3_resnet50_8xb512-amp-coslr-800e_in1k/mocov3_resnet50_8xb512-amp-coslr-800e_in1k_20220927-e043f51a.json) |
| `mocov3_vit-small-p16_16xb256-amp-coslr-300e_in1k` |   84.27    |   4.61    | [config](mocov3_vit-small-p16_16xb256-amp-coslr-300e_in1k.py) | [model](https://download.openmmlab.com/mmselfsup/1.x/mocov3/mocov3_vit-small-p16_16xb256-amp-coslr-300e_in1k/mocov3_vit-small-p16_16xb256-amp-coslr-300e_in1k-224_20220826-08bc52f7.pth) \| [log](https://download.openmmlab.com/mmselfsup/1.x/mocov3/mocov3_vit-small-p16_16xb256-amp-coslr-300e_in1k/mocov3_vit-small-p16_16xb256-amp-coslr-300e_in1k-224_20220826-08bc52f7.json) |
| `mocov3_vit-base-p16_16xb256-amp-coslr-300e_in1k`  |   215.68   |   17.58   | [config](mocov3_vit-base-p16_16xb256-amp-coslr-300e_in1k.py)  | [model](https://download.openmmlab.com/mmselfsup/1.x/mocov3/mocov3_vit-base-p16_16xb256-amp-coslr-300e_in1k/mocov3_vit-base-p16_16xb256-amp-coslr-300e_in1k-224_20220826-25213343.pth) \| [log](https://download.openmmlab.com/mmselfsup/1.x/mocov3/mocov3_vit-base-p16_16xb256-amp-coslr-300e_in1k/mocov3_vit-base-p16_16xb256-amp-coslr-300e_in1k-224_20220826-25213343.json) |
| `mocov3_vit-large-p16_64xb64-amp-coslr-300e_in1k`  |   652.78   |   61.60   | [config](mocov3_vit-large-p16_64xb64-amp-coslr-300e_in1k.py)  | [model](https://download.openmmlab.com/mmselfsup/1.x/mocov3/mocov3_vit-large-p16_64xb64-amp-coslr-300e_in1k/mocov3_vit-large-p16_64xb64-amp-coslr-300e_in1k-224_20220829-9b88a442.pth) \| [log](https://download.openmmlab.com/mmselfsup/1.x/mocov3/mocov3_vit-large-p16_64xb64-amp-coslr-300e_in1k/mocov3_vit-large-p16_64xb64-amp-coslr-300e_in1k-224_20220829-9b88a442.json) |

### Image Classification on ImageNet-1k

| Model                                     |                   Pretrain                   | Params (M) | Flops (G) | Top-1 (%) |                   Config                   |                   Download                    |
| :---------------------------------------- | :------------------------------------------: | :--------: | :-------: | :-------: | :----------------------------------------: | :-------------------------------------------: |
| `resnet50_mocov3-100e-pre_8xb128-linear-coslr-90e_in1k` | [MOCOV3 100-Epochs](https://download.openmmlab.com/mmselfsup/1.x/mocov3/mocov3_resnet50_8xb512-amp-coslr-100e_in1k/mocov3_resnet50_8xb512-amp-coslr-100e_in1k_20220927-f1144efa.pth) |   25.56    |   4.11    |   69.60   | [config](benchmarks/resnet50_8xb128-linear-coslr-90e_in1k.py) | [model](https://download.openmmlab.com/mmselfsup/1.x/mocov3/mocov3_resnet50_8xb512-amp-coslr-100e_in1k/resnet50_linear-8xb128-coslr-90e_in1k/resnet50_linear-8xb128-coslr-90e_in1k_20220927-8f7d937e.pth) \| [log](https://download.openmmlab.com/mmselfsup/1.x/mocov3/mocov3_resnet50_8xb512-amp-coslr-100e_in1k/resnet50_linear-8xb128-coslr-90e_in1k/resnet50_linear-8xb128-coslr-90e_in1k_20220927-8f7d937e.json) |
| `resnet50_mocov3-300e-pre_8xb128-linear-coslr-90e_in1k` | [MOCOV3 300-Epochs](https://download.openmmlab.com/mmselfsup/1.x/mocov3/mocov3_resnet50_8xb512-amp-coslr-300e_in1k/mocov3_resnet50_8xb512-amp-coslr-300e_in1k_20220927-1e4f3304.pth) |   25.56    |   4.11    |   72.80   | [config](benchmarks/resnet50_8xb128-linear-coslr-90e_in1k.py) | [model](https://download.openmmlab.com/mmselfsup/1.x/mocov3/mocov3_resnet50_8xb512-amp-coslr-300e_in1k/resnet50_linear-8xb128-coslr-90e_in1k/resnet50_linear-8xb128-coslr-90e_in1k_20220927-d21ddac2.pth) \| [log](https://download.openmmlab.com/mmselfsup/1.x/mocov3/mocov3_resnet50_8xb512-amp-coslr-300e_in1k/resnet50_linear-8xb128-coslr-90e_in1k/resnet50_linear-8xb128-coslr-90e_in1k_20220927-d21ddac2.json) |
| `resnet50_mocov3-800e-pre_8xb128-linear-coslr-90e_in1k` | [MOCOV3 800-Epochs](https://download.openmmlab.com/mmselfsup/1.x/mocov3/mocov3_resnet50_8xb512-amp-coslr-800e_in1k/mocov3_resnet50_8xb512-amp-coslr-800e_in1k_20220927-e043f51a.pth) |   25.56    |   4.11    |   74.40   | [config](benchmarks/resnet50_8xb128-linear-coslr-90e_in1k.py) | [model](https://download.openmmlab.com/mmselfsup/1.x/mocov3/mocov3_resnet50_8xb512-amp-coslr-800e_in1k/resnet50_linear-8xb128-coslr-90e_in1k/resnet50_linear-8xb128-coslr-90e_in1k_20220927-0e97a483.pth) \| [log](https://download.openmmlab.com/mmselfsup/1.x/mocov3/mocov3_resnet50_8xb512-amp-coslr-800e_in1k/resnet50_linear-8xb128-coslr-90e_in1k/resnet50_linear-8xb128-coslr-90e_in1k_20220927-0e97a483.json) |
| `vit-small-p16_mocov3-pre_8xb128-linear-coslr-90e_in1k` | [MOCOV3](https://download.openmmlab.com/mmselfsup/1.x/mocov3/mocov3_vit-small-p16_16xb256-amp-coslr-300e_in1k/mocov3_vit-small-p16_16xb256-amp-coslr-300e_in1k-224_20220826-08bc52f7.pth) |   22.05    |   4.61    |   73.60   | [config](benchmarks/vit-small-p16_8xb128-linear-coslr-90e_in1k.py) | [model](https://download.openmmlab.com/mmselfsup/1.x/mocov3/mocov3_vit-small-p16_16xb256-amp-coslr-300e_in1k/vit-small-p16_linear-8xb128-coslr-90e_in1k/vit-small-p16_linear-8xb128-coslr-90e_in1k_20220826-376674ef.pth) \| [log](https://download.openmmlab.com/mmselfsup/1.x/mocov3/mocov3_vit-small-p16_16xb256-amp-coslr-300e_in1k/vit-small-p16_linear-8xb128-coslr-90e_in1k/vit-small-p16_linear-8xb128-coslr-90e_in1k_20220826-376674ef.json) |
| `vit-base-p16_mocov3-pre_8xb64-coslr-150e_in1k` | [MOCOV3](https://download.openmmlab.com/mmselfsup/1.x/mocov3/mocov3_vit-base-p16_16xb256-amp-coslr-300e_in1k/mocov3_vit-base-p16_16xb256-amp-coslr-300e_in1k-224_20220826-25213343.pth) |   86.57    |   17.58   |   83.00   | [config](benchmarks/vit-base-p16_8xb64-coslr-150e_in1k.py) | [model](https://download.openmmlab.com/mmselfsup/1.x/mocov3/mocov3_vit-base-p16_16xb256-amp-coslr-300e_in1k/vit-base-p16_ft-8xb64-coslr-150e_in1k/vit-base-p16_ft-8xb64-coslr-150e_in1k_20220826-f1e6c442.pth) \| [log](https://download.openmmlab.com/mmselfsup/1.x/mocov3/mocov3_vit-base-p16_16xb256-amp-coslr-300e_in1k/vit-base-p16_ft-8xb64-coslr-150e_in1k/vit-base-p16_ft-8xb64-coslr-150e_in1k_20220826-f1e6c442.json) |
| `vit-base-p16_mocov3-pre_8xb128-linear-coslr-90e_in1k` | [MOCOV3](https://download.openmmlab.com/mmselfsup/1.x/mocov3/mocov3_vit-base-p16_16xb256-amp-coslr-300e_in1k/mocov3_vit-base-p16_16xb256-amp-coslr-300e_in1k-224_20220826-25213343.pth) |   86.57    |   17.58   |   76.90   | [config](benchmarks/vit-base-p16_8xb128-linear-coslr-90e_in1k.py) | [model](https://download.openmmlab.com/mmselfsup/1.x/mocov3/mocov3_vit-base-p16_16xb256-amp-coslr-300e_in1k/vit-base-p16_linear-8xb128-coslr-90e_in1k/vit-base-p16_linear-8xb128-coslr-90e_in1k_20220826-83be7758.pth) \| [log](https://download.openmmlab.com/mmselfsup/1.x/mocov3/mocov3_vit-base-p16_16xb256-amp-coslr-300e_in1k/vit-base-p16_linear-8xb128-coslr-90e_in1k/vit-base-p16_linear-8xb128-coslr-90e_in1k_20220826-83be7758.json) |
| `vit-large-p16_mocov3-pre_8xb64-coslr-100e_in1k` | [MOCOV3](https://download.openmmlab.com/mmselfsup/1.x/mocov3/mocov3_vit-large-p16_64xb64-amp-coslr-300e_in1k/mocov3_vit-large-p16_64xb64-amp-coslr-300e_in1k-224_20220829-9b88a442.pth) |   304.33   |   61.60   |   83.70   | [config](benchmarks/vit-large-p16_8xb64-coslr-100e_in1k.py) | [model](https://download.openmmlab.com/mmselfsup/1.x/mocov3/mocov3_vit-large-p16_64xb64-amp-coslr-300e_in1k/vit-large-p16_ft-8xb64-coslr-100e_in1k/vit-large-p16_ft-8xb64-coslr-100e_in1k_20220829-878a2f7f.pth) \| [log](https://download.openmmlab.com/mmselfsup/1.x/mocov3/mocov3_vit-large-p16_64xb64-amp-coslr-300e_in1k/vit-large-p16_ft-8xb64-coslr-100e_in1k/vit-large-p16_ft-8xb64-coslr-100e_in1k_20220829-878a2f7f.json) |

## Citation

```bibtex
@InProceedings{Chen_2021_ICCV,
    title     = {An Empirical Study of Training Self-Supervised Vision Transformers},
    author    = {Chen, Xinlei and Xie, Saining and He, Kaiming},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    year      = {2021}
}
```
