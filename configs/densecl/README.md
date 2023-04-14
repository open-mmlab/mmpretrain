# DenseCL

> [Dense contrastive learning for self-supervised visual pre-training](https://arxiv.org/abs/2011.09157)

<!-- [ALGORITHM] -->

## Abstract

To date, most existing self-supervised learning methods are designed and optimized for image classification. These pre-trained models can be sub-optimal for dense prediction tasks due to the discrepancy between image-level prediction and pixel-level prediction. To fill this gap, we aim to design an effective, dense self-supervised learning method that directly works at the level of pixels (or local features) by taking into account the correspondence between local features. We present dense contrastive learning (DenseCL), which implements self-supervised learning by optimizing a pairwise contrastive (dis)similarity loss at the pixel level between two views of input images.

<div align=center>
<img src="https://user-images.githubusercontent.com/36138628/149721111-bab03a6d-a30d-418e-b338-43c3689cfc65.png" width="900" />
</div>

## How to use it?

<!-- [TABS-BEGIN] -->

**Predict image**

```python
from mmpretrain import inference_model

predict = inference_model('resnet50_densecl-pre_8xb32-linear-steplr-100e_in1k', 'demo/bird.JPEG')
print(predict['pred_class'])
print(predict['pred_score'])
```

**Use the model**

```python
import torch
from mmpretrain import get_model

model = get_model('densecl_resnet50_8xb32-coslr-200e_in1k', pretrained=True)
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
python tools/train.py configs/densecl/densecl_resnet50_8xb32-coslr-200e_in1k.py
```

Test:

```shell
python tools/test.py configs/densecl/benchmarks/resnet50_8xb32-linear-steplr-100e_in1k.py https://download.openmmlab.com/mmselfsup/1.x/densecl/densecl_resnet50_8xb32-coslr-200e_in1k/resnet50_linear-8xb32-steplr-100e_in1k/resnet50_linear-8xb32-steplr-100e_in1k_20220825-f0f0a579.pth
```

<!-- [TABS-END] -->

## Models and results

### Pretrained models

| Model                                    | Params (M) | Flops (G) |                       Config                        |                                          Download                                          |
| :--------------------------------------- | :--------: | :-------: | :-------------------------------------------------: | :----------------------------------------------------------------------------------------: |
| `densecl_resnet50_8xb32-coslr-200e_in1k` |   64.85    |   4.11    | [config](densecl_resnet50_8xb32-coslr-200e_in1k.py) | [model](https://download.openmmlab.com/mmselfsup/1.x/densecl/densecl_resnet50_8xb32-coslr-200e_in1k/densecl_resnet50_8xb32-coslr-200e_in1k_20220825-3078723b.pth) \| [log](https://download.openmmlab.com/mmselfsup/1.x/densecl/densecl_resnet50_8xb32-coslr-200e_in1k/densecl_resnet50_8xb32-coslr-200e_in1k_20220825-3078723b.json) |

### Image Classification on ImageNet-1k

| Model                                     |                   Pretrain                   | Params (M) | Flops (G) | Top-1 (%) |                   Config                   |                   Download                    |
| :---------------------------------------- | :------------------------------------------: | :--------: | :-------: | :-------: | :----------------------------------------: | :-------------------------------------------: |
| `resnet50_densecl-pre_8xb32-linear-steplr-100e_in1k` | [DENSECL](https://download.openmmlab.com/mmselfsup/1.x/densecl/densecl_resnet50_8xb32-coslr-200e_in1k/densecl_resnet50_8xb32-coslr-200e_in1k_20220825-3078723b.pth) |   25.56    |   4.11    |   63.50   | [config](benchmarks/resnet50_8xb32-linear-steplr-100e_in1k.py) | [model](https://download.openmmlab.com/mmselfsup/1.x/densecl/densecl_resnet50_8xb32-coslr-200e_in1k/resnet50_linear-8xb32-steplr-100e_in1k/resnet50_linear-8xb32-steplr-100e_in1k_20220825-f0f0a579.pth) \| [log](https://download.openmmlab.com/mmselfsup/1.x/densecl/densecl_resnet50_8xb32-coslr-200e_in1k/resnet50_linear-8xb32-steplr-100e_in1k/resnet50_linear-8xb32-steplr-100e_in1k_20220825-f0f0a579.json) |

## Citation

```bibtex
@inproceedings{wang2021dense,
  title={Dense contrastive learning for self-supervised visual pre-training},
  author={Wang, Xinlong and Zhang, Rufeng and Shen, Chunhua and Kong, Tao and Li, Lei},
  booktitle={CVPR},
  year={2021}
}
```
