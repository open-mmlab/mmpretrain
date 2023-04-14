# MoCoV2

> [Improved Baselines with Momentum Contrastive Learning](https://arxiv.org/abs/2003.04297)

<!-- [ALGORITHM] -->

## Abstract

Contrastive unsupervised learning has recently shown encouraging progress, e.g., in Momentum Contrast (MoCo) and SimCLR. In this note, we verify the effectiveness of two of SimCLR’s design improvements by implementing them in the MoCo framework. With simple modifications to MoCo—namely, using an MLP projection head and more data augmentation—we establish stronger baselines that outperform SimCLR and do not require large training batches. We hope this will make state-of-the-art unsupervised learning research more accessible.

<div align=center>
<img  src="https://user-images.githubusercontent.com/36138628/149720067-b65e5736-d425-45b3-93ed-6f2427fc6217.png" width="500" />
</div>

## How to use it?

<!-- [TABS-BEGIN] -->

**Predict image**

```python
from mmpretrain import inference_model

predict = inference_model('resnet50_mocov2-pre_8xb32-linear-steplr-100e_in1k', 'demo/bird.JPEG')
print(predict['pred_class'])
print(predict['pred_score'])
```

**Use the model**

```python
import torch
from mmpretrain import get_model

model = get_model('mocov2_resnet50_8xb32-coslr-200e_in1k', pretrained=True)
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
python tools/train.py configs/mocov2/mocov2_resnet50_8xb32-coslr-200e_in1k.py
```

Test:

```shell
python tools/test.py configs/mocov2/benchmarks/resnet50_8xb32-linear-steplr-100e_in1k.py https://download.openmmlab.com/mmselfsup/1.x/mocov2/mocov2_resnet50_8xb32-coslr-200e_in1k/resnet50_linear-8xb32-steplr-100e_in1k/resnet50_linear-8xb32-steplr-100e_in1k_20220825-994c4128.pth
```

<!-- [TABS-END] -->

## Models and results

### Pretrained models

| Model                                   | Params (M) | Flops (G) |                       Config                       |                                           Download                                           |
| :-------------------------------------- | :--------: | :-------: | :------------------------------------------------: | :------------------------------------------------------------------------------------------: |
| `mocov2_resnet50_8xb32-coslr-200e_in1k` |   55.93    |   4.11    | [config](mocov2_resnet50_8xb32-coslr-200e_in1k.py) | [model](https://download.openmmlab.com/mmselfsup/1.x/mocov2/mocov2_resnet50_8xb32-coslr-200e_in1k/mocov2_resnet50_8xb32-coslr-200e_in1k_20220825-b6d23c86.pth) \| [log](https://download.openmmlab.com/mmselfsup/1.x/mocov2/mocov2_resnet50_8xb32-coslr-200e_in1k/mocov2_resnet50_8xb32-coslr-200e_in1k_20220825-b6d23c86.json) |

### Image Classification on ImageNet-1k

| Model                                     |                   Pretrain                   | Params (M) | Flops (G) | Top-1 (%) |                   Config                   |                   Download                    |
| :---------------------------------------- | :------------------------------------------: | :--------: | :-------: | :-------: | :----------------------------------------: | :-------------------------------------------: |
| `resnet50_mocov2-pre_8xb32-linear-steplr-100e_in1k` | [MOCOV2](https://download.openmmlab.com/mmselfsup/1.x/mocov2/mocov2_resnet50_8xb32-coslr-200e_in1k/mocov2_resnet50_8xb32-coslr-200e_in1k_20220825-b6d23c86.pth) |   25.56    |   4.11    |   67.50   | [config](benchmarks/resnet50_8xb32-linear-steplr-100e_in1k.py) | [model](https://download.openmmlab.com/mmselfsup/1.x/mocov2/mocov2_resnet50_8xb32-coslr-200e_in1k/resnet50_linear-8xb32-steplr-100e_in1k/resnet50_linear-8xb32-steplr-100e_in1k_20220825-994c4128.pth) \| [log](https://download.openmmlab.com/mmselfsup/1.x/mocov2/mocov2_resnet50_8xb32-coslr-200e_in1k/resnet50_linear-8xb32-steplr-100e_in1k/resnet50_linear-8xb32-steplr-100e_in1k_20220825-994c4128.json) |

## Citation

```bibtex
@article{chen2020improved,
  title={Improved baselines with momentum contrastive learning},
  author={Chen, Xinlei and Fan, Haoqi and Girshick, Ross and He, Kaiming},
  journal={arXiv preprint arXiv:2003.04297},
  year={2020}
}
```
