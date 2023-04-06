# SimCLR

> [A simple framework for contrastive learning of visual representations](https://arxiv.org/abs/2002.05709)

<!-- [ALGORITHM] -->

## Abstract

This paper presents SimCLR: a simple framework for contrastive learning of visual representations. We simplify recently proposed contrastive self-supervised learning algorithms without requiring specialized architectures or a memory bank. In order to understand what enables the contrastive prediction tasks to learn useful representations, we systematically study the major components of our framework. We show that (1) composition of data augmentations plays a critical role in defining effective predictive tasks, (2) introducing a learnable nonlinear transformation between the representation and the contrastive loss substantially improves the quality of the learned representations, and (3) contrastive learning benefits from larger batch sizes and more training steps compared to supervised learning. By combining these findings, we are able to considerably outperform previous methods for self-supervised and semi-supervised learning on ImageNet. A linear classifier trained on self-supervised representations learned by SimCLR achieves 76.5% top-1 accuracy, which is a 7% relative improvement over previous state-of-the-art, matching the performance of a supervised ResNet-50.

<div align=center>
<img  src="https://user-images.githubusercontent.com/36138628/149723851-cf5f309e-d891-454d-90c0-e5337e5a11ed.png" width="400" />
</div>

## How to use it?

<!-- [TABS-BEGIN] -->

**Predict image**

```python
from mmpretrain import inference_model

predict = inference_model('resnet50_simclr-200e-pre_8xb512-linear-coslr-90e_in1k', 'demo/bird.JPEG')
print(predict['pred_class'])
print(predict['pred_score'])
```

**Use the model**

```python
import torch
from mmpretrain import get_model

model = get_model('simclr_resnet50_16xb256-coslr-200e_in1k', pretrained=True)
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
python tools/train.py configs/simclr/simclr_resnet50_16xb256-coslr-200e_in1k.py
```

Test:

```shell
python tools/test.py configs/simclr/benchmarks/resnet50_8xb512-linear-coslr-90e_in1k.py https://download.openmmlab.com/mmselfsup/1.x/simclr/simclr_resnet50_16xb256-coslr-200e_in1k/resnet50_linear-8xb512-coslr-90e_in1k/resnet50_linear-8xb512-coslr-90e_in1k_20220825-f12c0457.pth
```

<!-- [TABS-END] -->

## Models and results

### Pretrained models

| Model                                     | Params (M) | Flops (G) |                        Config                        |                                         Download                                         |
| :---------------------------------------- | :--------: | :-------: | :--------------------------------------------------: | :--------------------------------------------------------------------------------------: |
| `simclr_resnet50_16xb256-coslr-200e_in1k` |   27.97    |   4.11    | [config](simclr_resnet50_16xb256-coslr-200e_in1k.py) | [model](https://download.openmmlab.com/mmselfsup/1.x/simclr/simclr_resnet50_16xb256-coslr-200e_in1k/simclr_resnet50_16xb256-coslr-200e_in1k_20220825-4d9cce50.pth) \| [log](https://download.openmmlab.com/mmselfsup/1.x/simclr/simclr_resnet50_16xb256-coslr-200e_in1k/simclr_resnet50_16xb256-coslr-200e_in1k_20220825-4d9cce50.json) |
| `simclr_resnet50_16xb256-coslr-800e_in1k` |   27.97    |   4.11    | [config](simclr_resnet50_16xb256-coslr-800e_in1k.py) | [model](https://download.openmmlab.com/mmselfsup/1.x/simclr/simclr_resnet50_16xb256-coslr-800e_in1k/simclr_resnet50_16xb256-coslr-800e_in1k_20220825-85fcc4de.pth) \| [log](https://download.openmmlab.com/mmselfsup/1.x/simclr/simclr_resnet50_16xb256-coslr-800e_in1k/simclr_resnet50_16xb256-coslr-800e_in1k_20220825-85fcc4de.json) |

### Image Classification on ImageNet-1k

| Model                                     |                   Pretrain                   | Params (M) | Flops (G) | Top-1 (%) |                   Config                   |                   Download                    |
| :---------------------------------------- | :------------------------------------------: | :--------: | :-------: | :-------: | :----------------------------------------: | :-------------------------------------------: |
| `resnet50_simclr-200e-pre_8xb512-linear-coslr-90e_in1k` | [SIMCLR 200-Epochs](https://download.openmmlab.com/mmselfsup/1.x/simclr/simclr_resnet50_16xb256-coslr-200e_in1k/simclr_resnet50_16xb256-coslr-200e_in1k_20220825-4d9cce50.pth) |   25.56    |   4.11    |   66.90   | [config](benchmarks/resnet50_8xb512-linear-coslr-90e_in1k.py) | [model](https://download.openmmlab.com/mmselfsup/1.x/simclr/simclr_resnet50_16xb256-coslr-200e_in1k/resnet50_linear-8xb512-coslr-90e_in1k/resnet50_linear-8xb512-coslr-90e_in1k_20220825-f12c0457.pth) \| [log](https://download.openmmlab.com/mmselfsup/1.x/simclr/simclr_resnet50_16xb256-coslr-200e_in1k/resnet50_linear-8xb512-coslr-90e_in1k/resnet50_linear-8xb512-coslr-90e_in1k_20220825-f12c0457.json) |
| `resnet50_simclr-800e-pre_8xb512-linear-coslr-90e_in1k` | [SIMCLR 800-Epochs](https://download.openmmlab.com/mmselfsup/1.x/simclr/simclr_resnet50_16xb256-coslr-800e_in1k/simclr_resnet50_16xb256-coslr-800e_in1k_20220825-85fcc4de.pth) |   25.56    |   4.11    |   69.20   | [config](benchmarks/resnet50_8xb512-linear-coslr-90e_in1k.py) | [model](https://download.openmmlab.com/mmselfsup/1.x/simclr/simclr_resnet50_16xb256-coslr-800e_in1k/resnet50_linear-8xb512-coslr-90e_in1k/resnet50_linear-8xb512-coslr-90e_in1k_20220825-b80ae1e5.pth) \| [log](https://download.openmmlab.com/mmselfsup/1.x/simclr/simclr_resnet50_16xb256-coslr-800e_in1k/resnet50_linear-8xb512-coslr-90e_in1k/resnet50_linear-8xb512-coslr-90e_in1k_20220825-b80ae1e5.json) |

## Citation

```bibtex
@inproceedings{chen2020simple,
  title={A simple framework for contrastive learning of visual representations},
  author={Chen, Ting and Kornblith, Simon and Norouzi, Mohammad and Hinton, Geoffrey},
  booktitle={ICML},
  year={2020},
}
```
