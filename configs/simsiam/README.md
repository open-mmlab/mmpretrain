# SimSiam

> [Exploring simple siamese representation learning](https://arxiv.org/abs/2011.10566)

<!-- [ALGORITHM] -->

## Abstract

Siamese networks have become a common structure in various recent models for unsupervised visual representation learning. These models maximize the similarity between two augmentations of one image, subject to certain conditions for avoiding collapsing solutions. In this paper, we report surprising empirical results that simple Siamese networks can learn meaningful representations even using none of the following: (i) negative sample pairs, (ii) large batches, (iii) momentum encoders. Our experiments show that collapsing solutions do exist for the loss and structure, but a stop-gradient operation plays an essential role in preventing collapsing. We provide a hypothesis on the implication of stop-gradient, and further show proof-of-concept experiments verifying it. Our “SimSiam” method achieves competitive results on ImageNet and downstream tasks. We hope this simple baseline will motivate people to rethink the roles of Siamese architectures for unsupervised representation learning.

<div align=center>
<img  src="https://user-images.githubusercontent.com/36138628/149724180-bc7bac6a-fcb8-421e-b8f1-9550c624d154.png" width="500" />
</div>

## How to use it?

<!-- [TABS-BEGIN] -->

**Predict image**

```python
from mmpretrain import inference_model

predict = inference_model('resnet50_simsiam-100e-pre_8xb512-linear-coslr-90e_in1k', 'demo/bird.JPEG')
print(predict['pred_class'])
print(predict['pred_score'])
```

**Use the model**

```python
import torch
from mmpretrain import get_model

model = get_model('simsiam_resnet50_8xb32-coslr-100e_in1k', pretrained=True)
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
python tools/train.py configs/simsiam/simsiam_resnet50_8xb32-coslr-100e_in1k.py
```

Test:

```shell
python tools/test.py configs/simsiam/benchmarks/resnet50_8xb512-linear-coslr-90e_in1k.py https://download.openmmlab.com/mmselfsup/1.x/simsiam/simsiam_resnet50_8xb32-coslr-100e_in1k/resnet50_linear-8xb512-coslr-90e_in1k/resnet50_linear-8xb512-coslr-90e_in1k_20220825-f53ba400.pth
```

<!-- [TABS-END] -->

## Models and results

### Pretrained models

| Model                                    | Params (M) | Flops (G) |                       Config                        |                                          Download                                          |
| :--------------------------------------- | :--------: | :-------: | :-------------------------------------------------: | :----------------------------------------------------------------------------------------: |
| `simsiam_resnet50_8xb32-coslr-100e_in1k` |   38.20    |   4.11    | [config](simsiam_resnet50_8xb32-coslr-100e_in1k.py) | [model](https://download.openmmlab.com/mmselfsup/1.x/simsiam/simsiam_resnet50_8xb32-coslr-100e_in1k/simsiam_resnet50_8xb32-coslr-100e_in1k_20220825-d07cb2e6.pth) \| [log](https://download.openmmlab.com/mmselfsup/1.x/simsiam/simsiam_resnet50_8xb32-coslr-100e_in1k/simsiam_resnet50_8xb32-coslr-100e_in1k_20220825-d07cb2e6.json) |
| `simsiam_resnet50_8xb32-coslr-200e_in1k` |   38.20    |   4.11    | [config](simsiam_resnet50_8xb32-coslr-200e_in1k.py) | [model](https://download.openmmlab.com/mmselfsup/1.x/simsiam/simsiam_resnet50_8xb32-coslr-200e_in1k/simsiam_resnet50_8xb32-coslr-200e_in1k_20220825-efe91299.pth) \| [log](https://download.openmmlab.com/mmselfsup/1.x/simsiam/simsiam_resnet50_8xb32-coslr-200e_in1k/simsiam_resnet50_8xb32-coslr-200e_in1k_20220825-efe91299.json) |

### Image Classification on ImageNet-1k

| Model                                     |                   Pretrain                   | Params (M) | Flops (G) | Top-1 (%) |                   Config                   |                   Download                    |
| :---------------------------------------- | :------------------------------------------: | :--------: | :-------: | :-------: | :----------------------------------------: | :-------------------------------------------: |
| `resnet50_simsiam-100e-pre_8xb512-linear-coslr-90e_in1k` | [SIMSIAM 100-Epochs](https://download.openmmlab.com/mmselfsup/1.x/simsiam/simsiam_resnet50_8xb32-coslr-100e_in1k/simsiam_resnet50_8xb32-coslr-100e_in1k_20220825-d07cb2e6.pth) |   25.56    |   4.11    |   68.30   | [config](benchmarks/resnet50_8xb512-linear-coslr-90e_in1k.py) | [model](https://download.openmmlab.com/mmselfsup/1.x/simsiam/simsiam_resnet50_8xb32-coslr-100e_in1k/resnet50_linear-8xb512-coslr-90e_in1k/resnet50_linear-8xb512-coslr-90e_in1k_20220825-f53ba400.pth) \| [log](https://download.openmmlab.com/mmselfsup/1.x/simsiam/simsiam_resnet50_8xb32-coslr-100e_in1k/resnet50_linear-8xb512-coslr-90e_in1k/resnet50_linear-8xb512-coslr-90e_in1k_20220825-f53ba400.json) |
| `resnet50_simsiam-200e-pre_8xb512-linear-coslr-90e_in1k` | [SIMSIAM 200-Epochs](https://download.openmmlab.com/mmselfsup/1.x/simsiam/simsiam_resnet50_8xb32-coslr-200e_in1k/simsiam_resnet50_8xb32-coslr-200e_in1k_20220825-efe91299.pth) |   25.56    |   4.11    |   69.80   | [config](benchmarks/resnet50_8xb512-linear-coslr-90e_in1k.py) | [model](https://download.openmmlab.com/mmselfsup/1.x/simsiam/simsiam_resnet50_8xb32-coslr-200e_in1k/resnet50_linear-8xb512-coslr-90e_in1k/resnet50_linear-8xb512-coslr-90e_in1k_20220825-519b5135.pth) \| [log](https://download.openmmlab.com/mmselfsup/1.x/simsiam/simsiam_resnet50_8xb32-coslr-200e_in1k/resnet50_linear-8xb512-coslr-90e_in1k/resnet50_linear-8xb512-coslr-90e_in1k_20220825-519b5135.json) |

## Citation

```bibtex
@inproceedings{chen2021exploring,
  title={Exploring simple siamese representation learning},
  author={Chen, Xinlei and He, Kaiming},
  booktitle={CVPR},
  year={2021}
}
```
