# SparK

> [Designing BERT for Convolutional Networks: Sparse and Hierarchical Masked Modeling](https://arxiv.org/abs/2301.03580)

<!-- [ALGORITHM] -->

## Abstract

We identify and overcome two key obstacles in extending the success of BERT-style pre-training, or the masked image modeling, to convolutional networks (convnets): (i) convolution operation cannot handle irregular, random-masked input images; (ii) the single-scale nature of BERT pre-training is inconsistent with convnet's hierarchical structure. For (i), we treat unmasked pixels as sparse voxels of 3D point clouds and use sparse convolution to encode. This is the first use of sparse convolution for 2D masked modeling. For (ii), we develop a hierarchical decoder to reconstruct images from multi-scale encoded features. Our method called Sparse masKed modeling (SparK) is general: it can be used directly on any convolutional model without backbone modifications. We validate it on both classical (ResNet) and modern (ConvNeXt) models: on three downstream tasks, it surpasses both state-of-the-art contrastive learning and transformer-based masked modeling by similarly large margins (around +1.0%). Improvements on object detection and instance segmentation are more substantial (up to +3.5%), verifying the strong transferability of features learned. We also find its favorable scaling behavior by observing more gains on larger models. All this evidence reveals a promising future of generative pre-training on convnets. Codes and models are released at https://github.com/keyu-tian/SparK.

<div align=center>
<img src="https://github.com/open-mmlab/mmpretrain/assets/36138628/b93e8d6f-ec1e-4f27-b986-da470fabe7df" width="80%"/>
</div>

## How to use it?

<!-- [TABS-BEGIN] -->

**Predict image**

```python
from mmpretrain import inference_model

predict = inference_model('resnet50_spark-pre_300e_in1k', 'demo/bird.JPEG')
print(predict['pred_class'])
print(predict['pred_score'])
```

**Use the model**

```python
import torch
from mmpretrain import get_model

model = get_model('spark_sparse-resnet50_800e_in1k', pretrained=True)
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
python tools/train.py configs/spark/spark_sparse-resnet50_8xb512-amp-coslr-800e_in1k.py
```

Test:

```shell
python tools/test.py configs/spark/benchmarks/resnet50_8xb256-coslr-300e_in1k.py https://download.openmmlab.com/mmpretrain/v1.0/spark/spark_sparse-resnet50_8xb512-amp-coslr-800e_in1k/resnet50_8xb256-coslr-300e_in1k/resnet50_8xb256-coslr-300e_in1k_20230612-f86aab51.pth
```

<!-- [TABS-END] -->

## Models and results

### Pretrained models

| Model                                    | Params (M) | Flops (G) |                                Config                                 |                                 Download                                 |
| :--------------------------------------- | :--------: | :-------: | :-------------------------------------------------------------------: | :----------------------------------------------------------------------: |
| `spark_sparse-resnet50_800e_in1k`        |   37.97    |   4.10    |     [config](spark_sparse-resnet50_8xb512-amp-coslr-800e_in1k.py)     | [model](https://download.openmmlab.com/mmpretrain/v1.0/spark/spark_sparse-resnet50_8xb512-amp-coslr-800e_in1k/spark_sparse-resnet50_8xb512-amp-coslr-800e_in1k_20230612-e403c28f.pth) \| [log](https://download.openmmlab.com/mmpretrain/v1.0/spark/spark_sparse-resnet50_8xb512-amp-coslr-800e_in1k/spark_sparse-resnet50_8xb512-amp-coslr-800e_in1k_20230612-e403c28f.json) |
| `spark_sparse-convnextv2-tiny_800e_in1k` |   39.73    |   4.47    | [config](spark_sparse-convnextv2-tiny_16xb256-amp-coslr-800e_in1k.py) | [model](https://download.openmmlab.com/mmpretrain/v1.0/spark/spark_sparse-convnextv2-tiny_16xb256-amp-coslr-800e_in1k/spark_sparse-convnextv2-tiny_16xb256-amp-coslr-800e_in1k_20230612-b0ea712e.pth) \| [log](https://download.openmmlab.com/mmpretrain/v1.0/spark/spark_sparse-convnextv2-tiny_16xb256-amp-coslr-800e_in1k/spark_sparse-convnextv2-tiny_16xb256-amp-coslr-800e_in1k_20230612-b0ea712e.json) |

### Image Classification on ImageNet-1k

| Model                                 |                  Pretrain                  | Params (M) | Flops (G) | Top-1 (%) | Top-5 (%) |                  Config                   |                  Download                   |
| :------------------------------------ | :----------------------------------------: | :--------: | :-------: | :-------: | :-------: | :---------------------------------------: | :-----------------------------------------: |
| `resnet50_spark-pre_300e_in1k`        | [SPARK](https://download.openmmlab.com/mmpretrain/v1.0/spark/spark_sparse-resnet50_8xb512-amp-coslr-800e_in1k/spark_sparse-resnet50_8xb512-amp-coslr-800e_in1k_20230612-e403c28f.pth) |   23.52    |   1.31    |   80.10   |   94.90   | [config](benchmarks/resnet50_8xb256-coslr-300e_in1k.py) | [model](https://download.openmmlab.com/mmpretrain/v1.0/spark/spark_sparse-resnet50_8xb512-amp-coslr-800e_in1k/resnet50_8xb256-coslr-300e_in1k/resnet50_8xb256-coslr-300e_in1k_20230612-f86aab51.pth) \| [log](https://download.openmmlab.com/mmpretrain/v1.0/spark/spark_sparse-resnet50_8xb512-amp-coslr-800e_in1k/resnet50_8xb256-coslr-300e_in1k/resnet50_8xb256-coslr-300e_in1k_20230612-f86aab51.json) |
| `convnextv2-tiny_spark-pre_300e_in1k` | [SPARK](https://download.openmmlab.com/mmpretrain/v1.0/spark/spark_sparse-convnextv2-tiny_16xb256-amp-coslr-800e_in1k/spark_sparse-convnextv2-tiny_16xb256-amp-coslr-800e_in1k_20230612-b0ea712e.pth) |   28.64    |   4.47    |   82.80   |   96.30   | [config](benchmarks/convnextv2-tiny_8xb256-coslr-300e_in1k.py) | [model](https://download.openmmlab.com/mmpretrain/v1.0/spark//spark_sparse-convnextv2-tiny_16xb256-amp-coslr-800e_in1k/convnextv2-tiny_8xb256-coslr-300e_in1k/convnextv2-tiny_8xb256-coslr-300e_in1k_20230612-ffc78743.pth) \| [log](https://download.openmmlab.com/mmpretrain/v1.0/spark//spark_sparse-convnextv2-tiny_16xb256-amp-coslr-800e_in1k/convnextv2-tiny_8xb256-coslr-300e_in1k/convnextv2-tiny_8xb256-coslr-300e_in1k_20230612-ffc78743.json) |

## Citation

```bibtex
@Article{tian2023designing,
  author  = {Keyu Tian and Yi Jiang and Qishuai Diao and Chen Lin and Liwei Wang and Zehuan Yuan},
  title   = {Designing BERT for Convolutional Networks: Sparse and Hierarchical Masked Modeling},
  journal = {arXiv:2301.03580},
  year    = {2023},
}
```
