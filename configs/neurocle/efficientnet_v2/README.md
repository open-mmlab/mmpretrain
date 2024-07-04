# EfficientNetV2

> [EfficientNetV2: Smaller Models and Faster Training](https://arxiv.org/abs/2104.00298)

<!-- [ALGORITHM] -->

## Abstract

This paper introduces EfficientNetV2, a new family of convolutional networks that have faster training speed and better parameter efficiency than previous models. To develop this family of models, we use a combination of training-aware neural architecture search and scaling, to jointly optimize training speed and parameter efficiency. The models were searched from the search space enriched with new ops such as Fused-MBConv. Our experiments show that EfficientNetV2 models train much faster than state-of-the-art models while being up to 6.8x smaller.   Our training can be further sped up by progressively increasing the image size during training, but it often causes a drop in accuracy. To compensate for this accuracy drop, we propose to adaptively adjust regularization (e.g., dropout and data augmentation) as well, such that we can achieve both fast training and good accuracy.   With progressive learning, our EfficientNetV2 significantly outperforms previous models on ImageNet and CIFAR/Cars/Flowers datasets. By pretraining on the same ImageNet21k, our EfficientNetV2 achieves 87.3% top-1 accuracy on ImageNet ILSVRC2012, outperforming the recent ViT by 2.0% accuracy while training 5x-11x faster using the same computing resources. Code will be available at https://github.com/google/automl/tree/master/efficientnetv2.

<div align=center>
<img src="https://user-images.githubusercontent.com/18586273/208616931-0c5107f1-f08c-48d3-8694-7a6eaf227dc2.png" width="50%"/>
</div>

## How to use it?

<!-- [TABS-BEGIN] -->

**Predict image**

```python
from mmpretrain import inference_model

predict = inference_model('efficientnetv2-b0_3rdparty_in1k', 'demo/bird.JPEG')
print(predict['pred_class'])
print(predict['pred_score'])
```

**Use the model**

```python
import torch
from mmpretrain import get_model

model = get_model('efficientnetv2-b0_3rdparty_in1k', pretrained=True)
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
python tools/test.py configs/efficientnet_v2/efficientnetv2-b0_8xb32_in1k.py https://download.openmmlab.com/mmclassification/v0/efficientnetv2/efficientnetv2-b0_3rdparty_in1k_20221221-9ef6e736.pth
```

<!-- [TABS-END] -->

## Models and results

### Pretrained models

| Model                                | Params (M) | Flops (G) |                   Config                   |                                                Download                                                 |
| :----------------------------------- | :--------: | :-------: | :----------------------------------------: | :-----------------------------------------------------------------------------------------------------: |
| `efficientnetv2-s_3rdparty_in21k`\*  |   48.16    |   3.31    | [config](efficientnetv2-s_8xb32_in21k.py)  | [model](https://download.openmmlab.com/mmclassification/v0/efficientnetv2/efficientnetv2-s_3rdparty_in21k_20221220-c0572b56.pth) |
| `efficientnetv2-m_3rdparty_in21k`\*  |   80.84    |   5.86    | [config](efficientnetv2-m_8xb32_in21k.py)  | [model](https://download.openmmlab.com/mmclassification/v0/efficientnetv2/efficientnetv2-m_3rdparty_in21k_20221220-073e944c.pth) |
| `efficientnetv2-l_3rdparty_in21k`\*  |   145.22   |   13.11   | [config](efficientnetv2-l_8xb32_in21k.py)  | [model](https://download.openmmlab.com/mmclassification/v0/efficientnetv2/efficientnetv2-l_3rdparty_in21k_20221220-f28f91e1.pth) |
| `efficientnetv2-xl_3rdparty_in21k`\* |   234.82   |   18.86   | [config](efficientnetv2-xl_8xb32_in21k.py) | [model](https://download.openmmlab.com/mmclassification/v0/efficientnetv2/efficientnetv2-xl_3rdparty_in21k_20221220-b2c9329c.pth) |

*Models with * are converted from the [timm](https://github.com/rwightman/pytorch-image-models/blob/main/timm/models/efficientnet.py). The config files of these models are only for inference. We haven't reproduce the training results.*

### Image Classification on ImageNet-1k

| Model                                         |   Pretrain   | Params (M) | Flops (G) | Top-1 (%) | Top-5 (%) |                     Config                      |                          Download                           |
| :-------------------------------------------- | :----------: | :--------: | :-------: | :-------: | :-------: | :---------------------------------------------: | :---------------------------------------------------------: |
| `efficientnetv2-b0_3rdparty_in1k`\*           | From scratch |    7.14    |   0.92    |   78.52   |   94.44   |    [config](efficientnetv2-b0_8xb32_in1k.py)    | [model](https://download.openmmlab.com/mmclassification/v0/efficientnetv2/efficientnetv2-b0_3rdparty_in1k_20221221-9ef6e736.pth) |
| `efficientnetv2-b1_3rdparty_in1k`\*           | From scratch |    8.14    |   1.44    |   79.80   |   94.89   |    [config](efficientnetv2-b1_8xb32_in1k.py)    | [model](https://download.openmmlab.com/mmclassification/v0/efficientnetv2/efficientnetv2-b1_3rdparty_in1k_20221221-6955d9ce.pth) |
| `efficientnetv2-b2_3rdparty_in1k`\*           | From scratch |   10.10    |   1.99    |   80.63   |   95.30   |    [config](efficientnetv2-b2_8xb32_in1k.py)    | [model](https://download.openmmlab.com/mmclassification/v0/efficientnetv2/efficientnetv2-b2_3rdparty_in1k_20221221-74f7d493.pth) |
| `efficientnetv2-b3_3rdparty_in1k`\*           | From scratch |   14.36    |   3.50    |   82.03   |   95.88   |    [config](efficientnetv2-b3_8xb32_in1k.py)    | [model](https://download.openmmlab.com/mmclassification/v0/efficientnetv2/efficientnetv2-b3_3rdparty_in1k_20221221-b6f07a36.pth) |
| `efficientnetv2-s_3rdparty_in1k`\*            | From scratch |   21.46    |   9.72    |   83.82   |   96.67   | [config](efficientnetv2-s_8xb32_in1k-384px.py)  | [model](https://download.openmmlab.com/mmclassification/v0/efficientnetv2/efficientnetv2-s_3rdparty_in1k_20221220-f0eaff9d.pth) |
| `efficientnetv2-m_3rdparty_in1k`\*            | From scratch |   54.14    |   26.88   |   85.01   |   97.26   | [config](efficientnetv2-m_8xb32_in1k-480px.py)  | [model](https://download.openmmlab.com/mmclassification/v0/efficientnetv2/efficientnetv2-m_3rdparty_in1k_20221220-9dc0c729.pth) |
| `efficientnetv2-l_3rdparty_in1k`\*            | From scratch |   118.52   |   60.14   |   85.43   |   97.31   | [config](efficientnetv2-l_8xb32_in1k-480px.py)  | [model](https://download.openmmlab.com/mmclassification/v0/efficientnetv2/efficientnetv2-l_3rdparty_in1k_20221220-5c3bac0f.pth) |
| `efficientnetv2-s_in21k-pre_3rdparty_in1k`\*  | ImageNet-21k |   21.46    |   9.72    |   84.29   |   97.26   | [config](efficientnetv2-s_8xb32_in1k-384px.py)  | [model](https://download.openmmlab.com/mmclassification/v0/efficientnetv2/efficientnetv2-s_in21k-pre-3rdparty_in1k_20221220-7a7c8475.pth) |
| `efficientnetv2-m_in21k-pre_3rdparty_in1k`\*  | ImageNet-21k |   54.14    |   26.88   |   85.47   |   97.76   | [config](efficientnetv2-m_8xb32_in1k-480px.py)  | [model](https://download.openmmlab.com/mmclassification/v0/efficientnetv2/efficientnetv2-m_in21k-pre-3rdparty_in1k_20221220-a1013a04.pth) |
| `efficientnetv2-l_in21k-pre_3rdparty_in1k`\*  | ImageNet-21k |   118.52   |   60.14   |   86.31   |   97.99   | [config](efficientnetv2-l_8xb32_in1k-480px.py)  | [model](https://download.openmmlab.com/mmclassification/v0/efficientnetv2/efficientnetv2-l_in21k-pre-3rdparty_in1k_20221220-63df0efd.pth) |
| `efficientnetv2-xl_in21k-pre_3rdparty_in1k`\* | ImageNet-21k |   208.12   |   98.34   |   86.39   |   97.83   | [config](efficientnetv2-xl_8xb32_in1k-512px.py) | [model](https://download.openmmlab.com/mmclassification/v0/efficientnetv2/efficientnetv2-xl_in21k-pre-3rdparty_in1k_20221220-583ac18b.pth) |

*Models with * are converted from the [timm](https://github.com/rwightman/pytorch-image-models/blob/main/timm/models/efficientnet.py). The config files of these models are only for inference. We haven't reproduce the training results.*

## Citation

```bibtex
@inproceedings{tan2021efficientnetv2,
  title={Efficientnetv2: Smaller models and faster training},
  author={Tan, Mingxing and Le, Quoc},
  booktitle={International Conference on Machine Learning},
  pages={10096--10106},
  year={2021},
  organization={PMLR}
}
```
