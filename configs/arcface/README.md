# ArcFace

> [ArcFace: Additive Angular Margin Loss for Deep Face Recognition](https://arxiv.org/abs/1801.07698)

<!-- [ALGORITHM] -->

## Abstract

Recently, a popular line of research in face recognition is adopting margins in the well-established softmax loss function to maximize class separability. In this paper, we first introduce an Additive Angular Margin Loss (ArcFace), which not only has a clear geometric interpretation but also significantly enhances the discriminative power. Since ArcFace is susceptible to the massive label noise, we further propose sub-center ArcFace, in which each class contains K sub-centers and training samples only need to be close to any of the K positive sub-centers. Sub-center ArcFace encourages one dominant sub-class that contains the majority of clean faces and non-dominant sub-classes that include hard or noisy faces. Based on this self-propelled isolation, we boost the performance through automatically purifying raw web faces under massive real-world noise. Besides discriminative feature embedding, we also explore the inverse problem, mapping feature vectors to face images. Without training any additional generator or discriminator, the pre-trained ArcFace model can generate identity-preserved face images for both subjects inside and outside the training data only by using the network gradient and Batch Normalization (BN) priors. Extensive experiments demonstrate that ArcFace can enhance the discriminative feature embedding as well as strengthen the generative face synthesis.

<div align=center>
<img src="https://user-images.githubusercontent.com/24734142/212606212-8ffc3cd2-dbc1-4abf-8924-22167f3f6e34.png" width="80%"/>
</div>

## How to use it?

<!-- [TABS-BEGIN] -->

**Retrieve image**

```python
from mmpretrain import ImageRetrievalInferencer

inferencer = ImageRetrievalInferencer('resnet50-arcface_inshop', prototype='demo/')
predict = inferencer('demo/dog.jpg', topk=2)[0]
print(predict[0])
print(predict[1])
```

**Use the model**

```python
import torch
from mmpretrain import get_model

model = get_model('resnet50-arcface_inshop', pretrained=True)
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
python tools/train.py configs/arcface/resnet50-arcface_8xb32_inshop.py
```

Test:

```shell
python tools/test.py configs/arcface/resnet50-arcface_8xb32_inshop.py https://download.openmmlab.com/mmclassification/v0/arcface/resnet50-arcface_inshop_20230202-b766fe7f.pth
```

<!-- [TABS-END] -->

## Models and results

### Image Retrieval on InShop

|           Model           |                      Pretrain                      | Params(M) | Flops(G) | Recall@1 | mAP@10 |                    Config                    |                      Download                      |
| :-----------------------: | :------------------------------------------------: | :-------: | :------: | :------: | :----: | :------------------------------------------: | :------------------------------------------------: |
| `resnet50-arcface_inshop` | [ImageNet-21k-mill](https://download.openmmlab.com/mmclassification/v0/resnet/resnet50_3rdparty-mill_in21k_20220331-faac000b.pth) |   31.69   |  16.48   |  90.18   | 69.30  | [config](./resnet50-arcface_8xb32_inshop.py) | [model](https://download.openmmlab.com/mmclassification/v0/arcface/resnet50-arcface_inshop_20230202-b766fe7f.pth) \| [log](https://download.openmmlab.com/mmclassification/v0/arcface/resnet50-arcface_inshop_20230202-b766fe7f.log) |

## Citation

```bibtex
@inproceedings{deng2018arcface,
title={ArcFace: Additive Angular Margin Loss for Deep Face Recognition},
author={Deng, Jiankang and Guo, Jia and Niannan, Xue and Zafeiriou, Stefanos},
booktitle={CVPR},
year={2019}
}
```
