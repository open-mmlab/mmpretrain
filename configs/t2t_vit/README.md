# Tokens-to-Token ViT

> [Tokens-to-Token ViT: Training Vision Transformers from Scratch on ImageNet](https://arxiv.org/abs/2101.11986)

<!-- [ALGORITHM] -->

## Abstract

Transformers, which are popular for language modeling, have been explored for solving vision tasks recently, e.g., the Vision Transformer (ViT) for image classification. The ViT model splits each image into a sequence of tokens with fixed length and then applies multiple Transformer layers to model their global relation for classification. However, ViT achieves inferior performance to CNNs when trained from scratch on a midsize dataset like ImageNet. We find it is because: 1) the simple tokenization of input images fails to model the important local structure such as edges and lines among neighboring pixels, leading to low training sample efficiency; 2) the redundant attention backbone design of ViT leads to limited feature richness for fixed computation budgets and limited training samples. To overcome such limitations, we propose a new Tokens-To-Token Vision Transformer (T2T-ViT), which incorporates 1) a layer-wise Tokens-to-Token (T2T) transformation to progressively structurize the image to tokens by recursively aggregating neighboring Tokens into one Token (Tokens-to-Token), such that local structure represented by surrounding tokens can be modeled and tokens length can be reduced; 2) an efficient backbone with a deep-narrow structure for vision transformer motivated by CNN architecture design after empirical study. Notably, T2T-ViT reduces the parameter count and MACs of vanilla ViT by half, while achieving more than 3.0% improvement when trained from scratch on ImageNet. It also outperforms ResNets and achieves comparable performance with MobileNets by directly training on ImageNet. For example, T2T-ViT with comparable size to ResNet50 (21.5M parameters) can achieve 83.3% top1 accuracy in image resolution 384×384 on ImageNet.

<div align=center>
<img src="https://user-images.githubusercontent.com/26739999/142578381-e9040610-05d9-457c-8bf5-01c2fa94add2.png" width="60%"/>
</div>

## How to use it?

<!-- [TABS-BEGIN] -->

**Predict image**

```python
from mmpretrain import inference_model

predict = inference_model('t2t-vit-t-14_8xb64_in1k', 'demo/bird.JPEG')
print(predict['pred_class'])
print(predict['pred_score'])
```

**Use the model**

```python
import torch
from mmpretrain import get_model

model = get_model('t2t-vit-t-14_8xb64_in1k', pretrained=True)
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
python tools/train.py configs/t2t_vit/t2t-vit-t-14_8xb64_in1k.py
```

Test:

```shell
python tools/test.py configs/t2t_vit/t2t-vit-t-14_8xb64_in1k.py https://download.openmmlab.com/mmclassification/v0/t2t-vit/t2t-vit-t-14_8xb64_in1k_20211220-f7378dd5.pth
```

<!-- [TABS-END] -->

## Models and results

### Image Classification on ImageNet-1k

| Model                     |   Pretrain   | Params (M) | Flops (G) | Top-1 (%) | Top-5 (%) |                Config                |                                          Download                                          |
| :------------------------ | :----------: | :--------: | :-------: | :-------: | :-------: | :----------------------------------: | :----------------------------------------------------------------------------------------: |
| `t2t-vit-t-14_8xb64_in1k` | From scratch |   21.47    |   4.34    |   81.83   |   95.84   | [config](t2t-vit-t-14_8xb64_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/t2t-vit/t2t-vit-t-14_8xb64_in1k_20211220-f7378dd5.pth) \| [log](https://download.openmmlab.com/mmclassification/v0/t2t-vit/t2t-vit-t-14_8xb64_in1k_20211220-f7378dd5.json) |
| `t2t-vit-t-19_8xb64_in1k` | From scratch |   39.08    |   7.80    |   82.63   |   96.18   | [config](t2t-vit-t-19_8xb64_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/t2t-vit/t2t-vit-t-19_8xb64_in1k_20211214-7f5e3aaf.pth) \| [log](https://download.openmmlab.com/mmclassification/v0/t2t-vit/t2t-vit-t-19_8xb64_in1k_20211214-7f5e3aaf.json) |
| `t2t-vit-t-24_8xb64_in1k` | From scratch |   64.00    |   12.69   |   82.71   |   96.09   | [config](t2t-vit-t-24_8xb64_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/t2t-vit/t2t-vit-t-24_8xb64_in1k_20211214-b2a68ae3.pth) \| [log](https://download.openmmlab.com/mmclassification/v0/t2t-vit/t2t-vit-t-24_8xb64_in1k_20211214-b2a68ae3.json) |

## Citation

```bibtex
@article{yuan2021tokens,
  title={Tokens-to-token vit: Training vision transformers from scratch on imagenet},
  author={Yuan, Li and Chen, Yunpeng and Wang, Tao and Yu, Weihao and Shi, Yujun and Tay, Francis EH and Feng, Jiashi and Yan, Shuicheng},
  journal={arXiv preprint arXiv:2101.11986},
  year={2021}
}
```
