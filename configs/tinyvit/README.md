# TinyViT

> [TinyViT: Fast Pretraining Distillation for Small Vision Transformers](https://arxiv.org/abs/2207.10666)

<!-- [ALGORITHM] -->

## Abstract

Vision transformer (ViT) recently has drawn great attention in computer vision due to its remarkable model capability. However, most prevailing ViT models suffer from huge number of parameters, restricting their applicability on devices with limited resources. To alleviate this issue, we propose TinyViT, a new family of tiny and efficient small vision transformers pretrained on large-scale datasets with our proposed fast distillation framework. The central idea is to transfer knowledge from large pretrained models to small ones, while enabling small models to get the dividends of massive pretraining data. More specifically, we apply distillation during pretraining for knowledge transfer. The logits of large teacher models are sparsified and stored in disk in advance to save the memory cost and computation overheads. The tiny student transformers are automatically scaled down from a large pretrained model with computation and parameter constraints. Comprehensive experiments demonstrate the efficacy of TinyViT. It achieves a top-1 accuracy of 84.8% on ImageNet-1k with only 21M parameters, being comparable to SwinB pretrained on ImageNet-21k while using 4.2 times fewer parameters. Moreover, increasing image resolutions, TinyViT can reach 86.5% accuracy, being slightly better than Swin-L while using only 11% parameters. Last but not the least, we demonstrate a good transfer ability of TinyViT on various downstream tasks.

<div align=center>
<img src="https://github.com/microsoft/Cream/raw/main/TinyViT/.figure/framework.png" width="100%">
</div>

## How to use it?

<!-- [TABS-BEGIN] -->

**Predict image**

```python
from mmpretrain import inference_model

predict = inference_model('tinyvit-5m_3rdparty_in1k', 'demo/bird.JPEG')
print(predict['pred_class'])
print(predict['pred_score'])
```

**Use the model**

```python
import torch
from mmpretrain import get_model

model = get_model('tinyvit-5m_3rdparty_in1k', pretrained=True)
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
python tools/test.py configs/tinyvit/tinyvit-5m_8xb256_in1k.py https://download.openmmlab.com/mmclassification/v0/tinyvit/tinyvit-5m_3rdparty_in1k_20221021-62cb5abf.pth
```

<!-- [TABS-END] -->

## Models and results

### Image Classification on ImageNet-1k

| Model                                          |       Pretrain       | Params (M) | Flops (G) | Top-1 (%) | Top-5 (%) |                     Config                      |                      Download                      |
| :--------------------------------------------- | :------------------: | :--------: | :-------: | :-------: | :-------: | :---------------------------------------------: | :------------------------------------------------: |
| `tinyvit-5m_3rdparty_in1k`\*                   |     From scratch     |    5.39    |   1.29    |   79.02   |   94.74   |       [config](tinyvit-5m_8xb256_in1k.py)       | [model](https://download.openmmlab.com/mmclassification/v0/tinyvit/tinyvit-5m_3rdparty_in1k_20221021-62cb5abf.pth) |
| `tinyvit-5m_in21k-distill-pre_3rdparty_in1k`\* | ImageNet-21k DISTILL |    5.39    |   1.29    |   80.71   |   95.57   |   [config](tinyvit-5m-distill_8xb256_in1k.py)   | [model](https://download.openmmlab.com/mmclassification/v0/tinyvit/tinyvit-5m_in21k-distill-pre_3rdparty_in1k_20221021-d4b010a8.pth) |
| `tinyvit-11m_3rdparty_in1k`\*                  |     From scratch     |   11.00    |   2.05    |   81.44   |   95.79   |      [config](tinyvit-11m_8xb256_in1k.py)       | [model](https://download.openmmlab.com/mmclassification/v0/tinyvit/tinyvit-11m_3rdparty_in1k_20221021-11ccef16.pth) |
| `tinyvit-11m_in21k-distill-pre_3rdparty_in1k`\* | ImageNet-21k DISTILL |   11.00    |   2.05    |   83.19   |   96.53   |  [config](tinyvit-11m-distill_8xb256_in1k.py)   | [model](https://download.openmmlab.com/mmclassification/v0/tinyvit/tinyvit-11m_in21k-distill-pre_3rdparty_in1k_20221021-5d3bc0dc.pth) |
| `tinyvit-21m_3rdparty_in1k`\*                  |     From scratch     |   21.20    |   4.30    |   83.08   |   96.58   |      [config](tinyvit-21m_8xb256_in1k.py)       | [model](https://download.openmmlab.com/mmclassification/v0/tinyvit/tinyvit-21m_3rdparty_in1k_20221021-5346ba34.pth) |
| `tinyvit-21m_in21k-distill-pre_3rdparty_in1k`\* | ImageNet-21k DISTILL |   21.20    |   4.30    |   84.85   |   97.27   |  [config](tinyvit-21m-distill_8xb256_in1k.py)   | [model](https://download.openmmlab.com/mmclassification/v0/tinyvit/tinyvit-21m_in21k-distill-pre_3rdparty_in1k_20221021-3d9b30a2.pth) |
| `tinyvit-21m_in21k-distill-pre_3rdparty_in1k-384px`\* | ImageNet-21k DISTILL |   21.23    |   13.85   |   86.21   |   97.77   | [config](tinyvit-21m-distill_8xb256_in1k-384px.py) | [model](https://download.openmmlab.com/mmclassification/v0/tinyvit/tinyvit-21m_in21k-distill-pre_3rdparty_in1k-384px_20221021-65be6b3f.pth) |
| `tinyvit-21m_in21k-distill-pre_3rdparty_in1k-512px`\* | ImageNet-21k DISTILL |   21.27    |   27.15   |   86.44   |   97.89   | [config](tinyvit-21m-distill_8xb256_in1k-512px.py) | [model](https://download.openmmlab.com/mmclassification/v0/tinyvit/tinyvit-21m_in21k-distill-pre_3rdparty_in1k-512px_20221021-e42a9bea.pth) |

*Models with * are converted from the [official repo](https://github.com/microsoft/Cream/tree/main/TinyViT). The config files of these models are only for inference. We haven't reproduce the training results.*

## Citation

```bibtex
@InProceedings{tiny_vit,
  title={TinyViT: Fast Pretraining Distillation for Small Vision Transformers},
  author={Wu, Kan and Zhang, Jinnian and Peng, Houwen and Liu, Mengchen and Xiao, Bin and Fu, Jianlong and Yuan, Lu},
  booktitle={European conference on computer vision (ECCV)},
  year={2022}
}
```
