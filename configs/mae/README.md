# MAE

> [Masked Autoencoders Are Scalable Vision Learners](https://arxiv.org/abs/2111.06377)

<!-- [ALGORITHM] -->

## Abstract

This paper shows that masked autoencoders (MAE) are
scalable self-supervised learners for computer vision. Our
MAE approach is simple: we mask random patches of the
input image and reconstruct the missing pixels. It is based
on two core designs. First, we develop an asymmetric
encoder-decoder architecture, with an encoder that operates only on the
visible subset of patches (without mask tokens), along with a lightweight
decoder that reconstructs the original image from the latent representation
and mask tokens. Second, we find that masking a high proportion
of the input image, e.g., 75%, yields a nontrivial and
meaningful self-supervisory task. Coupling these two designs enables us to
train large models efficiently and effectively: we accelerate
training (by 3× or more) and improve accuracy. Our scalable approach allows
for learning high-capacity models that generalize well: e.g., a vanilla
ViT-Huge model achieves the best accuracy (87.8%) among
methods that use only ImageNet-1K data. Transfer performance in downstream tasks outperforms supervised pretraining and shows promising scaling behavior.

<div align=center>
<img src="https://user-images.githubusercontent.com/30762564/150733959-2959852a-c7bd-4d3f-911f-3e8d8839fe67.png" width="80%"/>
</div>

## How to use it?

<!-- [TABS-BEGIN] -->

**Predict image**

```python
from mmpretrain import inference_model

predict = inference_model('vit-base-p16_mae-300e-pre_8xb128-coslr-100e_in1k', 'demo/bird.JPEG')
print(predict['pred_class'])
print(predict['pred_score'])
```

**Use the model**

```python
import torch
from mmpretrain import get_model

model = get_model('mae_vit-base-p16_8xb512-amp-coslr-300e_in1k', pretrained=True)
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
python tools/train.py configs/mae/mae_vit-base-p16_8xb512-amp-coslr-300e_in1k.py
```

Test:

```shell
python tools/test.py configs/mae/benchmarks/vit-base-p16_8xb128-coslr-100e_in1k.py None
```

<!-- [TABS-END] -->

## Models and results

### Pretrained models

| Model                                           | Params (M) | Flops (G) |                           Config                           |                                   Download                                   |
| :---------------------------------------------- | :--------: | :-------: | :--------------------------------------------------------: | :--------------------------------------------------------------------------: |
| `mae_vit-base-p16_8xb512-amp-coslr-300e_in1k`   |   111.91   |   17.58   |  [config](mae_vit-base-p16_8xb512-amp-coslr-300e_in1k.py)  | [model](https://download.openmmlab.com/mmselfsup/1.x/mae/mae_vit-base-p16_8xb512-fp16-coslr-300e_in1k/mae_vit-base-p16_8xb512-coslr-300e-fp16_in1k_20220829-c2cf66ba.pth) \| [log](https://download.openmmlab.com/mmselfsup/1.x/mae/mae_vit-base-p16_8xb512-fp16-coslr-300e_in1k/mae_vit-base-p16_8xb512-coslr-300e-fp16_in1k_20220829-c2cf66ba.json) |
| `mae_vit-base-p16_8xb512-amp-coslr-400e_in1k`   |   111.91   |   17.58   |  [config](mae_vit-base-p16_8xb512-amp-coslr-400e_in1k.py)  | [model](https://download.openmmlab.com/mmselfsup/1.x/mae/mae_vit-base-p16_8xb512-fp16-coslr-400e_in1k/mae_vit-base-p16_8xb512-coslr-400e-fp16_in1k_20220825-bc79e40b.pth) \| [log](https://download.openmmlab.com/mmselfsup/1.x/mae/mae_vit-base-p16_8xb512-fp16-coslr-400e_in1k/mae_vit-base-p16_8xb512-coslr-400e-fp16_in1k_20220825-bc79e40b.json) |
| `mae_vit-base-p16_8xb512-amp-coslr-800e_in1k`   |   111.91   |   17.58   |  [config](mae_vit-base-p16_8xb512-amp-coslr-800e_in1k.py)  | [model](https://download.openmmlab.com/mmselfsup/1.x/mae/mae_vit-base-p16_8xb512-fp16-coslr-800e_in1k/mae_vit-base-p16_8xb512-coslr-800e-fp16_in1k_20220825-5d81fbc4.pth) \| [log](https://download.openmmlab.com/mmselfsup/1.x/mae/mae_vit-base-p16_8xb512-fp16-coslr-800e_in1k/mae_vit-base-p16_8xb512-coslr-800e-fp16_in1k_20220825-5d81fbc4.json) |
| `mae_vit-base-p16_8xb512-amp-coslr-1600e_in1k`  |   111.91   |   17.58   | [config](mae_vit-base-p16_8xb512-amp-coslr-1600e_in1k.py)  | [model](https://download.openmmlab.com/mmselfsup/1.x/mae/mae_vit-base-p16_8xb512-fp16-coslr-1600e_in1k/mae_vit-base-p16_8xb512-fp16-coslr-1600e_in1k_20220825-f7569ca2.pth) \| [log](https://download.openmmlab.com/mmselfsup/1.x/mae/mae_vit-base-p16_8xb512-fp16-coslr-1600e_in1k/mae_vit-base-p16_8xb512-fp16-coslr-1600e_in1k_20220825-f7569ca2.json) |
| `mae_vit-large-p16_8xb512-amp-coslr-400e_in1k`  |   329.54   |   61.60   | [config](mae_vit-large-p16_8xb512-amp-coslr-400e_in1k.py)  | [model](https://download.openmmlab.com/mmselfsup/1.x/mae/mae_vit-large-p16_8xb512-fp16-coslr-400e_in1k/mae_vit-large-p16_8xb512-fp16-coslr-400e_in1k_20220825-b11d0425.pth) \| [log](https://download.openmmlab.com/mmselfsup/1.x/mae/mae_vit-large-p16_8xb512-fp16-coslr-400e_in1k/mae_vit-large-p16_8xb512-fp16-coslr-400e_in1k_20220825-b11d0425.json) |
| `mae_vit-large-p16_8xb512-amp-coslr-800e_in1k`  |   329.54   |   61.60   | [config](mae_vit-large-p16_8xb512-amp-coslr-800e_in1k.py)  | [model](https://download.openmmlab.com/mmselfsup/1.x/mae/mae_vit-large-p16_8xb512-fp16-coslr-800e_in1k/mae_vit-large-p16_8xb512-fp16-coslr-800e_in1k_20220825-df72726a.pth) \| [log](https://download.openmmlab.com/mmselfsup/1.x/mae/mae_vit-large-p16_8xb512-fp16-coslr-800e_in1k/mae_vit-large-p16_8xb512-fp16-coslr-800e_in1k_20220825-df72726a.json) |
| `mae_vit-large-p16_8xb512-amp-coslr-1600e_in1k` |   329.54   |   61.60   | [config](mae_vit-large-p16_8xb512-amp-coslr-1600e_in1k.py) | [model](https://download.openmmlab.com/mmselfsup/1.x/mae/mae_vit-large-p16_8xb512-fp16-coslr-1600e_in1k/mae_vit-large-p16_8xb512-fp16-coslr-1600e_in1k_20220825-cc7e98c9.pth) \| [log](https://download.openmmlab.com/mmselfsup/1.x/mae/mae_vit-large-p16_8xb512-fp16-coslr-1600e_in1k/mae_vit-large-p16_8xb512-fp16-coslr-1600e_in1k_20220825-cc7e98c9.json) |
| `mae_vit-huge-p16_8xb512-amp-coslr-1600e_in1k`  |   657.07   |  167.40   | [config](mae_vit-huge-p14_8xb512-amp-coslr-1600e_in1k.py)  | [model](https://download.openmmlab.com/mmselfsup/1.x/mae/mae_vit-huge-p16_8xb512-fp16-coslr-1600e_in1k/mae_vit-huge-p16_8xb512-fp16-coslr-1600e_in1k_20220916-ff848775.pth) \| [log](https://download.openmmlab.com/mmselfsup/1.x/mae/mae_vit-huge-p16_8xb512-fp16-coslr-1600e_in1k/mae_vit-huge-p16_8xb512-fp16-coslr-1600e_in1k_20220916-ff848775.json) |

### Image Classification on ImageNet-1k

| Model                                     |                   Pretrain                   | Params (M) | Flops (G) | Top-1 (%) |                   Config                   |                   Download                    |
| :---------------------------------------- | :------------------------------------------: | :--------: | :-------: | :-------: | :----------------------------------------: | :-------------------------------------------: |
| `vit-base-p16_mae-300e-pre_8xb128-coslr-100e_in1k` | [MAE 300-Epochs](https://download.openmmlab.com/mmselfsup/1.x/mae/mae_vit-base-p16_8xb512-fp16-coslr-300e_in1k/mae_vit-base-p16_8xb512-coslr-300e-fp16_in1k_20220829-c2cf66ba.pth) |   86.57    |   17.58   |   83.10   | [config](benchmarks/vit-base-p16_8xb128-coslr-100e_in1k.py) |                      N/A                      |
| `vit-base-p16_mae-400e-pre_8xb128-coslr-100e_in1k` | [MAE 400-Epochs](https://download.openmmlab.com/mmselfsup/1.x/mae/mae_vit-base-p16_8xb512-fp16-coslr-400e_in1k/mae_vit-base-p16_8xb512-coslr-400e-fp16_in1k_20220825-bc79e40b.pth) |   86.57    |   17.58   |   83.30   | [config](benchmarks/vit-base-p16_8xb128-coslr-100e_in1k.py) |                      N/A                      |
| `vit-base-p16_mae-800e-pre_8xb128-coslr-100e_in1k` | [MAE 800-Epochs](https://download.openmmlab.com/mmselfsup/1.x/mae/mae_vit-base-p16_8xb512-fp16-coslr-800e_in1k/mae_vit-base-p16_8xb512-coslr-800e-fp16_in1k_20220825-5d81fbc4.pth) |   86.57    |   17.58   |   83.30   | [config](benchmarks/vit-base-p16_8xb128-coslr-100e_in1k.py) |                      N/A                      |
| `vit-base-p16_mae-1600e-pre_8xb128-coslr-100e_in1k` | [MAE 1600-Epochs](https://download.openmmlab.com/mmselfsup/1.x/mae/mae_vit-base-p16_8xb512-fp16-coslr-1600e_in1k/mae_vit-base-p16_8xb512-fp16-coslr-1600e_in1k_20220825-f7569ca2.pth) |   86.57    |   17.58   |   83.50   | [config](benchmarks/vit-base-p16_8xb128-coslr-100e_in1k.py) | [model](https://download.openmmlab.com/mmselfsup/1.x/mae/mae_vit-base-p16_8xb512-fp16-coslr-1600e_in1k/vit-base-p16_ft-8xb128-coslr-100e_in1k/vit-base-p16_ft-8xb128-coslr-100e_in1k_20220825-cf70aa21.pth) \| [log](https://download.openmmlab.com/mmselfsup/1.x/mae/mae_vit-base-p16_8xb512-fp16-coslr-1600e_in1k/vit-base-p16_ft-8xb128-coslr-100e_in1k/vit-base-p16_ft-8xb128-coslr-100e_in1k_20220825-cf70aa21.json) |
| `vit-base-p16_mae-300e-pre_8xb2048-linear-coslr-90e_in1k` | [MAE 300-Epochs](https://download.openmmlab.com/mmselfsup/1.x/mae/mae_vit-base-p16_8xb512-fp16-coslr-300e_in1k/mae_vit-base-p16_8xb512-coslr-300e-fp16_in1k_20220829-c2cf66ba.pth) |   86.57    |   17.58   |   60.80   | [config](benchmarks/vit-base-p16_8xb2048-linear-coslr-90e_in1k.py) |                      N/A                      |
| `vit-base-p16_mae-400e-pre_8xb2048-linear-coslr-90e_in1k` | [MAE 400-Epochs](https://download.openmmlab.com/mmselfsup/1.x/mae/mae_vit-base-p16_8xb512-fp16-coslr-400e_in1k/mae_vit-base-p16_8xb512-coslr-400e-fp16_in1k_20220825-bc79e40b.pth) |   86.57    |   17.58   |   62.50   | [config](benchmarks/vit-base-p16_8xb2048-linear-coslr-90e_in1k.py) |                      N/A                      |
| `vit-base-p16_mae-800e-pre_8xb2048-linear-coslr-90e_in1k` | [MAE 800-Epochs](https://download.openmmlab.com/mmselfsup/1.x/mae/mae_vit-base-p16_8xb512-fp16-coslr-800e_in1k/mae_vit-base-p16_8xb512-coslr-800e-fp16_in1k_20220825-5d81fbc4.pth) |   86.57    |   17.58   |   65.10   | [config](benchmarks/vit-base-p16_8xb2048-linear-coslr-90e_in1k.py) |                      N/A                      |
| `vit-base-p16_mae-1600e-pre_8xb2048-linear-coslr-90e_in1k` | [MAE 1600-Epochs](https://download.openmmlab.com/mmselfsup/1.x/mae/mae_vit-base-p16_8xb512-fp16-coslr-1600e_in1k/mae_vit-base-p16_8xb512-fp16-coslr-1600e_in1k_20220825-f7569ca2.pth) |   86.57    |   17.58   |   67.10   | [config](benchmarks/vit-base-p16_8xb2048-linear-coslr-90e_in1k.py) |                      N/A                      |
| `vit-large-p16_mae-400e-pre_8xb128-coslr-50e_in1k` | [MAE 400-Epochs](https://download.openmmlab.com/mmselfsup/1.x/mae/mae_vit-large-p16_8xb512-fp16-coslr-400e_in1k/mae_vit-large-p16_8xb512-fp16-coslr-400e_in1k_20220825-b11d0425.pth) |   304.32   |   61.60   |   85.20   | [config](benchmarks/vit-large-p16_8xb128-coslr-50e_in1k.py) |                      N/A                      |
| `vit-large-p16_mae-800e-pre_8xb128-coslr-50e_in1k` | [MAE 800-Epochs](https://download.openmmlab.com/mmselfsup/1.x/mae/mae_vit-large-p16_8xb512-fp16-coslr-800e_in1k/mae_vit-large-p16_8xb512-fp16-coslr-800e_in1k_20220825-df72726a.pth) |   304.32   |   61.60   |   85.40   | [config](benchmarks/vit-large-p16_8xb128-coslr-50e_in1k.py) |                      N/A                      |
| `vit-large-p16_mae-1600e-pre_8xb128-coslr-50e_in1k` | [MAE 1600-Epochs](https://download.openmmlab.com/mmselfsup/1.x/mae/mae_vit-large-p16_8xb512-fp16-coslr-1600e_in1k/mae_vit-large-p16_8xb512-fp16-coslr-1600e_in1k_20220825-cc7e98c9.pth) |   304.32   |   61.60   |   85.70   | [config](benchmarks/vit-large-p16_8xb128-coslr-50e_in1k.py) |                      N/A                      |
| `vit-large-p16_mae-400e-pre_8xb2048-linear-coslr-90e_in1k` | [MAE 400-Epochs](https://download.openmmlab.com/mmselfsup/1.x/mae/mae_vit-large-p16_8xb512-fp16-coslr-400e_in1k/mae_vit-large-p16_8xb512-fp16-coslr-400e_in1k_20220825-b11d0425.pth) |   304.33   |   61.60   |   70.70   | [config](benchmarks/vit-large-p16_8xb2048-linear-coslr-90e_in1k.py) |                      N/A                      |
| `vit-large-p16_mae-800e-pre_8xb2048-linear-coslr-90e_in1k` | [MAE 800-Epochs](https://download.openmmlab.com/mmselfsup/1.x/mae/mae_vit-large-p16_8xb512-fp16-coslr-800e_in1k/mae_vit-large-p16_8xb512-fp16-coslr-800e_in1k_20220825-df72726a.pth) |   304.33   |   61.60   |   73.70   | [config](benchmarks/vit-large-p16_8xb2048-linear-coslr-90e_in1k.py) |                      N/A                      |
| `vit-large-p16_mae-1600e-pre_8xb2048-linear-coslr-90e_in1k` | [MAE 1600-Epochs](https://download.openmmlab.com/mmselfsup/1.x/mae/mae_vit-large-p16_8xb512-fp16-coslr-1600e_in1k/mae_vit-large-p16_8xb512-fp16-coslr-1600e_in1k_20220825-cc7e98c9.pth) |   304.33   |   61.60   |   75.50   | [config](benchmarks/vit-large-p16_8xb2048-linear-coslr-90e_in1k.py) |                      N/A                      |
| `vit-huge-p14_mae-1600e-pre_8xb128-coslr-50e_in1k` | [MAE 1600-Epochs](https://download.openmmlab.com/mmselfsup/1.x/mae/mae_vit-huge-p16_8xb512-fp16-coslr-1600e_in1k/mae_vit-huge-p16_8xb512-fp16-coslr-1600e_in1k_20220916-ff848775.pth) |   632.04   |  167.40   |   86.90   | [config](benchmarks/vit-huge-p14_8xb128-coslr-50e_in1k.py) | [model](https://download.openmmlab.com/mmselfsup/1.x/mae/mae_vit-huge-p16_8xb512-fp16-coslr-1600e_in1k/vit-huge-p16_ft-8xb128-coslr-50e_in1k/vit-huge-p16_ft-8xb128-coslr-50e_in1k_20220916-0bfc9bfd.pth) \| [log](https://download.openmmlab.com/mmselfsup/1.x/mae/mae_vit-huge-p16_8xb512-fp16-coslr-1600e_in1k/vit-huge-p16_ft-8xb128-coslr-50e_in1k/vit-huge-p16_ft-8xb128-coslr-50e_in1k_20220916-0bfc9bfd.json) |
| `vit-huge-p14_mae-1600e-pre_32xb8-coslr-50e_in1k-448px` | [MAE 1600-Epochs](https://download.openmmlab.com/mmselfsup/1.x/mae/mae_vit-huge-p16_8xb512-fp16-coslr-1600e_in1k/mae_vit-huge-p16_8xb512-fp16-coslr-1600e_in1k_20220916-ff848775.pth) |   633.03   |  732.13   |   87.30   | [config](benchmarks/vit-huge-p14_32xb8-coslr-50e_in1k-448px.py) | [model](https://download.openmmlab.com/mmselfsup/1.x/mae/mae_vit-huge-p16_8xb512-fp16-coslr-1600e_in1k/vit-huge-p16_ft-32xb8-coslr-50e_in1k-448/vit-huge-p16_ft-32xb8-coslr-50e_in1k-448_20220916-95b6a0ce.pth) \| [log](https://download.openmmlab.com/mmselfsup/1.x/mae/mae_vit-huge-p16_8xb512-fp16-coslr-1600e_in1k/vit-huge-p16_ft-32xb8-coslr-50e_in1k-448/vit-huge-p16_ft-32xb8-coslr-50e_in1k-448_20220916-95b6a0ce.json) |

## Citation

```bibtex
@article{He2021MaskedAA,
  title={Masked Autoencoders Are Scalable Vision Learners},
  author={Kaiming He and Xinlei Chen and Saining Xie and Yanghao Li and
  Piotr Doll'ar and Ross B. Girshick},
  journal={arXiv},
  year={2021}
}
```
