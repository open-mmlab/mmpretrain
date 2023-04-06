# MixMIM

> [MixMIM: Mixed and Masked Image Modeling for Efficient Visual Representation Learning](https://arxiv.org/abs/2205.13137)

<!-- [ALGORITHM] -->

## Abstract

In this study, we propose Mixed and Masked Image Modeling (MixMIM), a
simple but efficient MIM method that is applicable to various hierarchical Vision
Transformers. Existing MIM methods replace a random subset of input tokens with
a special [MASK] symbol and aim at reconstructing original image tokens from
the corrupted image. However, we find that using the [MASK] symbol greatly
slows down the training and causes training-finetuning inconsistency, due to the
large masking ratio (e.g., 40% in BEiT). In contrast, we replace the masked tokens
of one image with visible tokens of another image, i.e., creating a mixed image.
We then conduct dual reconstruction to reconstruct the original two images from
the mixed input, which significantly improves efficiency. While MixMIM can
be applied to various architectures, this paper explores a simpler but stronger
hierarchical Transformer, and scales with MixMIM-B, -L, and -H. Empirical
results demonstrate that MixMIM can learn high-quality visual representations
efficiently. Notably, MixMIM-B with 88M parameters achieves 85.1% top-1
accuracy on ImageNet-1K by pretraining for 600 epochs, setting a new record for
neural networks with comparable model sizes (e.g., ViT-B) among MIM methods.
Besides, its transferring performances on the other 6 datasets show MixMIM has
better FLOPs / performance tradeoff than previous MIM methods

<div align=center>
<img src="https://user-images.githubusercontent.com/56866854/202853730-d26fb3d7-e5e8-487a-aad5-e3d4600cef87.png"/>
</div>

## How to use it?

<!-- [TABS-BEGIN] -->

**Predict image**

```python
from mmpretrain import inference_model

predict = inference_model('mixmim-base_mixmim-pre_8xb128-coslr-100e_in1k', 'demo/bird.JPEG')
print(predict['pred_class'])
print(predict['pred_score'])
```

**Use the model**

```python
import torch
from mmpretrain import get_model

model = get_model('mixmim_mixmim-base_16xb128-coslr-300e_in1k', pretrained=True)
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
python tools/train.py configs/mixmim/mixmim_mixmim-base_16xb128-coslr-300e_in1k.py
```

Test:

```shell
python tools/test.py configs/mixmim/benchmarks/mixmim-base_8xb128-coslr-100e_in1k.py https://download.openmmlab.com/mmselfsup/1.x/mixmim/mixmim-base-p16_16xb128-coslr-300e_in1k/mixmim-base-p16_ft-8xb128-coslr-100e_in1k/mixmim-base-p16_ft-8xb128-coslr-100e_in1k_20221208-41ecada9.pth
```

<!-- [TABS-END] -->

## Models and results

### Pretrained models

| Model                                        | Params (M) | Flops (G) |                         Config                          |                                      Download                                      |
| :------------------------------------------- | :--------: | :-------: | :-----------------------------------------------------: | :--------------------------------------------------------------------------------: |
| `mixmim_mixmim-base_16xb128-coslr-300e_in1k` |   114.67   |   16.35   | [config](mixmim_mixmim-base_16xb128-coslr-300e_in1k.py) | [model](https://download.openmmlab.com/mmselfsup/1.x/mixmim/mixmim-base-p16_16xb128-coslr-300e_in1k/mixmim-base-p16_16xb128-coslr-300e_in1k_20221208-44fe8d2c.pth) \| [log](https://download.openmmlab.com/mmselfsup/1.x/mixmim/mixmim-base-p16_16xb128-coslr-300e_in1k/mixmim-base-p16_16xb128-coslr-300e_in1k_20221208-44fe8d2c.json) |

### Image Classification on ImageNet-1k

| Model                                     |                   Pretrain                   | Params (M) | Flops (G) | Top-1 (%) |                   Config                   |                   Download                    |
| :---------------------------------------- | :------------------------------------------: | :--------: | :-------: | :-------: | :----------------------------------------: | :-------------------------------------------: |
| `mixmim-base_mixmim-pre_8xb128-coslr-100e_in1k` | [MIXMIM](https://download.openmmlab.com/mmselfsup/1.x/mixmim/mixmim-base-p16_16xb128-coslr-300e_in1k/mixmim-base-p16_16xb128-coslr-300e_in1k_20221208-44fe8d2c.pth) |   88.34    |   16.35   |   84.63   | [config](benchmarks/mixmim-base_8xb128-coslr-100e_in1k.py) | [model](https://download.openmmlab.com/mmselfsup/1.x/mixmim/mixmim-base-p16_16xb128-coslr-300e_in1k/mixmim-base-p16_ft-8xb128-coslr-100e_in1k/mixmim-base-p16_ft-8xb128-coslr-100e_in1k_20221208-41ecada9.pth) \| [log](https://download.openmmlab.com/mmselfsup/1.x/mixmim/mixmim-base-p16_16xb128-coslr-300e_in1k/mixmim-base-p16_ft-8xb128-coslr-100e_in1k/mixmim-base-p16_ft-8xb128-coslr-100e_in1k_20221208-41ecada9.json) |

## Citation

```bibtex
@article{MixMIM2022,
  author  = {Jihao Liu, Xin Huang, Yu Liu, Hongsheng Li},
  journal = {arXiv:2205.13137},
  title   = {MixMIM: Mixed and Masked Image Modeling for Efficient Visual Representation Learning},
  year    = {2022},
}
```
