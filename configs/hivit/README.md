# HiViT

> [HiViT: A Simple and More Efficient Design of Hierarchical Vision Transformer](https://arxiv.org/abs/2205.14949)

<!-- [ALGORITHM] -->

## Abstract

Recently, masked image modeling (MIM) has offered a new methodology of self-supervised pre-training of vision transformers. A key idea of efficient implementation is to discard the masked image patches (or tokens) throughout the target network (encoder), which requires the encoder to be a plain vision transformer (e.g., ViT), albeit hierarchical vision transformers (e.g., Swin Transformer) have potentially better properties in formulating vision inputs. In this paper, we offer a new design of hierarchical vision transformers named HiViT (short for Hierarchical ViT) that enjoys both high efficiency and good performance in MIM. The key is to remove the unnecessary "local inter-unit operations", deriving structurally simple hierarchical vision transformers in which mask-units can be serialized like plain vision transformers. For this purpose, we start with Swin Transformer and (i) set the masking unit size to be the token size in the main stage of Swin Transformer, (ii) switch off inter-unit self-attentions before the main stage, and (iii) eliminate all operations after the main stage. Empirical studies demonstrate the advantageous performance of HiViT in terms of fully-supervised, self-supervised, and transfer learning. In particular, in running MAE on ImageNet-1K, HiViT-B reports a +0.6% accuracy gain over ViT-B and a 1.9$\times$ speed-up over Swin-B, and the performance gain generalizes to downstream tasks of detection and segmentation. Code will be made publicly available.

<div align=center>
<img src="https://github.com/open-mmlab/mmpretrain/assets/36138628/4a99cf9d-15df-4866-8750-bd2c3db5d894" width="80%"/>
</div>

## How to use it?

<!-- [TABS-BEGIN] -->

<!-- **Predict image**

```python
from mmpretrain import inference_model

predict = inference_model('hivit-tiny-p16_16xb64_in1k', 'demo/bird.JPEG')
print(predict['pred_class'])
print(predict['pred_score'])
```

<!-- **Use the model** -->

<!-- ```python
import torch
from mmpretrain import get_model

model = get_model('hivit-tiny-p16_16xb64_in1k', pretrained=True)
inputs = torch.rand(1, 3, 224, 224)
out = model(inputs)
print(type(out))
# To extract features.
feats = model.extract_feat(inputs)
print(type(feats))
``` -->

**Train/Test Command**

Prepare your dataset according to the [docs](https://mmpretrain.readthedocs.io/en/latest/user_guides/dataset_prepare.html#prepare-dataset).

Train:

```shell
python tools/train.py configs/hivit/hivit-tiny-p16_16xb64_in1k.py
```

<!-- Test:

```shell
python tools/test.py configs/hivit/hivit-tiny-p16_16xb64_in1k.py None
``` -->

<!-- [TABS-END] -->

## Models and results

### Image Classification on ImageNet-1k

| Model                         |   Pretrain   | Params (M) | Flops (G) | Top-1 (%) |                  Config                  | Download |
| :---------------------------- | :----------: | :--------: | :-------: | :-------: | :--------------------------------------: | :------: |
| `hivit-tiny-p16_16xb64_in1k`  | From scratch |   19.18    |   4.60    |   82.10   | [config](hivit-tiny-p16_16xb64_in1k.py)  |   N/A    |
| `hivit-small-p16_16xb64_in1k` | From scratch |   37.53    |   9.07    |    N/A    | [config](hivit-small-p16_16xb64_in1k.py) |   N/A    |
| `hivit-base-p16_16xb64_in1k`  | From scratch |   79.05    |   18.47   |    N/A    | [config](hivit-base-p16_16xb64_in1k.py)  |   N/A    |

## Citation

```bibtex
@inproceedings{zhanghivit,
  title={HiViT: A Simpler and More Efficient Design of Hierarchical Vision Transformer},
  author={Zhang, Xiaosong and Tian, Yunjie and Xie, Lingxi and Huang, Wei and Dai, Qi and Ye, Qixiang and Tian, Qi},
  booktitle={International Conference on Learning Representations},
  year={2023},
}
```
