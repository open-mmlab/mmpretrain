# iTPN

> [Integrally Pre-Trained Transformer Pyramid Networks](https://arxiv.org/abs/2211.12735)

<!-- [ALGORITHM] -->

## Abstract

In this paper, we present an integral pre-training framework based on masked image modeling (MIM). We advocate for pre-training the backbone and neck jointly so that the transfer gap between MIM and downstream recognition tasks is minimal. We make two technical contributions. First, we unify the reconstruction and recognition necks by inserting a feature pyramid into the pre-training stage. Second, we complement mask image modeling (MIM) with masked feature modeling (MFM) that offers multi-stage supervision to the feature pyramid. The pre-trained models, termed integrally pre-trained transformer pyramid networks (iTPNs), serve as powerful foundation models for visual recognition. In particular, the base/large-level iTPN achieves an 86.2%/87.8% top-1 accuracy on ImageNet-1K, a 53.2%/55.6% box AP on COCO object detection with 1x training schedule using Mask-RCNN, and a 54.7%/57.7% mIoU on ADE20K semantic segmentation using UPerHead -- all these results set new records. Our work inspires the community to work on unifying upstream pre-training and downstream fine-tuning tasks. Code and the pre-trained models will be released at https://github.com/sunsmarterjie/iTPN.

<div align=center>
<img src="https://github.com/open-mmlab/mmpretrain/assets/36138628/2e53d5b5-300e-4640-8507-c1173965ca62" width="80%"/>
</div>

## How to use it?

<!-- [TABS-BEGIN] -->

<!-- **Use the model**

```python
import torch
from mmpretrain import get_model

model = get_model('itpn-clip-b_hivit-base-p16_8xb256-amp-coslr-800e_in1k', pretrained=True)
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
python tools/train.py configs/itpn/itpn-pixel_hivit-base-p16_8xb512-amp-coslr-800e_in1k.py
```

<!-- [TABS-END] -->

## Models and results

### Pretrained models

| Model                                                   | Params (M) | Flops (G) |                               Config                               | Download |
| :------------------------------------------------------ | :--------: | :-------: | :----------------------------------------------------------------: | :------: |
| `itpn-clip-b_hivit-base-p16_8xb256-amp-coslr-800e_in1k` |   233.00   |   18.47   | [config](itpn-clip-b_hivit-base-p16_8xb256-amp-coslr-800e_in1k.py) |   N/A    |
| `itpn-pixel_hivit-base-p16_8xb512-amp-coslr-800e_in1k`  |   103.00   |   18.47   | [config](itpn-pixel_hivit-base-p16_8xb512-amp-coslr-800e_in1k.py)  |   N/A    |
| `itpn-pixel_hivit-large-p16_8xb512-amp-coslr-800e_in1k` |   314.00   |   63.98   | [config](itpn-pixel_hivit-large-p16_8xb512-amp-coslr-800e_in1k.py) |   N/A    |

## Citation

```bibtex
@article{tian2022integrally,
  title={Integrally Pre-Trained Transformer Pyramid Networks},
  author={Tian, Yunjie and Xie, Lingxi and Wang, Zhaozhi and Wei, Longhui and Zhang, Xiaopeng and Jiao, Jianbin and Wang, Yaowei and Tian, Qi and Ye, Qixiang},
  journal={arXiv preprint arXiv:2211.12735},
  year={2022}
}
```
