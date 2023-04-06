# MILAN

> [MILAN: Masked Image Pretraining on Language Assisted Representation](https://arxiv.org/pdf/2208.06049)

<!-- [ALGORITHM] -->

## Abstract

Self-attention based transformer models have been dominating many computer
vision tasks in the past few years. Their superb model qualities heavily depend
on the excessively large labeled image datasets. In order to reduce the reliance
on large labeled datasets, reconstruction based masked autoencoders are gaining
popularity, which learn high quality transferable representations from unlabeled
images. For the same purpose, recent weakly supervised image pretraining methods
explore language supervision from text captions accompanying the images. In this
work, we propose masked image pretraining on language assisted representation,
dubbed as MILAN. Instead of predicting raw pixels or low level features, our
pretraining objective is to reconstruct the image features with substantial semantic
signals that are obtained using caption supervision. Moreover, to accommodate our
reconstruction target, we propose a more efficient prompting decoder architecture
and a semantic aware mask sampling mechanism, which further advance the
transfer performance of the pretrained model. Experimental results demonstrate
that MILAN delivers higher accuracy than the previous works. When the masked
autoencoder is pretrained and finetuned on ImageNet-1K dataset with an input
resolution of 224×224, MILAN achieves a top-1 accuracy of 85.4% on ViTB/16, surpassing previous state-of-the-arts by 1%. In the downstream semantic
segmentation task, MILAN achieves 52.7 mIoU using ViT-B/16 backbone on
ADE20K dataset, outperforming previous masked pretraining results by 4 points.

<div align=center>
<img src="https://user-images.githubusercontent.com/30762564/205210369-41a65c4c-bcd4-4147-91ea-c6c9061ab455.png" width="80%"/>
</div>

## How to use it?

<!-- [TABS-BEGIN] -->

**Predict image**

```python
from mmpretrain import inference_model

predict = inference_model('vit-base-p16_milan-pre_8xb128-coslr-100e_in1k', 'demo/bird.JPEG')
print(predict['pred_class'])
print(predict['pred_score'])
```

**Use the model**

```python
import torch
from mmpretrain import get_model

model = get_model('milan_vit-base-p16_16xb256-amp-coslr-400e_in1k', pretrained=True)
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
python tools/train.py configs/milan/milan_vit-base-p16_16xb256-amp-coslr-400e_in1k.py
```

Test:

```shell
python tools/test.py configs/milan/benchmarks/vit-base-p16_8xb128-coslr-100e_in1k.py https://download.openmmlab.com/mmselfsup/1.x/milan/milan_vit-base-p16_16xb256-amp-coslr-400e_in1k/vit-base-p16_ft-8xb128-coslr-100e_in1k/vit-base-p16_ft-8xb128-coslr-100e_in1k-milan_20221129-74ac94fa.pth
```

<!-- [TABS-END] -->

## Models and results

### Pretrained models

| Model                                            | Params (M) | Flops (G) |                           Config                            |                                  Download                                  |
| :----------------------------------------------- | :--------: | :-------: | :---------------------------------------------------------: | :------------------------------------------------------------------------: |
| `milan_vit-base-p16_16xb256-amp-coslr-400e_in1k` |   111.91   |   17.58   | [config](milan_vit-base-p16_16xb256-amp-coslr-400e_in1k.py) | [model](https://download.openmmlab.com/mmselfsup/1.x/milan/milan_vit-base-p16_16xb256-amp-coslr-400e_in1k/milan_vit-base-p16_16xb256-amp-coslr-400e_in1k_20221129-180922e8.pth) \| [log](https://download.openmmlab.com/mmselfsup/1.x/milan/milan_vit-base-p16_16xb256-amp-coslr-400e_in1k/milan_vit-base-p16_16xb256-amp-coslr-400e_in1k_20221129-180922e8.json) |

### Image Classification on ImageNet-1k

| Model                                     |                   Pretrain                   | Params (M) | Flops (G) | Top-1 (%) |                   Config                   |                   Download                    |
| :---------------------------------------- | :------------------------------------------: | :--------: | :-------: | :-------: | :----------------------------------------: | :-------------------------------------------: |
| `vit-base-p16_milan-pre_8xb128-coslr-100e_in1k` | [MILAN](https://download.openmmlab.com/mmselfsup/1.x/milan/milan_vit-base-p16_16xb256-amp-coslr-400e_in1k/milan_vit-base-p16_16xb256-amp-coslr-400e_in1k_20221129-180922e8.pth) |   86.57    |   17.58   |   85.30   | [config](benchmarks/vit-base-p16_8xb128-coslr-100e_in1k.py) | [model](https://download.openmmlab.com/mmselfsup/1.x/milan/milan_vit-base-p16_16xb256-amp-coslr-400e_in1k/vit-base-p16_ft-8xb128-coslr-100e_in1k/vit-base-p16_ft-8xb128-coslr-100e_in1k-milan_20221129-74ac94fa.pth) \| [log](https://download.openmmlab.com/mmselfsup/1.x/milan/milan_vit-base-p16_16xb256-amp-coslr-400e_in1k/vit-base-p16_ft-8xb128-coslr-100e_in1k/vit-base-p16_ft-8xb128-coslr-100e_in1k-milan_20221129-74ac94fa.json) |
| `vit-base-p16_milan-pre_8xb2048-linear-coslr-100e_in1k` | [MILAN](https://download.openmmlab.com/mmselfsup/1.x/milan/milan_vit-base-p16_16xb256-amp-coslr-400e_in1k/milan_vit-base-p16_16xb256-amp-coslr-400e_in1k_20221129-180922e8.pth) |   86.57    |   17.58   |   78.90   | [config](benchmarks/vit-base-p16_8xb2048-linear-coslr-100e_in1k.py) | [model](https://download.openmmlab.com/mmselfsup/1.x/milan/milan_vit-base-p16_16xb256-amp-coslr-400e_in1k/vit-base-p16_linear-8xb2048-coslr-100e_in1k/vit-base-p16_linear-8xb2048-coslr-100e_in1k_20221129-03f26f85.pth) \| [log](https://download.openmmlab.com/mmselfsup/1.x/milan/milan_vit-base-p16_16xb256-amp-coslr-400e_in1k/vit-base-p16_linear-8xb2048-coslr-100e_in1k/vit-base-p16_linear-8xb2048-coslr-100e_in1k_20221129-03f26f85.json) |

## Citation

```bibtex
@article{Hou2022MILANMI,
  title={MILAN: Masked Image Pretraining on Language Assisted Representation},
  author={Zejiang Hou and Fei Sun and Yen-Kuang Chen and Yuan Xie and S. Y. Kung},
  journal={ArXiv},
  year={2022}
}
```
