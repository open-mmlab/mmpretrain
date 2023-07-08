# BEiTv2

> [BEiT v2: Masked Image Modeling with Vector-Quantized Visual Tokenizers](https://arxiv.org/abs/2208.06366)

<!-- [ALGORITHM] -->

## Abstract

Masked image modeling (MIM) has demonstrated impressive results in self-supervised representation learning by recovering corrupted image patches. However, most existing studies operate on low-level image pixels, which hinders the exploitation of high-level semantics for representation models. In this work, we propose to use a semantic-rich visual tokenizer as the reconstruction target for masked prediction, providing a systematic way to promote MIM from pixel-level to semantic-level. Specifically, we propose vector-quantized knowledge distillation to train the tokenizer, which discretizes a continuous semantic space to compact codes. We then pretrain vision Transformers by predicting the original visual tokens for the masked image patches. Furthermore, we introduce a patch aggregation strategy which associates discrete image patches to enhance global semantic representation. Experiments on image classification and semantic segmentation show that BEiT v2 outperforms all compared MIM methods. On ImageNet-1K (224 size), the base-size BEiT v2 achieves 85.5% top-1 accuracy for fine-tuning and 80.1% top-1 accuracy for linear probing. The large-size BEiT v2 obtains 87.3% top-1 accuracy for ImageNet-1K (224 size) fine-tuning, and 56.7% mIoU on ADE20K for semantic segmentation.

<div align=center>
<img src="https://user-images.githubusercontent.com/36138628/203912182-5967a520-d455-49ea-bc67-dcbd500d76bf.png" width="70%"/>
</div>

## How to use it?

<!-- [TABS-BEGIN] -->

**Predict image**

```python
from mmpretrain import inference_model

predict = inference_model('beit-base-p16_beitv2-pre_8xb128-coslr-100e_in1k', 'demo/bird.JPEG')
print(predict['pred_class'])
print(predict['pred_score'])
```

**Use the model**

```python
import torch
from mmpretrain import get_model

model = get_model('beitv2_beit-base-p16_8xb256-amp-coslr-300e_in1k', pretrained=True)
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
python tools/train.py configs/beitv2/beitv2_beit-base-p16_8xb256-amp-coslr-300e_in1k.py
```

Test:

```shell
python tools/test.py configs/beitv2/benchmarks/beit-base-p16_8xb128-coslr-100e_in1k.py https://download.openmmlab.com/mmselfsup/1.x/beitv2/beitv2_vit-base-p16_8xb256-amp-coslr-300e_in1k/vit-base-p16_ft-8xb128-coslr-100e_in1k/vit-base-p16_ft-8xb128-coslr-100e_in1k_20221212-d1c0789e.pth
```

<!-- [TABS-END] -->

## Models and results

### Pretrained models

| Model                                             | Params (M) | Flops (G) |                            Config                            |                                 Download                                 |
| :------------------------------------------------ | :--------: | :-------: | :----------------------------------------------------------: | :----------------------------------------------------------------------: |
| `beitv2_beit-base-p16_8xb256-amp-coslr-300e_in1k` |   192.81   |   17.58   | [config](beitv2_beit-base-p16_8xb256-amp-coslr-300e_in1k.py) | [model](https://download.openmmlab.com/mmselfsup/1.x/beitv2/beitv2_vit-base-p16_8xb256-amp-coslr-300e_in1k/beitv2_vit-base-p16_8xb256-amp-coslr-300e_in1k_20221212-a157be30.pth) \| [log](https://download.openmmlab.com/mmselfsup/1.x/beitv2/beitv2_vit-base-p16_8xb256-amp-coslr-300e_in1k/beitv2_vit-base-p16_8xb256-amp-coslr-300e_in1k_20221212-a157be30.json) |

### Image Classification on ImageNet-1k

| Model                                   |                  Pretrain                  | Params (M) | Flops (G) | Top-1 (%) | Top-5 (%) |                  Config                  |                  Download                  |
| :-------------------------------------- | :----------------------------------------: | :--------: | :-------: | :-------: | :-------: | :--------------------------------------: | :----------------------------------------: |
| `beit-base-p16_beitv2-pre_8xb128-coslr-100e_in1k` | [BEITV2](https://download.openmmlab.com/mmselfsup/1.x/beitv2/beitv2_vit-base-p16_8xb256-amp-coslr-300e_in1k/beitv2_vit-base-p16_8xb256-amp-coslr-300e_in1k_20221212-a157be30.pth) |   86.53    |   17.58   |   85.00   |    N/A    | [config](benchmarks/beit-base-p16_8xb128-coslr-100e_in1k.py) | [model](https://download.openmmlab.com/mmselfsup/1.x/beitv2/beitv2_vit-base-p16_8xb256-amp-coslr-300e_in1k/vit-base-p16_ft-8xb128-coslr-100e_in1k/vit-base-p16_ft-8xb128-coslr-100e_in1k_20221212-d1c0789e.pth) \| [log](https://download.openmmlab.com/mmselfsup/1.x/beitv2/beitv2_vit-base-p16_8xb256-amp-coslr-300e_in1k/vit-base-p16_ft-8xb128-coslr-100e_in1k/vit-base-p16_ft-8xb128-coslr-100e_in1k_20221212-d1c0789e.json) |
| `beit-base-p16_beitv2-in21k-pre_3rdparty_in1k`\* |            BEITV2 ImageNet-21k             |   86.53    |   17.58   |   86.47   |   97.99   | [config](benchmarks/beit-base-p16_8xb64_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/beit/beitv2-base_3rdparty_in1k_20221114-73e11905.pth) |

*Models with * are converted from the [official repo](https://github.com/microsoft/unilm/tree/master/beit2). The config files of these models are only for inference. We haven't reproduce the training results.*

## Citation

```bibtex
@article{beitv2,
    title={{BEiT v2}: Masked Image Modeling with Vector-Quantized Visual Tokenizers},
    author={Zhiliang Peng and Li Dong and Hangbo Bao and Qixiang Ye and Furu Wei},
    year={2022},
    eprint={2208.06366},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```
