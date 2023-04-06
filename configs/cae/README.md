# CAE

> [Context Autoencoder for Self-Supervised Representation Learning](https://arxiv.org/abs/2202.03026)

<!-- [ALGORITHM] -->

## Abstract

We present a novel masked image modeling (MIM) approach, context autoencoder (CAE), for self-supervised learning. We randomly partition the image into two sets: visible patches and masked patches. The CAE architecture consists of: (i) an encoder that takes visible patches as input and outputs their latent representations, (ii) a latent context regressor that predicts the masked patch representations from the visible patch representations that are not updated in this regressor, (iii) a decoder that takes the estimated masked patch representations as input and makes predictions for the masked patches, and (iv) an alignment module that aligns the masked patch representation estimation with the masked patch representations computed from the encoder. In comparison to previous MIM methods that couple the encoding and decoding roles, e.g., using a single module in BEiT, our approach attempts to separate the encoding role (content understanding) from the decoding role (making predictions for masked patches) using different modules, improving the content understanding capability. In addition, our approach makes predictions from the visible patches to the masked patches in the latent representation space that is expected to take on semantics. In addition, we present the explanations about why contrastive pretraining and supervised pretraining perform similarly and why MIM potentially performs better. We demonstrate the effectiveness of our CAE through superior transfer performance in downstream tasks: semantic segmentation, and object detection and instance segmentation.

<div align=center>
<img src="https://user-images.githubusercontent.com/30762564/165459947-6c6ef13c-0593-4765-b44e-6da0a079802a.png" width="70%"/>
</div>

## How to use it?

<!-- [TABS-BEGIN] -->

**Predict image**

```python
from mmpretrain import inference_model

predict = inference_model('beit-base-p16_cae-pre_8xb128-coslr-100e_in1k', 'demo/bird.JPEG')
print(predict['pred_class'])
print(predict['pred_score'])
```

**Use the model**

```python
import torch
from mmpretrain import get_model

model = get_model('cae_beit-base-p16_8xb256-amp-coslr-300e_in1k', pretrained=True)
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
python tools/train.py configs/cae/cae_beit-base-p16_8xb256-amp-coslr-300e_in1k.py
```

Test:

```shell
python tools/test.py configs/cae/benchmarks/beit-base-p16_8xb128-coslr-100e_in1k.py https://download.openmmlab.com/mmselfsup/1.x/cae/cae_vit-base-p16_16xb128-fp16-coslr-300e_in1k/vit-base-p16_ft-8xb128-coslr-100e-rpe_in1k/vit-base-p16_ft-8xb128-coslr-100e-rpe_in1k_20220825-f3d234cd.pth
```

<!-- [TABS-END] -->

## Models and results

### Pretrained models

| Model                                          | Params (M) | Flops (G) |                          Config                           |                                    Download                                    |
| :--------------------------------------------- | :--------: | :-------: | :-------------------------------------------------------: | :----------------------------------------------------------------------------: |
| `cae_beit-base-p16_8xb256-amp-coslr-300e_in1k` |   288.43   |   17.58   | [config](cae_beit-base-p16_8xb256-amp-coslr-300e_in1k.py) | [model](https://download.openmmlab.com/mmselfsup/1.x/cae/cae_vit-base-p16_8xb256-amp-coslr-300e_in1k/cae_vit-base-p16_8xb256-amp-coslr-300e_in1k_20221230-808170f3.pth) \| [log](https://download.openmmlab.com/mmselfsup/1.x/cae/cae_vit-base-p16_8xb256-amp-coslr-300e_in1k/cae_vit-base-p16_8xb256-amp-coslr-300e_in1k_20221230-808170f3.json) |

### Image Classification on ImageNet-1k

| Model                                     |                   Pretrain                   | Params (M) | Flops (G) | Top-1 (%) |                   Config                   |                   Download                    |
| :---------------------------------------- | :------------------------------------------: | :--------: | :-------: | :-------: | :----------------------------------------: | :-------------------------------------------: |
| `beit-base-p16_cae-pre_8xb128-coslr-100e_in1k` | [CAE](https://download.openmmlab.com/mmselfsup/1.x/cae/cae_vit-base-p16_8xb256-amp-coslr-300e_in1k/cae_vit-base-p16_8xb256-amp-coslr-300e_in1k_20221230-808170f3.pth) |   86.68    |   17.58   |   83.20   | [config](benchmarks/beit-base-p16_8xb128-coslr-100e_in1k.py) | [model](https://download.openmmlab.com/mmselfsup/1.x/cae/cae_vit-base-p16_16xb128-fp16-coslr-300e_in1k/vit-base-p16_ft-8xb128-coslr-100e-rpe_in1k/vit-base-p16_ft-8xb128-coslr-100e-rpe_in1k_20220825-f3d234cd.pth) \| [log](https://download.openmmlab.com/mmselfsup/1.x/cae/cae_vit-base-p16_16xb128-fp16-coslr-300e_in1k/vit-base-p16_ft-8xb128-coslr-100e-rpe_in1k/vit-base-p16_ft-8xb128-coslr-100e-rpe_in1k_20220825-f3d234cd.json) |

## Citation

```bibtex
@article{CAE,
  title={Context Autoencoder for Self-Supervised Representation Learning},
  author={Xiaokang Chen, Mingyu Ding, Xiaodi Wang, Ying Xin, Shentong Mo,
  Yunhao Wang, Shumin Han, Ping Luo, Gang Zeng, Jingdong Wang},
  journal={ArXiv},
  year={2022}
}
```
