# PoolFormer

> [MetaFormer is Actually What You Need for Vision](https://arxiv.org/abs/2111.11418)

<!-- [ALGORITHM] -->

## Abstract

Transformers have shown great potential in computer vision tasks. A common belief is their attention-based token mixer module contributes most to their competence. However, recent works show the attention-based module in transformers can be replaced by spatial MLPs and the resulted models still perform quite well. Based on this observation, we hypothesize that the general architecture of the transformers, instead of the specific token mixer module, is more essential to the model's performance. To verify this, we deliberately replace the attention module in transformers with an embarrassingly simple spatial pooling operator to conduct only basic token mixing. Surprisingly, we observe that the derived model, termed as PoolFormer, achieves competitive performance on multiple computer vision tasks. For example, on ImageNet-1K, PoolFormer achieves 82.1% top-1 accuracy, surpassing well-tuned vision transformer/MLP-like baselines DeiT-B/ResMLP-B24 by 0.3%/1.1% accuracy with 35%/52% fewer parameters and 49%/61% fewer MACs. The effectiveness of PoolFormer verifies our hypothesis and urges us to initiate the concept of "MetaFormer", a general architecture abstracted from transformers without specifying the token mixer. Based on the extensive experiments, we argue that MetaFormer is the key player in achieving superior results for recent transformer and MLP-like models on vision tasks. This work calls for more future research dedicated to improving MetaFormer instead of focusing on the token mixer modules. Additionally, our proposed PoolFormer could serve as a starting baseline for future MetaFormer architecture design.

<div align=center>
<img src="https://user-images.githubusercontent.com/15921929/144710761-1635f59a-abde-4946-984c-a2c3f22a19d2.png" width="100%"/>
</div>

## How to use it?

<!-- [TABS-BEGIN] -->

**Predict image**

```python
from mmpretrain import inference_model

predict = inference_model('poolformer-s12_3rdparty_32xb128_in1k', 'demo/bird.JPEG')
print(predict['pred_class'])
print(predict['pred_score'])
```

**Use the model**

```python
import torch
from mmpretrain import get_model

model = get_model('poolformer-s12_3rdparty_32xb128_in1k', pretrained=True)
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
python tools/test.py configs/poolformer/poolformer-s12_32xb128_in1k.py https://download.openmmlab.com/mmclassification/v0/poolformer/poolformer-s12_3rdparty_32xb128_in1k_20220414-f8d83051.pth
```

<!-- [TABS-END] -->

## Models and results

### Image Classification on ImageNet-1k

| Model                                    |   Pretrain   | Params (M) | Flops (G) | Top-1 (%) | Top-5 (%) |                  Config                  |                                Download                                 |
| :--------------------------------------- | :----------: | :--------: | :-------: | :-------: | :-------: | :--------------------------------------: | :---------------------------------------------------------------------: |
| `poolformer-s12_3rdparty_32xb128_in1k`\* | From scratch |   11.92    |   1.87    |   77.24   |   93.51   | [config](poolformer-s12_32xb128_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/poolformer/poolformer-s12_3rdparty_32xb128_in1k_20220414-f8d83051.pth) |
| `poolformer-s24_3rdparty_32xb128_in1k`\* | From scratch |   21.39    |   3.51    |   80.33   |   95.05   | [config](poolformer-s24_32xb128_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/poolformer/poolformer-s24_3rdparty_32xb128_in1k_20220414-d7055904.pth) |
| `poolformer-s36_3rdparty_32xb128_in1k`\* | From scratch |   30.86    |   5.15    |   81.43   |   95.45   | [config](poolformer-s36_32xb128_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/poolformer/poolformer-s36_3rdparty_32xb128_in1k_20220414-d78ff3e8.pth) |
| `poolformer-m36_3rdparty_32xb128_in1k`\* | From scratch |   56.17    |   8.96    |   82.14   |   95.71   | [config](poolformer-m36_32xb128_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/poolformer/poolformer-m36_3rdparty_32xb128_in1k_20220414-c55e0949.pth) |
| `poolformer-m48_3rdparty_32xb128_in1k`\* | From scratch |   73.47    |   11.80   |   82.51   |   95.95   | [config](poolformer-m48_32xb128_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/poolformer/poolformer-m48_3rdparty_32xb128_in1k_20220414-9378f3eb.pth) |

*Models with * are converted from the [official repo](https://github.com/sail-sg/poolformer). The config files of these models are only for inference. We haven't reproduce the training results.*

## Citation

```bibtex
@inproceedings{yu2022metaformer,
  title={Metaformer is actually what you need for vision},
  author={Yu, Weihao and Luo, Mi and Zhou, Pan and Si, Chenyang and Zhou, Yichen and Wang, Xinchao and Feng, Jiashi and Yan, Shuicheng},
  booktitle={Proceedings of the IEEE/CVF conference on computer vision and pattern recognition},
  pages={10819--10829},
  year={2022}
}
```
