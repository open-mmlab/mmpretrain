# Res2Net

> [Res2Net: A New Multi-scale Backbone Architecture](https://arxiv.org/abs/1904.01169)

<!-- [ALGORITHM] -->

## Abstract

Representing features at multiple scales is of great importance for numerous vision tasks. Recent advances in backbone convolutional neural networks (CNNs) continually demonstrate stronger multi-scale representation ability, leading to consistent performance gains on a wide range of applications. However, most existing methods represent the multi-scale features in a layer-wise manner. In this paper, we propose a novel building block for CNNs, namely Res2Net, by constructing hierarchical residual-like connections within one single residual block. The Res2Net represents multi-scale features at a granular level and increases the range of receptive fields for each network layer. The proposed Res2Net block can be plugged into the state-of-the-art backbone CNN models, e.g., ResNet, ResNeXt, and DLA. We evaluate the Res2Net block on all these models and demonstrate consistent performance gains over baseline models on widely-used datasets, e.g., CIFAR-100 and ImageNet. Further ablation studies and experimental results on representative computer vision tasks, i.e., object detection, class activation mapping, and salient object detection, further verify the superiority of the Res2Net over the state-of-the-art baseline methods.

<div align=center>
<img src="https://user-images.githubusercontent.com/26739999/142573547-cde68abf-287b-46db-a848-5cffe3068faf.png" width="50%"/>
</div>

## How to use it?

<!-- [TABS-BEGIN] -->

**Predict image**

```python
from mmpretrain import inference_model

predict = inference_model('res2net50-w14-s8_3rdparty_8xb32_in1k', 'demo/bird.JPEG')
print(predict['pred_class'])
print(predict['pred_score'])
```

**Use the model**

```python
import torch
from mmpretrain import get_model

model = get_model('res2net50-w14-s8_3rdparty_8xb32_in1k', pretrained=True)
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
python tools/test.py configs/res2net/res2net50-w14-s8_8xb32_in1k.py https://download.openmmlab.com/mmclassification/v0/res2net/res2net50-w14-s8_3rdparty_8xb32_in1k_20210927-bc967bf1.pth
```

<!-- [TABS-END] -->

## Models and results

### Image Classification on ImageNet-1k

| Model                                     |   Pretrain   | Params (M) | Flops (G) | Top-1 (%) | Top-5 (%) |                  Config                   |                               Download                                |
| :---------------------------------------- | :----------: | :--------: | :-------: | :-------: | :-------: | :---------------------------------------: | :-------------------------------------------------------------------: |
| `res2net50-w14-s8_3rdparty_8xb32_in1k`\*  | From scratch |   25.06    |   4.22    |   78.14   |   93.85   | [config](res2net50-w14-s8_8xb32_in1k.py)  | [model](https://download.openmmlab.com/mmclassification/v0/res2net/res2net50-w14-s8_3rdparty_8xb32_in1k_20210927-bc967bf1.pth) |
| `res2net50-w26-s8_3rdparty_8xb32_in1k`\*  | From scratch |   48.40    |   8.39    |   79.20   |   94.36   | [config](res2net50-w26-s8_8xb32_in1k.py)  | [model](https://download.openmmlab.com/mmclassification/v0/res2net/res2net50-w26-s8_3rdparty_8xb32_in1k_20210927-f547a94b.pth) |
| `res2net101-w26-s4_3rdparty_8xb32_in1k`\* | From scratch |   45.21    |   8.12    |   79.19   |   94.44   | [config](res2net101-w26-s4_8xb32_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/res2net/res2net101-w26-s4_3rdparty_8xb32_in1k_20210927-870b6c36.pth) |

*Models with * are converted from the [official repo](https://github.com/Res2Net/Res2Net-PretrainedModels/blob/master/res2net.py#L181). The config files of these models are only for inference. We haven't reproduce the training results.*

## Citation

```bibtex
@article{gao2019res2net,
  title={Res2Net: A New Multi-scale Backbone Architecture},
  author={Gao, Shang-Hua and Cheng, Ming-Ming and Zhao, Kai and Zhang, Xin-Yu and Yang, Ming-Hsuan and Torr, Philip},
  journal={IEEE TPAMI},
  year={2021},
  doi={10.1109/TPAMI.2019.2938758},
}
```
