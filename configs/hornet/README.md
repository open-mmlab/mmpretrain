# HorNet

> [HorNet: Efficient High-Order Spatial Interactions with Recursive Gated Convolutions](https://arxiv.org/abs/2207.14284)

<!-- [ALGORITHM] -->

## Abstract

Recent progress in vision Transformers exhibits great success in various tasks driven by the new spatial modeling mechanism based on dot-product self-attention. In this paper, we show that the key ingredients behind the vision Transformers, namely input-adaptive, long-range and high-order spatial interactions, can also be efficiently implemented with a convolution-based framework. We present the Recursive Gated Convolution (g nConv) that performs high-order spatial interactions with gated convolutions and recursive designs. The new operation is highly flexible and customizable, which is compatible with various variants of convolution and extends the two-order interactions in self-attention to arbitrary orders without introducing significant extra computation. g nConv can serve as a plug-and-play module to improve various vision Transformers and convolution-based models. Based on the operation, we construct a new family of generic vision backbones named HorNet. Extensive experiments on ImageNet classification, COCO object detection and ADE20K semantic segmentation show HorNet outperform Swin Transformers and ConvNeXt by a significant margin with similar overall architecture and training configurations. HorNet also shows favorable scalability to more training data and a larger model size. Apart from the effectiveness in visual encoders, we also show g nConv can be applied to task-specific decoders and consistently improve dense prediction performance with less computation. Our results demonstrate that g nConv can be a new basic module for visual modeling that effectively combines the merits of both vision Transformers and CNNs. Code is available at https://github.com/raoyongming/HorNet.

<div align=center>
<img src="https://user-images.githubusercontent.com/24734142/188356236-b8e3db94-eaa6-48e9-b323-15e5ba7f2991.png" width="80%"/>
</div>

## How to use it?

<!-- [TABS-BEGIN] -->

**Predict image**

```python
from mmpretrain import inference_model

predict = inference_model('hornet-tiny_3rdparty_in1k', 'demo/bird.JPEG')
print(predict['pred_class'])
print(predict['pred_score'])
```

**Use the model**

```python
import torch
from mmpretrain import get_model

model = get_model('hornet-tiny_3rdparty_in1k', pretrained=True)
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
python tools/test.py configs/hornet/hornet-tiny_8xb128_in1k.py https://download.openmmlab.com/mmclassification/v0/hornet/hornet-tiny_3rdparty_in1k_20220915-0e8eedff.pth
```

<!-- [TABS-END] -->

## Models and results

### Image Classification on ImageNet-1k

| Model                             |   Pretrain   | Params (M) | Flops (G) | Top-1 (%) | Top-5 (%) |                 Config                  |                                    Download                                     |
| :-------------------------------- | :----------: | :--------: | :-------: | :-------: | :-------: | :-------------------------------------: | :-----------------------------------------------------------------------------: |
| `hornet-tiny_3rdparty_in1k`\*     | From scratch |   22.41    |   3.98    |   82.84   |   96.24   |  [config](hornet-tiny_8xb128_in1k.py)   | [model](https://download.openmmlab.com/mmclassification/v0/hornet/hornet-tiny_3rdparty_in1k_20220915-0e8eedff.pth) |
| `hornet-tiny-gf_3rdparty_in1k`\*  | From scratch |   22.99    |   3.90    |   82.98   |   96.38   | [config](hornet-tiny-gf_8xb128_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/hornet/hornet-tiny-gf_3rdparty_in1k_20220915-4c35a66b.pth) |
| `hornet-small_3rdparty_in1k`\*    | From scratch |   49.53    |   8.83    |   83.79   |   96.75   |  [config](hornet-small_8xb64_in1k.py)   | [model](https://download.openmmlab.com/mmclassification/v0/hornet/hornet-small_3rdparty_in1k_20220915-5935f60f.pth) |
| `hornet-small-gf_3rdparty_in1k`\* | From scratch |   50.40    |   8.71    |   83.98   |   96.77   | [config](hornet-small-gf_8xb64_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/hornet/hornet-small-gf_3rdparty_in1k_20220915-649ca492.pth) |
| `hornet-base_3rdparty_in1k`\*     | From scratch |   87.26    |   15.58   |   84.24   |   96.94   |   [config](hornet-base_8xb64_in1k.py)   | [model](https://download.openmmlab.com/mmclassification/v0/hornet/hornet-base_3rdparty_in1k_20220915-a06176bb.pth) |
| `hornet-base-gf_3rdparty_in1k`\*  | From scratch |   88.42    |   15.42   |   84.32   |   96.95   | [config](hornet-base-gf_8xb64_in1k.py)  | [model](https://download.openmmlab.com/mmclassification/v0/hornet/hornet-base-gf_3rdparty_in1k_20220915-82c06fa7.pth) |

*Models with * are converted from the [official repo](https://github.com/raoyongming/HorNet). The config files of these models are only for inference. We haven't reproduce the training results.*

## Citation

```bibtex
@article{rao2022hornet,
  title={HorNet: Efficient High-Order Spatial Interactions with Recursive Gated Convolutions},
  author={Rao, Yongming and Zhao, Wenliang and Tang, Yansong and Zhou, Jie and Lim, Ser-Lam and Lu, Jiwen},
  journal={arXiv preprint arXiv:2207.14284},
  year={2022}
}
```
