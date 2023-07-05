# Conformer

> [Conformer: Local Features Coupling Global Representations for Visual Recognition](https://arxiv.org/abs/2105.03889)

<!-- [ALGORITHM] -->

## Abstract

Within Convolutional Neural Network (CNN), the convolution operations are good at extracting local features but experience difficulty to capture global representations. Within visual transformer, the cascaded self-attention modules can capture long-distance feature dependencies but unfortunately deteriorate local feature details. In this paper, we propose a hybrid network structure, termed Conformer, to take advantage of convolutional operations and self-attention mechanisms for enhanced representation learning. Conformer roots in the Feature Coupling Unit (FCU), which fuses local features and global representations under different resolutions in an interactive fashion. Conformer adopts a concurrent structure so that local features and global representations are retained to the maximum extent. Experiments show that Conformer, under the comparable parameter complexity, outperforms the visual transformer (DeiT-B) by 2.3% on ImageNet. On MSCOCO, it outperforms ResNet-101 by 3.7% and 3.6% mAPs for object detection and instance segmentation, respectively, demonstrating the great potential to be a general backbone network.

<div align=center>
<img src="https://user-images.githubusercontent.com/26739999/144957687-926390ed-6119-4e4c-beaa-9bc0017fe953.png" width="90%"/>
</div>

## How to use it?

<!-- [TABS-BEGIN] -->

**Predict image**

```python
from mmpretrain import inference_model

predict = inference_model('conformer-tiny-p16_3rdparty_in1k', 'demo/bird.JPEG')
print(predict['pred_class'])
print(predict['pred_score'])
```

**Use the model**

```python
import torch
from mmpretrain import get_model

model = get_model('conformer-tiny-p16_3rdparty_in1k', pretrained=True)
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
python tools/train.py configs/conformer/conformer-small-p32_8xb128_in1k.py
```

Test:

```shell
python tools/test.py configs/conformer/conformer-tiny-p16_8xb128_in1k.py https://download.openmmlab.com/mmclassification/v0/conformer/conformer-tiny-p16_3rdparty_8xb128_in1k_20211206-f6860372.pth
```

<!-- [TABS-END] -->

## Models and results

### Image Classification on ImageNet-1k

| Model                                 |   Pretrain   | Params (M) | Flops (G) | Top-1 (%) | Top-5 (%) |                    Config                    |                                Download                                |
| :------------------------------------ | :----------: | :--------: | :-------: | :-------: | :-------: | :------------------------------------------: | :--------------------------------------------------------------------: |
| `conformer-tiny-p16_3rdparty_in1k`\*  | From scratch |   23.52    |   4.90    |   81.31   |   95.60   | [config](conformer-tiny-p16_8xb128_in1k.py)  | [model](https://download.openmmlab.com/mmclassification/v0/conformer/conformer-tiny-p16_3rdparty_8xb128_in1k_20211206-f6860372.pth) |
| `conformer-small-p16_3rdparty_in1k`\* | From scratch |   37.67    |   10.31   |   83.32   |   96.46   | [config](conformer-small-p16_8xb128_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/conformer/conformer-small-p16_3rdparty_8xb128_in1k_20211206-3065dcf5.pth) |
| `conformer-small-p32_8xb128_in1k`     | From scratch |   38.85    |   7.09    |   81.96   |   96.02   | [config](conformer-small-p32_8xb128_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/conformer/conformer-small-p32_8xb128_in1k_20211206-947a0816.pth) |
| `conformer-base-p16_3rdparty_in1k`\*  | From scratch |   83.29    |   22.89   |   83.82   |   96.59   | [config](conformer-base-p16_8xb128_in1k.py)  | [model](https://download.openmmlab.com/mmclassification/v0/conformer/conformer-base-p16_3rdparty_8xb128_in1k_20211206-bfdf8637.pth) |

*Models with * are converted from the [official repo](https://github.com/pengzhiliang/Conformer/blob/main/models.py#L89). The config files of these models are only for inference. We haven't reproduce the training results.*

## Citation

```bibtex
@article{peng2021conformer,
      title={Conformer: Local Features Coupling Global Representations for Visual Recognition},
      author={Zhiliang Peng and Wei Huang and Shanzhi Gu and Lingxi Xie and Yaowei Wang and Jianbin Jiao and Qixiang Ye},
      journal={arXiv preprint arXiv:2105.03889},
      year={2021},
}
```
