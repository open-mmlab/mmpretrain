# MLP-Mixer

> [MLP-Mixer: An all-MLP Architecture for Vision](https://arxiv.org/abs/2105.01601)

<!-- [ALGORITHM] -->

## Abstract

Convolutional Neural Networks (CNNs) are the go-to model for computer vision. Recently, attention-based networks, such as the Vision Transformer, have also become popular. In this paper we show that while convolutions and attention are both sufficient for good performance, neither of them are necessary. We present MLP-Mixer, an architecture based exclusively on multi-layer perceptrons (MLPs). MLP-Mixer contains two types of layers: one with MLPs applied independently to image patches (i.e. "mixing" the per-location features), and one with MLPs applied across patches (i.e. "mixing" spatial information). When trained on large datasets, or with modern regularization schemes, MLP-Mixer attains competitive scores on image classification benchmarks, with pre-training and inference cost comparable to state-of-the-art models. We hope that these results spark further research beyond the realms of well established CNNs and Transformers.

<div align=center>
<img src="https://user-images.githubusercontent.com/26739999/143178327-7118b48a-5f5f-4844-a614-a571917384ca.png" width="90%"/>
</div>

## How to use it?

<!-- [TABS-BEGIN] -->

**Predict image**

```python
from mmpretrain import inference_model

predict = inference_model('mlp-mixer-base-p16_3rdparty_64xb64_in1k', 'demo/bird.JPEG')
print(predict['pred_class'])
print(predict['pred_score'])
```

**Use the model**

```python
import torch
from mmpretrain import get_model

model = get_model('mlp-mixer-base-p16_3rdparty_64xb64_in1k', pretrained=True)
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
python tools/test.py configs/mlp_mixer/mlp-mixer-base-p16_64xb64_in1k.py https://download.openmmlab.com/mmclassification/v0/mlp-mixer/mixer-base-p16_3rdparty_64xb64_in1k_20211124-1377e3e0.pth
```

<!-- [TABS-END] -->

## Models and results

### Image Classification on ImageNet-1k

| Model                                        |   Pretrain   | Params (M) | Flops (G) | Top-1 (%) | Top-5 (%) |                    Config                    |                            Download                             |
| :------------------------------------------- | :----------: | :--------: | :-------: | :-------: | :-------: | :------------------------------------------: | :-------------------------------------------------------------: |
| `mlp-mixer-base-p16_3rdparty_64xb64_in1k`\*  | From scratch |   59.88    |   12.61   |   76.68   |   92.25   | [config](mlp-mixer-base-p16_64xb64_in1k.py)  | [model](https://download.openmmlab.com/mmclassification/v0/mlp-mixer/mixer-base-p16_3rdparty_64xb64_in1k_20211124-1377e3e0.pth) |
| `mlp-mixer-large-p16_3rdparty_64xb64_in1k`\* | From scratch |   208.20   |   44.57   |   72.34   |   88.02   | [config](mlp-mixer-large-p16_64xb64_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/mlp-mixer/mixer-large-p16_3rdparty_64xb64_in1k_20211124-5a2519d2.pth) |

*Models with * are converted from the [timm](https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/mlp_mixer.py). The config files of these models are only for inference. We haven't reproduce the training results.*

## Citation

```bibtex
@misc{tolstikhin2021mlpmixer,
      title={MLP-Mixer: An all-MLP Architecture for Vision},
      author={Ilya Tolstikhin and Neil Houlsby and Alexander Kolesnikov and Lucas Beyer and Xiaohua Zhai and Thomas Unterthiner and Jessica Yung and Andreas Steiner and Daniel Keysers and Jakob Uszkoreit and Mario Lucic and Alexey Dosovitskiy},
      year={2021},
      eprint={2105.01601},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
