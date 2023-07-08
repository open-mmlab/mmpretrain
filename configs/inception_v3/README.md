# Inception V3

> [Rethinking the Inception Architecture for Computer Vision](http://arxiv.org/abs/1512.00567)

<!-- [ALGORITHM] -->

## Abstract

Convolutional networks are at the core of most state-of-the-art computer vision solutions for a wide variety of tasks. Since 2014 very deep convolutional networks started to become mainstream, yielding substantial gains in various benchmarks. Although increased model size and computational cost tend to translate to immediate quality gains for most tasks (as long as enough labeled data is provided for training), computational efficiency and low parameter count are still enabling factors for various use cases such as mobile vision and big-data scenarios. Here we explore ways to scale up networks in ways that aim at utilizing the added computation as efficiently as possible by suitably factorized convolutions and aggressive regularization. We benchmark our methods on the ILSVRC 2012 classification challenge validation set demonstrate substantial gains over the state of the art: 21.2% top-1 and 5.6% top-5 error for single frame evaluation using a network with a computational cost of 5 billion multiply-adds per inference and with using less than 25 million parameters. With an ensemble of 4 models and multi-crop evaluation, we report 3.5% top-5 error on the validation set (3.6% error on the test set) and 17.3% top-1 error on the validation set.

<div align=center>
<img src="https://user-images.githubusercontent.com/26739999/177241797-c103eff4-79bb-414d-aef6-eac323b65a50.png" width="40%"/>
</div>

## How to use it?

<!-- [TABS-BEGIN] -->

**Predict image**

```python
from mmpretrain import inference_model

predict = inference_model('inception-v3_3rdparty_8xb32_in1k', 'demo/bird.JPEG')
print(predict['pred_class'])
print(predict['pred_score'])
```

**Use the model**

```python
import torch
from mmpretrain import get_model

model = get_model('inception-v3_3rdparty_8xb32_in1k', pretrained=True)
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
python tools/test.py configs/inception_v3/inception-v3_8xb32_in1k.py https://download.openmmlab.com/mmclassification/v0/inception-v3/inception-v3_3rdparty_8xb32_in1k_20220615-dcd4d910.pth
```

<!-- [TABS-END] -->

## Models and results

### Image Classification on ImageNet-1k

| Model                                |   Pretrain   | Params (M) | Flops (G) | Top-1 (%) | Top-5 (%) |                Config                |                                    Download                                     |
| :----------------------------------- | :----------: | :--------: | :-------: | :-------: | :-------: | :----------------------------------: | :-----------------------------------------------------------------------------: |
| `inception-v3_3rdparty_8xb32_in1k`\* | From scratch |   23.83    |   5.75    |   77.57   |   93.58   | [config](inception-v3_8xb32_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/inception-v3/inception-v3_3rdparty_8xb32_in1k_20220615-dcd4d910.pth) |

*Models with * are converted from the [official repo](https://github.com/pytorch/vision/blob/main/torchvision/models/inception.py#L28). The config files of these models are only for inference. We haven't reproduce the training results.*

## Citation

```bibtex
@inproceedings{szegedy2016rethinking,
  title={Rethinking the inception architecture for computer vision},
  author={Szegedy, Christian and Vanhoucke, Vincent and Ioffe, Sergey and Shlens, Jon and Wojna, Zbigniew},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={2818--2826},
  year={2016}
}
```
