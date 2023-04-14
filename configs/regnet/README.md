# RegNet

> [Designing Network Design Spaces](https://arxiv.org/abs/2003.13678)

<!-- [ALGORITHM] -->

## Abstract

In this work, we present a new network design paradigm. Our goal is to help advance the understanding of network design and discover design principles that generalize across settings. Instead of focusing on designing individual network instances, we design network design spaces that parametrize populations of networks. The overall process is analogous to classic manual design of networks, but elevated to the design space level. Using our methodology we explore the structure aspect of network design and arrive at a low-dimensional design space consisting of simple, regular networks that we call RegNet. The core insight of the RegNet parametrization is surprisingly simple: widths and depths of good networks can be explained by a quantized linear function. We analyze the RegNet design space and arrive at interesting findings that do not match the current practice of network design. The RegNet design space provides simple and fast networks that work well across a wide range of flop regimes. Under comparable training settings and flops, the RegNet models outperform the popular EfficientNet models while being up to 5x faster on GPUs.

<div align=center>
<img src="https://user-images.githubusercontent.com/26739999/142572813-5dad3317-9d58-4177-971f-d346e01fb3c4.png" width=60%/>
</div>

## How to use it?

<!-- [TABS-BEGIN] -->

**Predict image**

```python
from mmpretrain import inference_model

predict = inference_model('regnetx-400mf_8xb128_in1k', 'demo/bird.JPEG')
print(predict['pred_class'])
print(predict['pred_score'])
```

**Use the model**

```python
import torch
from mmpretrain import get_model

model = get_model('regnetx-400mf_8xb128_in1k', pretrained=True)
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
python tools/train.py configs/regnet/regnetx-400mf_8xb128_in1k.py
```

Test:

```shell
python tools/test.py configs/regnet/regnetx-400mf_8xb128_in1k.py https://download.openmmlab.com/mmclassification/v0/regnet/regnetx-400mf_8xb128_in1k_20211213-89bfc226.pth
```

<!-- [TABS-END] -->

## Models and results

### Image Classification on ImageNet-1k

| Model                       |   Pretrain   | Params (M) | Flops (G) | Top-1 (%) | Top-5 (%) |                 Config                 |                                        Download                                        |
| :-------------------------- | :----------: | :--------: | :-------: | :-------: | :-------: | :------------------------------------: | :------------------------------------------------------------------------------------: |
| `regnetx-400mf_8xb128_in1k` | From scratch |    5.16    |   0.41    |   72.56   |   90.78   | [config](regnetx-400mf_8xb128_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/regnet/regnetx-400mf_8xb128_in1k_20211213-89bfc226.pth) \| [log](https://download.openmmlab.com/mmclassification/v0/regnet/regnetx-400mf_8xb128_in1k_20211208_143316.json) |
| `regnetx-800mf_8xb128_in1k` | From scratch |    7.26    |   0.81    |   74.76   |   92.32   | [config](regnetx-800mf_8xb128_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/regnet/regnetx-800mf_8xb128_in1k_20211213-222b0f11.pth) \| [log](https://download.openmmlab.com/mmclassification/v0/regnet/regnetx-800mf_8xb128_in1k_20211207_143037.log.json) |
| `regnetx-1.6gf_8xb128_in1k` | From scratch |    9.19    |   1.63    |   76.84   |   93.31   | [config](regnetx-1.6gf_8xb128_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/regnet/regnetx-1.6gf_8xb128_in1k_20211213-d1b89758.pth) \| [log](https://download.openmmlab.com/mmclassification/v0/regnet/regnetx-1.6gf_8xb128_in1k_20211208_143018.log.json) |
| `regnetx-3.2gf_8xb64_in1k`  | From scratch |    3.21    |   1.53    |   78.09   |   94.08   | [config](regnetx-3.2gf_8xb64_in1k.py)  | [model](https://download.openmmlab.com/mmclassification/v0/regnet/regnetx-3.2gf_8xb64_in1k_20211213-1fdd82ae.pth) \| [log](https://download.openmmlab.com/mmclassification/v0/regnet/regnetx-3.2gf_8xb64_in1k_20211208_142720.log.json) |
| `regnetx-4.0gf_8xb64_in1k`  | From scratch |   22.12    |   4.00    |   78.60   |   94.17   | [config](regnetx-4.0gf_8xb64_in1k.py)  | [model](https://download.openmmlab.com/mmclassification/v0/regnet/regnetx-4.0gf_8xb64_in1k_20211213-efed675c.pth) \| [log](https://download.openmmlab.com/mmclassification/v0/regnet/regnetx-4.0gf_8xb64_in1k_20211207_150431.log.json) |
| `regnetx-6.4gf_8xb64_in1k`  | From scratch |   26.21    |   6.51    |   79.38   |   94.65   | [config](regnetx-6.4gf_8xb64_in1k.py)  | [model](https://download.openmmlab.com/mmclassification/v0/regnet/regnetx-6.4gf_8xb64_in1k_20211215-5c6089da.pth) \| [log](https://download.openmmlab.com/mmclassification/v0/regnet/regnetx-6.4gf_8xb64_in1k_20211213_172748.log.json) |
| `regnetx-8.0gf_8xb64_in1k`  | From scratch |   39.57    |   8.03    |   79.12   |   94.51   | [config](regnetx-8.0gf_8xb64_in1k.py)  | [model](https://download.openmmlab.com/mmclassification/v0/regnet/regnetx-8.0gf_8xb64_in1k_20211213-9a9fcc76.pth) \| [log](https://download.openmmlab.com/mmclassification/v0/regnet/regnetx-8.0gf_8xb64_in1k_20211208_103250.log.json) |
| `regnetx-12gf_8xb64_in1k`   | From scratch |   46.11    |   12.15   |   79.67   |   95.03   |  [config](regnetx-12gf_8xb64_in1k.py)  | [model](https://download.openmmlab.com/mmclassification/v0/regnet/regnetx-12gf_8xb64_in1k_20211213-5df8c2f8.pth) \| [log](https://download.openmmlab.com/mmclassification/v0/regnet/regnetx-12gf_8xb64_in1k_20211208_143713.log.json) |

## Citation

```bibtex
@article{radosavovic2020designing,
    title={Designing Network Design Spaces},
    author={Ilija Radosavovic and Raj Prateek Kosaraju and Ross Girshick and Kaiming He and Piotr Dollár},
    year={2020},
    eprint={2003.13678},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```
