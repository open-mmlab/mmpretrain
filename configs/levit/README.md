# LeViT

> [LeViT: a Vision Transformer in ConvNet’s Clothing for Faster Inference](https://arxiv.org/abs/2104.01136)

<!-- [ALGORITHM] -->

## Abstract

We design a family of image classification architectures that optimize the trade-off between accuracy and efficiency in a high-speed regime. Our work exploits recent findings in attention-based architectures, which are competitive on highly parallel processing hardware. We revisit principles from the extensive literature on convolutional neural networks to apply them to transformers, in particular activation maps with decreasing resolutions. We also introduce the attention bias, a new way to integrate positional information in vision transformers. As a result, we propose LeVIT: a hybrid neural network for fast inference image classification. We consider different measures of efficiency on different hardware platforms, so as to best reflect a wide range of application scenarios. Our extensive experiments empirically validate our technical choices and show they are suitable to most architectures. Overall, LeViT significantly outperforms existing convnets and vision transformers with respect to the speed/accuracy tradeoff. For example, at 80% ImageNet top-1 accuracy, LeViT is 5 times faster than EfficientNet on CPU.

<div align=center>
<img src="https://raw.githubusercontent.com/facebookresearch/LeViT/main/.github/levit.png" width="90%"/>
</div>

## How to use it?

<!-- [TABS-BEGIN] -->

**Predict image**

```python
from mmpretrain import inference_model

predict = inference_model('levit-128s_3rdparty_in1k', 'demo/bird.JPEG')
print(predict['pred_class'])
print(predict['pred_score'])
```

**Use the model**

```python
import torch
from mmpretrain import get_model

model = get_model('levit-128s_3rdparty_in1k', pretrained=True)
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
python tools/test.py configs/levit/levit-128s_8xb256_in1k.py https://download.openmmlab.com/mmclassification/v0/levit/levit-128s_3rdparty_in1k_20230117-e9fbd209.pth
```

<!-- [TABS-END] -->

## Models and results

### Image Classification on ImageNet-1k

| Model                        |   Pretrain   | Params (M) | Flops (G) | Top-1 (%) | Top-5 (%) |               Config                |                                         Download                                         |
| :--------------------------- | :----------: | :--------: | :-------: | :-------: | :-------: | :---------------------------------: | :--------------------------------------------------------------------------------------: |
| `levit-128s_3rdparty_in1k`\* | From scratch |    7.39    |   0.31    |   76.51   |   92.90   | [config](levit-128s_8xb256_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/levit/levit-128s_3rdparty_in1k_20230117-e9fbd209.pth) |
| `levit-128_3rdparty_in1k`\*  | From scratch |    8.83    |   0.41    |   78.58   |   93.95   | [config](levit-128_8xb256_in1k.py)  | [model](https://download.openmmlab.com/mmclassification/v0/levit/levit-128_3rdparty_in1k_20230117-3be02a02.pth) |
| `levit-192_3rdparty_in1k`\*  | From scratch |   10.56    |   0.67    |   79.86   |   94.75   | [config](levit-192_8xb256_in1k.py)  | [model](https://download.openmmlab.com/mmclassification/v0/levit/levit-192_3rdparty_in1k_20230117-8217a0f9.pth) |
| `levit-256_3rdparty_in1k`\*  | From scratch |   18.38    |   1.14    |   81.59   |   95.46   | [config](levit-256_8xb256_in1k.py)  | [model](https://download.openmmlab.com/mmclassification/v0/levit/levit-256_3rdparty_in1k_20230117-5ae2ce7d.pth) |
| `levit-384_3rdparty_in1k`\*  | From scratch |   38.36    |   2.37    |   82.59   |   95.95   | [config](levit-384_8xb256_in1k.py)  | [model](https://download.openmmlab.com/mmclassification/v0/levit/levit-384_3rdparty_in1k_20230117-f3539cce.pth) |

*Models with * are converted from the [official repo](https://github.com/facebookresearch/LeViT). The config files of these models are only for inference. We haven't reproduce the training results.*

## Citation

```bibtex
@InProceedings{Graham_2021_ICCV,
    author    = {Graham, Benjamin and El-Nouby, Alaaeldin and Touvron, Hugo and Stock, Pierre and Joulin, Armand and Jegou, Herve and Douze, Matthijs},
    title     = {LeViT: A Vision Transformer in ConvNet's Clothing for Faster Inference},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2021},
    pages     = {12259-12269}
}
```
