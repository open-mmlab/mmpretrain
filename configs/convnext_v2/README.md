# ConvNeXt V2

> [Co-designing and Scaling ConvNets with Masked Autoencoders](http://arxiv.org/abs/2301.00808)

<!-- [ALGORITHM] -->

## Abstract

Driven by improved architectures and better representation learning frameworks, the field of visual recognition has enjoyed rapid modernization and performance boost in the early 2020s. For example, modern ConvNets, represented by ConvNeXt, have demonstrated strong performance in various scenarios. While these models were originally designed for supervised learning with ImageNet labels, they can also potentially benefit from self-supervised learning techniques such as masked autoencoders (MAE). However, we found that simply combining these two approaches leads to subpar performance. In this paper, we propose a fully convolutional masked autoencoder framework and a new Global Response Normalization (GRN) layer that can be added to the ConvNeXt architecture to enhance inter-channel feature competition. This co-design of self-supervised learning techniques and architectural improvement results in a new model family called ConvNeXt V2, which significantly improves the performance of pure ConvNets on various recognition benchmarks, including ImageNet classification, COCO detection, and ADE20K segmentation. We also provide pre-trained ConvNeXt V2 models of various sizes, ranging from an efficient 3.7M-parameter Atto model with 76.7% top-1 accuracy on ImageNet, to a 650M Huge model that achieves a state-of-the-art 88.9% accuracy using only public training data.

<div align=center>
<img src="https://user-images.githubusercontent.com/26739999/210496285-f235083f-218f-4153-8e21-c8a64481a2f5.png" width="50%"/>
</div>

## How to use it?

<!-- [TABS-BEGIN] -->

**Predict image**

```python
from mmpretrain import inference_model

predict = inference_model('convnext-v2-atto_fcmae-pre_3rdparty_in1k', 'demo/bird.JPEG')
print(predict['pred_class'])
print(predict['pred_score'])
```

**Use the model**

```python
import torch
from mmpretrain import get_model

model = get_model('convnext-v2-atto_3rdparty-fcmae_in1k', pretrained=True)
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
python tools/test.py configs/convnext_v2/convnext-v2-atto_32xb32_in1k.py https://download.openmmlab.com/mmclassification/v0/convnext-v2/convnext-v2-atto_fcmae-pre_3rdparty_in1k_20230104-23765f83.pth
```

<!-- [TABS-END] -->

## Models and results

### Pretrained models

| Model                                     | Params (M) | Flops (G) |                   Config                   |                                              Download                                              |
| :---------------------------------------- | :--------: | :-------: | :----------------------------------------: | :------------------------------------------------------------------------------------------------: |
| `convnext-v2-atto_3rdparty-fcmae_in1k`\*  |    3.71    |   0.55    | [config](convnext-v2-atto_32xb32_in1k.py)  | [model](https://download.openmmlab.com/mmclassification/v0/convnext-v2/convnext-v2-atto_3rdparty-fcmae_in1k_20230104-07514db4.pth) |
| `convnext-v2-femto_3rdparty-fcmae_in1k`\* |    5.23    |   0.78    | [config](convnext-v2-femto_32xb32_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/convnext-v2/convnext-v2-femto_3rdparty-fcmae_in1k_20230104-adbe2082.pth) |
| `convnext-v2-pico_3rdparty-fcmae_in1k`\*  |    9.07    |   1.37    | [config](convnext-v2-pico_32xb32_in1k.py)  | [model](https://download.openmmlab.com/mmclassification/v0/convnext-v2/convnext-v2-pico_3rdparty-fcmae_in1k_20230104-147b1b59.pth) |
| `convnext-v2-nano_3rdparty-fcmae_in1k`\*  |   15.62    |   2.45    | [config](convnext-v2-nano_32xb32_in1k.py)  | [model](https://download.openmmlab.com/mmclassification/v0/convnext-v2/convnext-v2-nano_3rdparty-fcmae_in1k_20230104-3dd1f29e.pth) |
| `convnext-v2-tiny_3rdparty-fcmae_in1k`\*  |   28.64    |   4.47    | [config](convnext-v2-tiny_32xb32_in1k.py)  | [model](https://download.openmmlab.com/mmclassification/v0/convnext-v2/convnext-v2-tiny_3rdparty-fcmae_in1k_20230104-80513adc.pth) |
| `convnext-v2-base_3rdparty-fcmae_in1k`\*  |   88.72    |   15.38   | [config](convnext-v2-base_32xb32_in1k.py)  | [model](https://download.openmmlab.com/mmclassification/v0/convnext-v2/convnext-v2-base_3rdparty-fcmae_in1k_20230104-8a798eaf.pth) |
| `convnext-v2-large_3rdparty-fcmae_in1k`\* |   197.96   |   34.40   | [config](convnext-v2-large_32xb32_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/convnext-v2/convnext-v2-large_3rdparty-fcmae_in1k_20230104-bf38df92.pth) |
| `convnext-v2-huge_3rdparty-fcmae_in1k`\*  |   660.29   |  115.00   | [config](convnext-v2-huge_32xb32_in1k.py)  | [model](https://download.openmmlab.com/mmclassification/v0/convnext-v2/convnext-v2-huge_3rdparty-fcmae_in1k_20230104-fe43ae6c.pth) |

*Models with * are converted from the [official repo](https://github.com/facebookresearch/ConvNeXt-V2). The config files of these models are only for inference. We haven't reproduce the training results.*

### Image Classification on ImageNet-1k

| Model                                           |      Pretrain      | Params (M) | Flops (G) | Top-1 (%) | Top-5 (%) |                      Config                      |                      Download                      |
| :---------------------------------------------- | :----------------: | :--------: | :-------: | :-------: | :-------: | :----------------------------------------------: | :------------------------------------------------: |
| `convnext-v2-atto_fcmae-pre_3rdparty_in1k`\*    |       FCMAE        |    3.71    |   0.55    |   76.64   |   93.04   |    [config](convnext-v2-atto_32xb32_in1k.py)     | [model](https://download.openmmlab.com/mmclassification/v0/convnext-v2/convnext-v2-atto_fcmae-pre_3rdparty_in1k_20230104-23765f83.pth) |
| `convnext-v2-femto_fcmae-pre_3rdparty_in1k`\*   |       FCMAE        |    5.23    |   0.78    |   78.48   |   93.98   |    [config](convnext-v2-femto_32xb32_in1k.py)    | [model](https://download.openmmlab.com/mmclassification/v0/convnext-v2/convnext-v2-femto_fcmae-pre_3rdparty_in1k_20230104-92a75d75.pth) |
| `convnext-v2-pico_fcmae-pre_3rdparty_in1k`\*    |       FCMAE        |    9.07    |   1.37    |   80.31   |   95.08   |    [config](convnext-v2-pico_32xb32_in1k.py)     | [model](https://download.openmmlab.com/mmclassification/v0/convnext-v2/convnext-v2-pico_fcmae-pre_3rdparty_in1k_20230104-d20263ca.pth) |
| `convnext-v2-nano_fcmae-pre_3rdparty_in1k`\*    |       FCMAE        |   15.62    |   2.45    |   81.86   |   95.75   |    [config](convnext-v2-nano_32xb32_in1k.py)     | [model](https://download.openmmlab.com/mmclassification/v0/convnext-v2/convnext-v2-nano_fcmae-pre_3rdparty_in1k_20230104-fe1aaaf2.pth) |
| `convnext-v2-nano_fcmae-in21k-pre_3rdparty_in1k`\* | FCMAE ImageNet-21k |   15.62    |   2.45    |   82.04   |   96.16   |    [config](convnext-v2-nano_32xb32_in1k.py)     | [model](https://download.openmmlab.com/mmclassification/v0/convnext-v2/convnext-v2-nano_fcmae-in21k-pre_3rdparty_in1k_20230104-91fa8ae2.pth) |
| `convnext-v2-tiny_fcmae-pre_3rdparty_in1k`\*    |       FCMAE        |   28.64    |   4.47    |   82.94   |   96.29   |    [config](convnext-v2-tiny_32xb32_in1k.py)     | [model](https://download.openmmlab.com/mmclassification/v0/convnext-v2/convnext-v2-tiny_fcmae-pre_3rdparty_in1k_20230104-471a86de.pth) |
| `convnext-v2-tiny_fcmae-in21k-pre_3rdparty_in1k`\* | FCMAE ImageNet-21k |   28.64    |   4.47    |   83.89   |   96.96   |    [config](convnext-v2-tiny_32xb32_in1k.py)     | [model](https://download.openmmlab.com/mmclassification/v0/convnext-v2/convnext-v2-tiny_fcmae-in21k-pre_3rdparty_in1k_20230104-8cc8b8f2.pth) |
| `convnext-v2-nano_fcmae-in21k-pre_3rdparty_in1k-384px`\* | FCMAE ImageNet-21k |   15.62    |   7.21    |   83.36   |   96.75   | [config](convnext-v2-nano_32xb32_in1k-384px.py)  | [model](https://download.openmmlab.com/mmclassification/v0/convnext-v2/convnext-v2-nano_fcmae-in21k-pre_3rdparty_in1k-384px_20230104-f951ae87.pth) |
| `convnext-v2-tiny_fcmae-in21k-pre_3rdparty_in1k-384px`\* | FCMAE ImageNet-21k |   28.64    |   13.14   |   85.09   |   97.63   | [config](convnext-v2-tiny_32xb32_in1k-384px.py)  | [model](https://download.openmmlab.com/mmclassification/v0/convnext-v2/convnext-v2-tiny_fcmae-in21k-pre_3rdparty_in1k-384px_20230104-d8579f84.pth) |
| `convnext-v2-base_fcmae-pre_3rdparty_in1k`\*    |       FCMAE        |   88.72    |   15.38   |   84.87   |   97.08   |    [config](convnext-v2-base_32xb32_in1k.py)     | [model](https://download.openmmlab.com/mmclassification/v0/convnext-v2/convnext-v2-base_fcmae-pre_3rdparty_in1k_20230104-00a70fa4.pth) |
| `convnext-v2-base_fcmae-in21k-pre_3rdparty_in1k`\* | FCMAE ImageNet-21k |   88.72    |   15.38   |   86.74   |   98.02   |    [config](convnext-v2-base_32xb32_in1k.py)     | [model](https://download.openmmlab.com/mmclassification/v0/convnext-v2/convnext-v2-base_fcmae-in21k-pre_3rdparty_in1k_20230104-c48d16a5.pth) |
| `convnext-v2-large_fcmae-pre_3rdparty_in1k`\*   |       FCMAE        |   197.96   |   34.40   |   85.76   |   97.59   |    [config](convnext-v2-large_32xb32_in1k.py)    | [model](https://download.openmmlab.com/mmclassification/v0/convnext-v2/convnext-v2-large_fcmae-pre_3rdparty_in1k_20230104-ef393013.pth) |
| `convnext-v2-large_fcmae-in21k-pre_3rdparty_in1k`\* | FCMAE ImageNet-21k |   197.96   |   34.40   |   87.26   |   98.24   |    [config](convnext-v2-large_32xb32_in1k.py)    | [model](https://download.openmmlab.com/mmclassification/v0/convnext-v2/convnext-v2-large_fcmae-in21k-pre_3rdparty_in1k_20230104-d9c4dc0c.pth) |
| `convnext-v2-base_fcmae-in21k-pre_3rdparty_in1k-384px`\* | FCMAE ImageNet-21k |   88.72    |   45.21   |   87.63   |   98.42   | [config](convnext-v2-base_32xb32_in1k-384px.py)  | [model](https://download.openmmlab.com/mmclassification/v0/convnext-v2/convnext-v2-base_fcmae-in21k-pre_3rdparty_in1k-384px_20230104-379425cc.pth) |
| `convnext-v2-large_fcmae-in21k-pre_3rdparty_in1k-384px`\* | FCMAE ImageNet-21k |   197.96   |  101.10   |   88.18   |   98.52   | [config](convnext-v2-large_32xb32_in1k-384px.py) | [model](https://download.openmmlab.com/mmclassification/v0/convnext-v2/convnext-v2-large_fcmae-in21k-pre_3rdparty_in1k-384px_20230104-9139a1f3.pth) |
| `convnext-v2-huge_fcmae-pre_3rdparty_in1k`\*    |       FCMAE        |   660.29   |  115.00   |   86.25   |   97.75   |    [config](convnext-v2-huge_32xb32_in1k.py)     | [model](https://download.openmmlab.com/mmclassification/v0/convnext-v2/convnext-v2-huge_fcmae-pre_3rdparty_in1k_20230104-f795e5b8.pth) |
| `convnext-v2-huge_fcmae-in21k-pre_3rdparty_in1k-384px`\* | FCMAE ImageNet-21k |   660.29   |  337.96   |   88.68   |   98.73   | [config](convnext-v2-huge_32xb32_in1k-384px.py)  | [model](https://download.openmmlab.com/mmclassification/v0/convnext-v2/convnext-v2-huge_fcmae-in21k-pre_3rdparty_in1k-384px_20230104-02a4eb35.pth) |
| `convnext-v2-huge_fcmae-in21k-pre_3rdparty_in1k-512px`\* | FCMAE ImageNet-21k |   660.29   |  600.81   |   88.86   |   98.74   | [config](convnext-v2-huge_32xb32_in1k-512px.py)  | [model](https://download.openmmlab.com/mmclassification/v0/convnext-v2/convnext-v2-huge_fcmae-in21k-pre_3rdparty_in1k-512px_20230104-ce32e63c.pth) |

*Models with * are converted from the [official repo](https://github.com/facebookresearch/ConvNeXt-V2). The config files of these models are only for inference. We haven't reproduce the training results.*

## Citation

```bibtex
@article{Woo2023ConvNeXtV2,
  title={ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders},
  author={Sanghyun Woo, Shoubhik Debnath, Ronghang Hu, Xinlei Chen, Zhuang Liu, In So Kweon and Saining Xie},
  year={2023},
  journal={arXiv preprint arXiv:2301.00808},
}
```
