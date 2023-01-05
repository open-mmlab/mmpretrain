# MobileNet V3

> [Searching for MobileNetV3](https://arxiv.org/abs/1905.02244)

<!-- [ALGORITHM] -->

## Introduction

**MobileNet V3** is initially described in [the paper](https://arxiv.org/pdf/1905.02244.pdf). MobileNetV3 parameters are obtained by NAS (network architecture search) search, and some practical results of V1 and V2 are inherited, and the attention mechanism of SE channel is attracted, which can be considered as a masterpiece. The author create two new MobileNet models for release: MobileNetV3-Large and MobileNetV3-Small which are targeted for high and low resource use cases. These models are then adapted and applied to the tasks of object detection and semantic segmentation. The author of MobileNet V3 measure its performance on Imagenet classification, COCO object detection, and Cityscapes segmentation.

<div align=center>
<img src="https://user-images.githubusercontent.com/26739999/142563801-ef4feacc-ecd7-4d14-a411-8c9d63571749.png" width="60%"/>
</div>

## Abstract

<details>

<summary>Show the paper's abstract</summary>

<br>
We present the next generation of MobileNets based on a combination of complementary search techniques as well as a novel architecture design. MobileNetV3 is tuned to mobile phone CPUs through a combination of hardware-aware network architecture search (NAS) complemented by the NetAdapt algorithm and then subsequently improved through novel architecture advances. This paper starts the exploration of how automated search algorithms and network design can work together to harness complementary approaches improving the overall state of the art. Through this process we create two new MobileNet models for release: MobileNetV3-Large and MobileNetV3-Small which are targeted for high and low resource use cases. These models are then adapted and applied to the tasks of object detection and semantic segmentation. For the task of semantic segmentation (or any dense pixel prediction), we propose a new efficient segmentation decoder Lite Reduced Atrous Spatial Pyramid Pooling (LR-ASPP). We achieve new state of the art results for mobile classification, detection and segmentation. MobileNetV3-Large is 3.2% more accurate on ImageNet classification while reducing latency by 15% compared to MobileNetV2. MobileNetV3-Small is 4.6% more accurate while reducing latency by 5% compared to MobileNetV2. MobileNetV3-Large detection is 25% faster at roughly the same accuracy as MobileNetV2 on COCO detection. MobileNetV3-Large LR-ASPP is 30% faster than MobileNetV2 R-ASPP at similar accuracy for Cityscapes segmentation.
</br>

</details>

## How to use it?

<!-- [TABS-BEGIN] -->

**Predict image**

```python
>>> import torch
>>> from mmcls.apis import init_model, inference_model
>>>
>>> model = init_model('configs/mobilenet_v3/mobilenet-v3-small-050_8xb128_in1k.py', 'https://download.openmmlab.com/mmclassification/v0/mobilenet_v3/mobilenet-v3-small-050_3rdparty_in1k_20221114-e0b86be1.pth')
>>> predict = inference_model(model, 'demo/demo.JPEG')
>>> print(predict['pred_class'])
print(predict['pred_class'])
>>> print(predict['pred_score'])
print(predict['pred_score'])
```

**Use the model**

```python
>>> import torch
>>> from mmcls.apis import init_model
>>>
>>> model = init_model('configs/mobilenet_v3/mobilenet-v3-small-050_8xb128_in1k.py', 'https://download.openmmlab.com/mmclassification/v0/mobilenet_v3/mobilenet-v3-small-050_3rdparty_in1k_20221114-e0b86be1.pth')
>>> inputs = torch.rand(1, 3, 224, 224).to(model.data_preprocessor.device)
>>> # To get classification scores.
>>> out = model(inputs)
>>> print(out.shape)
torch.Size([1, 1000])
>>> # To extract features.
>>> outs = model.extract_feat(inputs)
>>> print(outs[0].shape)
torch.Size([1, 288])
```

**Train/Test Command**

Place the ImageNet dataset to the `data/imagenet/` directory, or prepare datasets according to the [docs](https://mmclassification.readthedocs.io/en/1.x/user_guides/dataset_prepare.html#prepare-dataset).

Train:

```shell
python tools/train.py configs/mobilenet_v3/mobilenet-v3-small_8xb128_in1k.py
```

Test:

```shell
python tools/test.py configs/mobilenet_v3/mobilenet-v3-small_8xb128_in1k.py https://download.openmmlab.com/mmclassification/v0/mobilenet_v3/mobilenet-v3-small_8xb128_in1k_20221114-bd1bfcde.pth
```

<!-- [TABS-END] -->

For more configurable parameters, please refer to the [API](https://mmclassification.readthedocs.io/en/1.x/api/generated/mmcls.models.backbones.MobileNetV3.html#mmcls.models.backbones.MobileNetV3).

## Results and models

### ImageNet-1k

|           Model           | Params(M) | Flops(G) | Top-1 (%) | Top-5 (%) |                      Config                       |                                          Download                                           |
| :-----------------------: | :-------: | :------: | :-------: | :-------: | :-----------------------------------------------: | :-----------------------------------------------------------------------------------------: |
| MobileNetV3-Small-050\*\* |   1.59    |   0.02   |   57.91   |   80.19   | [config](./mobilenet-v3-small-050_8xb128_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/mobilenet_v3/mobilenet-v3-small-050_3rdparty_in1k_20221114-e0b86be1.pth) |
| MobileNetV3-Small-075\*\* |   2.04    |   0.04   |   65.23   |   85.44   | [config](./mobilenet-v3-small-075_8xb128_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/mobilenet_v3/mobilenet-v3-small-075_3rdparty_in1k_20221114-2011fa76.pth) |
|     MobileNetV3-Small     |   2.54    |   0.06   |   66.68   |   86.74   |   [config](./mobilenet-v3-small_8xb128_in1k.py)   | [model](https://download.openmmlab.com/mmclassification/v0/mobilenet_v3/mobilenet-v3-small_8xb128_in1k_20221114-bd1bfcde.pth) \| [log](https://download.openmmlab.com/mmclassification/v0/mobilenet_v3/mobilenet-v3-small_8xb128_in1k_20221114-bd1bfcde.log.json) |
|    MobileNetV3-Small\*    |   2.54    |   0.06   |   67.66   |   87.41   |   [config](./mobilenet-v3-small_8xb128_in1k.py)   | [model](https://download.openmmlab.com/mmclassification/v0/mobilenet_v3/convert/mobilenet_v3_small-8427ecf0.pth) |
|     MobileNetV3-Large     |   5.48    |   0.23   |   73.49   |   91.31   |   [config](./mobilenet-v3-large_8xb128_in1k.py)   | [model](https://download.openmmlab.com/mmclassification/v0/mobilenet_v3/mobilenet-v3-large_8xb128_in1k_20221114-0ed9ed9a.pth) \| [log](https://download.openmmlab.com/mmclassification/v0/mobilenet_v3/mobilenet-v3-large_8xb128_in1k_20221114-0ed9ed9a.log.json) |
|    MobileNetV3-Large\*    |   5.48    |   0.23   |   74.04   |   91.34   |   [config](./mobilenet-v3-large_8xb128_in1k.py)   | [model](https://download.openmmlab.com/mmclassification/v0/mobilenet_v3/convert/mobilenet_v3_large-3ea3c186.pth) |

We cannot reproduce the performances provided by TorchVision with the [training script](https://github.com/pytorch/vision/tree/master/references/classification#mobilenetv3-large--small), and the accuracy results we got are 65.5±0.5% and 73.39% for small and large model respectively. Here we provide checkpoints trained by MMClassification that outperform the aforementioned results, and the original checkpoints provided by TorchVision.

*Models with * are converted from [torchvision](https://pytorch.org/vision/stable/_modules/torchvision/models/mobilenetv3.html). Models with \*\* are converted from [timm](https://github.com/rwightman/pytorch-image-models/blob/main/timm/models/mobilenetv3.py). The config files of these models are only for validation. We don't ensure these config files' training accuracy and welcome you to contribute your reproduction results.*

## Citation

```
@inproceedings{Howard_2019_ICCV,
    author = {Howard, Andrew and Sandler, Mark and Chu, Grace and Chen, Liang-Chieh and Chen, Bo and Tan, Mingxing and Wang, Weijun and Zhu, Yukun and Pang, Ruoming and Vasudevan, Vijay and Le, Quoc V. and Adam, Hartwig},
    title = {Searching for MobileNetV3},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month = {October},
    year = {2019}
}
```
