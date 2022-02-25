# MobileNet V3

> [Searching for MobileNetV3](https://arxiv.org/abs/1905.02244)
<!-- [ALGORITHM] -->

## Abstract

We present the next generation of MobileNets based on a combination of complementary search techniques as well as a novel architecture design. MobileNetV3 is tuned to mobile phone CPUs through a combination of hardware-aware network architecture search (NAS) complemented by the NetAdapt algorithm and then subsequently improved through novel architecture advances. This paper starts the exploration of how automated search algorithms and network design can work together to harness complementary approaches improving the overall state of the art. Through this process we create two new MobileNet models for release: MobileNetV3-Large and MobileNetV3-Small which are targeted for high and low resource use cases. These models are then adapted and applied to the tasks of object detection and semantic segmentation. For the task of semantic segmentation (or any dense pixel prediction), we propose a new efficient segmentation decoder Lite Reduced Atrous Spatial Pyramid Pooling (LR-ASPP). We achieve new state of the art results for mobile classification, detection and segmentation. MobileNetV3-Large is 3.2\% more accurate on ImageNet classification while reducing latency by 15\% compared to MobileNetV2. MobileNetV3-Small is 4.6\% more accurate while reducing latency by 5\% compared to MobileNetV2. MobileNetV3-Large detection is 25\% faster at roughly the same accuracy as MobileNetV2 on COCO detection. MobileNetV3-Large LR-ASPP is 30\% faster than MobileNetV2 R-ASPP at similar accuracy for Cityscapes segmentation.

<div align=center>
<img src="https://user-images.githubusercontent.com/26739999/142563801-ef4feacc-ecd7-4d14-a411-8c9d63571749.png" width="70%"/>
</div>

## Results and models

### ImageNet-1k

|         Model         | Params(M) | Flops(G) | Top-1 (%) | Top-5 (%) | Config | Download |
|:---------------------:|:---------:|:--------:|:---------:|:---------:|:------:|:--------:|
| MobileNetV3-Small\*   |   2.54    |   0.06   |   67.66   |   87.41   | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/mobilenet_v3/mobilenet-v3-small_8xb32_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/mobilenet_v3/convert/mobilenet_v3_small-8427ecf0.pth) |
| MobileNetV3-Large\*   |   5.48    |   0.23   |   74.04   |   91.34   | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/mobilenet_v3/mobilenet-v3-large_8xb32_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/mobilenet_v3/convert/mobilenet_v3_large-3ea3c186.pth) |

*Models with \* are converted from [torchvision](https://pytorch.org/vision/stable/_modules/torchvision/models/mobilenetv3.html). The config files of these models are only for validation. We don't ensure these config files' training accuracy and welcome you to contribute your reproduction results.*

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
