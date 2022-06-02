# ShuffleNet V1

> [ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices](https://openaccess.thecvf.com/content_cvpr_2018/html/Zhang_ShuffleNet_An_Extremely_CVPR_2018_paper.html)

<!-- [ALGORITHM] -->

## Abstract

We introduce an extremely computation-efficient CNN architecture named ShuffleNet, which is designed specially for mobile devices with very limited computing power (e.g., 10-150 MFLOPs). The new architecture utilizes two new operations, pointwise group convolution and channel shuffle, to greatly reduce computation cost while maintaining accuracy. Experiments on ImageNet classification and MS COCO object detection demonstrate the superior performance of ShuffleNet over other structures, e.g. lower top-1 error (absolute 7.8%) than recent MobileNet on ImageNet classification task, under the computation budget of 40 MFLOPs. On an ARM-based mobile device, ShuffleNet achieves ~13x actual speedup over AlexNet while maintaining comparable accuracy.

<div align=center>
<img src="https://user-images.githubusercontent.com/26739999/142575730-dc2f616d-80df-4fb1-93e1-77ebb2b835cf.png" width="70%"/>
</div>

## Results and models

### ImageNet-1k

|            Model            | Params(M) | Flops(G) | Top-1 (%) | Top-5 (%) |                                Config                                |                                Download                                |
| :-------------------------: | :-------: | :------: | :-------: | :-------: | :------------------------------------------------------------------: | :--------------------------------------------------------------------: |
| ShuffleNetV1 1.0x (group=3) |   1.87    |  0.146   |   68.13   |   87.81   | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/shufflenet_v1/shufflenet-v1-1x_16xb64_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/shufflenet_v1/shufflenet_v1_batch1024_imagenet_20200804-5d6cec73.pth) \| [log](https://download.openmmlab.com/mmclassification/v0/shufflenet_v1/shufflenet_v1_batch1024_imagenet_20200804-5d6cec73.log.json) |

## Citation

```
@inproceedings{zhang2018shufflenet,
  title={Shufflenet: An extremely efficient convolutional neural network for mobile devices},
  author={Zhang, Xiangyu and Zhou, Xinyu and Lin, Mengxiao and Sun, Jian},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={6848--6856},
  year={2018}
}
```
