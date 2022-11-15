# NPU (HUAWEI Ascend)

## Usage

Please install MMCV with NPU device support according to {external+mmcv:doc}`the tutorial <get_started/build>`.

Here we use 8 NPUs on your computer to train the model with the following command:

```shell
bash tools/dist_train.sh configs/cspnet/resnet50_8xb32_in1k.py 8 --device npu
```

Also, you can use only one NPU to trian the model with the following command:

```shell
python tools/train.py configs/cspnet/resnet50_8xb32_in1k.py --device npu
```

## Verified Models

|                           Model                            | Top-1 (%) | Top-5 (%) |                            Config                             |                            Download                             |
| :--------------------------------------------------------: | :-------: | :-------: | :-----------------------------------------------------------: | :-------------------------------------------------------------: |
|            [CSPResNeXt50](../papers/cspnet.md)             |   77.10   |   93.55   | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/cspnet/cspresnext50_8xb32_in1k.py) | [model](<>) \| [log](https://download.openmmlab.com/mmclassification/v0/device/npu/cspresnext50_8xb32_in1k.log.json) |
|            [DenseNet121](../papers/densenet.md)            |   72.62   |   91.04   | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/densenet/densenet121_4xb256_in1k.py) | [model](<>) \| [log](https://download.openmmlab.com/mmclassification/v0/device/npu/densenet121_4xb256_in1k.log.json) |
| [EfficientNet-B4(AA + AdvProp)](../papers/efficientnet.md) |   75.55   |   92.86   | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/efficientnet/efficientnet-b4_8xb32-01norm_in1k.py) | [model](<>) \| [log](https://download.openmmlab.com/mmclassification/v0/device/npu/efficientnet-b4_8xb32-01norm_in1k.log.json) |
|              [HRNet-W18](../papers/hrnet.md)               |   77.01   |   93.46   | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/hrnet/hrnet-w18_4xb32_in1k.py) | [model](<>) \| [log](https://download.openmmlab.com/mmclassification/v0/device/npu/hrnet-w18_4xb32_in1k.log.json) |
|            [ResNetV1D-152](../papers/resnet.md)            |   77.11   |   94.54   | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/resnet/resnetv1d152_8xb32_in1k.py) | [model](<>) \| [log](https://download.openmmlab.com/mmclassification/v0/device/npu/resnetv1d152_8xb32_in1k.log.json) |
|              [ResNet-50](../papers/resnet.md)              |   76.38   |   93.22   | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/resnet/resnet50_8xb32_in1k.py) | [model](<>) \| [log](https://download.openmmlab.com/mmclassification/v0/device/npu/resnet50_8xb32_in1k.log) |
|         [ResNetXt-32x4d-50](../papers/resnext.md)          |   77.55   |   93.75   | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/resnext/resnext50-32x4d_8xb32_in1k.py) | [model](<>) \| [log](https://download.openmmlab.com/mmclassification/v0/device/npu/resnext50-32x4d_8xb32_in1k.log.json) |
|           [SE-ResNet-50](../papers/seresnet.md)            |   77.64   |   93.76   | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/seresnet/seresnet50_8xb32_in1k.py) | [model](<>) \| [log](https://download.openmmlab.com/mmclassification/v0/device/npu/seresnet50_8xb32_in1k.log.json) |
|                 [VGG-11](../papers/vgg.md)                 |   68.92   |   88.83   | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/vgg/vgg11_8xb32_in1k.py) | [model](<>) \| [log](https://download.openmmlab.com/mmclassification/v0/device/npu/vgg11_8xb32_in1k.log.json) |
|      [ShuffleNetV2 1.0x](../papers/shufflenet_v2.md)       |   69.53   |   88.82   | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/shufflenet_v2/shufflenet-v2-1x_16xb64_in1k.py) | [model](<>) \| [log](https://download.openmmlab.com/mmclassification/v0/device/npu/shufflenet-v2-1x_16xb64_in1k.json) |

**All above models are provided by Huawei Ascend group.**
