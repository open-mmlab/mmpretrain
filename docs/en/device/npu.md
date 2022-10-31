# NPU (HUAWEI Ascend)

## Usage

Please refer to [link](https://github.com/open-mmlab/mmcv/blob/master/docs/zh_cn/get_started/build.md) installing mmcv on NPU Devices.

Here we use 8 NPUs on your computer to train the model with the following command:

```shell
bash tools/dist_train.sh configs/cspnet/resnet50_8xb32_in1k.py 8 --device npu
```

Also, you can use only one NPU to trian the model with the following command:

```shell
python tools/train.py configs/cspnet/resnet50_8xb32_in1k.py --device npu
```

## Verified Models

|                Model                | Top-1 (%) | Top-5 (%) | Config                                                                                                                         |  Download   |
| :---------------------------------: | :-------: | :-------: | :----------------------------------------------------------------------------------------------------------------------------- | :---------: |
|         [CSPResNeXt50](<>)          |   77.1    |   93.55   | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/cspnet/cspresnext50_8xb32_in1k.py)                 | [model](<>) |
|          [DenseNet121](<>)          |   72.62   |   91.04   | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/densenet/densenet121_4xb256_in1k.py)               | [model](<>) |
| [EfficientNet-B4(AA + AdvProp)](<>) |  75.552   |   92.86   | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/efficientnet/efficientnet-b4_8xb32-01norm_in1k.py) | [model](<>) |
|           [HRNet-W18](<>)           |   77.01   |  93.462   | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/hrnet/hrnet-w18_4xb32_in1k.py)                     | [model](<>) |
|         [ResNetV1D-152](<>)         |  77.114   |  94.542   | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/resnet/resnetv1d152_8xb32_in1k.py)                 | [model](<>) |
|           [ResNet-50](<>)           |   76.4    |     -     | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/resnet/resnet50_8xb32_in1k.py)                     | [model](<>) |
|       [ResNetXt-32x4d-50](<>)       |  77.548   |  93.752   | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/resnext/resnext50-32x4d_8xb32_in1k.py)             | [model](<>) |
|         [SE-ResNet-50](<>)          |  77.642   |  93.756   | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/seresnet/seresnet50_8xb32_in1k.py)                 | [model](<>) |
|            [VGG-11](<>)             |  68.916   |  88.832   | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/vgg/vgg11_8xb32_in1k.py)                           | [model](<>) |
|       [ShuffleNetV2 1.0x](<>)       |   69.53   |   88.82   | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/shufflenet_v2/shufflenet-v2-1x_16xb64_in1k.py)     | [model](<>) |

**All above models are provided by Huawei Ascend group.**
