# NPU (华为昇腾)

## 使用方法

首先，请参考[链接](https://mmcv.readthedocs.io/zh_CN/latest/get_started/build.html#npu-mmcv-full)安装带有 NPU 支持的 MMCV 和[链接](https://mmengine.readthedocs.io/en/latest/get_started/installation.html#build-from-source)安装 MMEngine。

使用如下命令，可以利用 8 个 NPU 在机器上训练模型（以 ResNet 为例）：

```shell
bash tools/dist_train.sh configs/cspnet/resnet50_8xb32_in1k.py 8
```

或者，使用如下命令，在一个 NPU 上训练模型（以 ResNet 为例）：

```shell
python tools/train.py configs/cspnet/resnet50_8xb32_in1k.py
```

## 经过验证的模型

|                            Model                            | Top-1 (%) | Top-5 (%) |                            Config                            |                            Download                             |
| :---------------------------------------------------------: | :-------: | :-------: | :----------------------------------------------------------: | :-------------------------------------------------------------: |
| [ResNet-50](https://github.com/open-mmlab/mmclassification/blob/1.x/configs/resnet/README.md) |   76.40   |   93.21   | [config](https://github.com/open-mmlab/mmclassification/blob/1.x/configs/resnet/resnet50_8xb32_in1k.py) | [model](<>) \| [log](https://download.openmmlab.com/mmclassification/v1/device/npu/resnet50_8xb32_in1k.log) |
| [ResNetXt-32x4d-50](https://github.com/open-mmlab/mmclassification/blob/1.x/configs/resnext/README.md) |   77.48   |   93.75   | [config](https://github.com/open-mmlab/mmclassification/blob/1.x/configs/resnext/resnext50-32x4d_8xb32_in1k.py) | [model](<>) \| [log](https://download.openmmlab.com/mmclassification/v1/device/npu/resnext50-32x4d_8xb32_in1k.log) |
| [HRNet-W18](https://github.com/open-mmlab/mmclassification/blob/master/configs/hrnet/README.md) |   77.06   |   93.57   | [config](https://github.com/open-mmlab/mmclassification/blob/1.x/configs/hrnet/hrnet-w18_4xb32_in1k.py) | [model](<>) \| [log](https://download.openmmlab.com/mmclassification/v1/device/npu/hrnet-w18_4xb32_in1k.log) |
| [ResNetV1D-152](https://github.com/open-mmlab/mmclassification/blob/1.x/configs/resnet/README.md) |   79.41   |   94.48   | [config](https://github.com/open-mmlab/mmclassification/blob/1.x/configs/resnet/resnetv1d152_8xb32_in1k.py) | [model](<>) \| [log](https://download.openmmlab.com/mmclassification/v1/device/npu/resnetv1d152_8xb32_in1k.log) |
| [SE-ResNet-50](https://github.com/open-mmlab/mmclassification/blob/1.x/configs/seresnet/README.md) |   77.65   |   93.74   | [config](https://github.com/open-mmlab/mmclassification/blob/1.x/configs/seresnet/seresnet50_8xb32_in1k.py) | [model](<>) \|[log](https://download.openmmlab.com/mmclassification/v1/device/npu/seresnet50_8xb32_in1k.log) |
| [ShuffleNetV2 1.0x](https://github.com/open-mmlab/mmclassification/blob/1.x/configs/shufflenet_v2/README.md) |   69.52   |   88.79   | [config](https://github.com/open-mmlab/mmclassification/blob/1.x/configs/shufflenet_v2/shufflenet-v2-1x_16xb64_in1k.py) | [model](<>) \| [log](https://download.openmmlab.com/mmclassification/v1/device/npu/shufflenet-v2-1x_16xb64_in1k.log) |
| [MobileNetV2](https://github.com/open-mmlab/mmclassification/tree/1.x/configs/mobilenet_v2) |   71.74   |   90.28   | [config](https://github.com/open-mmlab/mmclassification/blob/1.x/configs/mobilenet_v2/mobilenet-v2_8xb32_in1k.py) | [model](<>) \| [log](https://download.openmmlab.com/mmclassification/v1/device/npu/mobilenet-v2_8xb32_in1k.log) |
| [MobileNetV3-Small](https://github.com/open-mmlab/mmclassification/blob/1.x/configs/mobilenet_v3/README.md) |   67.09   |   87.17   | [config](https://github.com/open-mmlab/mmclassification/blob/1.x/configs/mobilenet_v3/mobilenet-v3-small_8xb128_in1k.py) | [model](<>) \| [log](https://download.openmmlab.com/mmclassification/v1/device/npu/mobilenet-v3-small.log) |
| [\*CSPResNeXt50](https://github.com/open-mmlab/mmclassification/blob/1.x/configs/cspnet/README.md) |   77.25   |   93.46   | [config](https://github.com/open-mmlab/mmclassification/blob/1.x/configs/cspnet/cspresnext50_8xb32_in1k.py) | [model](<>) \| [log](https://download.openmmlab.com/mmclassification/v1/device/npu/cspresnext50_8xb32_in1k.log) |
| [\*EfficientNet-B4](https://github.com/open-mmlab/mmclassification/blob/1.x/configs/efficientnet/README.md) |   75.73   |  92.9100  | [config](https://github.com/open-mmlab/mmclassification/blob/1.x/configs/efficientnet/efficientnet-b4_8xb32_in1k.py) | [model](<>) \|[log](https://download.openmmlab.com/mmclassification/v1/device/npu/efficientnet-b4_8xb32_in1k.log) |
| [\*\*DenseNet121](https://github.com/open-mmlab/mmclassification/blob/1.x/configs/densenet/README.md) |   72.53   |   90.85   | [config](https://github.com/open-mmlab/mmclassification/blob/1.x/configs/densenet/densenet121_4xb256_in1k.py) | [model](<>) \| [log](https://download.openmmlab.com/mmclassification/v1/device/npu/densenet121_4xb256_in1k.log) |

**注意:**

- 如果没有特别标记，NPU 上的结果与使用 FP32 的 GPU 上的结果结果相同。
- (\*) 这些模型的训练结果低于相应模型中自述文件上的结果，主要是因为自述文件上的结果直接是 timm 训练得出的权重，而这边的结果是根据 mmcls 的配置重新训练得到的结果。GPU 上的配置训练结果与 NPU 的结果相同。
- (\*\*)这个模型的精度略低，因为 config 是 4 张卡的配置，我们使用 8 张卡来运行，用户可以调整超参数以获得最佳精度结果。

**以上所有模型权重及训练日志均由华为昇腾团队提供**
