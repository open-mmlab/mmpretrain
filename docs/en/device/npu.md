# NPU (HUAWEI Ascend)

## Usage

### General Usage

Please refer to the [building documentation of MMCV](https://mmcv.readthedocs.io/en/latest/get_started/build.html#build-mmcv-full-on-ascend-npu-machine) to install MMCV and [MMEngine](https://mmengine.readthedocs.io/en/latest/get_started/installation.html#build-from-source) on NPU devices.

Here we use 8 NPUs on your computer to train the model with the following command:

```shell
bash ./tools/dist_train.sh configs/resnet/resnet50_8xb32_in1k.py 8
```

Also, you can use only one NPU to train the model with the following command:

```shell
python ./tools/train.py configs/resnet/resnet50_8xb32_in1k.py
```

## Models Results

|                            Model                            | Top-1 (%) | Top-5 (%) |                            Config                            |                            Download                             |
| :---------------------------------------------------------: | :-------: | :-------: | :----------------------------------------------------------: | :-------------------------------------------------------------: |
| [ResNet-50](https://github.com/open-mmlab/mmclassification/blob/1.x/configs/resnet/README.md) |   76.40   |   93.21   | [config](https://github.com/open-mmlab/mmclassification/blob/1.x/configs/resnet/resnet50_8xb32_in1k.py) | [log](https://download.openmmlab.com/mmclassification/v1/device/npu/resnet50_8xb32_in1k.log) |
| [ResNetXt-32x4d-50](https://github.com/open-mmlab/mmclassification/blob/1.x/configs/resnext/README.md) |   77.48   |   93.75   | [config](https://github.com/open-mmlab/mmclassification/blob/1.x/configs/resnext/resnext50-32x4d_8xb32_in1k.py) | [log](https://download.openmmlab.com/mmclassification/v1/device/npu/resnext50-32x4d_8xb32_in1k.log) |
| [HRNet-W18](https://github.com/open-mmlab/mmclassification/blob/master/configs/hrnet/README.md) |   77.06   |   93.57   | [config](https://github.com/open-mmlab/mmclassification/blob/1.x/configs/hrnet/hrnet-w18_4xb32_in1k.py) | [log](https://download.openmmlab.com/mmclassification/v1/device/npu/hrnet-w18_4xb32_in1k.log) |
| [ResNetV1D-152](https://github.com/open-mmlab/mmclassification/blob/1.x/configs/resnet/README.md) |   79.41   |   94.48   | [config](https://github.com/open-mmlab/mmclassification/blob/1.x/configs/resnet/resnetv1d152_8xb32_in1k.py) | [log](https://download.openmmlab.com/mmclassification/v1/device/npu/resnetv1d152_8xb32_in1k.log) |
| [SE-ResNet-50](https://github.com/open-mmlab/mmclassification/blob/1.x/configs/seresnet/README.md) |   77.65   |   93.74   | [config](https://github.com/open-mmlab/mmclassification/blob/1.x/configs/seresnet/seresnet50_8xb32_in1k.py) | [log](https://download.openmmlab.com/mmclassification/v1/device/npu/seresnet50_8xb32_in1k.log) |
| [ShuffleNetV2 1.0x](https://github.com/open-mmlab/mmclassification/blob/1.x/configs/shufflenet_v2/README.md) |   69.52   |   88.79   | [config](https://github.com/open-mmlab/mmclassification/blob/1.x/configs/shufflenet_v2/shufflenet-v2-1x_16xb64_in1k.py) | [log](https://download.openmmlab.com/mmclassification/v1/device/npu/shufflenet-v2-1x_16xb64_in1k.log) |
| [MobileNetV2](https://github.com/open-mmlab/mmclassification/tree/1.x/configs/mobilenet_v2) |   71.74   |   90.28   | [config](https://github.com/open-mmlab/mmclassification/blob/1.x/configs/mobilenet_v2/mobilenet-v2_8xb32_in1k.py) | [log](https://download.openmmlab.com/mmclassification/v1/device/npu/mobilenet-v2_8xb32_in1k.log) |
| [MobileNetV3-Small](https://github.com/open-mmlab/mmclassification/blob/1.x/configs/mobilenet_v3/README.md) |   67.09   |   87.17   | [config](https://github.com/open-mmlab/mmclassification/blob/1.x/configs/mobilenet_v3/mobilenet-v3-small_8xb128_in1k.py) | [log](https://download.openmmlab.com/mmclassification/v1/device/npu/mobilenet-v3-small.log) |
| [\*CSPResNeXt50](https://github.com/open-mmlab/mmclassification/blob/1.x/configs/cspnet/README.md) |   77.25   |   93.46   | [config](https://github.com/open-mmlab/mmclassification/blob/1.x/configs/cspnet/cspresnext50_8xb32_in1k.py) | [log](https://download.openmmlab.com/mmclassification/v1/device/npu/cspresnext50_8xb32_in1k.log) |
| [\*EfficientNet-B4](https://github.com/open-mmlab/mmclassification/blob/1.x/configs/efficientnet/README.md) |   75.73   |   92.91   | [config](https://github.com/open-mmlab/mmclassification/blob/1.x/configs/efficientnet/efficientnet-b4_8xb32_in1k.py) | [log](https://download.openmmlab.com/mmclassification/v1/device/npu/efficientnet-b4_8xb32_in1k.log) |
| [\*\*DenseNet121](https://github.com/open-mmlab/mmclassification/blob/1.x/configs/densenet/README.md) |   72.53   |   90.85   | [config](https://github.com/open-mmlab/mmclassification/blob/1.x/configs/densenet/densenet121_4xb256_in1k.py) | [log](https://download.openmmlab.com/mmclassification/v1/device/npu/densenet121_4xb256_in1k.log) |

**Notes:**

- If not specially marked, the results are almost same between results on the NPU and results on the GPU with FP32.
- (\*) The training results of these models are lower than the results on the readme in the corresponding model, mainly
  because the results on the readme are directly the weight of the timm of the eval, and the results on this side are
  retrained according to the config with mmcls. The results of the config training on the GPU are consistent with the
  results of the NPU.
- (\*\*) The accuracy of this model is slightly lower because config is a 4-card config, we use 8 cards to run, and users
  can adjust hyperparameters to get the best accuracy results.

**All above models are provided by Huawei Ascend group.**
