# NPU (HUAWEI Ascend)

## Usage

### General Usage

Please install MMCV with NPU device support according to {external+mmcv:doc}`the tutorial <get_started/build>`.

Here we use 8 NPUs on your computer to train the model with the following command:

```shell
bash ./tools/dist_train.sh configs/resnet/resnet50_8xb32_in1k.py 8 --device npu
```

Also, you can use only one NPU to train the model with the following command:

```shell
python ./tools/train.py configs/resnet/resnet50_8xb32_in1k.py --device npu
```

### High-performance Usage on ARM server

Since the scheduling ability of ARM CPUs when processing resource preemption is not as good as that of X86 CPUs during multi-card training, we provide a high-performance startup script to accelerate training with the following command:

```shell
# The script under the 8 cards of a single machine is shown here
bash tools/dist_train_arm.sh configs/resnet/resnet50_8xb32_in1k.py 8 --device npu --cfg-options data.workers_per_gpu=$(($(nproc)/8))
```

For resnet50 8 NPUs training with batch_size(data.samples_per_gpu)=512, the performance data is shown below:

| CPU                 | Start Script              |   IterTime(s)    |
| :------------------ | :------------------------ | :--------------: |
| ARM(Kunpeng920 \*4) | ./tools/dist_train.sh     |  ~0.9(0.85-1.0)  |
| ARM(Kunpeng920 \*4) | ./tools/dist_train_arm.sh | ~0.8(0.78s-0.85) |

## Models Results

|                            Model                            | Top-1 (%) | Top-5 (%) |                            Config                            |                            Download                             |
| :---------------------------------------------------------: | :-------: | :-------: | :----------------------------------------------------------: | :-------------------------------------------------------------: |
|              [ResNet-50](../papers/resnet.md)               |   76.38   |   93.22   | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/resnet/resnet50_8xb32_in1k.py) | [model](<>) \| [log](https://download.openmmlab.com/mmclassification/v0/device/npu/resnet50_8xb32_in1k.log) |
|          [ResNetXt-32x4d-50](../papers/resnext.md)          |   77.55   |   93.75   | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/resnext/resnext50-32x4d_8xb32_in1k.py) | [model](<>) \| [log](https://download.openmmlab.com/mmclassification/v0/device/npu/resnext50-32x4d_8xb32_in1k.log.json) |
|               [HRNet-W18](../papers/hrnet.md)               |   77.01   |   93.46   | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/hrnet/hrnet-w18_4xb32_in1k.py) | [model](<>) \| [log](https://download.openmmlab.com/mmclassification/v0/device/npu/hrnet-w18_4xb32_in1k.log.json) |
|            [ResNetV1D-152](../papers/resnet.md)             |   79.11   |   94.54   | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/resnet/resnetv1d152_8xb32_in1k.py) | [model](<>) \| [log](https://download.openmmlab.com/mmclassification/v0/device/npu/resnetv1d152_8xb32_in1k.log.json) |
|            [SE-ResNet-50](../papers/seresnet.md)            |   77.64   |   93.76   | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/seresnet/seresnet50_8xb32_in1k.py) | [model](<>) \| [log](https://download.openmmlab.com/mmclassification/v0/device/npu/seresnet50_8xb32_in1k.log.json) |
|                 [VGG-11](../papers/vgg.md)                  |   68.92   |   88.83   | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/vgg/vgg11_8xb32_in1k.py) | [model](<>) \| [log](https://download.openmmlab.com/mmclassification/v0/device/npu/vgg11_8xb32_in1k.log.json) |
|       [ShuffleNetV2 1.0x](../papers/shufflenet_v2.md)       |   69.53   |   88.82   | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/shufflenet_v2/shufflenet-v2-1x_16xb64_in1k.py) | [model](<>) \| [log](https://download.openmmlab.com/mmclassification/v0/device/npu/shufflenet-v2-1x_16xb64_in1k.json) |
|          [MobileNetV2](../papers/mobilenet_v2.md)           |  71.758   |  90.394   | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/mobilenet_v2/mobilenet-v2_8xb32_in1k.py) | [model](<>) \| [log](https://download.openmmlab.com/mmclassification/v0/device/npu/mobilenet-v2_8xb32_in1k.json) |
|       [MobileNetV3-Small](../papers/mobilenet_v3.md)        |  67.522   |  87.316   | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/mobilenet_v3/mobilenet-v3-small_8xb32_in1k.py) | [model](<>) \| [log](https://download.openmmlab.com/mmclassification/v0/device/npu/mobilenet-v3-small_8xb32_in1k.json) |
|            [\*CSPResNeXt50](../papers/cspnet.md)            |   77.10   |   93.55   | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/cspnet/cspresnext50_8xb32_in1k.py) | [model](<>) \| [log](https://download.openmmlab.com/mmclassification/v0/device/npu/cspresnext50_8xb32_in1k.log.json) |
| [\*EfficientNet-B4(AA + AdvProp)](../papers/efficientnet.md) |   75.55   |   92.86   | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/efficientnet/efficientnet-b4_8xb32-01norm_in1k.py) | [model](<>) \| [log](https://download.openmmlab.com/mmclassification/v0/device/npu/efficientnet-b4_8xb32-01norm_in1k.log.json) |
|          [\*\*DenseNet121](../papers/densenet.md)           |   72.62   |   91.04   | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/densenet/densenet121_4xb256_in1k.py) | [model](<>) \| [log](https://download.openmmlab.com/mmclassification/v0/device/npu/densenet121_4xb256_in1k.log.json) |

**Notes:**

- If not specially marked, the results are almost same between results on the NPU and results on the GPU with FP32.
- (\*) The training results of these models are lower than the results on the readme in the corresponding model, mainly
  because the results on the readme are directly the weight of the timm of the eval, and the results on this side are
  retrained according to the config with mmcls. The results of the config training on the GPU are consistent with the
  results of the NPU.
- (\*\*) The accuracy of this model is slightly lower because config is a 4-card config, we use 8 cards to run, and users
  can adjust hyperparameters to get the best accuracy results.

**All above models are provided by Huawei Ascend group.**
