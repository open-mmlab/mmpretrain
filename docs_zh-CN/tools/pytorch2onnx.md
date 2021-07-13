# Pytorch 转 ONNX （试验性的）

<!-- TOC -->

- [Pytorch 转 ONNX （试验性的）](#pytorch-onnx)
  - [如何将模型从 PyTorch 转换到 ONNX](#id1)
    - [准备工作](#id2)
    - [使用方法](#id3)
  - [支持导出至 ONNX 的模型列表](#onnx)
  - [提示](#id4)
  - [常见问题](#id5)

<!-- TOC -->

## 如何将模型从 PyTorch 转换到 ONNX

### 准备工作

1. 请参照 [安装指南](https://mmclassification.readthedocs.io/zh_CN/latest/install.html#mmclassification) 从源码安装 MMClassification。
2. 安装 onnx 和 onnxruntime。

  ```shell
  pip install onnx onnxruntime==1.5.1
  ```

### 使用方法

```bash
python tools/deployment/pytorch2onnx.py \
    ${CONFIG_FILE} \
    --checkpoint ${CHECKPOINT_FILE} \
    --output-file ${OUTPUT_FILE} \
    --shape ${IMAGE_SHAPE} \
    --opset-version ${OPSET_VERSION} \
    --dynamic-shape \
    --show \
    --simplify \
    --verify \
```

所有参数的说明：

- `config` : 模型配置文件的路径。
- `--checkpoint` : 模型权重文件的路径。
- `--output-file`: ONNX 模型的输出路径。如果没有指定，默认为当前脚本执行路径下的 `tmp.onnx`。
- `--shape`: 模型输入的高度和宽度。如果没有指定，默认为 `224 224`。
- `--opset-version` : ONNX 的 opset 版本。如果没有指定，默认为 `11`。
- `--dynamic-shape` : 是否以动态输入尺寸导出 ONNX。 如果没有指定，默认为 `False`。
- `--show`: 是否打印导出模型的架构。如果没有指定，默认为 `False`。
- `--simplify`: 是否精简导出的 ONNX 模型。如果没有指定，默认为 `False`。
- `--verify`: 是否验证导出模型的正确性。如果没有指定，默认为`False`。

示例：

```bash
python tools/deployment/pytorch2onnx.py \
    configs/resnet/resnet18_b16x8_cifar10.py \
    --checkpoint checkpoints/resnet/resnet18_b16x8_cifar10.pth \
    --output-file checkpoints/resnet/resnet18_b16x8_cifar10.onnx \
    --dynamic-shape \
    --show \
    --simplify \
    --verify \
```

## 支持导出至 ONNX 的模型列表

下表列出了保证可导出至 ONNX，并在 ONNX Runtime 中运行的模型。

|     模型     |                               配置文件                                       |     批推理      |  动态输入尺寸 | 备注 |
| :----------: | :--------------------------------------------------------------------------: | :-------------: | :-----------: | ---- |
| MobileNetV2  |    `configs/mobilenet_v2/mobilenet_v2_b32x8_imagenet.py`                     |        Y        |       Y       |      |
|    ResNet    |          `configs/resnet/resnet18_b16x8_cifar10.py`                          |        Y        |       Y       |      |
|   ResNeXt    |     `configs/resnext/resnext50_32x4d_b32x8_imagenet.py`                      |        Y        |       Y       |      |
|  SE-ResNet   |       `configs/seresnet/seresnet50_b32x8_imagenet.py`                        |        Y        |       Y       |      |
| ShuffleNetV1 | `configs/shufflenet_v1/shufflenet_v1_1x_b64x16_linearlr_bn_nowd_imagenet.py` |        Y        |       Y       |      |
| ShuffleNetV2 | `configs/shufflenet_v2/shufflenet_v2_1x_b64x16_linearlr_bn_nowd_imagenet.py` |        Y        |       Y       |      |

注：

- *以上所有模型转换测试基于 Pytorch==1.6.0 进行*

## 提示

- 如果你在上述模型的转换中遇到问题，请在 GitHub 中创建一个 issue，我们会尽快处理。未在上表中列出的模型，由于资源限制，我们可能无法提供很多帮助，如果遇到问题，请尝试自行解决。

## 常见问题

- 无
