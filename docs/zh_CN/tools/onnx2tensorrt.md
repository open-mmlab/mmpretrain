# ONNX 转 TensorRT（试验性的）

<!-- TOC -->

- [如何将模型从 ONNX 转换到 TensorRT](#如何将模型从-onnx-转换到-tensorrt)
  - [准备工作](#准备工作)
  - [使用方法](#使用方法)
- [支持转换至 TensorRT 的模型列表](#支持转换至-tensorrt-的模型列表)
- [提示](#提示)
- [常见问题](#常见问题)

<!-- TOC -->

## 如何将模型从 ONNX 转换到 TensorRT

### 准备工作

1. 请参照 [安装指南](https://mmclassification.readthedocs.io/zh_CN/latest/install.html#mmclassification) 从源码安装 MMClassification。
2. 使用我们的工具 [pytorch2onnx.md](./pytorch2onnx.md) 将 PyTorch 模型转换至 ONNX。

### 使用方法

```bash
python tools/deployment/onnx2tensorrt.py \
    ${MODEL} \
    --trt-file ${TRT_FILE} \
    --shape ${IMAGE_SHAPE} \
    --workspace-size {WORKSPACE_SIZE} \
    --show \
    --verify \
```

所有参数的说明：

- `model` : ONNX 模型的路径。
- `--trt-file`: TensorRT 引擎文件的输出路径。如果没有指定，默认为当前脚本执行路径下的 `tmp.trt`。
- `--shape`: 模型输入的高度和宽度。如果没有指定，默认为 `224 224`。
- `--workspace-size` : 构建 TensorRT 引擎所需要的 GPU 空间大小，单位为 GiB。如果没有指定，默认为 `1` GiB。
- `--show`: 是否展示模型的输出。如果没有指定，默认为 `False`。
- `--verify`: 是否使用 ONNXRuntime 和 TensorRT 验证模型转换的正确性。如果没有指定，默认为`False`。

示例：

```bash
python tools/deployment/onnx2tensorrt.py \
    checkpoints/resnet/resnet18_b16x8_cifar10.onnx \
    --trt-file checkpoints/resnet/resnet18_b16x8_cifar10.trt \
    --shape 224 224 \
    --show \
    --verify \
```

## 支持转换至 TensorRT 的模型列表

下表列出了保证可转换为 TensorRT 的模型。

|     模型     |                        配置文件                         | 状态 |
| :----------: | :-----------------------------------------------------: | :--: |
| MobileNetV2  |    `configs/mobilenet_v2/mobilenet-v2_8xb32_in1k.py`    |  Y   |
|    ResNet    |       `configs/resnet/resnet18_8xb16_cifar10.py`        |  Y   |
|   ResNeXt    |     `configs/resnext/resnext50-32x4d_8xb32_in1k.py`     |  Y   |
| ShuffleNetV1 | `configs/shufflenet_v1/shufflenet-v1-1x_16xb64_in1k.py` |  Y   |
| ShuffleNetV2 | `configs/shufflenet_v2/shufflenet-v2-1x_16xb64_in1k.py` |  Y   |

注：

- *以上所有模型转换测试基于 Pytorch==1.6.0 和 TensorRT-7.2.1.6.Ubuntu-16.04.x86_64-gnu.cuda-10.2.cudnn8.0 进行*

## 提示

- 如果你在上述模型的转换中遇到问题，请在 GitHub 中创建一个 issue，我们会尽快处理。未在上表中列出的模型，由于资源限制，我们可能无法提供很多帮助，如果遇到问题，请尝试自行解决。

## 常见问题

- 无
