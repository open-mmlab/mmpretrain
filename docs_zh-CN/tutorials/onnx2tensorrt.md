# Tutorial 6: ONNX to TensorRT (Experimental)

<!-- TOC -->

- [Tutorial 6: ONNX to TensorRT (Experimental)](#tutorial-6-onnx-to-tensorrt-experimental)
  - [How to convert models from ONNX to TensorRT](#how-to-convert-models-from-onnx-to-tensorrt)
    - [Prerequisite](#prerequisite)
    - [Usage](#usage)
  - [List of supported models convertable to TensorRT](#list-of-supported-models-convertable-to-tensorrt)
  - [Reminders](#reminders)
  - [FAQs](#faqs)

<!-- TOC -->

## How to convert models from ONNX to TensorRT

### Prerequisite

1. Please refer to [install.md](https://mmclassification.readthedocs.io/en/latest/install.html#install-mmclassification) for installation of MMClassification from source.
2. Use our tool [pytorch2onnx.md](./pytorch2onnx.md) to convert the model from PyTorch to ONNX.

### Usage

```bash
python tools/deployment/onnx2tensorrt.py \
    ${MODEL} \
    --trt-file ${TRT_FILE} \
    --shape ${IMAGE_SHAPE} \
    --workspace-size {WORKSPACE_SIZE} \
    --show \
    --verify \
```

Description of all arguments:

- `model` : The path of an ONNX model file.
- `--trt-file`: The Path of output TensorRT engine file. If not specified, it will be set to `tmp.trt`.
- `--shape`: The height and width of model input. If not specified, it will be set to `224 224`.
- `--workspace-size` : The required GPU workspace size in GiB to build TensorRT engine. If not specified, it will be set to `1` GiB.
- `--show`: Determines whether to show the outputs of the model. If not specified, it will be set to `False`.
- `--verify`: Determines whether to verify the correctness of models between ONNXRuntime and TensorRT. If not specified, it will be set to `False`.

Example:

```bash
python tools/onnx2tensorrt.py \
    checkpoints/resnet/resnet18_b16x8_cifar10.onnx \
    --trt-file checkpoints/resnet/resnet18_b16x8_cifar10.trt \
    --shape 224 224 \
    --show \
    --verify \
```

## List of supported models convertable to TensorRT

The table below lists the models that are guaranteed to be convertable to TensorRT.

|    Model     |                            Config                            | Status |
| :----------: | :----------------------------------------------------------: | :----: |
| MobileNetV2  |    `configs/mobilenet_v2/mobilenet_v2_b32x8_imagenet.py`     |   Y    |
|    ResNet    |          `configs/resnet/resnet18_b16x8_cifar10.py`          |   Y    |
|   ResNeXt    |     `configs/resnext/resnext50_32x4d_b32x8_imagenet.py`      |   Y    |
| ShuffleNetV1 | `configs/shufflenet_v1/shufflenet_v1_1x_b64x16_linearlr_bn_nowd_imagenet.py` |   Y    |
| ShuffleNetV2 | `configs/shufflenet_v2/shufflenet_v2_1x_b64x16_linearlr_bn_nowd_imagenet.py` |   Y    |

Notes:

- *All models above are tested with Pytorch==1.6.0 and TensorRT-7.2.1.6.Ubuntu-16.04.x86_64-gnu.cuda-10.2.cudnn8.0*

## Reminders

- If you meet any problem with the listed models above, please create an issue and it would be taken care of soon. For models not included in the list, we may not provide much help here due to the limited resources. Please try to dig a little deeper and debug by yourself.

## FAQs

- None
