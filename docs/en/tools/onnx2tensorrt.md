# ONNX to TensorRT (Experimental)

<!-- TOC -->

- [ONNX to TensorRT (Experimental)](#onnx-to-tensorrt-experimental)
  - [How to convert models from ONNX to TensorRT](#how-to-convert-models-from-onnx-to-tensorrt)
    - [Prerequisite](#prerequisite)
    - [Usage](#usage)
  - [List of supported models convertible to TensorRT](#list-of-supported-models-convertible-to-tensorrt)
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
    --max-batch-size ${MAX_BATCH_SIZE} \
    --workspace-size ${WORKSPACE_SIZE} \
    --fp16 \
    --show \
    --verify \
```

Description of all arguments:

- `model` : The path of an ONNX model file.
- `--trt-file`: The Path of output TensorRT engine file. If not specified, it will be set to `tmp.trt`.
- `--shape`: The height and width of model input. If not specified, it will be set to `224 224`.
- `--max-batch-size`: The max batch size of TensorRT model, should not be less than 1.
- `--fp16`: Enable fp16 mode.
- `--workspace-size` : The required GPU workspace size in GiB to build TensorRT engine. If not specified, it will be set to `1` GiB.
- `--show`: Determines whether to show the outputs of the model. If not specified, it will be set to `False`.
- `--verify`: Determines whether to verify the correctness of models between ONNXRuntime and TensorRT. If not specified, it will be set to `False`.

Example:

```bash
python tools/deployment/onnx2tensorrt.py \
    checkpoints/resnet/resnet18_b16x8_cifar10.onnx \
    --trt-file checkpoints/resnet/resnet18_b16x8_cifar10.trt \
    --shape 224 224 \
    --show \
    --verify \
```

## List of supported models convertible to TensorRT

The table below lists the models that are guaranteed to be convertible to TensorRT.

|    Model     |                         Config                          | Status |
| :----------: | :-----------------------------------------------------: | :----: |
| MobileNetV2  |    `configs/mobilenet_v2/mobilenet-v2_8xb32_in1k.py`    |   Y    |
|    ResNet    |       `configs/resnet/resnet18_8xb16_cifar10.py`        |   Y    |
|   ResNeXt    |     `configs/resnext/resnext50-32x4d_8xb32_in1k.py`     |   Y    |
| ShuffleNetV1 | `configs/shufflenet_v1/shufflenet-v1-1x_16xb64_in1k.py` |   Y    |
| ShuffleNetV2 | `configs/shufflenet_v2/shufflenet-v2-1x_16xb64_in1k.py` |   Y    |

Notes:

- *All models above are tested with Pytorch==1.6.0 and TensorRT-7.2.1.6.Ubuntu-16.04.x86_64-gnu.cuda-10.2.cudnn8.0*

## Reminders

- If you meet any problem with the listed models above, please create an issue and it would be taken care of soon. For models not included in the list, we may not provide much help here due to the limited resources. Please try to dig a little deeper and debug by yourself.

## FAQs

- None
