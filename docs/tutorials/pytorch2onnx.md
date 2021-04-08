# Tutorial 5: Pytorch to ONNX (Experimental)

<!-- TOC -->

- [Tutorial 5: Pytorch to ONNX (Experimental)](#tutorial-5-pytorch-to-onnx-experimental)
  - [How to convert models from Pytorch to ONNX](#how-to-convert-models-from-pytorch-to-onnx)
    - [Prerequisite](#prerequisite)
    - [Usage](#usage)
  - [List of supported models exportable to ONNX](#list-of-supported-models-exportable-to-onnx)
  - [Reminders](#reminders)
  - [FAQs](#faqs)

<!-- TOC -->

## How to convert models from Pytorch to ONNX

### Prerequisite

1. Please refer to [install](https://mmclassification.readthedocs.io/en/latest/install.html#install-mmclassification) for installation of MMClassification.
2. Install onnx and onnxruntime

  ```shell
  pip install onnx onnxruntime==1.5.1
  ```

### Usage

```bash
python tools/pytorch2onnx.py \
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

Description of all arguments:

- `config` : The path of a model config file.
- `--checkpoint` : The path of a model checkpoint file.
- `--output-file`: The path of output ONNX model. If not specified, it will be set to `tmp.onnx`.
- `--shape`: The height and width of input tensor to the model. If not specified, it will be set to `224 224`.
- `--opset-version` : The opset version of ONNX. If not specified, it will be set to `11`.
- `--dynamic-shape` : Determines whether to export ONNX with dynamic input shape.  If not specified, it will be set to `False`.
- `--show`: Determines whether to print the architecture of the exported model. If not specified, it will be set to `False`.
- `--simplify`: Determines whether to simplify the exported ONNX model. If not specified, it will be set to `False`.
- `--verify`: Determines whether to verify the correctness of an exported model. If not specified, it will be set to `False`.

Example:

```bash
python tools/pytorch2onnx.py \
    configs/resnet/resnet18_b16x8_cifar10.py \
    --checkpoint checkpoints/resnet/resnet18_b16x8_cifar10.pth \
    --output-file checkpoints/resnet/resnet18_b16x8_cifar10.onnx \
    --dynamic-shape \
    --show \
    --simplify \
    --verify \
```

## List of supported models exportable to ONNX

The table below lists the models that are guaranteed to be exportable to ONNX and runnable in ONNX Runtime.

|    Model     |                            Config                            | Batch Inference | Dynamic Shape | Note |
| :----------: | :----------------------------------------------------------: | :-------------: | :-----------: | ---- |
| MobileNetV2  |    `configs/mobilenet_v2/mobilenet_v2_b32x8_imagenet.py`     |        Y        |       Y       |      |
|    ResNet    |          `configs/resnet/resnet18_b16x8_cifar10.py`          |        Y        |       Y       |      |
|   ResNeXt    |     `configs/resnext/resnext50_32x4d_b32x8_imagenet.py`      |        Y        |       Y       |      |
|  SE-ResNet   |       `configs/seresnet/seresnet50_b32x8_imagenet.py`        |        Y        |       Y       |      |
| ShuffleNetV1 | `configs/shufflenet_v1/shufflenet_v1_1x_b64x16_linearlr_bn_nowd_imagenet.py` |        Y        |       Y       |      |
| ShuffleNetV2 | `configs/shufflenet_v2/shufflenet_v2_1x_b64x16_linearlr_bn_nowd_imagenet.py` |        Y        |       Y       |      |

Notes:

- *All models above are tested with Pytorch==1.6.0*

## Reminders

- If you meet any problem with the listed models above, please create an issue and it would be taken care of soon. For models not included in the list, please try to dig a little deeper and debug a little bit more and hopefully solve them by yourself.

## FAQs

- None
