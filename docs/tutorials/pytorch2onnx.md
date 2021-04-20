# Tutorial 5: Pytorch to ONNX (Experimental)

<!-- TOC -->

- [Tutorial 5: Pytorch to ONNX (Experimental)](#tutorial-5-pytorch-to-onnx-experimental)
  - [How to convert models from Pytorch to ONNX](#how-to-convert-models-from-pytorch-to-onnx)
    - [Prerequisite](#prerequisite)
    - [Usage](#usage)
    - [Description of all arguments](#description-of-all-arguments)
  - [How to evaluate ONNX models with ONNX Runtime](#how-to-evaluate-onnx-models-with-onnx-runtime)
    - [Prerequisite](#prerequisite-1)
    - [Usage](#usage-1)
    - [Description of all arguments](#description-of-all-arguments-1)
    - [Results and Models](#results-and-models)
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
    --dynamic-export \
    --show \
    --simplify \
    --verify \
```

### Description of all arguments:

- `config` : The path of a model config file.
- `--checkpoint` : The path of a model checkpoint file.
- `--output-file`: The path of output ONNX model. If not specified, it will be set to `tmp.onnx`.
- `--shape`: The height and width of input tensor to the model. If not specified, it will be set to `224 224`.
- `--opset-version` : The opset version of ONNX. If not specified, it will be set to `11`.
- `--dynamic-export` : Determines whether to export ONNX with dynamic input shape and output shapes. If not specified, it will be set to `False`.
- `--show`: Determines whether to print the architecture of the exported model. If not specified, it will be set to `False`.
- `--simplify`: Determines whether to simplify the exported ONNX model. If not specified, it will be set to `False`.
- `--verify`: Determines whether to verify the correctness of an exported model. If not specified, it will be set to `False`.

Example:

```bash
python tools/pytorch2onnx.py \
    configs/resnet/resnet18_b16x8_cifar10.py \
    --checkpoint checkpoints/resnet/resnet18_b16x8_cifar10.pth \
    --output-file checkpoints/resnet/resnet18_b16x8_cifar10.onnx \
    --dynamic-export \
    --show \
    --simplify \
    --verify \
```

## How to evaluate ONNX models with ONNX Runtime

We prepare a tool `tools/ort_test.py` to evaluate ONNX models with ONNX Runtime backend.

### Prerequisite

- Install onnx and onnxruntime-gpu

  ```shell
  pip install onnx onnxruntime-gpu
  ```

### Usage

```bash
python tools/ort_test.py \
    ${CONFIG_FILE} \
    ${ONNX_FILE} \
    --out ${OUTPUT_FILE} \
    --metrics ${EVALUATION_METRICS} \
    --metric-options ${EVALUATION_OPTIONS} \
    --show
    --show-dir ${SHOW_DIRECTORY} \
    --cfg-options ${CFG_OPTIONS} \
```

### Description of all arguments

- `config`: The path of a model config file.
- `model`: The path of a ONNX model file.
- `--out`: The path of output result file in pickle format.
- `--metrics`: Evaluation metrics, which depends on the dataset, e.g., "accuracy", "precision", "recall", "f1_score", "support" for single label dataset, and "mAP", "CP", "CR", "CF1", "OP", "OR", "OF1" for multi-label dataset.
- `--show`: Determines whether to show classifier outputs. If not specified, it will be set to `False`.
- `--show-dir`: Directory where painted images will be saved
- `--metrics-options`: Custom options for evaluation, the key-value pair in `xxx=yyy` format will be kwargs for `dataset.evaluate()` function
- `--cfg-options`: Override some settings in the used config file, the key-value pair in `xxx=yyy` format will be merged into config file.

### Results and Models

### ImageNet

ImageNet has multiple versions, but the most commonly used one is [ILSVRC 2012](http://www.image-net.org/challenges/LSVRC/2012/).

|     Model     |                            config                            | PyTorch Top1 / Top5 | ONNXRuntime Top-1 / Top-5 |
| :-----------: | :----------------------------------------------------------: | :-----------------: | :-----------------------: |
|   Resnet-50   |         `configs/resnet/resnet50_b32x8_imagenet.py`          |    76.55 / 93.15    |       76.49 / 93.22       |
|  Resnext-50   |     `configs/resnext/resnext50_32x4d_b32x8_imagenet.py`      |    77.92 / 93.74    |       77.15 / 93.36       |
| SE-Resnet-50  |       `configs/seresnet/seresnet50_b32x8_imagenet.py`        |    77.74 / 93.84    |       77.74 / 93.84       |
| Shufflenet_v1 | `configs/shufflenet_v1/shufflenet_v1_1x_b64x16_linearlr_bn_nowd_imagenet.py` |    68.13 / 87.81    |       68.13 / 87.81       |
| Shufflenet_v2 | `configs/shufflenet_v2/shufflenet_v2_1x_b64x16_linearlr_bn_nowd_imagenet.py` |    69.55 / 88.92    |       69.55 / 88.92       |
| MobileNet_v2  |    `configs/mobilenet_v2/mobilenet_v2_b32x8_imagenet.py`     |    71.86 / 90.42    |       71.86 / 90.42       |


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
