# 可视化

<!-- TOC -->

- [可视化](#可视化)
  - [数据流水线可视化](#数据流水线可视化)
    - [使用方法](#使用方法)
  - [常见问题](#常见问题)

<!-- TOC -->

## 数据流水线可视化

### 使用方法

```bash
python tools/visualizations/vis_pipeline.py \
    ${CONFIG_FILE} \
    --output-dir ${OUTPUT_DIR} \
    --phase ${DATASET_PHASE} \
    --number ${BUNBER_IMAGES_DISPLAY} \
    --skip-type ${SKIP_TRANSFORM_TYPE} \
    --mode ${DISPLAY_MODE} \
    --show \
    --adaptive \
    --min-edge-length ${MIN_EDGE_LENGTH} \
    --max-edge-length ${MAX_EDGE_LENGTH} \
    --bgr2rgb \
    --window-size ${WINDOW_SIZE}
```

**所有参数的说明**：

- `config` : 模型配置文件的路径。
- `--output-dir`: 保存图片文件夹，如果没有指定，默认为 `''`,表示不保存。
- `--phase`: 可视化数据集的阶段，只能为 `[train, val, test]` 之一，默认为 `train`。
- `--number`: 可视化样本数量。如果没有指定，默认展示数据集的所有图片。
- `--skip-type`: 预设跳过的数据流水线过程。如果没有指定，默认为 `['ToTensor', 'Normalize', 'ImageToTensor', 'Collect']`。
- `--mode`: 可视化的模式，只能为 `[original, pipeline, concat]` 之一，如果没有指定，默认为 `concat`。
- `--show`: 将可视化图片以弹窗形式展示。
- `--adaptive`: 自动调节可视化图片的大小。
- `--min-edge-length`: 最短边长度，当使用了 `--adaptive` 时有效。 当图片任意边小于 `${MIN_EDGE_LENGTH}` 时，会保持长宽比不变放大图片，短边对齐至 `${MIN_EDGE_LENGTH}`，默认为200。
- `--max-edge-length`: 最长边长度，当使用了 `--adaptive` 时有效。 当图片任意边大于 `${MAX_EDGE_LENGTH}` 时，会保持长宽比不变缩小图片，短边对齐至 `${MAX_EDGE_LENGTH}`，默认为1000。
- `--bgr2rgb`: 将图片的颜色通道翻转。
- `--window-size`: 可视化窗口大小，如果没有指定，默认为 `12*7`。如果需要指定，按照格式 `'W*H'`。

```{note}

1. 如果不指定 `--mode`，默认设置为 `concat`，获取原始图片和预处理后图片拼接的图片；如果 `--mode` 设置为 `original`，则获取原始图片； 如果  `--mode` 设置为 `pipeline`，则获取预处理后的图片。

2. 当指定了 `--adaptive` 选项时，会自动的调整尺寸过大和过小的图片，你可以通过设定 `--min-edge-length` 与 `--max-edge-length` 来指定自动调整的图片尺寸。
```

**示例**：

1. 可视化 `ImageNet` 训练集的所有经过预处理的图片，并以弹窗形式显示：

```shell
python ./tools/visualizations/vis_pipeline.py ./configs/resnet/resnet50_b32x8_imagenet.py --show --mode pipeline
```

<div align=center><img src="../_static/image/pipeline.JPEG" style=" width: auto; height: 40%; "></div>

2. 可视化 `ImageNet` 训练集的10张原始图片与预处理后图片对比图，保存在 `./tmp` 文件夹下：

```shell
python ./tools/visualizations/vis_pipeline.py configs/swin_transformer/swin_base_224_b16x64_300e_imagenet.py --phase train --output-dir tmp --number 10 --adaptive
```

<div align=center><img src="../_static/image/concat.JPEG" style=" width: auto; height: 40%; "></div>

3. 可视化 `CIFAR100` 验证集中的100张原始图片，显示并保存在 `./tmp` 文件夹下：

```shell
python ./tools/visualizations/vis_pipeline.py configs/resnet/resnet50_b16x8_cifar100.py --phase val --output-dir tmp --mode original --number 100 --show --adaptive --bgr2rgb
```

<div align=center><img src="../_static/image/original.JPEG" style=" width: auto; height: 40%; "></div>

## 常见问题

- 无
