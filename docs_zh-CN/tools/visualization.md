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
python tools/misc/vis_pipeline.py \
    ${CONFIG_FILE} \
    --output-dir ${OUTPUT_DIR} \
    --phase ${DATASET_PHASE} \
    --number ${BUNBER_IMAGES_DISPLAY} \
    --skip-type ${SKIP_TRANSFORM_TYPE} \
    --mode ${DISPLAY_MODE}\
    --show \
    --bgr2rgb \
    --original-display-shape ${RESIZE_ORIGINAL_SIZE}
```

**所有参数的说明**：

- `config` : 模型配置文件的路径。
- `--output-dir`: 保存图片文件夹，如果没有指定，默认为 `''`,表示不保存。
- `--phase`: 可视化数据集的阶段，只能为 `[train, val, test]` 之一，默认为 `train`。
- `--number`: 可视化样本数量。如果没有指定，默认为 `-1`，表示整个数据集。
- `--skip-type`: 预设跳过的数据流水线过程。如果没有指定，默认为 `['ToTensor', 'Normalize', 'ImageToTensor', 'Collect']`。
- `--mode`: 可视化的模式，只能为 `[original, pipeline, concat]` 之一，如果没有指定，默认为 `pipeline`。
- `--show`: 是否将预处理后的图片以弹窗形式展示。如果没有指定，默认为 `False`。
- `--bgr2rgb`: 是否将图片的颜色通道翻转。如果没有指定，默认为 `False`。
- `--original-display-shape`: 原图的放缩大小，如果没有指定，默认为 `''`,表示该表大小。如果需要指定，按照格式 `'W*H'`，例如 `'224*224'`。

**注意**：

1. 如果不指定`--mode`，默认设置为`pipeline`，获取预处理后的图片；如果`--mode`设置为`original`，则获取原始图片； 如果`--mode`设置为`concat`，则获取原始图片和预处理后图片拼接的图片。

2. 如果可视化时原图尺寸太小或太大，可以使用`--original-display-shape`改变形状，记得在值周围加上`'`或`"`。如果可视化时预处理后图像尺寸太小或太大, 可以修改配置文件中的数据流水线以更改形状。

3. 如果图片太小，标签信息将不显示。

**示例**：

1. 可视化 `ImageNet` 训练集的所有经过预处理的图片，并以弹窗形式显示：

`python ./tools/visualizations/vis_pipeline.py ./configs/resnet/resnet50_b32x8_imagenet.py --show`

2. 可视化 `ImageNet` 训练集的10张原始图片与预处理后图片对比图，保存在`test`文件夹下：

`python ./tools/visualizations/vis_pipeline.py configs/swin_transformer/swin_base_224_b16x64_300e_imagenet.py --phase test --output-dir test --mode 'concat' --number 10 --output-dir test`

3. 可视化 `CIFAR100` 验证集中的100张原始图片，将图片调整为`224*224`并显示并保存：

`python ./tools/visualizations/vis_pipeline.py configs/resnet/resnet50_b16x8_cifar100.py --phase val --output-dir val --mode original --number 100 --output-dir val --original-display-shape '224*224' --show`

## 常见问题

- 无
