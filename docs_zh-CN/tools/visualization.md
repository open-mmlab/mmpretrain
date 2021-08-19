# 可视化

<!-- TOC -->

- [可视化](#可视化)
  - [通用数据流水线可视化](#通用数据流水线可视化)
    - [使用方法](#使用方法)
  - [ImageNet 数据流水线可视化](#)
    - [增强功能](#增强功能)
  - [常见问题](#常见问题)

<!-- TOC -->

## 通用数据流水线可视化

### 使用方法

```bash
python tools/misc/vis_pipeline.py \
    ${CONFIG_FILE} \
    --output-dir ${OUTPUT_DIR} \
    --phase ${DATASET_PHASE} \
    --number ${BUNBER_IMAGES_DISPLAY} \
    --skip-type ${SKIP_TRANSFORM_TYPE}
    --show \
    --bgr2rgb \
```

所有参数的说明：

- `config` : 模型配置文件的路径。
- `--output-dir`: 当`--show`为False时，将图片保存的文件夹名。如果没有指定，默认为 `tmp`。
- `--phase`: 可视化数据集的阶段，只能为`['train', 'val', 'test']`之一，默认为 `train`。
- `--number`: 可视化样本数量。如果没有指定，默认为`-1`，表示可视化整个数据集。
- `--skip-type`: 预设跳过的数据流水线过程。如果没有指定，默认为 `['ToTensor', 'Normalize', 'ImageToTensor', 'Collect']`。
- `--show`: 是否将预处理后的图片以弹窗形式展示。如果没有指定，默认为 `False` ,表示保存图片至文件夹。
- `--bgr2rgb`: 是否将图片的颜色通道翻转。如果没有指定，默认为 `False`。

示例：

1. 可视化Cifar100验证集的所有图片，保存在文件夹 `vis_cifar100_val` 下:

`python ./tools/misc/vis_pipeline.py ./configs/resnet/resnet50_b16x8_cifar100.py --phase val --output-dir vis_cifar100_val`

2. 可视化ImageNet训练集的100张图片，并以弹窗形式显示：

`python ./tools/misc/vis_pipeline.py ./configs/resnet/resnet50_b32x8_imagenet.py --number 100 --show`

## ImageNet 数据流水线可视化

在通用数据流水线可视化的基础上增加可视化原始图片、原图与预处理后图片对比图。

### 增强功能

所增加参数的说明：

- `--original`: 是否可视化原始图片。如果没有指定，默认为 `False`。
- `--transform`: 是否预处理后图片。如果没有指定，默认为 `False`。

注意：

当只指定`--original`时，显示原始图片;当只指定`--transform`时，显示预处理后图片；当指定`--original`与`--transform`时，显示原始图片与预处理后图片合并图;`--original` 与 `--transform`至少指定一个，不然会报错；  

示例：
1. 可视化ImageNet训练集的100张原始图片，并以弹窗形式显示：

`python ./tools/misc/vis_imagenet.py ./configs/resnet/resnet50_b32x8_imagenet.py --original --number 100 --show`

2. 可视化ImageNet训练集的10张原始图片与预处理后图片对比图，保存在tmp文件夹下：

`python ./tools/misc/vis_imagenet.py configs/swin_transformer/swin_base_224_b16x64_300e_imagenet.py --original --transform --number 10`

## 常见问题

- 无
