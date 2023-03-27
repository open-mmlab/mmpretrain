# 数据集可视化

## 数据集可视化工具简介

```bash
python tools/visualization/browse_dataset.py \
    ${CONFIG_FILE} \
    [-o, --output-dir ${OUTPUT_DIR}] \
    [-p, --phase ${DATASET_PHASE}] \
    [-n, --show-number ${NUMBER_IMAGES_DISPLAY}] \
    [-i, --show-interval ${SHOW_INTERRVAL}] \
    [-m, --mode ${DISPLAY_MODE}] \
    [-r, --rescale-factor ${RESCALE_FACTOR}] \
    [-c, --channel-order ${CHANNEL_ORDER}] \
    [--cfg-options ${CFG_OPTIONS}]
```

**所有参数的说明**：

- `config` : 模型配置文件的路径。
- `-o, --output-dir`: 保存图片文件夹，如果没有指定，默认为 `''`,表示不保存。
- **`-p, --phase`**: 可视化数据集的阶段，只能为 `['train', 'val', 'test']` 之一，默认为 `'train'`。
- **`-n, --show-number`**: 可视化样本数量。如果没有指定，默认展示数据集的所有图片。
- `-i, --show-interval`: 浏览时，每张图片的停留间隔，单位为秒。
- **`-m, --mode`**: 可视化的模式，只能为 `['original', 'transformed', 'concat', 'pipeline']` 之一。 默认为`'transformed'`.
- `-r, --rescale-factor`: 在 `mode='original'` 下，可视化图片的放缩倍数，在图片过大或过小时设置。
- `-c, --channel-order`: 图片的通道顺序，为  `['BGR', 'RGB']` 之一，默认为 `'BGR'`。
- `--cfg-options` : 对配置文件的修改，参考[学习配置文件](../user_guides/config.md)。

```{note}

1. `-m, --mode` 用于设置可视化的模式，默认设置为 'transformed'。
- 如果 `--mode` 设置为 'original'，则获取原始图片；
- 如果 `--mode` 设置为 'transformed'，则获取预处理后的图片；
- 如果 `--mode` 设置为 'concat'，获取原始图片和预处理后图片拼接的图片；
- 如果 `--mode` 设置为 'pipeline'，则获得数据流水线所有中间过程图片。

2. `-r, --rescale-factor` 在数据集中图片的分辨率过大或者过小时设置。比如在可视化 CIFAR 数据集时，由于图片的分辨率非常小，可将 `-r, --rescale-factor` 设置为 10。
```

## 如何可视化原始图像

使用 **'original'** 模式 ：

```shell
python ./tools/visualization/browse_dataset.py ./configs/resnet/resnet101_8xb16_cifar10.py --phase val --output-dir tmp --mode original --show-number 100 --rescale-factor 10 --channel-order RGB
```

- `--phase val`: 可视化验证集，可简化为 `-p val`;
- `--output-dir tmp`: 可视化结果保存在 "tmp" 文件夹，可简化为 `-o tmp`;
- `--mode original`: 可视化原图，可简化为 `-m original`;
- `--show-number 100`: 可视化 100 张图，可简化为 `-n 100`;
- `--rescale-factor`: 图像放大 10 倍，可简化为 `-r 10`;
- `--channel-order RGB`: 可视化图像的通道顺序为 "RGB", 可简化为 `-c RGB`。

<div align=center><img src="https://user-images.githubusercontent.com/18586273/190993839-216a7a1e-590e-47b9-92ae-08f87a7d58df.jpg" style=" width: auto; height: 40%; "></div>

## 如何可视化处理后图像

使用 **'transformed'** 模式：

```shell
python ./tools/visualization/browse_dataset.py ./configs/resnet/resnet50_8xb32_in1k.py -n 100
```

<div align=center><img src="https://user-images.githubusercontent.com/18586273/190994696-737b09d9-d0fb-4593-94a2-4487121e0286.JPEG" style=" width: auto; height: 40%; "></div>

## 如何同时可视化原始图像与处理后图像

使用 **'concat'** 模式：

```shell
python ./tools/visualization/browse_dataset.py configs/swin_transformer/swin-small_16xb64_in1k.py -n 10 -m concat
```

<div align=center><img src="https://user-images.githubusercontent.com/18586273/190995078-3872feb2-d4e2-4727-a21b-7062d52f7d3e.JPEG" style=" width: auto; height: 40%; "></div>

使用 **'pipeline'** 模式：

```shell
python ./tools/visualization/browse_dataset.py configs/swin_transformer/swin-small_16xb64_in1k.py -m pipeline
```

<div align=center><img src="https://user-images.githubusercontent.com/18586273/190995525-fac0220f-6630-4013-b94a-bc6de4fdff7a.JPEG" style=" width: auto; height: 40%; "></div>

```shell
python ./tools/visualization/browse_dataset.py configs/beit/beit_beit-base-p16_8xb256-amp-coslr-300e_in1k.py -m pipeline
```

<div align=center><img src="https://user-images.githubusercontent.com/26739999/226542300-74216187-e3d0-4a6e-8731-342abe719721.png" style=" width: auto; height: 40%; "></div>
