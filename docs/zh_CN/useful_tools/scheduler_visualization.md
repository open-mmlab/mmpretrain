# 优化器参数策略可视化

该工具旨在帮助用户检查优化器的超参数调度器（无需训练），支持学习率（learning rate）和动量（momentum）。

## 工具简介

```bash
python tools/visualization/vis_scheduler.py \
    ${CONFIG_FILE} \
    [-p, --parameter ${PARAMETER_NAME}] \
    [-d, --dataset-size ${DATASET_SIZE}] \
    [-n, --ngpus ${NUM_GPUs}] \
    [-s, --save-path ${SAVE_PATH}] \
    [--title ${TITLE}] \
    [--style ${STYLE}] \
    [--window-size ${WINDOW_SIZE}] \
    [--cfg-options]
```

**所有参数的说明**：

- `config` : 模型配置文件的路径。
- **`-p, parameter`**: 可视化参数名，只能为 `["lr", "momentum"]` 之一， 默认为 `"lr"`.
- **`-d, --dataset-size`**: 数据集的大小。如果指定，`build_dataset` 将被跳过并使用这个大小作为数据集大小，默认使用 `build_dataset` 所得数据集的大小。
- **`-n, --ngpus`**: 使用 GPU 的数量, 默认为1。
- **`-s, --save-path`**: 保存的可视化图片的路径，默认不保存。
- `--title`: 可视化图片的标题，默认为配置文件名。
- `--style`: 可视化图片的风格，默认为 `whitegrid`。
- `--window-size`: 可视化窗口大小，如果没有指定，默认为 `12*7`。如果需要指定，按照格式 `'W*H'`。
- `--cfg-options`: 对配置文件的修改，参考[学习配置文件](../user_guides/config.md)。

```{note}
部分数据集在解析标注阶段比较耗时，可直接将 `-d, dataset-size` 指定数据集的大小，以节约时间。
```

## 如何在开始训练前可视化学习率曲线

你可以使用如下命令来绘制配置文件 `configs/resnet/resnet50_b16x8_cifar100.py` 将会使用的变化率曲线：

```bash
python tools/visualization/vis_scheduler.py configs/resnet/resnet50_b16x8_cifar100.py
```

<div align=center><img src="https://user-images.githubusercontent.com/18586273/191006713-023f065d-d366-4165-a52e-36176367506e.png" style=" width: auto; height: 40%; "></div>

当数据集为 ImageNet 时，通过直接指定数据集大小来节约时间，并保存图片：

```bash
python tools/visualization/vis_scheduler.py configs/repvgg/repvgg-B3g4_4xb64-autoaug-lbs-mixup-coslr-200e_in1k.py --dataset-size 1281167 --ngpus 4 --save-path ./repvgg-B3g4_4xb64-lr.jpg
```

<div align=center><img src="https://user-images.githubusercontent.com/18586273/191006721-0f680e07-355e-4cd6-889c-86c0cad9acb7.png" style=" width: auto; height: 40%; "></div>
