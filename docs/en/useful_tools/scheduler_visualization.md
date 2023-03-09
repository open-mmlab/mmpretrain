# Hyper-parameter Scheduler Visualization

This tool aims to help the user to check the hyper-parameter scheduler of the optimizer(without training), which support the "learning rate" or "momentum"

## Introduce the scheduler visualization tool

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

**Description of all arguments**：

- `config`: The path of a model config file.
- **`-p, --parameter`**: The param to visualize its change curve, choose from "lr" and "momentum". Default to use "lr".
- **`-d, --dataset-size`**: The size of the datasets. If set，`build_dataset` will be skipped and `${DATASET_SIZE}` will be used as the size. Default to use the function `build_dataset`.
- **`-n, --ngpus`**: The number of GPUs used in training, default to be 1.
- **`-s, --save-path`**: The learning rate curve plot save path, default not to save.
- `--title`: Title of figure. If not set, default to be config file name.
- `--style`: Style of plt. If not set, default to be `whitegrid`.
- `--window-size`: The shape of the display window. If not specified, it will be set to `12*7`. If used, it must be in the format `'W*H'`.
- `--cfg-options`: Modifications to the configuration file, refer to [Learn about Configs](../user_guides/config.md).

```{note}
Loading annotations maybe consume much time, you can directly specify the size of the dataset with `-d, dataset-size` to save time.
```

## How to plot the learning rate curve without training

You can use the following command to plot the step learning rate schedule used in the config `configs/resnet/resnet50_b16x8_cifar100.py`:

```bash
python tools/visualization/vis_scheduler.py configs/resnet/resnet50_b16x8_cifar100.py
```

<div align=center><img src="https://user-images.githubusercontent.com/18586273/191006713-023f065d-d366-4165-a52e-36176367506e.png" style=" width: auto; height: 40%; "></div>

When using ImageNet, directly specify the size of ImageNet, as below:

```bash
python tools/visualization/vis_scheduler.py configs/repvgg/repvgg-B3g4_4xb64-autoaug-lbs-mixup-coslr-200e_in1k.py --dataset-size 1281167 --ngpus 4 --save-path ./repvgg-B3g4_4xb64-lr.jpg
```

<div align=center><img src="https://user-images.githubusercontent.com/18586273/191006721-0f680e07-355e-4cd6-889c-86c0cad9acb7.png" style=" width: auto; height: 40%; "></div>
