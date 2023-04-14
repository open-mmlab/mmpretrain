# Hyper-parameter Scheduler Visualization

This tool aims to help the user to check the hyper-parameter scheduler of the optimizer (without training), which support the "learning rate" or "momentum"

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

You can use the following command to plot the step learning rate schedule used in the config `configs/swin_transformer/swin-base_16xb64_in1k.py`:

```bash
python tools/visualization/vis_scheduler.py configs/swin_transformer/swin-base_16xb64_in1k.py --dataset-size 1281167 --ngpus 16
```

<div align=center><img src="https://user-images.githubusercontent.com/26739999/226544329-cf3a3d45-6ab3-48aa-8972-2c2a58c35e62.png" style=" width: auto; height: 40%; "></div>
