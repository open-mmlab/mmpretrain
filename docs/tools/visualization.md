# Visualization

<!-- TOC -->

- [Visualization](#visualization)
  - [Pipeline Visualization](#pipeline-visualization)
  - [Learning Rate Schedule Visualization](#learning-rate-schedule-visualization)
  - [FAQs](#faqs)

<!-- TOC -->
## Pipeline Visualization

```bash
python tools/visualizations/vis_pipeline.py \
    ${CONFIG_FILE} \
    [--output-dir ${OUTPUT_DIR}] \
    [--phase ${DATASET_PHASE}] \
    [--number ${BUNBER_IMAGES_DISPLAY}] \
    [--skip-type ${SKIP_TRANSFORM_TYPE}] \
    [--mode ${DISPLAY_MODE}] \
    [--show] \
    [--adaptive] \
    [--min-edge-length ${MIN_EDGE_LENGTH}] \
    [--max-edge-length ${MAX_EDGE_LENGTH}] \
    [--bgr2rgb] \
    [--window-size ${WINDOW_SIZE}] \
    [--cfg-options ${CFG_OPTIONS}]
```

**Description of all arguments**：

- `config` : The path of a model config file.
- `--output-dir`: The output path for visualized images. If not specified, it will be set to `''`, which means not to save.
- `--phase`: Phase of visualizing dataset，must be one of `[train, val, test]`. If not specified, it will be set to `train`.
- `--number`: The number of samples to visualize. If not specified, display all images in the dataset.
- `--skip-type`: The pipelines to be skipped. If not specified, it will be set to `['ToTensor', 'Normalize', 'ImageToTensor', 'Collect']`.
- `--mode`: The display mode, can be one of `[original, pipeline, concat]`. If not specified, it will be set to `concat`.
- `--show`: If set, display pictures in pop-up windows.
- `--adaptive`: If set, automatically adjust the size of the visualization images.
- `--min-edge-length`: The minimum edge length, used when `--adaptive` is set. When any side of the picture is smaller than `${MIN_EDGE_LENGTH}`, the picture will be enlarged while keeping the aspect ratio unchanged, and the short side will be aligned to `${MIN_EDGE_LENGTH}`. If not specified, it will be set to 200.
- `--max-edge-length`: The maximum edge length, used when `--adaptive` is set. When any side of the picture is larger than `${MAX_EDGE_LENGTH}`, the picture will be reduced while keeping the aspect ratio unchanged, and the long side will be aligned to `${MAX_EDGE_LENGTH}`. If not specified, it will be set to 1000.
- `--bgr2rgb`: If set, flip the color channel order of images.
- `--window-size`: The shape of the display window. If not specified, it will be set to `12*7`. If used, it must be in the format `'W*H'`.
- `cfg-options` : Modifications to the configuration file, refer to [Tutorial 1: Learn about Configs](https://mmclassification.readthedocs.io/en/latest/tutorials/config.html).

```{note}

1. If the `--mode` is not specified, it will be set to `concat` as default, get the pictures stitched together by original pictures and transformed pictures; if the `--mode` is set to `original`, get the original pictures; if the `--mode` is set to `transformed`, get the transformed pictures; if the `--mode` is set to `pipeline`, get all the intermediate images through the pipeline.

2. When `--adaptive` option is set, images that are too large or too small will be automatically adjusted, you can use `--min-edge-length` and `--max-edge-length` to set the adjust size.
```

**Examples**：

- 1) In **'original'** mode, visualize 100 original pictures in the `CIFAR100` validation set, then display and save them in the `./tmp` folder：

  ```shell
  python ./tools/visualizations/vis_pipeline.py configs/resnet/resnet50_8xb16_cifar100.py --phase val --output-dir tmp --mode original --number 100  --show --adaptive --bgr2rgb
  ```

  <div align=center><img src="https://user-images.githubusercontent.com/18586273/146117528-1ec2d918-57f8-4ae4-8ca3-a8d31b602f64.jpg" style=" width: auto; height: 40%; "></div>

- 2) In **'transformed'** mode, visualize all the transformed pictures of the `ImageNet` training set and display them in pop-up windows：

  ```shell
  python ./tools/visualizations/vis_pipeline.py ./configs/resnet/resnet50_8xb32_in1k.py --show --mode transformed
  ```

  <div align=center><img src="https://user-images.githubusercontent.com/18586273/146117553-8006a4ba-e2fa-4f53-99bc-42a4b06e413f.jpg" style=" width: auto; height: 40%; "></div>

- 3) In **'concat'** mode, visualize 10 comparison pictures in the `ImageNet` train set and save them in the `./tmp` folder：

  ```shell
  python ./tools/visualizations/vis_pipeline.py configs/swin_transformer/swin_base_224_b16x64_300e_imagenet.py --phase train --output-dir tmp --number 10 --adaptive
  ```

  <div align=center><img src="https://user-images.githubusercontent.com/18586273/146128259-0a369991-7716-411d-8c27-c6863e6d76ea.JPEG" style=" width: auto; height: 40%; "></div>

- 4) In **'pipeline'** mode, visualize all the intermediate pictures in the `ImageNet` train set through the pipeline：

  ```shell
  python ./tools/visualizations/vis_pipeline.py configs/swin_transformer/swin_base_224_b16x64_300e_imagenet.py --phase train --adaptive --pipeline
  ```

  <div align=center><img src="https://user-images.githubusercontent.com/18586273/146128201-eb97c2aa-a615-4a81-a649-38db1c315d0e.JPEG" style=" width: auto; height: 40%; "></div>

## Learning Rate Schedule Visualization

```bash
python tools/visualizations/vis_lr.py \
    ${CONFIG_FILE} \
    --dataset-size ${DATASET_SIZE} \
    --ngpus ${NUM_GPUs}
    --save-path ${SAVE_PATH} \
    --title ${TITLE} \
    --style ${STYLE} \
    --window-size ${WINDOW_SIZE}
    --cfg-options
```

**Description of all arguments**：

- `config` :  The path of a model config file.
- `dataset-size` : The size of the datasets. If set，`build_dataset` will be skipped and `${DATASET_SIZE}` will be used as the size. Default to use the function `build_dataset`.
- `ngpus` : The number of GPUs used in training, default to be 1.
- `save-path` : The learning rate curve plot save path, default not to save.
- `title` : Title of figure. If not set, default to be config file name.
- `style` : Style of plt. If not set, default to be `whitegrid`.
- `window-size`: The shape of the display window. If not specified, it will be set to `12*7`. If used, it must be in the format `'W*H'`.
- `cfg-options` : Modifications to the configuration file, refer to [Tutorial 1: Learn about Configs](https://mmclassification.readthedocs.io/en/latest/tutorials/config.html).

```{note}
Loading annotations maybe consume much time, you can directly specify the size of the dataset with `dataset-size` to save time.
```

**Examples**：

```bash
python tools/visualizations/vis_lr.py configs/resnet/resnet50_b16x8_cifar100.py
```

<div align=center><img src="../_static/image/tools/visualization/lr_schedule1.png" style=" width: auto; height: 40%; "></div>

When using ImageNet, directly specify the size of ImageNet, as below:

```bash
python tools/visualizations/vis_lr.py configs/repvgg/repvgg-B3g4_4xb64-autoaug-lbs-mixup-coslr-200e_in1k.py --dataset-size 1281167 --ngpus 4 --save-path ./repvgg-B3g4_4xb64-lr.jpg
```

<div align=center><img src="../_static/image/tools/visualization/lr_schedule2.png" style=" width: auto; height: 40%; "></div>

## FAQs

- None
