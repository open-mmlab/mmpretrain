# Visualization

<!-- TOC -->

- [Visualization](#visualization)
  - [Pipeline Visualization](#pipeline-visualization)
    - [Usage](#usage)
  - [FAQs](#faqs)

<!-- TOC -->

## pipeline visualization

### Usage

```bash
python tools/visualizations/vis_pipeline.py \
    ${CONFIG_FILE} \
    --output-dir ${OUTPUT_DIR} \
    --phase ${DATASET_PHASE} \
    --number ${BUNBER_IMAGES_DISPLAY} \
    --skip-type ${SKIP_TRANSFORM_TYPE}
    --mode ${DISPLAY_MODE} \
    --show \
    --adaptive \
    --min-edge-length ${MIN-EDGE-LENGTH} \
    --max-edge-length ${MAX-EDGE-LENGTH} \
    --bgr2rgb \
    --window-size ${WINDOW_SIZE}
```

**Description of all arguments**：

- `config` : The path of a model config file.
- `--output-dir`: The path of output visualizd images. If not specified, it will be set to `''`, means not to save.
- `--phase`: Phase of visualizing dataset，must be one of `[train, val, test]`. If not specified, it will be set to `train`.
- `--number`: The number of samples to visualize. If not specified, it will be set to be `sys.maxsize`.
- `--skip-type`: The pipeline process to be skipped. If not specified, it will be set to `['ToTensor', 'Normalize', 'ImageToTensor', 'Collect']`.
- `--mode`: The display mode. can be one of `[original, pipeline, concat]`. If not specified, it will be set to `pipeline`.
- `--show`: Whether to display preprocessed pictures in pop-up windows. If not specified, it will be set to `False`.
- `--adaptive`: Whether to automatically adjust the size of the visual image. If not specified, the default is `False`.
- `--min-edge-length`: The minium edge length, used when `--adaptivethe` is `True`. When any side of the picture is smaller than `${MIN-EDGE-LENGTH}`, the picture will be enlarged while keeping the aspect ratio unchanged, and the short side will be aligned to `${MIN-EDGE-LENGTH}`. If not specified, it will be set to 200.
- `--max-edge-length`: The maxium edge length, used when `--adaptivethe` is `True`. When any side of the picture is larger than `${MAX-EDGE-LENGTH}`, the picture will be reduced while keeping the aspect ratio unchanged, and the long side will be aligned to `${MAX-EDGE-LENGTH}`. If not specified, it will be set to 1000.
- `--bgr2rgb`: Whether to flip the color channel order of images. If not specified, it will be set to `False`.
- `--window-size`: The shape of display window. If not specified, it will be set to `12*7`, If used, it must be in format `'W*H'`.

**Notice**:

1. If the `--mode` not specified, it will be set to `pipeline` as default, get the transformed pictures; if the `--mode` set to `original`, get the original pictures; if the `--mode` set to `concat`, get the pictures stitched together by original pictures and transformed pictures.

2. When `--adaptive` is set to `True`, images that are too large or too small will be automatically adjusted; `--min-edge-length` and `--max-edge-length` affect in this process.

**Examples**：

1. Visualizing all the transformed pictures of the `ImageNet` training set and display them in pop-up windows：

`python ./tools/visualizations/vis_pipeline.py ./configs/resnet/resnet50_b32x8_imagenet.py --show --adaptive`

2. Visualizing 10 comparison pictures in the `ImageNet` train set and save them in the `./tmp` folder：

`python ./tools/visualizations/vis_pipeline.py configs/swin_transformer/swin_base_224_b16x64_300e_imagenet.py --phase train --output-dir tmp --mode concat --number 10 --adaptive`

3. Visualizing 100 original pictures in the `CIFAR100` val set, then display and save them in the `./tmp` folder：

`python ./tools/visualizations/vis_pipeline.py configs/resnet/resnet50_b16x8_cifar100.py --phase val --output-dir tmp --mode original --number 100  --show --adaptive`

## FAQs

- None
