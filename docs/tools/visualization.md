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
    --bgr2rgb \
    --original-display-shape ${RESIZE_ORIGINAL_SIZE}
```

**Description of all arguments**：

- `config` : The path of a model config file.
- `--output-dir`: The path of output visualizd images. If not specified, it will be set to `''`, means not to save.
- `--phase`: Phase of visualizing dataset，can be one of `[train, val, test]`. If not specified, it will be set to `train`.
- `--number`: The number of samples to visualize. If not specified, it will be set to `-1`，which means the entire dataset.
- `--skip-type`: The pipeline process to be skipped. If not specified, it will be set to `['ToTensor', 'Normalize', 'ImageToTensor', 'Collect']`.
- `--mode`: The display mode. can be one of `[original, pipeline, concat]`. If not specified, it will be set to `pipeline`.
- `--show`: Whether to display preprocessed pictures in pop-up windows. If not specified, it will be set to `False`.
- `--bgr2rgb`: Whether to flip the color channel order of images. If not specified, it will be set to `False`.
- `--original-display-shape`: The resize shape of original picture to display. If not specified, it will be set to `''`, means not to change the shape of original pictures. If used, it should be `'W*H'`, eg. `'224*224'`.

**Notice**:

1. If the `--mode` not specified, it will be set to `pipeline` as default, get the transformed pictures; if the `--mode` set to `original`, get the original pictures; if the `--mode` set to `concat`, get the pictures stitched together by original pictures and transformed pictures.

2. If the original pictures are too small or too big, use `--original-display-shape` to change shapes, remember to add `'` or `"` sround the size value. If the transformed pictures are too small or too big, modify the pipeline in source config file to change shapes.

3. If the pictures are too small, the label infomations will not show.

**Examples**：

1. Visualizing all the transformed pictures of the `ImageNet` training set and display them in pop-up windows：

`python ./tools/visualizations/vis_pipeline.py ./configs/resnet/resnet50_b32x8_imagenet.py --show`

2. Visualizing 10 comparison pictures in the `ImageNet` test set and save them in the default `test` folder：

`python ./tools/visualizations/vis_pipeline.py configs/swin_transformer/swin_base_224_b16x64_300e_imagenet.py --phase test --output-dir test --mode 'concat' --number 10 --output-dir test`

3. Visualizing 100 original pictures in the `CIFAR100` val set, resize pictures to 224*224 and display and save them：

`python ./tools/visualizations/vis_pipeline.py configs/resnet/resnet50_b16x8_cifar100.py --phase val --output-dir val --mode original --number 100 --output-dir val --original-display-shape '224*224' --show`

## FAQs

- None
