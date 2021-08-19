# Visualization

<!-- TOC -->

- [Visualization](#visualization)
  - [General dataset pipeline visualization](#general-dataset-pipeline-visualization)
    - [Usage](#usage)
  - [ImageNet dataset pipeline visualization](#imagenet-dataset-pipeline-visualization)
    - [Enhance](#enhance)
  - [FAQs](#faqs)

<!-- TOC -->

## General dataset pipeline visualization

### Usage

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

Description of all arguments：

- `config` : The path of a model config file.
- `--output-dir`: The path of output visualizd images,when `--show` is `False`. If not specified, it will be set to `tmp`。
- `--phase`: phase of visualizing the dataset，could be one of `['train', 'val', 'test']`. If not specified, it will be set to `train`.
- `--number`: the number of samples to visualize. If not specified, it will be set to `-1`，means the entire dataset.
- `--skip-type`: The pipeline process to be skipped. If not specified, it will be set to `['ToTensor', 'Normalize', 'ImageToTensor', 'Collect']`。
- `--show`: Whether to display preprocessed pictures in pop-up windows. If not specified, it will be set to `False`, means to save the pictures into the folder of `--output-dir`.
- `--bgr2rgb`: Whether to flip the color channel. If not specified, it will be set to `False`.

Examples：

1. Visualize all the pictures of the Cifar100 verification set and save them into the folder of `vis_cifar100_val`:

`python ./tools/misc/vis_pipeline.py ./configs/resnet/resnet50_b16x8_cifar100.py --phase val --output-dir vis_cifar100_val`

2. Visualize 100 images of the ImageNet training set and display them in pop-up windows：

`python ./tools/misc/vis_pipeline.py ./configs/resnet/resnet50_b32x8_imagenet.py --number 100 --show`

## ImageNet dataset pipeline visualization

On the basis of the general data pipeline visualization, visualizing the original images and visualizing the pictures stitched together by original pictures and transformed pictures are added.

### Enhance

Description of all arguments：

- `--original`: Whether to visualize the original pictures. If not specified, it will be set to `False`.
- `--transform`: Whether to visualize the transformed pictures. If not specified, it will be set to `False`.

Notice:

When only `--original` is specified, the original pictures will displayed;When only `--transform` is specified, the transformed pictures will be displayed; When `--original` and `--transform` are specified, the pictures stitched together by original pictures and transformed pictures will be displayed; Specify at least one of `--original` and `--transform`, otherwise an error will be reported;  

Examples：

1. Visualize 100 pictures of the ImageNet training set and display them in pop-up windows：

`python ./tools/misc/vis_imagenet.py ./configs/resnet/resnet50_b32x8_imagenet.py --original --number 100 --show`

2. Visualize 10 comparison pictures stitched together by original pictures and transformed pictures in the ImageNet training set and save them in the default `tmp` folder：

`python ./tools/misc/vis_imagenet.py configs/swin_transformer/swin_base_224_b16x64_300e_imagenet.py --original --transform --number 10` 

## FAQs

- None
