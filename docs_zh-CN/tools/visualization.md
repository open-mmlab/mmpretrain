# 可视化

<!-- TOC -->

- [可视化](#可视化)
  - [数据流水线可视化](#数据流水线可视化)
  - [学习率策略可视化](#学习率策略可视化)
  - [Grad-CAM可视化](#grad-cam可视化)
  - [常见问题](#常见问题)

<!-- TOC -->

## 数据流水线可视化

```bash
python tools/visualizations/vis_pipeline.py \
    ${CONFIG_FILE} \
    --output-dir ${OUTPUT_DIR} \
    --phase ${DATASET_PHASE} \
    --number ${BUNBER_IMAGES_DISPLAY} \
    --skip-type ${SKIP_TRANSFORM_TYPE} \
    --mode ${DISPLAY_MODE} \
    --show \
    --adaptive \
    --min-edge-length ${MIN_EDGE_LENGTH} \
    --max-edge-length ${MAX_EDGE_LENGTH} \
    --bgr2rgb \
    --window-size ${WINDOW_SIZE}
```

**所有参数的说明**：

- `config` : 模型配置文件的路径。
- `--output-dir`: 保存图片文件夹，如果没有指定，默认为 `''`,表示不保存。
- `--phase`: 可视化数据集的阶段，只能为 `[train, val, test]` 之一，默认为 `train`。
- `--number`: 可视化样本数量。如果没有指定，默认展示数据集的所有图片。
- `--skip-type`: 预设跳过的数据流水线过程。如果没有指定，默认为 `['ToTensor', 'Normalize', 'ImageToTensor', 'Collect']`。
- `--mode`: 可视化的模式，只能为 `[original, pipeline, concat]` 之一，如果没有指定，默认为 `concat`。
- `--show`: 将可视化图片以弹窗形式展示。
- `--adaptive`: 自动调节可视化图片的大小。
- `--min-edge-length`: 最短边长度，当使用了 `--adaptive` 时有效。 当图片任意边小于 `${MIN_EDGE_LENGTH}` 时，会保持长宽比不变放大图片，短边对齐至 `${MIN_EDGE_LENGTH}`，默认为200。
- `--max-edge-length`: 最长边长度，当使用了 `--adaptive` 时有效。 当图片任意边大于 `${MAX_EDGE_LENGTH}` 时，会保持长宽比不变缩小图片，短边对齐至 `${MAX_EDGE_LENGTH}`，默认为1000。
- `--bgr2rgb`: 将图片的颜色通道翻转。
- `--window-size`: 可视化窗口大小，如果没有指定，默认为 `12*7`。如果需要指定，按照格式 `'W*H'`。

```{note}

1. 如果不指定 `--mode`，默认设置为 `concat`，获取原始图片和预处理后图片拼接的图片；如果 `--mode` 设置为 `original`，则获取原始图片； 如果  `--mode` 设置为 `pipeline`，则获取预处理后的图片。

2. 当指定了 `--adaptive` 选项时，会自动的调整尺寸过大和过小的图片，你可以通过设定 `--min-edge-length` 与 `--max-edge-length` 来指定自动调整的图片尺寸。
```

**示例**：

1. 可视化 `ImageNet` 训练集的所有经过预处理的图片，并以弹窗形式显示：

```shell
python ./tools/visualizations/vis_pipeline.py ./configs/resnet/resnet50_8xb32_in1k.py --show --mode pipeline
```

<div align=center><img src="../_static/image/tools/visualization/pipeline-pipeline.jpg" style=" width: auto; height: 40%; "></div>

2. 可视化 `ImageNet` 训练集的10张原始图片与预处理后图片对比图，保存在 `./tmp` 文件夹下：

```shell
python ./tools/visualizations/vis_pipeline.py configs/swin_transformer/swin_base_224_b16x64_300e_imagenet.py --phase train --output-dir tmp --number 10 --adaptive
```

<div align=center><img src="../_static/image/tools/visualization/pipeline-concat.jpg" style=" width: auto; height: 40%; "></div>

3. 可视化 `CIFAR100` 验证集中的100张原始图片，显示并保存在 `./tmp` 文件夹下：

```shell
python ./tools/visualizations/vis_pipeline.py configs/resnet/resnet50_8xb16_cifar100.py --phase val --output-dir tmp --mode original --number 100 --show --adaptive --bgr2rgb
```

<div align=center><img src="../_static/image/tools/visualization/pipeline-original.jpg" style=" width: auto; height: 40%; "></div>

## 学习率策略可视化

```bash
python tools/visualizations/vis_lr.py \
    ${CONFIG_FILE} \
    --dataset-size ${Dataset_Size} \
    --ngpus ${NUM_GPUs}
    --save-path ${SAVE_PATH} \
    --title ${TITLE} \
    --style ${STYLE} \
    --window-size ${WINDOW_SIZE}
    --cfg-options
```

**所有参数的说明**：

- `config` : 模型配置文件的路径。
- `dataset-size` : 数据集的大小。如果指定，`build_dataset` 将被跳过并使用这个大小作为数据集大小，默认使用 `build_dataset` 所得数据集的大小。
- `ngpus` : 使用 GPU 的数量。
- `save-path` : 保存的可视化图片的路径，默认不保存。
- `title` : 可视化图片的标题，默认为配置文件名。
- `style` : 可视化图片的风格，默认为 `whitegrid`。
- `window-size`: 可视化窗口大小，如果没有指定，默认为 `12*7`。如果需要指定，按照格式 `'W*H'`。
- `cfg-options` : 对配置文件的修改，参考[教程 1：如何编写配置文件](https://mmclassification.readthedocs.io/zh_CN/latest/tutorials/config.html)。

```{note}

部分数据集在解析标注阶段比较耗时，可直接将 `dataset-size` 指定数据集的大小，以节约时间。

```

**示例**：

```bash
python tools/visualizations/vis_lr.py configs/resnet/resnet50_b16x8_cifar100.py
```

<div align=center><img src="../_static/image/tools/visualization/lr_schedule1.png" style=" width: auto; height: 40%; "></div>

当数据集为 ImageNet 时，通过直接指定数据集大小来节约时间，并保存图片：

```bash
python tools/visualizations/vis_lr.py configs/repvgg/repvgg-B3g4_4xb64-autoaug-lbs-mixup-coslr-200e_in1k.py --dataset-size 1281167 --ngpus 4 --save-path ./repvgg-B3g4_4xb64-lr.jpg
```

<div align=center><img src="../_static/image/tools/visualization/lr_schedule2.png" style=" width: auto; height: 40%; "></div>

## 类别激活图可视化

MMClassification 提供 `tools\visualizations\vis_cam.py` 工具来可视化类别激活图。请使用 `pip install grad-cam` 安装依赖库。工具基于 [pytorch-grad-cam](https://github.com/jacobgil/pytorch-grad-cam)。支持的方法有：

| Method   | What it does |
|----------|--------------|
| GradCAM  | Weight the 2D activations by the average gradient |
| GradCAM++  | Like GradCAM but uses second order gradients |
| XGradCAM  | Like GradCAM but scale the gradients by the normalized activations |
| EigenCAM  | Takes the first principle component of the 2D Activations (no class discrimination, but seems to give great results)|
| EigenGradCAM  | Like EigenCAM but with class discrimination: First principle component of Activations*Grad. Looks like GradCAM, but cleaner|
| LayerCAM  | Spatially weight the activations by positive gradients. Works better especially in lower layers |

```bash
python tools/visualizations/vis_cam.py \
    ${IMG-PATH} \
    ${CONFIG_FILE} \
    ${CHECKPOINT} \
    --target-layers ${TARGET-LAYERS} \
    [--preview-model] \
    [--method ${CAM-TYPE}] \
    [--target-category ${TARGET-CATEGORY}] \
    [--save-path ${SAVE_PATH}] \
    [--aug-smooth] \
    [--eigen-smooth] \
    [--device ${DEVICE}] \
    [--cfg-options ${CFG-OPTIONS}]
```

**所有参数的说明**：

- `img`：目标图片路径。
- `config`：模型配置文件的路径。
- `checkpoint`：权重路径。
- `--target-layers`：所查看的网络层名称，可输入一个或者多个网络层名称。
- `--preview-model`：是否查看模型所有网络层。
- `--method`：热力图可视化的算法名称，目前支持 `GradCAM`, `GradCAM++`, `XGradCAM`, `EigenCAM`, `EigenGradCAM`, `LayerCAM`(不区分大小写)，如果不设置，默认为 `GradCAM`。
- `--target-category`：查看的目标类别，如果不设置，使用模型检测出来的类别做为目标类别。
- `--save-path`：保存的可视化图片的路径，默认不保存。
- `--eigen-smooth`：是否使用主成分降低噪音，默认不开启。
- `--aug-smooth`：是否使用测试时增强，默认不开启。
- `--device`：使用的计算设备，如果不设置，默认为'cpu'。
- `--cfg-options`：对配置文件的修改，参考[教程 1：如何编写配置文件](https://mmclassification.readthedocs.io/zh_CN/latest/tutorials/config.html)。

```{note}
1. 不知道模型中有哪些层，可以在命令行中添加 '--preview-model' 查看所有网络层名称；
2. 'target-layers' 都是以 'model'开始；
3. `--eigen-smooth` 以及 `--aug-smooth` 可以获得更加平滑的效果图。
```

**示例（CNN）**：

`target-layers` 不能为 `bn` 或者 `relu`。可以为:

- `model.backbone.layer4`
- `model.backbone.layer4.1.conv`

1.使用不同算法可视化 `ResNet50` 的 `layer4`，默认 `target-category` 为模型结果类别。

```shell
python tools/visualizations/vis_cam.py demo/bird.JPEG configs/resnet/resnet50_8xb32_in1k.py \
    https://download.openmmlab.com/mmclassification/v0/resnet/resnet50_batch256_imagenet_20200708-cfb998bf.pth \
    --target-layers model.backbone.layer4.2 \
    --method GradCAM
    # GradCAM++, XGradCAM, EigenCAM, EigenGradCAM, LayerCAM
```

| Category  | Image | GradCAM  |  GradCAM++ |  EigenGradCAM |  LayerCAM  |
| --------- |-------|----------|------------|-------------- |------------|
| Bird | ![](../../demo/bird.JPEG) | ![](../_static/image/tools/visualization/cam_resnet50_bird_gradcam.jpg)   |  ![](../_static/image/tools/visualization/cam_resnet50_bird_gradcamplusplus.jpg)   |![](../_static/image/tools/visualization/cam_resnet50_bird_eigengradcam.jpg) |![](../_static/image/tools/visualization/cam_resnet50_bird_layercam.jpg)   |

2.同一张图不同类别的激活图效果图, 238 为 '', 281 为 ''。

```shell
python tools/visualizations/vis_cam.py demo/cat-dog.png configs/resnet/resnet50_8xb32_in1k.py \
    https://download.openmmlab.com/mmclassification/v0/resnet/resnet50_batch256_imagenet_20200708-cfb998bf.pth \
    --target-layers model.backbone.layer4.2 \
    --method GradCAM \
    --target-category 238
    # --target-category 281
```

| Category  | Image | GradCAM  |  GradCAM++ |  XGradCAM |  LayerCAM  |
| --------- |-------|----------|------------|-------------- |------------|
|   Dog     | ![原图](../../demo/cat-dog.png) | ![GradCAM](../_static/image/tools/visualization/cam_dog_gradcam.jpg)   |  ![](../_static/image/tools/visualization/cam_dog_gradcamplusplus.jpg)   |![](../_static/image/tools/visualization/cam_dog_xgradcam.jpg) |![](../_static/image/tools/visualization/cam_dog_layercam.jpg)   |
|   Cat     | ![原图](../../demo/cat-dog.png) | ![GradCAM](../_static/image/tools/visualization/cam_cat_gradcam.jpg)   |  ![](../_static/image/tools/visualization/cam_cat_gradcamplusplus.jpg)   |![](../_static/image/tools/visualization/cam_cat_xgradcam.jpg) |![](../_static/image/tools/visualization/cam_cat_layercam.jpg)   |

3.使用 `--eigen-smooth` 以及 `--aug-smooth` 获取更好的效果。

```shell
python tools/visualizations/vis_cam.py demo/bird.JPEG  \
    configs/mobilenet_v3/mobilenet-v3-large_8xb32_in1k.py \
    https://download.openmmlab.com/mmclassification/v0/mobilenet_v3/convert/mobilenet_v3_large-3ea3c186.pth \
    --target-layers model.backbone.layer16 \
    --method GradCAM \
    --eigen-smooth --aug-smooth
```

| Category  | Image | GradCAM  |  eigen-smooth |  aug-smooth |  eigen-smooth & aug-smooth  |
| --------- |-------|----------|------------|-------------- |------------|
| Bird | ![](../../demo/bird.JPEG) | ![](../_static/image/tools/visualization/cam_mobilenetv3_bird_gradcam.jpg)   |  ![](../_static/image/tools/visualization/cam_mobilenetv3_bird_gradcam_eigen.jpg)   |![](../_static/image/tools/visualization/cam_mobilenetv3_bird_gradcam_aug.jpg) |![](../_static/image/tools/visualization/cam_mobilenetv3_bird_gradcam_eigen_aug.jpg)   |

**示例（Transformer）**：

Transformer 类的网络，目前只支持 `SwinTransformer` 和 `VisionTransformer`，`target-layers` 需要设置为 `layer norm`,如：

- `model.backbone.norm3`
- `model.backbone.layers.11.ln1`

1.对 `Swin Transformer` 进行 CAM 可视化：

```shell
python tools/visualizations/vis_cam.py demo/bird.JPEG  \
    configs/swin_transformer/swin-tiny_16xb64_in1k.py \
    https://download.openmmlab.com/mmclassification/v0/swin-transformer/swin_tiny_224_b16x64_300e_imagenet_20210616_090925-66df6be6.pth \
    --target-layers model.backbone.norm3
```

2.对 `Vision Transformer(ViT)` 进行 CAM 可视化：

```shell
python tools/visualizations/vis_cam.py demo/bird.JPEG  \
    configs/vision_transformer/vit-base-p16_ft-64xb64_in1k-384.py \
    https://download.openmmlab.com/mmclassification/v0/vit/finetune/vit-base-p16_in21k-pre-3rdparty_ft-64xb64_in1k-384_20210928-98e8652b.pth \
    --target-layers model.backbone.layers.11.ln1
```

3.对 `T2T-ViT` 进行 CAM 可视化：

```shell
python tools/visualizations/vis_cam.py demo/bird.JPEG  \
    configs/t2t_vit/t2t-vit-t-14_8xb64_in1k.py \
    https://download.openmmlab.com/mmclassification/v0/t2t-vit/t2t-vit-t-14_3rdparty_8xb64_in1k_20210928-b7c09b62.pth \
    --target-layers model.backbone.encoder.13.ln1
```

| Image | ResNet50  |  ViT |  Swin |  T2T-Vit  |
|-------|----------|------------|-------------- |------------|
| ![](../../demo/bird.JPEG) | ![](../_static/image/tools/visualization/cam_resnet50_bird_gradcam.jpg)   |  ![](../_static/image/tools/visualization/cam_vit_bird_gradcam.jpg)   |![](../_static/image/tools/visualization/cam_swin_bird_gradcam.jpg) |![](../_static/image/tools/visualization/cam_t2tvit_bird_gradcam.jpg)   |

## 常见问题

- 无
