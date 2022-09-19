# 可视化工具（待更新）

<!-- TOC -->

- [浏览数据集](#浏览数据集)
- [学习率策略可视化](#学习率策略可视化)
- [类别激活图可视化](#类别激活图可视化)
- [常见问题](#常见问题)

<!-- TOC -->

## 浏览数据集

```bash
python tools/visualizations/browse_dataset.py \
    ${CONFIG_FILE} \
    [--output-dir ${OUTPUT_DIR}] \
    [--phase ${DATASET_PHASE}] \
    [--show-number ${NUMBER_IMAGES_DISPLAY}] \
    [--show-interval ${SHOW-INTERRVAL}] \
    [--mode ${DISPLAY_MODE}] \
    [--bgr2rgb]
    [--rescale-factor ${RESCALE-FACTOR}] \
    [--cfg-options ${CFG_OPTIONS}]
```

**所有参数的说明**：

- `config` : 模型配置文件的路径。
- `--output-dir`: 保存图片文件夹，如果没有指定，默认为 `''`,表示不保存。
- **`--phase`**: 可视化数据集的阶段，只能为 `['train', 'val', 'test']` 之一，默认为 `'train'`。
- **`--show-number`**: 可视化样本数量。如果没有指定，默认展示数据集的所有图片。
- `--show-interval`: 浏览时，每张图片的停留间隔，单位为秒。
- **`--mode`**: 可视化的模式，只能为 `['original', 'transformed', 'concat', 'pipeline']` 之一。 默认为`'transformed'`.
- **`--rescale-factor`**: 对可视化图片的放缩倍数，在图片过大或过小时设置。
- `--bgr2rgb`: 将图片的颜色通道翻转。
- `--cfg-options` : 对配置文件的修改，参考[学习配置文件](./config.html)。

```{note}

1. `--mode` 用于设置可视化的模式，默认设置为 'transformed'。
- 如果 `--mode` 设置为 'original'，则获取原始图片；
- 如果 `--mode` 设置为 'transformed'，则获取预处理后的图片；
- 如果 `--mode` 设置为 'concat'，获取原始图片和预处理后图片拼接的图片；
- 如果 `--mode` 设置为 'pipeline'，则获得数据流水线所有中间过程图片。

2. `--rescale-factor` 在数据集中图片的分辨率过大或者过小时设置。比如在可视化 CIFAR 数据集时，由于图片的分辨率非常小，可将 `--rescale-factor` 设置为 10。
```

**示例**：

1. **'original'** 模式 ：

```shell
python ./tools/visualizations/browse_dataset.py ./configs/resnet/ --phase val --output-dir tmp --mode original --show-number 100 --rescale-factor 10 --bgr2rgb
```

<div align=center><img src="https://user-images.githubusercontent.com/18586273/190993839-216a7a1e-590e-47b9-92ae-08f87a7d58df.jpg" style=" width: auto; height: 40%; "></div>

2. **'transformed'** 模式 ：

```shell
python ./tools/visualizations/browse_dataset.py ./configs/resnet/resnet50_8xb32_in1k.py --show-number 100 --rescale-factor 2
```

<div align=center><img src="https://user-images.githubusercontent.com/18586273/190994696-737b09d9-d0fb-4593-94a2-4487121e0286.JPEG" style=" width: auto; height: 40%; "></div>

3. **'concat'** 模式 ：

```shell
python ./tools/visualizations/browse_dataset.py configs/swin_transformer/swin-small_16xb64_in1k.py --show-number 10 --output-dir tmp  --mode concat
```

<div align=center><img src="https://user-images.githubusercontent.com/18586273/190995078-3872feb2-d4e2-4727-a21b-7062d52f7d3e.JPEG" style=" width: auto; height: 40%; "></div>

4. **'pipeline'** 模式 ：

```shell
python ./tools/visualizations/browse_dataset.py configs/swin_transformer/swin-small_16xb64_in1k.py --mode pipeline
```

<div align=center><img src="https://user-images.githubusercontent.com/18586273/190995525-fac0220f-6630-4013-b94a-bc6de4fdff7a.JPEG" style=" width: auto; height: 40%; "></div>

## 学习率策略可视化

```bash
python tools/visualizations/vis_lr.py \
    ${CONFIG_FILE} \
    [--dataset-size ${Dataset_Size}] \
    [--ngpus ${NUM_GPUs}] \
    [--save-path ${SAVE_PATH}] \
    [--title ${TITLE}] \
    [--style ${STYLE}] \
    [--window-size ${WINDOW_SIZE}] \
    [--cfg-options ${CFG_OPTIONS}] \
```

**所有参数的说明**：

- `config` : 模型配置文件的路径。
- `--dataset-size` : 数据集的大小。如果指定，`build_dataset` 将被跳过并使用这个大小作为数据集大小，默认使用 `build_dataset` 所得数据集的大小。
- `--ngpus` : 使用 GPU 的数量。
- `--save-path` : 保存的可视化图片的路径，默认不保存。
- `--title` : 可视化图片的标题，默认为配置文件名。
- `--style` : 可视化图片的风格，默认为 `whitegrid`。
- `--window-size`: 可视化窗口大小，如果没有指定，默认为 `12*7`。如果需要指定，按照格式 `'W*H'`。
- `--cfg-options` : 对配置文件的修改，参考[教程 1：如何编写配置文件](https://mmclassification.readthedocs.io/zh_CN/latest/tutorials/config.html)。

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

MMClassification 提供 `tools\visualizations\vis_cam.py` 工具来可视化类别激活图。请使用 `pip install "grad-cam>=1.3.6"` 安装依赖的 [pytorch-grad-cam](https://github.com/jacobgil/pytorch-grad-cam)。

目前支持的方法有：

|    Method    |                                           What it does                                            |
| :----------: | :-----------------------------------------------------------------------------------------------: |
|   GradCAM    |                                  使用平均梯度对 2D 激活进行加权                                   |
|  GradCAM++   |                                  类似 GradCAM，但使用了二阶梯度                                   |
|   XGradCAM   |                         类似 GradCAM，但通过归一化的激活对梯度进行了加权                          |
|   EigenCAM   |                     使用 2D 激活的第一主成分（无法区分类别，但效果似乎不错）                      |
| EigenGradCAM | 类似 EigenCAM，但支持类别区分，使用了激活 * 梯度的第一主成分，看起来和 GradCAM 差不多，但是更干净 |
|   LayerCAM   |                        使用正梯度对激活进行空间加权，对于浅层有更好的效果                         |

**命令行**：

```bash
python tools/visualizations/vis_cam.py \
    ${IMG} \
    ${CONFIG_FILE} \
    ${CHECKPOINT} \
    [--target-layers ${TARGET-LAYERS}] \
    [--preview-model] \
    [--method ${METHOD}] \
    [--target-category ${TARGET-CATEGORY}] \
    [--save-path ${SAVE_PATH}] \
    [--vit-like] \
    [--num-extra-tokens ${NUM-EXTRA-TOKENS}]
    [--aug_smooth] \
    [--eigen_smooth] \
    [--device ${DEVICE}] \
    [--cfg-options ${CFG-OPTIONS}]
```

**所有参数的说明**：

- `img`：目标图片路径。
- `config`：模型配置文件的路径。
- `checkpoint`：权重路径。
- `--target-layers`：所查看的网络层名称，可输入一个或者多个网络层, 如果不设置，将使用最后一个`block`中的`norm`层。
- `--preview-model`：是否查看模型所有网络层。
- `--method`：类别激活图图可视化的方法，目前支持 `GradCAM`, `GradCAM++`, `XGradCAM`, `EigenCAM`, `EigenGradCAM`, `LayerCAM`，不区分大小写。如果不设置，默认为 `GradCAM`。
- `--target-category`：查看的目标类别，如果不设置，使用模型检测出来的类别做为目标类别。
- `--save-path`：保存的可视化图片的路径，默认不保存。
- `--eigen-smooth`：是否使用主成分降低噪音，默认不开启。
- `--vit-like`: 是否为 `ViT` 类似的 Transformer-based 网络
- `--num-extra-tokens`: `ViT` 类网络的额外的 tokens 通道数，默认使用主干网络的 `num_extra_tokens`。
- `--aug-smooth`：是否使用测试时增强
- `--device`：使用的计算设备，如果不设置，默认为'cpu'。
- `--cfg-options`：对配置文件的修改，参考[教程 1：如何编写配置文件](https://mmclassification.readthedocs.io/zh_CN/latest/tutorials/config.html)。

```{note}
在指定 `--target-layers` 时，如果不知道模型有哪些网络层，可使用命令行添加 `--preview-model` 查看所有网络层名称；
```

**示例（CNN）**：

`--target-layers` 在 `Resnet-50` 中的一些示例如下:

- `'backbone.layer4'`，表示第四个 `ResLayer` 层的输出。
- `'backbone.layer4.2'` 表示第四个 `ResLayer` 层中第三个 `BottleNeck` 块的输出。
- `'backbone.layer4.2.conv1'` 表示上述 `BottleNeck` 块中 `conv1` 层的输出。

```{note}
对于 `ModuleList` 或者 `Sequential` 类型的网络层，可以直接使用索引的方式指定子模块。比如 `backbone.layer4[-1]` 和 `backbone.layer4.2` 是相同的，因为 `layer4` 是一个拥有三个子模块的 `Sequential`。
```

1. 使用不同方法可视化 `ResNet50`，默认 `target-category` 为模型检测的结果，使用默认推导的 `target-layers`。

   ```shell
   python tools/visualizations/vis_cam.py \
       demo/bird.JPEG \
       configs/resnet/resnet50_8xb32_in1k.py \
       https://download.openmmlab.com/mmclassification/v0/resnet/resnet50_batch256_imagenet_20200708-cfb998bf.pth \
       --method GradCAM
       # GradCAM++, XGradCAM, EigenCAM, EigenGradCAM, LayerCAM
   ```

   | Image                                | GradCAM                                 | GradCAM++                                 | EigenGradCAM                                 | LayerCAM                                 |
   | ------------------------------------ | --------------------------------------- | ----------------------------------------- | -------------------------------------------- | ---------------------------------------- |
   | <div align=center><img src='https://user-images.githubusercontent.com/18586273/144429496-628d3fb3-1f6e-41ff-aa5c-1b08c60c32a9.JPEG' height="auto" width="160" ></div> | <div align=center><img src='https://user-images.githubusercontent.com/18586273/147065002-f1c86516-38b2-47ba-90c1-e00b49556c70.jpg' height="auto" width="150" ></div> | <div align=center><img src='https://user-images.githubusercontent.com/18586273/147065119-82581fa1-3414-4d6c-a849-804e1503c74b.jpg' height="auto" width="150"></div> | <div align=center><img src='https://user-images.githubusercontent.com/18586273/147065096-75a6a2c1-6c57-4789-ad64-ebe5e38765f4.jpg' height="auto" width="150"></div> | <div align=center><img src='https://user-images.githubusercontent.com/18586273/147065129-814d20fb-98be-4106-8c5e-420adcc85295.jpg' height="auto" width="150"></div> |

2. 同一张图不同类别的激活图效果图，在 `ImageNet` 数据集中，类别238为 'Greater Swiss Mountain dog'，类别281为 'tabby, tabby cat'。

   ```shell
   python tools/visualizations/vis_cam.py \
       demo/cat-dog.png configs/resnet/resnet50_8xb32_in1k.py \
       https://download.openmmlab.com/mmclassification/v0/resnet/resnet50_batch256_imagenet_20200708-cfb998bf.pth \
       --target-layers 'backbone.layer4.2' \
       --method GradCAM \
       --target-category 238
       # --target-category 281
   ```

   | Category | Image                                          | GradCAM                                          | XGradCAM                                          | LayerCAM                                          |
   | -------- | ---------------------------------------------- | ------------------------------------------------ | ------------------------------------------------- | ------------------------------------------------- |
   | Dog      | <div align=center><img src='https://user-images.githubusercontent.com/18586273/144429526-f27f4cce-89b9-4117-bfe6-55c2ca7eaba6.png' height="auto" width="165" ></div> | <div align=center><img src='https://user-images.githubusercontent.com/18586273/144433562-968a57bc-17d9-413e-810e-f91e334d648a.jpg' height="auto" width="150" ></div> | <div align=center><img src='https://user-images.githubusercontent.com/18586273/144433853-319f3a8f-95f2-446d-b84f-3028daca5378.jpg' height="auto" width="150" ></div> | <div align=center><img src='https://user-images.githubusercontent.com/18586273/144433937-daef5a69-fd70-428f-98a3-5e7747f4bb88.jpg' height="auto" width="150" ></div> |
   | Cat      | <div align=center><img src='https://user-images.githubusercontent.com/18586273/144429526-f27f4cce-89b9-4117-bfe6-55c2ca7eaba6.png' height="auto" width="165" ></div> | <div align=center><img src='https://user-images.githubusercontent.com/18586273/144434518-867ae32a-1cb5-4dbd-b1b9-5e375e94ea48.jpg' height="auto" width="150" ></div> | <div align=center><img src='https://user-images.githubusercontent.com/18586273/144434603-0a2fd9ec-c02e-4e6c-a17b-64c234808c56.jpg' height="auto" width="150" ></div> | <div align=center><img src='https://user-images.githubusercontent.com/18586273/144434623-b4432cc2-c663-4b97-aed3-583d9d3743e6.jpg' height="auto" width="150" ></div> |

3. 使用 `--eigen-smooth` 以及 `--aug-smooth` 获取更好的可视化效果。

   ```shell
   python tools/visualizations/vis_cam.py \
       demo/dog.jpg  \
       configs/mobilenet_v3/mobilenet-v3-large_8xb32_in1k.py \
       https://download.openmmlab.com/mmclassification/v0/mobilenet_v3/convert/mobilenet_v3_large-3ea3c186.pth \
       --target-layers 'backbone.layer16' \
       --method LayerCAM \
       --eigen-smooth --aug-smooth
   ```

   | Image                                | LayerCAM                                | eigen-smooth                                | aug-smooth                                | eigen&aug                                 |
   | ------------------------------------ | --------------------------------------- | ------------------------------------------- | ----------------------------------------- | ----------------------------------------- |
   | <div align=center><img src='https://user-images.githubusercontent.com/18586273/144557492-98ac5ce0-61f9-4da9-8ea7-396d0b6a20fa.jpg' height="auto" width="160"></div> | <div align=center><img src='https://user-images.githubusercontent.com/18586273/144557541-a4cf7d86-7267-46f9-937c-6f657ea661b4.jpg'  height="auto" width="145" ></div> | <div align=center><img src='https://user-images.githubusercontent.com/18586273/144557547-2731b53e-e997-4dd2-a092-64739cc91959.jpg'  height="auto" width="145" ></div> | <div align=center><img src='https://user-images.githubusercontent.com/18586273/144557545-8189524a-eb92-4cce-bf6a-760cab4a8065.jpg'  height="auto" width="145" ></div> | <div align=center><img src='https://user-images.githubusercontent.com/18586273/144557548-c1e3f3ec-3c96-43d4-874a-3b33cd3351c5.jpg'  height="auto" width="145" ></div> |

**示例（Transformer）**：

`--target-layers` 在 Transformer-based 网络中的一些示例如下:

- Swin-Transformer 中：`'backbone.norm3'`
- ViT 中：`'backbone.layers[-1].ln1'`

对于 Transformer-based 的网络，比如 ViT、T2T-ViT 和 Swin-Transformer，特征是被展平的。为了绘制 CAM 图，我们需要指定 `--vit-like` 选项，从而让被展平的特征恢复方形的特征图。

除了特征被展平之外，一些类 ViT 的网络还会添加额外的 tokens。比如 ViT 和 T2T-ViT 中添加了分类 token，DeiT 中还添加了蒸馏 token。在这些网络中，分类计算在最后一个注意力模块之后就已经完成了，分类得分也只和这些额外的 tokens 有关，与特征图无关，也就是说，分类得分对这些特征图的导数为 0。因此，我们不能使用最后一个注意力模块的输出作为 CAM 绘制的目标层。

另外，为了去除这些额外的 toekns 以获得特征图，我们需要知道这些额外 tokens 的数量。MMClassification 中几乎所有 Transformer-based 的网络都拥有 `num_extra_tokens` 属性。而如果你希望将此工具应用于新的，或者第三方的网络，而且该网络没有指定 `num_extra_tokens` 属性，那么可以使用 `--num-extra-tokens` 参数手动指定其数量。

1. 对 `Swin Transformer` 使用默认 `target-layers` 进行 CAM 可视化：

   ```shell
   python tools/visualizations/vis_cam.py \
       demo/bird.JPEG  \
       configs/swin_transformer/swin-tiny_16xb64_in1k.py \
       https://download.openmmlab.com/mmclassification/v0/swin-transformer/swin_tiny_224_b16x64_300e_imagenet_20210616_090925-66df6be6.pth \
       --vit-like
   ```

2. 对 `Vision Transformer(ViT)` 进行 CAM 可视化：

   ```shell
   python tools/visualizations/vis_cam.py \
       demo/bird.JPEG  \
       configs/vision_transformer/vit-base-p16_ft-64xb64_in1k-384.py \
       https://download.openmmlab.com/mmclassification/v0/vit/finetune/vit-base-p16_in21k-pre-3rdparty_ft-64xb64_in1k-384_20210928-98e8652b.pth \
       --vit-like \
       --target-layers 'backbone.layers[-1].ln1'
   ```

3. 对 `T2T-ViT` 进行 CAM 可视化：

   ```shell
   python tools/visualizations/vis_cam.py \
       demo/bird.JPEG  \
       configs/t2t_vit/t2t-vit-t-14_8xb64_in1k.py \
       https://download.openmmlab.com/mmclassification/v0/t2t-vit/t2t-vit-t-14_3rdparty_8xb64_in1k_20210928-b7c09b62.pth \
       --vit-like \
       --target-layers 'backbone.encoder[-1].ln1'
   ```

| Image                                   | ResNet50                                   | ViT                                    | Swin                                    | T2T-ViT                                    |
| --------------------------------------- | ------------------------------------------ | -------------------------------------- | --------------------------------------- | ------------------------------------------ |
| <div align=center><img src='https://user-images.githubusercontent.com/18586273/144429496-628d3fb3-1f6e-41ff-aa5c-1b08c60c32a9.JPEG' height="auto" width="165" ></div> | <div align=center><img src=https://user-images.githubusercontent.com/18586273/144431491-a2e19fe3-5c12-4404-b2af-a9552f5a95d9.jpg  height="auto" width="150" ></div> | <div align=center><img src='https://user-images.githubusercontent.com/18586273/144436218-245a11de-6234-4852-9c08-ff5069f6a739.jpg' height="auto" width="150" ></div> | <div align=center><img src='https://user-images.githubusercontent.com/18586273/144436168-01b0e565-442c-4e1e-910c-17c62cff7cd3.jpg' height="auto" width="150" ></div> | <div align=center><img src='https://user-images.githubusercontent.com/18586273/144436198-51dbfbda-c48d-48cc-ae06-1a923d19b6f6.jpg' height="auto" width="150" ></div> |

## 常见问题

- 无
