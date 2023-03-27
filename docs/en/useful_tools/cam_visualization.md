# Class Activation Map (CAM) Visualization

## Introduction of the CAM visualization tool

MMPretrain provides `tools/visualization/vis_cam.py` tool to visualize class activation map. Please use `pip install "grad-cam>=1.3.6"` command to install [pytorch-grad-cam](https://github.com/jacobgil/pytorch-grad-cam).

The supported methods are as follows:

| Method       | What it does                                                                                                                 |
| ------------ | ---------------------------------------------------------------------------------------------------------------------------- |
| GradCAM      | Weight the 2D activations by the average gradient                                                                            |
| GradCAM++    | Like GradCAM but uses second order gradients                                                                                 |
| XGradCAM     | Like GradCAM but scale the gradients by the normalized activations                                                           |
| EigenCAM     | Takes the first principle component of the 2D Activations (no class discrimination, but seems to give great results)         |
| EigenGradCAM | Like EigenCAM but with class discrimination: First principle component of Activations\*Grad. Looks like GradCAM, but cleaner |
| LayerCAM     | Spatially weight the activations by positive gradients. Works better especially in lower layers                              |

More CAM methods supported by the new version `pytorch-grad-cam` can also be used but we haven't verified the availability.

**Command**：

```bash
python tools/visualization/vis_cam.py \
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

**Description of all arguments**：

- `img`: The target picture path.
- `config`: The path of the model config file.
- `checkpoint`: The path of the checkpoint.
- `--target-layers`: The target layers to get activation maps, one or more network layers can be specified. If not set, use the norm layer of the last block.
- `--preview-model`: Whether to print all network layer names in the model.
- `--method`: Visualization method, supports `GradCAM`, `GradCAM++`, `XGradCAM`, `EigenCAM`, `EigenGradCAM`, `LayerCAM`, which is case insensitive. Defaults to `GradCAM`.
- `--target-category`: Target category, if not set, use the category detected by the given model.
- `--eigen-smooth`: Whether to use the principal component to reduce noise.
- `--aug-smooth`: Whether to use TTA(Test Time Augment) to get CAM.
- `--save-path`: The path to save the CAM visualization image. If not set, the CAM image will not be saved.
- `--vit-like`: Whether the network is ViT-like network.
- `--num-extra-tokens`: The number of extra tokens in ViT-like backbones. If not set, use num_extra_tokens the backbone.
- `--device`: The computing device used. Default to 'cpu'.
- `--cfg-options`: Modifications to the configuration file, refer to [Learn about Configs](../user_guides/config.md).

```{note}
The argument `--preview-model` can view all network layers names in the given model. It will be helpful if you know nothing about the model layers when setting `--target-layers`.
```

## How to visualize the CAM of CNN (ResNet-50)

Here are some examples of `target-layers` in ResNet-50, which can be any module or layer:

- `'backbone.layer4'` means the output of the forth ResLayer.
- `'backbone.layer4.2'` means the output of the third BottleNeck block in the forth ResLayer.
- `'backbone.layer4.2.conv1'` means the output of the `conv1` layer in above BottleNeck block.

1. Use different methods to visualize CAM for `ResNet50`, the `target-category` is the predicted result by the given checkpoint, using the default `target-layers`.

   ```shell
   python tools/visualization/vis_cam.py \
       demo/bird.JPEG \
       configs/resnet/resnet50_8xb32_in1k.py \
       https://download.openmmlab.com/mmclassification/v0/resnet/resnet50_batch256_imagenet_20200708-cfb998bf.pth \
       --method GradCAM
       # GradCAM++, XGradCAM, EigenCAM, EigenGradCAM, LayerCAM
   ```

   | Image                                | GradCAM                                 | GradCAM++                                 | EigenGradCAM                                 | LayerCAM                                 |
   | ------------------------------------ | --------------------------------------- | ----------------------------------------- | -------------------------------------------- | ---------------------------------------- |
   | <div align=center><img src='https://user-images.githubusercontent.com/18586273/144429496-628d3fb3-1f6e-41ff-aa5c-1b08c60c32a9.JPEG' height="auto" width="160" ></div> | <div align=center><img src='https://user-images.githubusercontent.com/18586273/147065002-f1c86516-38b2-47ba-90c1-e00b49556c70.jpg' height="auto" width="150" ></div> | <div align=center><img src='https://user-images.githubusercontent.com/18586273/147065119-82581fa1-3414-4d6c-a849-804e1503c74b.jpg' height="auto" width="150"></div> | <div align=center><img src='https://user-images.githubusercontent.com/18586273/147065096-75a6a2c1-6c57-4789-ad64-ebe5e38765f4.jpg' height="auto" width="150"></div> | <div align=center><img src='https://user-images.githubusercontent.com/18586273/147065129-814d20fb-98be-4106-8c5e-420adcc85295.jpg' height="auto" width="150"></div> |

2. Use different `target-category` to get CAM from the same picture. In `ImageNet` dataset, the category 238 is 'Greater Swiss Mountain dog', the category 281 is 'tabby, tabby cat'.

   ```shell
   python tools/visualization/vis_cam.py \
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

3. Use `--eigen-smooth` and `--aug-smooth` to improve visual effects.

   ```shell
   python tools/visualization/vis_cam.py \
       demo/dog.jpg  \
       configs/mobilenet_v3/mobilenet-v3-large_8xb128_in1k.py \
       https://download.openmmlab.com/mmclassification/v0/mobilenet_v3/convert/mobilenet_v3_large-3ea3c186.pth \
       --target-layers 'backbone.layer16' \
       --method LayerCAM \
       --eigen-smooth --aug-smooth
   ```

   | Image                                | LayerCAM                                | eigen-smooth                                | aug-smooth                                | eigen&aug                                 |
   | ------------------------------------ | --------------------------------------- | ------------------------------------------- | ----------------------------------------- | ----------------------------------------- |
   | <div align=center><img src='https://user-images.githubusercontent.com/18586273/144557492-98ac5ce0-61f9-4da9-8ea7-396d0b6a20fa.jpg' height="auto" width="160"></div> | <div align=center><img src='https://user-images.githubusercontent.com/18586273/144557541-a4cf7d86-7267-46f9-937c-6f657ea661b4.jpg'  height="auto" width="145" ></div> | <div align=center><img src='https://user-images.githubusercontent.com/18586273/144557547-2731b53e-e997-4dd2-a092-64739cc91959.jpg'  height="auto" width="145" ></div> | <div align=center><img src='https://user-images.githubusercontent.com/18586273/144557545-8189524a-eb92-4cce-bf6a-760cab4a8065.jpg'  height="auto" width="145" ></div> | <div align=center><img src='https://user-images.githubusercontent.com/18586273/144557548-c1e3f3ec-3c96-43d4-874a-3b33cd3351c5.jpg'  height="auto" width="145" ></div> |

## How to visualize the CAM of vision transformer

Here are some examples:

- `'backbone.norm3'` for Swin-Transformer;
- `'backbone.layers.11.ln1'` for ViT;

For ViT-like networks, such as ViT, T2T-ViT and Swin-Transformer, the features are flattened. And for drawing the CAM, we need to specify the `--vit-like` argument to reshape the features into square feature maps.

Besides the flattened features, some ViT-like networks also add extra tokens like the class token in ViT and T2T-ViT, and the distillation token in DeiT. In these networks, the final classification is done on the tokens computed in the last attention block, and therefore, the classification score will not be affected by other features and the gradient of the classification score with respect to them, will be zero. Therefore, you shouldn't use the output of the last attention block as the target layer in these networks.

To exclude these extra tokens, we need know the number of extra tokens. Almost all transformer-based backbones in MMPretrain have the `num_extra_tokens` attribute. If you want to use this tool in a new or third-party network that don't have the `num_extra_tokens` attribute, please specify it the `--num-extra-tokens` argument.

1. Visualize CAM for `Swin Transformer`, using default `target-layers`:

   ```shell
   python tools/visualization/vis_cam.py \
       demo/bird.JPEG  \
       configs/swin_transformer/swin-tiny_16xb64_in1k.py \
       https://download.openmmlab.com/mmclassification/v0/swin-transformer/swin_tiny_224_b16x64_300e_imagenet_20210616_090925-66df6be6.pth \
       --vit-like
   ```

2. Visualize CAM for `Vision Transformer(ViT)`:

   ```shell
   python tools/visualization/vis_cam.py \
       demo/bird.JPEG  \
       configs/vision_transformer/vit-base-p16_64xb64_in1k-384px.py \
       https://download.openmmlab.com/mmclassification/v0/vit/finetune/vit-base-p16_in21k-pre-3rdparty_ft-64xb64_in1k-384_20210928-98e8652b.pth \
       --vit-like \
       --target-layers 'backbone.layers.11.ln1'
   ```

3. Visualize CAM for `T2T-ViT`:

   ```shell
   python tools/visualization/vis_cam.py \
       demo/bird.JPEG  \
       configs/t2t_vit/t2t-vit-t-14_8xb64_in1k.py \
       https://download.openmmlab.com/mmclassification/v0/t2t-vit/t2t-vit-t-14_3rdparty_8xb64_in1k_20210928-b7c09b62.pth \
       --vit-like \
       --target-layers 'backbone.encoder.12.ln1'
   ```

| Image                                   | ResNet50                                   | ViT                                    | Swin                                    | T2T-ViT                                    |
| --------------------------------------- | ------------------------------------------ | -------------------------------------- | --------------------------------------- | ------------------------------------------ |
| <div align=center><img src='https://user-images.githubusercontent.com/18586273/144429496-628d3fb3-1f6e-41ff-aa5c-1b08c60c32a9.JPEG' height="auto" width="165" ></div> | <div align=center><img src=https://user-images.githubusercontent.com/18586273/144431491-a2e19fe3-5c12-4404-b2af-a9552f5a95d9.jpg  height="auto" width="150" ></div> | <div align=center><img src='https://user-images.githubusercontent.com/18586273/144436218-245a11de-6234-4852-9c08-ff5069f6a739.jpg' height="auto" width="150" ></div> | <div align=center><img src='https://user-images.githubusercontent.com/18586273/144436168-01b0e565-442c-4e1e-910c-17c62cff7cd3.jpg' height="auto" width="150" ></div> | <div align=center><img src='https://user-images.githubusercontent.com/18586273/144436198-51dbfbda-c48d-48cc-ae06-1a923d19b6f6.jpg' height="auto" width="150" ></div> |
