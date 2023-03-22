# Convention in MMPretrain

## Model Naming Convention

We follow the below convention to name models. Contributors are advised to follow the same style. The model names are divided into five parts: algorithm info, module information, pretrain information, training information and data information. Logically, different parts are concatenated by underscores `'_'`, and words in the same part are concatenated by dashes `'-'`.

```text
{algorithm info}_{module info}_{pretrain info}_{training info}_{data info}
```

- `algorithm info` (optional): The main algorithm information, it's includes the main training algorithms like MAE, BEiT, etc.
- `module info`:  The module information, it usually includes the backbone name, such as resnet, vit, etc.
- `pretrain info`: (optional): The pretrain model information, such as the pretrain model is trained on ImageNet-21k.
- `training info`: The training information, some training schedule, including batch size, lr schedule, data augment and the like.
- `data info`: The data information, it usually includes the dataset name, input size and so on, such as imagenet, cifar, etc.

### Algorithm information

The main algorithm name to train the model. For example:

- `simclr`
- `mocov2`
- `eva-mae-style`

The model trained by supervised image classification can omit this field.

### Module information

The modules of the model, usually, the backbone must be included in this field, and the neck and head
information can be omitted. For example:

- `resnet50`
- `vit-base-p16`
- `swin-base`

### Pretrain information

If the model is a fine-tuned model from a pre-trained model, we need to record some information of the
pre-trained model. For example:

- The source of the pre-trained model: `fb`, `openai`, etc.
- The method to train the pre-trained model: `clip`, `mae`, `distill`, etc.
- The dataset used for pre-training: `in21k`, `laion2b`, etc. (`in1k` can be omitted.)
- The training duration: `300e`, `1600e`, etc.

Not all information is necessary, only select the necessary information to distinguish different pre-trained
models.

At the end of this field, use a `-pre` as an identifier, like `mae-in21k-pre`.

### Training information

Training schedule, including training type, `batch size`, `lr schedule`, data augment, special loss functions and so on:

- format `{gpu x batch_per_gpu}`, such as `8xb32`

Training type (mainly seen in the transformer network, such as the `ViT` algorithm, which is usually divided into two training type: pre-training and fine-tuning):

- `ft` : configuration file for fine-tuning
- `pt` : configuration file for pretraining

Training recipe. Usually, only the part that is different from the original paper will be marked. These methods will be arranged in the order `{pipeline aug}-{train aug}-{loss trick}-{scheduler}-{epochs}`.

- `coslr-200e` : use cosine scheduler to train 200 epochs
- `autoaug-mixup-lbs-coslr-50e` : use `autoaug`, `mixup`, `label smooth`, `cosine scheduler` to train 50 epochs

If the model is converted from a third-party repository like the official repository, the training information
can be omitted and use a `3rdparty` as an identifier.

### Data information

- `in1k` : `ImageNet1k` dataset, default to use the input image size of 224x224;
- `in21k` : `ImageNet21k` dataset, also called `ImageNet22k` dataset, default to use the input image size of 224x224;
- `in1k-384px` : Indicates that the input image size is 384x384;
- `cifar100`

### Model Name Example

```text
vit-base-p32_clip-openai-pre_3rdparty_in1k
```

- `vit-base-p32`: The module information
- `clip-openai-pre`: The pre-train information.
  - `clip`: The pre-train method is clip.
  - `openai`: The pre-trained model is come from OpenAI.
  - `pre`: The pre-train identifier.
- `3rdparty`: The model is converted from a third-party repository.
- `in1k`: Dataset information. The model is trained from ImageNet-1k dataset and the input size is `224x224`.

```text
beit_beit-base-p16_8xb256-amp-coslr-300e_in1k
```

- `beit`: The algorithm information
- `beit-base`: The module information, since the backbone is a modified ViT from BEiT, the backbone name is
  also `beit`.
- `8xb256-amp-coslr-300e`: The training information.
  - `8xb256`: Use 8 GPUs and the batch size on each GPU is 256.
  - `amp`: Use automatic-mixed-precision training.
  - `coslr`: Use cosine annealing learning rate scheduler.
  - `300e`: To train 300 epochs.
- `in1k`: Dataset information. The model is trained from ImageNet-1k dataset and the input size is `224x224`.

## Config File Naming Convention

The naming of the config file is almost the same with the model name, with several difference:

- The training information is necessary, and cannot be `3rdparty`.
- If the config file only includes backbone settings, without neither head settings nor dataset settings. We
  will name it as `{module info}_headless.py`. This kind of config files are usually used for third-party
  pre-trained models on large datasets.

## Checkpoint Naming Convention

The naming of the weight mainly includes the model name, date and hash value.

```text
{model_name}_{date}-{hash}.pth
```
