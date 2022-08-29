# Convention in MMCLS

## Config files Naming Convention

We follow the below convention to name config files. Contributors are advised to follow the same style. The config file names are divided into four parts: algorithm info, module information, training information and data information. Logically, different parts are concatenated by underscores `'_'`, and words in the same part are concatenated by dashes `'-'`.

```text
{algorithm info}_{module info}_{training info}_{data info}.py
```

- `algorithm info`：algorithm information, model name and neural network architecture, such as resnet, etc.;
- `module info`： module information is used to represent some special neck, head and pretrain information;
- `training info`：Training information, some training schedule, including batch size, lr schedule, data augment and the like;
- `data info`：Data information, dataset name, input size and so on, such as imagenet, cifar, etc.;

### Algorithm information

The main algorithm name and the corresponding branch architecture information. E.g：

- `resnet50`
- `mobilenet-v3-large`
- `vit-small-patch32`   : `patch32` represents the size of the partition in `ViT` algorithm;
- `seresnext101-32x4d`  : `SeResNet101` network structure, `32x4d` means that `groups` and `width_per_group` are 32 and 4 respectively in `Bottleneck`;

### Module information

Some special `neck`, `head` and `pretrain` information. In classification tasks, `pretrain` information is the most commonly used:

- `in21k-pre` : pre-trained on ImageNet21k;
- `in21k-pre-3rd-party` : pre-trained on ImageNet21k and the checkpoint is converted from a third-party repository;

### Training information

Training schedule, including training type, `batch size`, `lr schedule`, data augment, special loss functions and so on:

- format `{gpu x batch_per_gpu}`, such as `8xb32`

Training type (mainly seen in the transformer network, such as the `ViT` algorithm, which is usually divided into two training type: pre-training and fine-tuning):

- `ft` : configuration file for fine-tuning
- `pt` : configuration file for pretraining

Training recipe. Usually, only the part that is different from the original paper will be marked. These methods will be arranged in the order `{pipeline aug}-{train aug}-{loss trick}-{scheduler}-{epochs}`.

- `coslr-200e` : use cosine scheduler to train 200 epochs
- `autoaug-mixup-lbs-coslr-50e` : use `autoaug`, `mixup`, `label smooth`, `cosine scheduler` to train 50 epochs

### Data information

- `in1k` : `ImageNet1k` dataset, default to use the input image size of 224x224;
- `in21k` : `ImageNet21k` dataset, also called `ImageNet22k` dataset, default to use the input image size of 224x224;
- `in1k-384px` : Indicates that the input image size is 384x384;
- `cifar100`

### Config File Name Example

```text
repvgg-D2se_deploy_4xb64-autoaug-lbs-mixup-coslr-200e_in1k.py
```

- `repvgg-D2se`:  Algorithm information
  - `repvgg`: The main algorithm.
  - `D2se`: The architecture.
- `deploy`: Module information, means the backbone is in the deploy state.
- `4xb64-autoaug-lbs-mixup-coslr-200e`: Training information.
  - `4xb64`: Use 4 GPUs and the size of batches per GPU is 64.
  - `autoaug`: Use `AutoAugment` in training pipeline.
  - `lbs`: Use label smoothing loss.
  - `mixup`: Use `mixup` training augment method.
  - `coslr`: Use cosine learning rate scheduler.
  - `200e`: Train the model for 200 epochs.
- `in1k`: Dataset information. The config is for `ImageNet1k` dataset and the input size is `224x224`.

## Checkpoint Naming Convention

The naming of the weight mainly includes the configuration file name, date and hash value.

```text
{config_name}_{date}-{hash}.pth
```
