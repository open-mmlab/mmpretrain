# MMCLS 中的约定

## 配置文件命名规则

MMClassification 按照以下风格进行配置文件命名，代码库的贡献者需要遵循相同的命名规则。文件名总体分为四部分：算法信息，模块信息，训练信息和数据信息。逻辑上属于不同部分的单词之间用下划线 `'_'` 连接，同一部分有多个单词用短横线 `'-'` 连接。

```text
{algorithm info}_{module info}_{training info}_{data info}.py
```

- `algorithm info`：算法信息，算法名称或者网络架构，如 resnet 等；
- `module info`： 模块信息，因任务而异，用以表示一些特殊的 neck、head 和 pretrain 信息；
- `training info`：一些训练信息，训练策略设置，包括 batch size，schedule 以及数据增强等；
- `data info`：数据信息，数据集名称、模态、输入尺寸等，如 imagenet, cifar 等；

### 算法信息

指论文中的算法名称缩写，以及相应的分支架构信息。例如：

- `resnet50`
- `mobilenet-v3-large`
- `vit-small-patch32`   : `patch32` 表示 `ViT` 切分的分块大小
- `seresnext101-32x4d`  : `SeResNet101` 基本网络结构，`32x4d` 表示在 `Bottleneck` 中  `groups` 和 `width_per_group` 分别为32和4

### 模块信息

指一些特殊的 `neck` 、`head` 或者 `pretrain` 的信息， 在分类中常见为预训练信息，比如：

- `in21k-pre` : 在 `ImageNet21k` 上预训练
- `in21k-pre-3rd-party` : 在 `ImageNet21k` 上预训练，其权重来自其他仓库

### 训练信息

训练策略的一些设置，包括训练类型、 `batch size`、 `lr schedule`、 数据增强以及特殊的损失函数等等,比如:
Batch size 信息：

- 格式为`{gpu x batch_per_gpu}`, 如 `8xb32`

训练类型(主要见于 transformer 网络，如 `ViT` 算法，这类算法通常分为预训练和微调两种模式):

- `ft` : Finetune config，用于微调的配置文件
- `pt` : Pretrain config，用于预训练的配置文件

训练策略信息，训练策略以复现配置文件为基础，此基础不必标注训练策略。但如果在此基础上进行改进，则需注明训练策略，按照应用点位顺序排列，如：`{pipeline aug}-{train aug}-{loss trick}-{scheduler}-{epochs}`

- `coslr-200e` : 使用 cosine scheduler, 训练 200 个 epoch
- `autoaug-mixup-lbs-coslr-50e` : 使用了 `autoaug`、`mixup`、`label smooth`、`cosine scheduler`, 训练了 50 个轮次

### 数据信息

- `in1k` : `ImageNet1k` 数据集，默认使用 `224x224` 大小的图片
- `in21k` : `ImageNet21k` 数据集，有些地方也称为 `ImageNet22k` 数据集，默认使用 `224x224` 大小的图片
- `in1k-384px` : 表示训练的输出图片大小为 `384x384`
- `cifar100`

### 配置文件命名案例

```text
repvgg-D2se_deploy_4xb64-autoaug-lbs-mixup-coslr-200e_in1k.py
```

- `repvgg-D2se`:  算法信息
  - `repvgg`: 主要算法名称。
  - `D2se`: 模型的结构。
- `deploy`:模块信息，该模型为推理状态。
- `4xb64-autoaug-lbs-mixup-coslr-200e`: 训练信息
  - `4xb64`: 使用4块 GPU 并且 每块 GPU 的批大小为64。
  - `autoaug`: 使用 `AutoAugment` 数据增强方法。
  - `lbs`: 使用 `label smoothing` 损失函数。
  - `mixup`: 使用 `mixup` 训练增强方法。
  - `coslr`: 使用 `cosine scheduler` 优化策略。
  - `200e`: 训练 200 轮次。
- `in1k`: 数据信息。 配置文件用于 `ImageNet1k` 数据集上使用 `224x224` 大小图片训练。

### 权重命名规则

权重的命名主要包括配置文件名，日期和哈希值。

```text
{config_name}_{date}-{hash}.pth
```
