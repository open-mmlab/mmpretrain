# MMPretrain 中的约定

## 模型命名规则

MMPretrain 按照以下风格进行模型命名，代码库的贡献者需要遵循相同的命名规则。模型名总体分为五个部分：算法信息，模块信息，预训练信息，训练信息和数据信息。逻辑上属于不同部分的单词之间用下划线 `'_'` 连接，同一部分有多个单词用短横线 `'-'` 连接。

```text
{algorithm info}_{module info}_{pretrain info}_{training info}_{data info}
```

- `algorithm info`（可选）：算法信息，表示用以训练该模型的主要算法，如 MAE、BEiT 等
- `module info`：模块信息，主要包含模型的主干网络名称，如 resnet、vit 等
- `pretrain info`（可选）：预训练信息，比如预训练模型是在 ImageNet-21k 数据集上训练的等
- `training info`：训练信息，训练策略设置，包括 batch size，schedule 以及数据增强等；
- `data info`：数据信息，数据集名称、模态、输入尺寸等，如 imagenet, cifar 等；

### 算法信息

指用以训练该模型的算法名称，例如：

- `simclr`
- `mocov2`
- `eva-mae-style`

使用监督图像分类任务训练的模型可以省略这个字段。

### 模块信息

指模型的结构信息，一般主要包含模型的主干网络结构，`neck` 和 `head` 信息一般被省略。例如：

- `resnet50`
- `vit-base-p16`
- `swin-base`

### 预训练信息

如果该模型是在预训练模型基础上，通过微调获得的，我们需要记录预训练模型的一些信息。例如：

- 预训练模型的来源：`fb`、`openai`等。
- 训练预训练模型的方法：`clip`、`mae`、`distill` 等。
- 用于预训练的数据集：`in21k`、`laion2b`等（`in1k`可以省略）
- 训练时长：`300e`、`1600e` 等。

并非所有信息都是必要的，只需要选择用以区分不同的预训练模型的信息即可。

在此字段的末尾，使用 `-pre` 作为标识符，例如 `mae-in21k-pre`。

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

如果模型是从官方仓库等第三方仓库转换过来的，训练信息可以省略，使用 `3rdparty` 作为标识符。

### 数据信息

- `in1k` : `ImageNet1k` 数据集，默认使用 `224x224` 大小的图片
- `in21k` : `ImageNet21k` 数据集，有些地方也称为 `ImageNet22k` 数据集，默认使用 `224x224` 大小的图片
- `in1k-384px` : 表示训练的输出图片大小为 `384x384`
- `cifar100`

### 模型命名案例

```text
vit-base-p32_clip-openai-pre_3rdparty_in1k
```

- `vit-base-p32`: 模块信息
- `clip-openai-pre`：预训练信息
  - `clip`：预训练方法是 clip
  - `openai`：预训练模型来自 OpenAI
  - `pre`：预训练标识符
- `3rdparty`：模型是从第三方仓库转换而来的
- `in1k`：数据集信息。该模型是从 ImageNet-1k 数据集训练而来的，输入大小为 `224x224`

```text
beit_beit-base-p16_8xb256-amp-coslr-300e_in1k
```

- `beit`: 算法信息
- `beit-base`：模块信息，由于主干网络来自 BEiT 中提出的修改版 ViT，主干网络名称也是 `beit`
- `8xb256-amp-coslr-300e`：训练信息
  - `8xb256`：使用 8 个 GPU，每个 GPU 的批量大小为 256
  - `amp`：使用自动混合精度训练
  - `coslr`：使用余弦退火学习率调度器
  - `300e`：训练 300 个 epoch
- `in1k`：数据集信息。该模型是从 ImageNet-1k 数据集训练而来的，输入大小为 `224x224`

## 配置文件命名规则

配置文件的命名与模型名称几乎相同，有几点不同：

- 训练信息是必要的，不能是 `3rdparty`
- 如果配置文件只包含主干网络设置，既没有头部设置也没有数据集设置，我们将其命名为`{module info}_headless.py`。这种配置文件通常用于大型数据集上的第三方预训练模型。

### 权重命名规则

权重的命名主要包括模型名称，日期和哈希值。

```text
{model_name}_{date}-{hash}.pth
```
