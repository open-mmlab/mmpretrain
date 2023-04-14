# 如何在自定义数据集上微调模型

在很多场景下，我们需要快速地将模型应用到新的数据集上，但从头训练模型通常很难快速收敛，这种不确定性会浪费额外的时间。
通常，已有的、在大数据集上训练好的模型会比随机初始化提供更为有效的先验信息，粗略来讲，在此基础上的学习我们称之为模型微调。

已经证明，在 ImageNet 数据集上预训练的模型对于其他数据集和其他下游任务有很好的效果。
因此，该教程提供了如何将 [Model Zoo](../modelzoo_statistics.md) 中提供的预训练模型用于其他数据集，已获得更好的效果。

在本教程中，我们提供了一个实践示例和一些关于如何在自己的数据集上微调模型的技巧。

## 第一步：准备你的数据集

按照 [准备数据集](../user_guides/dataset_prepare.md) 准备你的数据集。
假设我们的数据集根文件夹路径为 `data/custom_dataset/`

假设我们想进行有监督图像分类训练，并使用子文件夹格式的 `CustomDataset` 来组织数据集：

```text
data/custom_dataset/
├── train
│   ├── class_x
│   │   ├── x_1.png
│   │   ├── x_2.png
│   │   ├── x_3.png
│   │   └── ...
│   ├── class_y
│   └── ...
└── test
    ├── class_x
    │   ├── test_x_1.png
    │   ├── test_x_2.png
    │   ├── test_x_3.png
    │   └── ...
    ├── class_y
    └── ...
```

## 第二步：选择一个配置文件作为模板

在这里，我们使用 `configs/resnet/resnet50_8xb32_in1k.py` 作为示例。
首先在同一文件夹下复制一份配置文件，并将其重命名为 `resnet50_8xb32-ft_custom.py`。

```{tip}
按照惯例，配置名称的最后一个字段是数据集，例如，`in1k` 表示 ImageNet-1k，`coco` 表示 coco 数据集
```

这个配置的内容是：

```python
_base_ = [
    '../_base_/models/resnet50.py',           # 模型设置
    '../_base_/datasets/imagenet_bs32.py',    # 数据设置
    '../_base_/schedules/imagenet_bs256.py',  # 训练策略设置
    '../_base_/default_runtime.py',           # 运行设置
]
```

## 第三步：修改模型设置

在进行模型微调时，我们通常希望在主干网络（backbone）加载预训练模型，再用我们的数据集训练一个新的分类头（head）。

为了在主干网络加载预训练模型，我们需要修改主干网络的初始化设置，使用
`Pretrained` 类型的初始化函数。另外，在初始化设置中，我们使用 `prefix='backbone'`
来告诉初始化函数需要加载的子模块的前缀，`backbone`即指加载模型中的主干网络。
方便起见，我们这里使用一个在线的权重文件链接，它
会在训练前自动下载对应的文件，你也可以提前下载这个模型，然后使用本地路径。

接下来，新的配置文件需要按照新数据集的类别数目来修改分类头的配置。只需要修改分
类头中的 `num_classes` 设置即可。

另外，当新的小数据集和原本预训练的大数据集中的数据分布较为类似的话，我们在进行微调时会希望
冻结主干网络前面几层的参数，只训练后面层以及分类头的参数，这么做有助于在后续训练中，
保持网络从预训练权重中获得的提取低阶特征的能力。在 MMPretrain 中，
这一功能可以通过简单的一个 `frozen_stages` 参数来实现。比如我们需要冻结前两层网
络的参数，只需要在上面的配置中添加一行：

```{note}
注意，目前并非所有的主干网络都支持 `frozen_stages` 参数。请检查[文档](https://mmpretrain.readthedocs.io/en/latest/api.html#module-mmpretrain.models.backbones)
确认使用的主干网络是否支持这一参数。
```

```python
_base_ = [
    '../_base_/models/resnet50.py',           # 模型设置
    '../_base_/datasets/imagenet_bs32.py',    # 数据设置
    '../_base_/schedules/imagenet_bs256.py',  # 训练策略设置
    '../_base_/default_runtime.py',           # 运行设置
]

# >>>>>>>>>>>>>>> 在这里重载模型相关配置 >>>>>>>>>>>>>>>>>>>
model = dict(
    backbone=dict(
        frozen_stages=2,
        init_cfg=dict(
            type='Pretrained',
            checkpoint='https://download.openmmlab.com/mmclassification/v0/resnet/resnet50_8xb32_in1k_20210831-ea4938fc.pth',
            prefix='backbone',
        )),
    head=dict(num_classes=10),
)
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
```

```{tip}
这里我们只需要设定我们想要修改的部分配置，其他配置将会自动从我们的基配置文件中获取。
```

## 第四步：修改数据集设置

为了在新数据集上进行微调，我们需要覆盖一些数据集设置，例如数据集类型、数据流水线等。

```python
_base_ = [
    '../_base_/models/resnet50.py',           # 模型设置
    '../_base_/datasets/imagenet_bs32.py',    # 数据设置
    '../_base_/schedules/imagenet_bs256.py',  # 训练策略设置
    '../_base_/default_runtime.py',           # 运行设置
]

# 模型设置
model = dict(
    backbone=dict(
        frozen_stages=2,
        init_cfg=dict(
            type='Pretrained',
            checkpoint='https://download.openmmlab.com/mmclassification/v0/resnet/resnet50_8xb32_in1k_20210831-ea4938fc.pth',
            prefix='backbone',
        )),
    head=dict(num_classes=10),
)

# >>>>>>>>>>>>>>> 在这里重载数据配置 >>>>>>>>>>>>>>>>>>>
data_root = 'data/custom_dataset'
train_dataloader = dict(
    dataset=dict(
        type='CustomDataset',
        data_root=data_root,
        ann_file='',       # 我们假定使用子文件夹格式，因此需要将标注文件置空
        data_prefix='train',
    ))
val_dataloader = dict(
    dataset=dict(
        type='CustomDataset',
        data_root=data_root,
        ann_file='',       # 我们假定使用子文件夹格式，因此需要将标注文件置空
        data_prefix='test',
    ))
test_dataloader = val_dataloader
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
```

## 第五步：修改训练策略设置（可选）

微调所使用的训练超参数一般与默认的超参数不同，它通常需要更小的学习率和更快的学习率衰减。

```python
_base_ = [
    '../_base_/models/resnet50.py',           # 模型设置
    '../_base_/datasets/imagenet_bs32.py',    # 数据设置
    '../_base_/schedules/imagenet_bs256.py',  # 训练策略设置
    '../_base_/default_runtime.py',           # 运行设置
]

# 模型设置
model = dict(
    backbone=dict(
        frozen_stages=2,
        init_cfg=dict(
            type='Pretrained',
            checkpoint='https://download.openmmlab.com/mmclassification/v0/resnet/resnet50_8xb32_in1k_20210831-ea4938fc.pth',
            prefix='backbone',
        )),
    head=dict(num_classes=10),
)

# 数据设置
data_root = 'data/custom_dataset'
train_dataloader = dict(
    dataset=dict(
        type='CustomDataset',
        data_root=data_root,
        ann_file='',
        data_prefix='train',
    ))
val_dataloader = dict(
    dataset=dict(
        type='CustomDataset',
        data_root=data_root,
        ann_file='',
        data_prefix='test',
    ))
test_dataloader = val_dataloader

# >>>>>>>>>>>>>>> 在这里重载训练策略设置 >>>>>>>>>>>>>>>>>>>
# 优化器超参数
optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001))
# 学习率策略
param_scheduler = dict(
    type='MultiStepLR', by_epoch=True, milestones=[15], gamma=0.1)
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
```

```{tip}
更多关于配置文件的信息，请参阅[学习配置文件](../user_guides/config.md)
```

## 开始训练

现在，我们完成了用于微调的配置文件，完整的文件如下：

```python
_base_ = [
    '../_base_/models/resnet50.py',           # 模型设置
    '../_base_/datasets/imagenet_bs32.py',    # 数据设置
    '../_base_/schedules/imagenet_bs256.py',  # 训练策略设置
    '../_base_/default_runtime.py',           # 运行设置
]

# 模型设置
model = dict(
    backbone=dict(
        frozen_stages=2,
        init_cfg=dict(
            type='Pretrained',
            checkpoint='https://download.openmmlab.com/mmclassification/v0/resnet/resnet50_8xb32_in1k_20210831-ea4938fc.pth',
            prefix='backbone',
        )),
    head=dict(num_classes=10),
)

# 数据设置
data_root = 'data/custom_dataset'
train_dataloader = dict(
    dataset=dict(
        type='CustomDataset',
        data_root=data_root,
        ann_file='',
        data_prefix='train',
    ))
val_dataloader = dict(
    dataset=dict(
        type='CustomDataset',
        data_root=data_root,
        ann_file='',
        data_prefix='test',
    ))
test_dataloader = val_dataloader

# 训练策略设置
optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001))
param_scheduler = dict(
    type='MultiStepLR', by_epoch=True, milestones=[15], gamma=0.1)
```

接下来，我们使用一台 8 张 GPU 的电脑来训练我们的模型，指令如下：

```shell
bash tools/dist_train.sh configs/resnet/resnet50_8xb32-ft_custom.py 8
```

当然，我们也可以使用单张 GPU 来进行训练，使用如下命令：

```shell
python tools/train.py configs/resnet/resnet50_8xb32-ft_custom.py
```

但是如果我们使用单张 GPU 进行训练的话，需要在数据集设置部分作如下修改：

```python
data_root = 'data/custom_dataset'
train_dataloader = dict(
    batch_size=256,
    dataset=dict(
        type='CustomDataset',
        data_root=data_root,
        ann_file='',
        data_prefix='train',
    ))
val_dataloader = dict(
    dataset=dict(
        type='CustomDataset',
        data_root=data_root,
        ann_file='',
        data_prefix='test',
    ))
test_dataloader = val_dataloader
```

这是因为我们的训练策略是针对批次大小（batch size）为 256 设置的。在父配置文件中，
设置了单张 `batch_size=32`，如果使用 8 张 GPU，总的批次大小就是 256。而如果使
用单张 GPU，就必须手动修改 `batch_size=256` 来匹配训练策略。

然而，更大的批次大小需要更大的 GPU 显存，这里有几个简单的技巧来节省显存：

1. 启用自动混合精度训练

   ```shell
   python tools/train.py configs/resnet/resnet50_8xb32-ft_custom.py --amp
   ```

2. 使用较小的批次大小，例如仍然使用 `batch_size=32`，而不是 256，并启用学习率自动缩放

   ```shell
   python tools/train.py configs/resnet/resnet50_8xb32-ft_custom.py --auto-scale-lr
   ```

   学习率自动缩放功能会根据实际的 batch size 和配置文件中的 `auto_scale_lr.base_batch_size`
   字段对学习率进行线性调整（你可以在基配置文件 `configs/_base_/schedules/imagenet_bs256.py`
   中找到这一字段）

```{note}
以上技巧都有可能对训练效果造成轻微影响。
```

### 在命令行指定预训练模型

如果您不想修改配置文件，您可以使用 `--cfg-options` 将您的预训练模型文件添加到 `init_cfg`.

例如，以下命令也会加载预训练模型：

```shell
bash tools/dist_train.sh configs/tutorial/resnet50_finetune_cifar.py 8 \
    --cfg-options model.backbone.init_cfg.type='Pretrained' \
    model.backbone.init_cfg.checkpoint='https://download.openmmlab.com/mmselfsup/1.x/mocov3/mocov3_resnet50_8xb512-amp-coslr-100e_in1k/mocov3_resnet50_8xb512-amp-coslr-100e_in1k_20220927-f1144efa.pth' \
    model.backbone.init_cfg.prefix='backbone' \
```
