# 教程 5：如何增加新模块

## 开发新组件

我们基本上将模型组件分为 3 种类型。

- 主干网络：通常是一个特征提取网络，例如 ResNet、MobileNet
- 颈部：用于连接主干网络和头部的组件，例如 GlobalAveragePooling
- 头部：用于执行特定任务的组件，例如分类和回归

### 添加新的主干网络

这里，我们以 ResNet_CIFAR 为例，展示了如何开发一个新的主干网络组件。

ResNet_CIFAR 针对 CIFAR 32x32 的图像输入，将 ResNet 中 `kernel_size=7,
stride=2` 的设置替换为 `kernel_size=3, stride=1`，并移除了 stem 层之后的
`MaxPooling`，以避免传递过小的特征图到残差块中。

它继承自 `ResNet` 并只修改了 stem 层。

1. 创建一个新文件 `mmcls/models/backbones/resnet_cifar.py`。

```python
import torch.nn as nn

from ..builder import BACKBONES
from .resnet import ResNet


@BACKBONES.register_module()
class ResNet_CIFAR(ResNet):

    """ResNet backbone for CIFAR.

    （对这个主干网络的简短描述）

    Args:
        depth(int): Network depth, from {18, 34, 50, 101, 152}.
        ...
        （参数文档）
    """

    def __init__(self, depth, deep_stem=False, **kwargs):
        # 调用基类 ResNet 的初始化函数
        super(ResNet_CIFAR, self).__init__(depth, deep_stem=deep_stem **kwargs)
        # 其他特殊的初始化流程
        assert not self.deep_stem, 'ResNet_CIFAR do not support deep_stem'

    def _make_stem_layer(self, in_channels, base_channels):
        # 重载基类的方法，以实现对网络结构的修改
        self.conv1 = build_conv_layer(
            self.conv_cfg,
            in_channels,
            base_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False)
        self.norm1_name, norm1 = build_norm_layer(
            self.norm_cfg, base_channels, postfix=1)
        self.add_module(self.norm1_name, norm1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):  # 需要返回一个元组
        pass  # 此处省略了网络的前向实现

    def init_weights(self, pretrained=None):
        pass  # 如果有必要的话，重载基类 ResNet 的参数初始化函数

    def train(self, mode=True):
        pass  # 如果有必要的话，重载基类 ResNet 的训练状态函数
```

2. 在 `mmcls/models/backbones/__init__.py` 中导入新模块

```python
...
from .resnet_cifar import ResNet_CIFAR

__all__ = [
    ..., 'ResNet_CIFAR'
]
```

3. 在配置文件中使用新的主干网络

```python
model = dict(
    ...
    backbone=dict(
        type='ResNet_CIFAR',
        depth=18,
        other_arg=xxx),
    ...
```

### 添加新的颈部组件

这里我们以 `GlobalAveragePooling` 为例。这是一个非常简单的颈部组件，没有任何参数。

要添加新的颈部组件，我们主要需要实现 `forward` 函数，该函数对主干网络的输出进行
一些操作并将结果传递到头部。

1. 创建一个新文件 `mmcls/models/necks/gap.py`

    ```python
    import torch.nn as nn

    from ..builder import NECKS

    @NECKS.register_module()
    class GlobalAveragePooling(nn.Module):

        def __init__(self):
            self.gap = nn.AdaptiveAvgPool2d((1, 1))

        def forward(self, inputs):
            # 简单起见，我们默认输入是一个张量
            outs = self.gap(inputs)
            outs = outs.view(inputs.size(0), -1)
            return outs
    ```

2. 在 `mmcls/models/necks/__init__.py` 中导入新模块

    ```python
    ...
    from .gap import GlobalAveragePooling

    __all__ = [
        ..., 'GlobalAveragePooling'
    ]
    ```

3. 修改配置文件以使用新的颈部组件

    ```python
    model = dict(
        neck=dict(type='GlobalAveragePooling'),
    )
    ```

### 添加新的头部组件

在此，我们以 `LinearClsHead` 为例，说明如何开发新的头部组件。

要添加一个新的头部组件，基本上我们需要实现 `forward_train` 函数，它接受来自颈部
或主干网络的特征图作为输入，并基于真实标签计算。

1. 创建一个文件 `mmcls/models/heads/linear_head.py`.

    ```python
    from ..builder import HEADS
    from .cls_head import ClsHead


    @HEADS.register_module()
    class LinearClsHead(ClsHead):

        def __init__(self,
                  num_classes,
                  in_channels,
                  loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
                  topk=(1, )):
            super(LinearClsHead, self).__init__(loss=loss, topk=topk)
            self.in_channels = in_channels
            self.num_classes = num_classes

            if self.num_classes <= 0:
                raise ValueError(
                    f'num_classes={num_classes} must be a positive integer')

            self._init_layers()

        def _init_layers(self):
            self.fc = nn.Linear(self.in_channels, self.num_classes)

        def init_weights(self):
            normal_init(self.fc, mean=0, std=0.01, bias=0)

        def forward_train(self, x, gt_label):
            cls_score = self.fc(x)
            losses = self.loss(cls_score, gt_label)
            return losses

    ```

2. 在 `mmcls/models/heads/__init__.py` 中导入这个模块

    ```python
    ...
    from .linear_head import LinearClsHead

    __all__ = [
        ..., 'LinearClsHead'
    ]
    ```

3. 修改配置文件以使用新的头部组件。

连同 `GlobalAveragePooling` 颈部组件，完整的模型配置如下：

```python
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(3, ),
        style='pytorch'),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=1000,
        in_channels=2048,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5),
    ))

```

### 添加新的损失函数

要添加新的损失函数，我们主要需要在损失函数模块中 `forward` 函数。另外，利用装饰器 `weighted_loss` 可以方便的实现对每个元素的损失进行加权平均。

假设我们要模拟从另一个分类模型生成的概率分布，需要添加 `L1loss` 来实现该目的。

1. 创建一个新文件 `mmcls/models/losses/l1_loss.py`

    ```python
    import torch
    import torch.nn as nn

    from ..builder import LOSSES
    from .utils import weighted_loss

    @weighted_loss
    def l1_loss(pred, target):
        assert pred.size() == target.size() and target.numel() > 0
        loss = torch.abs(pred - target)
        return loss

    @LOSSES.register_module()
    class L1Loss(nn.Module):

        def __init__(self, reduction='mean', loss_weight=1.0):
            super(L1Loss, self).__init__()
            self.reduction = reduction
            self.loss_weight = loss_weight

        def forward(self,
                    pred,
                    target,
                    weight=None,
                    avg_factor=None,
                    reduction_override=None):
            assert reduction_override in (None, 'none', 'mean', 'sum')
            reduction = (
                reduction_override if reduction_override else self.reduction)
            loss = self.loss_weight * l1_loss(
                pred, target, weight, reduction=reduction, avg_factor=avg_factor)
            return loss
    ```

2. 在文件 `mmcls/models/losses/__init__.py` 中导入这个模块

    ```python
    ...
    from .l1_loss import L1Loss, l1_loss

    __all__ = [
        ..., 'L1Loss', 'l1_loss'
    ]
    ```

3. 修改配置文件中的 `loss` 字段以使用新的损失函数

    ```python
    loss=dict(type='L1Loss', loss_weight=1.0))
    ```
