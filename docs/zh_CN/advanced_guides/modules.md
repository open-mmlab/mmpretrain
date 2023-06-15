# 自定义模型

在我们的设计中，我们定义一个完整的模型为顶层模块，根据功能的不同，基本几种不同类型的模型组件组成。

- 模型：顶层模块定义了具体的任务类型，例如 `ImageClassifier` 用在图像分类任务中， `MAE` 用在自监督学习中， `ImageToImageRetriever` 用在图像检索中。
- 主干网络：通常是一个特征提取网络，涵盖了模型之间绝大多数的差异，例如 `ResNet`、`MobileNet`。
- 颈部：用于连接主干网络和头部的组件，例如 `GlobalAveragePooling`。
- 头部：用于执行特定任务的组件，例如 `ClsHead`、 `ContrastiveHead`。
- 损失函数：在头部用于计算损失函数的组件，例如 `CrossEntropyLoss`、`LabelSmoothLoss`。
- 目标生成器: 用于自监督学习任务的组件，例如 `VQKD`、 `HOGGenerator`。

## 添加新的顶层模型

通常来说，对于图像分类和图像检索任务来说，模型顶层模型流程基本一致。但是不同的自监督学习算法却用不同的计算流程，像 `MAE` 和 `BEiT` 就大不相同。 所以在这个部分，我们将简单介绍如何添加一个新的自监督学习算法。

### 添加新的自监督学习算法

1. 创建新文件 `mmpretrain/models/selfsup/new_algorithm.py` 以及实现 `NewAlgorithm`

   ```python
   from mmpretrain.registry import MODELS
   from .base import BaseSelfSupvisor


   @MODELS.register_module()
   class NewAlgorithm(BaseSelfSupvisor):

       def __init__(self, backbone, neck=None, head=None, init_cfg=None):
           super().__init__(init_cfg)
           pass

       # ``extract_feat`` function is defined in BaseSelfSupvisor, you could
       # overwrite it if needed
       def extract_feat(self, inputs, **kwargs):
           pass

       # the core function to compute the loss
       def loss(self, inputs, data_samples, **kwargs):
           pass

   ```

2. 在 `mmpretrain/models/selfsup/__init__.py` 中导入对应的新算法

   ```python
   ...
   from .new_algorithm import NewAlgorithm

   __all__ = [
       ...,
       'NewAlgorithm',
       ...
   ]
   ```

3. 在配置文件中使用新算法

   ```python
   model = dict(
       type='NewAlgorithm',
       backbone=...,
       neck=...,
       head=...,
       ...
   )
   ```

## 添加新的主干网络

这里，我们以 `ResNet_CIFAR` 为例，展示了如何开发一个新的主干网络组件。

`ResNet_CIFAR` 针对 CIFAR 32x32 的图像输入，远小于大多数模型使用的ImageNet默认的224x224输入配置，所以我们将骨干网络中 `kernel_size=7,stride=2`
的设置替换为 `kernel_size=3, stride=1`，并移除了 stem 层之后的
`MaxPooling`，以避免传递过小的特征图到残差块中。

最简单的方式就是继承自 `ResNet` 并只修改 stem 层。

1. 创建一个新文件 `mmpretrain/models/backbones/resnet_cifar.py`。

   ```python
   import torch.nn as nn

   from mmpretrain.registry import MODELS
   from .resnet import ResNet


   @MODELS.register_module()
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

       def forward(self, x):
           # 如果需要的话，可以自定义forward方法
           x = self.conv1(x)
           x = self.norm1(x)
           x = self.relu(x)
           outs = []
           for i, layer_name in enumerate(self.res_layers):
               res_layer = getattr(self, layer_name)
               x = res_layer(x)
               if i in self.out_indices:
                   outs.append(x)
           # 输出值需要是一个包含不同层多尺度输出的元组
           # 如果不需要多尺度特征，可以直接在最终输出上包一层元组
           return tuple(outs)

       def init_weights(self):
           # 如果需要的话，可以自定义权重初始化的方法
           super().init_weights()

           # 如果有预训练模型，则不需要进行权重初始化
           if self.init_cfg is not None and self.init_cfg['type'] == 'Pretrained':
               return

           # 通常来说，我们建议用`init_cfg`去列举不同层权重初始化方法
           # 包括卷积层，线性层，归一化层等等
           # 如果有特殊需要，可以在这里进行额外的初始化操作
           ...
   ```

```{note}
在 OpenMMLab 2.0 的设计中，将原有的`BACKBONES`、`NECKS`、`HEADS`、`LOSSES`等注册名统一为`MODELS`.
```

2. 在 `mmpretrain/models/backbones/__init__.py` 中导入新模块

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

### 为自监督学习添加新的主干网络

对于一部分自监督学习算法，主干网络做了一定修改，例如 `MAE`、`BEiT` 等。 这些主干网络需要处理 `mask` 相关的逻辑，以此从可见的图像块中提取对应的特征信息。

以 [MAEViT](mmpretrain.models.selfsup.MAEViT) 作为例子，我们需要重写 `forward` 函数，进行基于 `mask` 的计算。我们实现了 `init_weights` 进行特定权重的初始化和 `random_masking` 函数来生成 `MAE` 预训练所需要的 `mask`。

```python
class MAEViT(VisionTransformer):
    """Vision Transformer for MAE pre-training"""

    def __init__(mask_ratio, **kwargs) -> None:
        super().__init__(**kwargs)
        # position embedding is not learnable during pretraining
        self.pos_embed.requires_grad = False
        self.mask_ratio = mask_ratio
        self.num_patches = self.patch_resolution[0] * self.patch_resolution[1]

    def init_weights(self) -> None:
        """Initialize position embedding, patch embedding and cls token."""
        super().init_weights()
        # define what if needed
        pass

    def random_masking(
        self,
        x: torch.Tensor,
        mask_ratio: float = 0.75
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate the mask for MAE Pre-training."""
        pass

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[bool] = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate features for masked images.

        The function supports two kind of forward behaviors. If the ``mask`` is
        ``True``, the function will generate mask to masking some patches
        randomly and get the hidden features for visible patches, which means
        the function will be executed as masked imagemodeling pre-training;
        if the ``mask`` is ``None`` or ``False``, the forward function will
        call ``super().forward()``, which extract features from images without
        mask.
        """
        if mask is None or False:
            return super().forward(x)

        else:
            B = x.shape[0]
            x = self.patch_embed(x)[0]
            # add pos embed w/o cls token
            x = x + self.pos_embed[:, 1:, :]

            # masking: length -> length * mask_ratio
            x, mask, ids_restore = self.random_masking(x, self.mask_ratio)

            # append cls token
            cls_token = self.cls_token + self.pos_embed[:, :1, :]
            cls_tokens = cls_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)

            for _, layer in enumerate(self.layers):
                x = layer(x)
            # Use final norm
            x = self.norm1(x)

            return (x, mask, ids_restore)

```

## 添加新的颈部组件

这里我们以 `GlobalAveragePooling` 为例。这是一个非常简单的颈部组件，没有任何参数。

要添加新的颈部组件，我们主要需要实现 `forward` 函数，该函数对主干网络的输出进行
一些操作并将结果传递到头部。

1. 创建一个新文件 `mmpretrain/models/necks/gap.py`

   ```python
   import torch.nn as nn

   from mmpretrain.registry import MODELS

   @MODELS.register_module()
   class GlobalAveragePooling(nn.Module):

       def __init__(self):
           self.gap = nn.AdaptiveAvgPool2d((1, 1))

       def forward(self, inputs):
           # 简单起见，我们默认输入是一个张量
           outs = self.gap(inputs)
           outs = outs.view(inputs.size(0), -1)
           return outs
   ```

2. 在 `mmpretrain/models/necks/__init__.py` 中导入新模块

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

## 添加新的头部组件

### 基于分类头

在此，我们以一个简化的 `VisionTransformerClsHead` 为例，说明如何开发新的头部组件。

要添加一个新的头部组件，基本上我们需要实现 `pre_logits` 函数用于进入最后的分类头之前需要的处理，
以及 `forward` 函数。

1. 创建一个文件 `mmpretrain/models/heads/vit_head.py`.

   ```python
   import torch.nn as nn

   from mmpretrain.registry import MODELS
   from .cls_head import ClsHead


   @MODELS.register_module()
   class LinearClsHead(ClsHead):

       def __init__(self, num_classes, in_channels, hidden_dim, **kwargs):
           super().__init__(**kwargs)
           self.in_channels = in_channels
           self.num_classes = num_classes
           self.hidden_dim = hidden_dim

           self.fc1 = nn.Linear(in_channels, hidden_dim)
           self.act = nn.Tanh()
           self.fc2 = nn.Linear(hidden_dim, num_classes)

       def pre_logits(self, feats):
           # 骨干网络的输出通常包含多尺度信息的元组
           # 对于分类任务来说，我们只需要关注最后的输出
           feat = feats[-1]

           # VisionTransformer的最终输出是一个包含patch tokens和cls tokens的元组
           # 这里我们只需要cls tokens
           _, cls_token = feat

           # 完成除了最后的线性分类头以外的操作
           return self.act(self.fc1(cls_token))

       def forward(self, feats):
           pre_logits = self.pre_logits(feats)

           # 完成最后的分类头
           cls_score = self.fc(pre_logits)
           return cls_score
   ```

2. 在 `mmpretrain/models/heads/__init__.py` 中导入这个模块

   ```python
   ...
   from .vit_head import VisionTransformerClsHead

   __all__ = [
       ..., 'VisionTransformerClsHead'
   ]
   ```

3. 修改配置文件以使用新的头部组件。

   ```python
   model = dict(
       head=dict(
           type='VisionTransformerClsHead',
           ...,
       ))
   ```

### 基于 BaseModule 类

这是一个基于 MMEngine 中的 `BaseModule` 进行开发例子，`MAEPretrainHead`，主要是为了 `MAE` 掩码学习。我们需要实现 `loss` 函数来计算损失吗，不过其它的函数均为可选项。

```python
# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmengine.model import BaseModule

from mmpretrain.registry import MODELS


@MODELS.register_module()
class MAEPretrainHead(BaseModule):
    """Head for MAE Pre-training."""

    def __init__(self,
                 loss: dict,
                 norm_pix: bool = False,
                 patch_size: int = 16) -> None:
        super().__init__()
        self.norm_pix = norm_pix
        self.patch_size = patch_size
        self.loss_module = MODELS.build(loss)

    def patchify(self, imgs: torch.Tensor) -> torch.Tensor:
        """Split images into non-overlapped patches."""
        p = self.patch_size
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x

    def construct_target(self, target: torch.Tensor) -> torch.Tensor:
        """Construct the reconstruction target."""
        target = self.patchify(target)
        if self.norm_pix:
            # normalize the target image
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        return target

    def loss(self, pred: torch.Tensor, target: torch.Tensor,
             mask: torch.Tensor) -> torch.Tensor:
        """Generate loss."""
        target = self.construct_target(target)
        loss = self.loss_module(pred, target, mask)

        return loss
```

完成实现后，之后的步骤和 [基于分类头](#基于分类头) 中的步骤 2 和步骤 3 一致。

## 添加新的损失函数

要添加新的损失函数，我们主要需要在损失函数模块中 `forward` 函数。这里需要注意的是，损失模块也应该注册到`MODELS`中。另外，利用装饰器 `weighted_loss` 可以方便的实现对每个元素的损失进行加权平均。

假设我们要模拟从另一个分类模型生成的概率分布，需要添加 `L1loss` 来实现该目的。

1. 创建一个新文件 `mmpretrain/models/losses/l1_loss.py`

   ```python
   import torch
   import torch.nn as nn

   from mmpretrain.registry import MODELS
   from .utils import weighted_loss

   @weighted_loss
   def l1_loss(pred, target):
       assert pred.size() == target.size() and target.numel() > 0
       loss = torch.abs(pred - target)
       return loss

   @MODELS.register_module()
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

2. 在文件 `mmpretrain/models/losses/__init__.py` 中导入这个模块

   ```python
   ...
   from .l1_loss import L1Loss

   __all__ = [
       ..., 'L1Loss'
   ]
   ```

3. 修改配置文件中的 `loss` 字段以使用新的损失函数

   ```python
   model = dict(
       head=dict(
           loss=dict(type='L1Loss', loss_weight=1.0),
       ))
   ```

最后我们可以在配置文件中结合所有新增的模型组件来使用新的模型。由于`ResNet_CIFAR` 不是一个基于ViT的骨干网络，这里我们不用`VisionTransformerClsHead`的配置。

```python
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='ResNet_CIFAR',
        depth=18,
        num_stages=4,
        out_indices=(3, ),
        style='pytorch'),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=10,
        in_channels=512,
        loss=dict(type='L1Loss', loss_weight=1.0),
        topk=(1, 5),
    ))

```

```{tip}
为了方便，相同的模型组件可以直接从已有的config文件里继承，更多细节可以参考[学习配置文件](../user_guides/config.md)。
```
