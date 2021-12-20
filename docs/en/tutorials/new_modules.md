# Tutorial 5: Adding New Modules

## Develop new components

We basically categorize model components into 3 types.

- backbone: usually an feature extraction network, e.g., ResNet, MobileNet.
- neck: the component between backbones and heads, e.g., GlobalAveragePooling.
- head: the component for specific tasks, e.g., classification or regression.

### Add new backbones

Here we show how to develop new components with an example of ResNet_CIFAR.
As the input size of CIFAR is 32x32, this backbone replaces the `kernel_size=7, stride=2` to `kernel_size=3, stride=1` and remove the MaxPooling after stem, to avoid forwarding small feature maps to residual blocks.
It inherits from ResNet and only modifies the stem layers.

1. Create a new file `mmcls/models/backbones/resnet_cifar.py`.

```python
import torch.nn as nn

from ..builder import BACKBONES
from .resnet import ResNet


@BACKBONES.register_module()
class ResNet_CIFAR(ResNet):

    """ResNet backbone for CIFAR.

    short description of the backbone

    Args:
        depth(int): Network depth, from {18, 34, 50, 101, 152}.
        ...
    """

    def __init__(self, depth, deep_stem, **kwargs):
        # call ResNet init
        super(ResNet_CIFAR, self).__init__(depth, deep_stem=deep_stem, **kwargs)
        # other specific initialization
        assert not self.deep_stem, 'ResNet_CIFAR do not support deep_stem'

    def _make_stem_layer(self, in_channels, base_channels):
        # override ResNet method to modify the network structure
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

    def forward(self, x):  # should return a tuple
        pass  # implementation is ignored

    def init_weights(self, pretrained=None):
        pass  # override ResNet init_weights if necessary

    def train(self, mode=True):
        pass  # override ResNet train if necessary
```

2. Import the module in `mmcls/models/backbones/__init__.py`.

```python
...
from .resnet_cifar import ResNet_CIFAR

__all__ = [
    ..., 'ResNet_CIFAR'
]
```

3. Use it in your config file.

```python
model = dict(
    ...
    backbone=dict(
        type='ResNet_CIFAR',
        depth=18,
        other_arg=xxx),
    ...
```

### Add new necks

Here we take `GlobalAveragePooling` as an example. It is a very simple neck without any arguments.
To add a new neck, we mainly implement the `forward` function, which applies some operation on the output from backbone and forward the results to head.

1. Create a new file in `mmcls/models/necks/gap.py`.

    ```python
    import torch.nn as nn

    from ..builder import NECKS

    @NECKS.register_module()
    class GlobalAveragePooling(nn.Module):

        def __init__(self):
            self.gap = nn.AdaptiveAvgPool2d((1, 1))

        def forward(self, inputs):
            # we regard inputs as tensor for simplicity
            outs = self.gap(inputs)
            outs = outs.view(inputs.size(0), -1)
            return outs
    ```

2. Import the module in `mmcls/models/necks/__init__.py`.

    ```python
    ...
    from .gap import GlobalAveragePooling

    __all__ = [
        ..., 'GlobalAveragePooling'
    ]
    ```

3. Modify the config file.

    ```python
    model = dict(
        neck=dict(type='GlobalAveragePooling'),
    )
    ```

### Add new heads

Here we show how to develop a new head with the example of `LinearClsHead` as the following.
To implement a new head, basically we need to implement `forward_train`, which takes the feature maps from necks or backbones as input and compute loss based on ground-truth labels.

1. Create a new file in `mmcls/models/heads/linear_head.py`.

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

2. Import the module in `mmcls/models/heads/__init__.py`.

    ```python
    ...
    from .linear_head import LinearClsHead

    __all__ = [
        ..., 'LinearClsHead'
    ]
    ```

3. Modify the config file.

Together with the added GlobalAveragePooling neck, an entire config for a model is as follows.

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

### Add new loss

To add a new loss function, we mainly implement the `forward` function in the loss module.
In addition, it is helpful to leverage the decorator `weighted_loss` to weight the loss for each element.
Assuming that we want to mimic a probabilistic distribution generated from another classification model, we implement a L1Loss to fulfil the purpose as below.

1. Create a new file in `mmcls/models/losses/l1_loss.py`.

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

2. Import the module in `mmcls/models/losses/__init__.py`.

    ```python
    ...
    from .l1_loss import L1Loss, l1_loss

    __all__ = [
        ..., 'L1Loss', 'l1_loss'
    ]
    ```

3. Modify loss field in the config.

    ```python
    loss=dict(type='L1Loss', loss_weight=1.0))
    ```
