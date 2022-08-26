# Customize Modules

## Develop new components

In our design, a complete model is defined as an ImageClassifier which basically contains below 4 types of model components based on their functionalities.

- backbone: usually a feature extraction network which records the major differences between models, e.g., ResNet, MobileNet.
- neck: the component between backbone and head, e.g., GlobalAveragePooling.
- head: the component for specific tasks, e.g., classification or regression.
- loss: the component in head for calculating losses, e.g., CrossEntropyLoss, LabelSmoothLoss.

### Add a new backbone

Here we presents how to develop a new backbone component by an example of ResNet_CIFAR.
As the input size of CIFAR is 32x32, which is much smaller than the default size of 224x224 in ImageNet, this backbone replaces the `kernel_size=7, stride=2` to `kernel_size=3, stride=1` and removes the MaxPooling after the stem layer to avoid forwarding small feature maps to residual blocks.

The esaiest way is to inherit from `ResNet` and only modify the stem layer.

1. Create a new file `mmcls/models/backbones/resnet_cifar.py`.

   ```python
   import torch.nn as nn

   from mmcls.registry import MODELS
   from .resnet import ResNet


   @MODELS.register_module()
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

```{note}
Replace original registry names from BACKBONES, NECKS, HEADS, LOSSES, etc to MODELS in OpenMMLab 2.0 design.
```

2. Import the new backbone module in `mmcls/models/backbones/__init__.py`.

   ```python
   ...
   from .resnet_cifar import ResNet_CIFAR

   __all__ = [
       ..., 'ResNet_CIFAR'
   ]
   ```

3. Modify the correlated settings in your config file.

   ```python
   model = dict(
       ...
       backbone=dict(
           type='ResNet_CIFAR',
           depth=18,
           ...),
       ...
   ```

### Add a new neck

Here we take `GlobalAveragePooling` as an example. It is a very simple neck without any arguments.
To add a new neck, we mainly implement the `forward` function, which applies some operations on the output from backbone and forward the results to head.

1. Create a new file in `mmcls/models/necks/gap.py`.

   ```python
   import torch.nn as nn

   from mmcls.registry import MODELS

   @MODELS.register_module()
   class GlobalAveragePooling(nn.Module):

       def __init__(self):
           self.gap = nn.AdaptiveAvgPool2d((1, 1))

       def forward(self, inputs):
           # we regard inputs as tensor for simplicity
           outs = self.gap(inputs)
           outs = outs.view(inputs.size(0), -1)
           return outs
   ```

2. Import the new neck module in `mmcls/models/necks/__init__.py`.

   ```python
   ...
   from .gap import GlobalAveragePooling

   __all__ = [
       ..., 'GlobalAveragePooling'
   ]
   ```

3. Modify the correlated settings in your config file.

   ```python
   model = dict(
       neck=dict(type='GlobalAveragePooling'),
   )
   ```

### Add a new head

Here we presents how to develop a new head by the example of `LinearClsHead` as the following.
To implement a new head, basically we need to implement `pre_logits` method for processes before the final classification head and `forward` method.

1. Create a new file in `mmcls/models/heads/linear_head.py`.

   ```python
   import torch.nn as nn

   from mmcls.registry import MODELS
   from .cls_head import ClsHead


   @MODELS.register_module()
   class LinearClsHead(ClsHead):

       def __init__(self,
                    num_classes,
                    in_channels,
                    init_cfg=dict(
                        type='Normal', layer='Linear', std=0.01),
                    **kwargs):
           super(LinearClsHead, self).__init__(init_cfg=init_cfg, **kwargs)
           self.in_channels = in_channels
           self.num_classes = num_classes

           if self.num_classes <= 0:
               raise ValueError(
                   f'num_classes={num_classes} must be a positive integer')

           self.fc = nn.Linear(self.in_channels, self.num_classes)

       def pre_logits(self, feats):
           """The process before the final classification head."""
           # The LinearClsHead doesn't have other module, just return after
           # unpacking.
           return feats[-1]

       def forward(self, feats):
           pre_logits = self.pre_logits(feats)
           # The final classification head.
           cls_score = self.fc(pre_logits)
           return cls_score
   ```

2. Import the module in `mmcls/models/heads/__init__.py`.

   ```python
   ...
   from .linear_head import LinearClsHead

   __all__ = [
       ..., 'LinearClsHead'
   ]
   ```

3. Modify the correlated settings in your config file.

   ```python
   model = dict(
       head=dict(
           type='LinearClsHead',
           num_classes=10, # for cifar10 dataset
           in_channels=512, # for resnet_cifar with depth of 18
           ...,
       ))
   ```

### Add a new loss

To add a new loss function, we mainly implement the `forward` function in the loss module. We should register loss module as `MODELS` as well.
In addition, it is helpful to leverage the decorator `weighted_loss` to weight the loss for each element.
Assuming that we want to mimic a probabilistic distribution generated from another classification model, we implement a L1Loss to fulfil the purpose as below.

1. Create a new file in `mmcls/models/losses/l1_loss.py`.

   ```python
   import torch
   import torch.nn as nn

   from mmcls.registry import MODELS
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

2. Import the module in `mmcls/models/losses/__init__.py`.

   ```python
   ...
   from .l1_loss import L1Loss, l1_loss

   __all__ = [
       ..., 'L1Loss', 'l1_loss'
   ]
   ```

3. Modify loss field in the head configs.

   ```python
   model = dict(
       head=dict(
           loss=dict(type='L1Loss', loss_weight=1.0),
       ))
   ```

Finally we can combine all the new model components in config file to create a new model for best practices.

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
For conveniency, the same model components could inherit from existing config files, refers to [Learn about configs](../user_guides/config.md) for more details.
```
