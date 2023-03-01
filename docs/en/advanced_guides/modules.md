# Customize Models

In our design, a complete model is defined as an ImageClassifier which contains 4 types of model components based on their functionalities.

- backbone: usually a feature extraction network that records the major differences between models, e.g., ResNet, MobileNet.
- neck: the component between backbone and head, e.g., GlobalAveragePooling.
- head: the component for specific tasks, e.g., classification or regression.
- loss: the component in the head for calculating losses, e.g., CrossEntropyLoss, LabelSmoothLoss.

## Add a new backbone

Here we present how to develop a new backbone component by an example of `ResNet_CIFAR`.
As the input size of CIFAR is 32x32, which is much smaller than the default size of 224x224 in ImageNet, this backbone replaces the `kernel_size=7, stride=2` to `kernel_size=3, stride=1` and removes the MaxPooling after the stem layer to avoid forwarding small feature maps to residual blocks.

The easiest way is to inherit from `ResNet` and only modify the stem layer.

1. Create a new file `mmpretrain/models/backbones/resnet_cifar.py`.

   ```python
   import torch.nn as nn

   from mmpretrain.registry import MODELS
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
           # other specific initializations
           assert not self.deep_stem, 'ResNet_CIFAR do not support deep_stem'

       def _make_stem_layer(self, in_channels, base_channels):
           # override the ResNet method to modify the network structure
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
           # Customize the forward method if needed.
           x = self.conv1(x)
           x = self.norm1(x)
           x = self.relu(x)
           outs = []
           for i, layer_name in enumerate(self.res_layers):
               res_layer = getattr(self, layer_name)
               x = res_layer(x)
               if i in self.out_indices:
                   outs.append(x)
           # The return value needs to be a tuple with multi-scale outputs from different depths.
           # If you don't need multi-scale features, just wrap the output as a one-item tuple.
           return tuple(outs)

       def init_weights(self):
           # Customize the weight initialization method if needed.
           super().init_weights()

           # Disable the weight initialization if loading a pretrained model.
           if self.init_cfg is not None and self.init_cfg['type'] == 'Pretrained':
               return

           # Usually, we recommend using `init_cfg` to specify weight initialization methods
           # of convolution, linear, or normalization layers. If you have some special needs,
           # do these extra weight initialization here.
           ...
   ```

```{note}
Replace original registry names from `BACKBONES`, `NECKS`, `HEADS` and `LOSSES` to `MODELS` in OpenMMLab 2.0 design.
```

2. Import the new backbone module in `mmpretrain/models/backbones/__init__.py`.

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

## Add a new neck

Here we take `GlobalAveragePooling` as an example. It is a very simple neck without any arguments.
To add a new neck, we mainly implement the `forward` function, which applies some operations on the output from the backbone and forwards the results to the head.

1. Create a new file in `mmpretrain/models/necks/gap.py`.

   ```python
   import torch.nn as nn

   from mmpretrain.registry import MODELS

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

2. Import the new neck module in `mmpretrain/models/necks/__init__.py`.

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

## Add a new head

Here we present how to develop a new head by the example of simplified `VisionTransformerClsHead` as the following.
To implement a new head, we need to implement a `pre_logits` method for processes before the final classification head and a `forward` method.

:::{admonition} Why do we need the `pre_logits` method?
:class: note

In classification tasks, we usually use a linear layer to do the final classification. And sometimes, we need
to obtain the feature before the final classification, which is the output of the `pre_logits` method.
:::

1. Create a new file in `mmpretrain/models/heads/vit_head.py`.

   ```python
   import torch.nn as nn

   from mmpretrain.registry import MODELS
   from .cls_head import ClsHead


   @MODELS.register_module()
   class VisionTransformerClsHead(ClsHead):

       def __init__(self, num_classes, in_channels, hidden_dim, **kwargs):
           super().__init__(**kwargs)
           self.in_channels = in_channels
           self.num_classes = num_classes
           self.hidden_dim = hidden_dim

           self.fc1 = nn.Linear(in_channels, hidden_dim)
           self.act = nn.Tanh()
           self.fc2 = nn.Linear(hidden_dim, num_classes)

       def pre_logits(self, feats):
           # The output of the backbone is usually a tuple from multiple depths,
           # and for classification, we only need the final output.
           feat = feats[-1]

           # The final output of VisionTransformer is a tuple of patch tokens and
           # classification tokens. We need classification tokens here.
           _, cls_token = feat

           # Do all works except the final classification linear layer.
           return self.act(self.fc1(cls_token))

       def forward(self, feats):
           pre_logits = self.pre_logits(feats)

           # The final classification linear layer.
           cls_score = self.fc2(pre_logits)
           return cls_score
   ```

2. Import the module in `mmpretrain/models/heads/__init__.py`.

   ```python
   ...
   from .vit_head import VisionTransformerClsHead

   __all__ = [
       ..., 'VisionTransformerClsHead'
   ]
   ```

3. Modify the correlated settings in your config file.

   ```python
   model = dict(
       head=dict(
           type='VisionTransformerClsHead',
           ...,
       ))
   ```

## Add a new loss

To add a new loss function, we mainly implement the `forward` function in the loss module. We should register the loss module as `MODELS` as well.
In addition, it is helpful to leverage the decorator `weighted_loss` to weight the loss for each element.
Assuming that we want to mimic a probabilistic distribution generated from another classification model, we implement an L1Loss to fulfill the purpose as below.

1. Create a new file in `mmpretrain/models/losses/l1_loss.py`.

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

2. Import the module in `mmpretrain/models/losses/__init__.py`.

   ```python
   ...
   from .l1_loss import L1Loss

   __all__ = [
       ..., 'L1Loss'
   ]
   ```

3. Modify loss field in the head configs.

   ```python
   model = dict(
       head=dict(
           loss=dict(type='L1Loss', loss_weight=1.0),
       ))
   ```

Finally, we can combine all the new model components in a config file to create a new model for best practices. Because `ResNet_CIFAR` is not a ViT-based backbone, we do not implement `VisionTransformerClsHead` here.

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
For convenience, the same model components could inherit from existing config files, refers to [Learn about configs](../user_guides/config.md) for more details.
```
