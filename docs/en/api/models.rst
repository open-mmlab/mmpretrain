.. role:: hidden
    :class: hidden-section

mmcls.models
===================================

The ``models`` package contains several sub-packages for addressing the different components of a model.

- :ref:`classifiers`: The top-level module which defines the whole process of a classification model.
- :ref:`backbones`: Usually a feature extraction network, e.g., ResNet, MobileNet.
- :ref:`necks`: The component between backbones and heads, e.g., GlobalAveragePooling.
- :ref:`heads`: The component for specific tasks. In MMClassification, we provides heads for classification.
- :ref:`losses`: Loss functions.
- :ref:`datapreprocessors`: The component before model to preprocess the inputs, e.g., ClsDataPreprocessor.
- :ref:`models.utils`: Some helper functions and common components used in various networks.

  - :ref:`components`: Common components used in various networks.
  - :ref:`helpers`: Helper functions.

.. currentmodule:: mmcls.models

Build Functions
---------------

.. autosummary::
    :toctree: generated
    :nosignatures:

    build_classifier
    build_backbone
    build_neck
    build_head
    build_loss

.. _classifiers:

Classifier
------------------

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: classtemplate.rst

    BaseClassifier
    ImageClassifier

.. _backbones:

Backbones
------------------

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: classtemplate.rst

   AlexNet
   CSPDarkNet
   CSPNet
   CSPResNeXt
   CSPResNet
   Conformer
   ConvMixer
   ConvNeXt
   DenseNet
   DistilledVisionTransformer
   EfficientNet
   HRNet
   LeNet5
   MlpMixer
   MobileNetV2
   MobileNetV3
   PCPVT
   PoolFormer
   RegNet
   RepMLPNet
   RepVGG
   Res2Net
   ResNeSt
   ResNeXt
   ResNet
   ResNetV1c
   ResNetV1d
   ResNet_CIFAR
   SEResNeXt
   SEResNet
   SVT
   ShuffleNetV1
   ShuffleNetV2
   SwinTransformer
   T2T_ViT
   TIMMBackbone
   TNT
   VAN
   VGG
   VisionTransformer

.. _necks:

Necks
------------------

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: classtemplate.rst

   GlobalAveragePooling
   GeneralizedMeanPooling
   HRFuseScales

.. _heads:

Heads
------------------

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: classtemplate.rst

   ClsHead
   LinearClsHead
   StackedLinearClsHead
   VisionTransformerClsHead
   DeiTClsHead
   ConformerHead
   MultiLabelClsHead
   MultiLabelLinearClsHead

.. _losses:

Losses
------------------

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: classtemplate.rst

   CrossEntropyLoss
   LabelSmoothLoss
   FocalLoss
   AsymmetricLoss
   SeesawLoss

.. _models.utils:

models.utils
------------

This package includes some helper functions and common components used in various networks.

.. currentmodule:: mmcls.models.utils

.. _components:

Common Components
^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: classtemplate.rst

   InvertedResidual
   SELayer
   ShiftWindowMSA
   MultiheadAttention
   ConditionalPositionEncoding
   PatchEmbed
   PatchMerging
   HybridEmbed

.. _helpers:

Helper Functions
^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated
   :nosignatures:

   channel_shuffle
   make_divisible
   resize_pos_embed
   resize_relative_position_bias_table
   to_ntuple
   is_tracing
