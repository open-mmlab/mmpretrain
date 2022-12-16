.. role:: hidden
    :class: hidden-section

.. module:: mmcls.models

mmcls.models
===================================

The ``models`` package contains several sub-packages for addressing the different components of a model.

- :mod:`~mmcls.models.classifiers`: The top-level module which defines the whole process of a classification model.
- :mod:`~mmcls.models.backbones`: Usually a feature extraction network, e.g., ResNet, MobileNet.
- :mod:`~mmcls.models.necks`: The component between backbones and heads, e.g., GlobalAveragePooling.
- :mod:`~mmcls.models.heads`: The component for specific tasks. In MMClassification, we provides heads for classification.
- :mod:`~mmcls.models.losses`: Loss functions.
- :mod:`~mmcls.models.utils`: Some helper functions and common components used in various networks.

  - :mod:`~mmcls.models.utils.data_preprocessor`: The component before model to preprocess the inputs, e.g., ClsDataPreprocessor.
  - :ref:`components`: Common components used in various networks.
  - :ref:`helpers`: Helper functions.

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

.. module:: mmcls.models.classifiers

Classifiers
------------------

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: classtemplate.rst

    BaseClassifier
    ImageClassifier
    TimmClassifier
    HuggingFaceClassifier

.. module:: mmcls.models.backbones

Backbones
------------------

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: classtemplate.rst

   AlexNet
   BEiT
   CSPDarkNet
   CSPNet
   CSPResNeXt
   CSPResNet
   Conformer
   ConvMixer
   ConvNeXt
   DaViT
   DeiT3
   DenseNet
   DistilledVisionTransformer
   EdgeNeXt
   EfficientFormer
   EfficientNet
   HRNet
   HorNet
   InceptionV3
   LeNet5
   MViT
   MlpMixer
   MobileNetV2
   MobileNetV3
   MobileOne
   MobileViT
   PCPVT
   PoolFormer
   RegNet
   RepLKNet
   RepMLPNet
   RepVGG
   Res2Net
   ResNeSt
   ResNeXt
   ResNet
   ResNetV1c
   ResNetV1d
   ResNet_CIFAR
   RevVisionTransformer
   SEResNeXt
   SEResNet
   SVT
   ShuffleNetV1
   ShuffleNetV2
   SwinTransformer
   SwinTransformerV2
   T2T_ViT
   TIMMBackbone
   TNT
   VAN
   VGG
   VisionTransformer

.. module:: mmcls.models.necks

Necks
------------------

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: classtemplate.rst

   GlobalAveragePooling
   GeneralizedMeanPooling
   HRFuseScales

.. module:: mmcls.models.heads

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
   EfficientFormerClsHead
   DeiTClsHead
   ConformerHead
   ArcFaceClsHead
   MultiLabelClsHead
   MultiLabelLinearClsHead
   CSRAClsHead

.. module:: mmcls.models.losses

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

.. module:: mmcls.models.utils

models.utils
------------

This package includes some helper functions and common components used in various networks.

.. _components:

Common Components
^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: classtemplate.rst

   InvertedResidual
   SELayer
   WindowMSA
   WindowMSAV2
   ShiftWindowMSA
   MultiheadAttention
   ConditionalPositionEncoding
   PatchEmbed
   PatchMerging
   HybridEmbed
   LayerScale

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
