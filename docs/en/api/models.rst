.. role:: hidden
    :class: hidden-section

.. module:: mmpretrain.models

mmpretrain.models
===================================

The ``models`` package contains several sub-packages for addressing the different components of a model.

- :mod:`~mmpretrain.models.classifiers`: The top-level module which defines the whole process of a classification model.
- :mod:`~mmpretrain.models.selfsup`: The top-level module which defines the whole process of a self-supervised learning model.
- :mod:`~mmpretrain.models.backbones`: Usually a feature extraction network, e.g., ResNet, MobileNet.
- :mod:`~mmpretrain.models.necks`: The component between backbones and heads, e.g., GlobalAveragePooling.
- :mod:`~mmpretrain.models.heads`: The component for specific tasks. In MMClassification, we provides heads for classification.
- :mod:`~mmpretrain.models.losses`: Loss functions.
- :mod:`~mmpretrain.models.utils`: Some helper functions and common components used in various networks.

  - :mod:`~mmpretrain.models.utils.data_preprocessor`: The component before model to preprocess the inputs, e.g., ClsDataPreprocessor.
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

.. module:: mmpretrain.models.classifiers

Classifiers
------------------

.. autosummary::
   :toctree: generated
   :nosignatures:

    BaseClassifier
    ImageClassifier
    TimmClassifier
    HuggingFaceClassifier

.. module:: mmpretrain.models.selfsup

SelfSup
------------------

.. _selfsup_algorithms:

Self-Supervised Learning
^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated
   :nosignatures:

   BaseSelfSupervisor
   MoCo
   MoCoV3
   BYOL
   SimCLR
   SimSiam
   BEiT
   CAE
   MAE
   MaskFeat
   MILAN
   MixMIM
   SimMIM
   EVA
   DenseCL
   BarlowTwins
   SwAV

.. _selfsup_backbones:

SelfSup Backbones
^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated
   :nosignatures:

   BEiTPretrainViT
   CAEPretrainViT
   MAEViT
   MaskFeatViT
   MILANViT
   MixMIMPretrainTransformer
   MoCoV3ViT
   SimMIMSwinTransformer

.. _target_generators:

Target Generators
^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated
   :nosignatures:

   VQKD
   DALLEEncoder
   HOGGenerator
   CLIPGenerator

.. module:: mmpretrain.models.backbones

Backbones
------------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   AlexNet
   BEiTViT
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
   EfficientNetV2
   HRNet
   HorNet
   InceptionV3
   LeNet5
   LeViT
   MViT
   MlpMixer
   MobileNetV2
   MobileNetV3
   MobileOne
   MobileViT
   PCPVT
   PoolFormer
   PyramidVig
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
   Vig
   VisionTransformer
   XCiT

.. module:: mmpretrain.models.necks

Necks
------------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   GlobalAveragePooling
   GeneralizedMeanPooling
   HRFuseScales
   LinearNeck
   BEiTV2Neck
   CAENeck
   DenseCLNeck
   MAEPretrainDecoder
   ClsBatchNormNeck
   MILANPretrainDecoder
   MixMIMPretrainDecoder
   MoCoV2Neck
   NonLinearNeck
   SimMIMLinearDecoder
   SwAVNeck

.. module:: mmpretrain.models.heads

Heads
------------------

.. autosummary::
   :toctree: generated
   :nosignatures:

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
   MultiTaskHead
   LeViTClsHead
   VigClsHead
   BEiTV1Head
   BEiTV2Head
   CAEHead
   ContrastiveHead
   LatentCrossCorrelationHead
   LatentPredictHead
   MAEPretrainHead
   MixMIMPretrainHead
   SwAVHead
   MoCoV3Head
   MIMHead
   SimMIMHead

.. module:: mmpretrain.models.losses

Losses
------------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   CrossEntropyLoss
   LabelSmoothLoss
   FocalLoss
   AsymmetricLoss
   SeesawLoss
   CAELoss
   CosineSimilarityLoss
   CrossCorrelationLoss
   PixelReconstructionLoss
   SwAVLoss

.. module:: mmpretrain.models.utils

models.utils
------------

This package includes some helper functions and common components used in various networks.

.. _components:

Common Components
^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated
   :nosignatures:

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
