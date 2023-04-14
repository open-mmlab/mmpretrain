.. role:: hidden
    :class: hidden-section

.. module:: mmpretrain.models

mmpretrain.models
===================================

The ``models`` package contains several sub-packages for addressing the different components of a model.

- :mod:`~mmpretrain.models.classifiers`: The top-level module which defines the whole process of a classification model.
- :mod:`~mmpretrain.models.selfsup`: The top-level module which defines the whole process of a self-supervised learning model.
- :mod:`~mmpretrain.models.retrievers`: The top-level module which defines the whole process of a retrieval model.
- :mod:`~mmpretrain.models.backbones`: Usually a feature extraction network, e.g., ResNet, MobileNet.
- :mod:`~mmpretrain.models.necks`: The component between backbones and heads, e.g., GlobalAveragePooling.
- :mod:`~mmpretrain.models.heads`: The component for specific tasks.
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

Self-supervised Algorithms
--------------------------

.. _selfsup_algorithms:

.. autosummary::
   :toctree: generated
   :nosignatures:

   BaseSelfSupervisor
   BEiT
   BYOL
   BarlowTwins
   CAE
   DenseCL
   EVA
   MAE
   MILAN
   MaskFeat
   MixMIM
   MoCo
   MoCoV3
   SimCLR
   SimMIM
   SimSiam
   SwAV

.. _selfsup_backbones:

Some of above algorithms modified the backbone module to adapt the extra inputs
like ``mask``, and here is the a list of these **modified backbone** modules.

.. autosummary::
   :toctree: generated
   :nosignatures:

   BEiTPretrainViT
   CAEPretrainViT
   MAEViT
   MILANViT
   MaskFeatViT
   MixMIMPretrainTransformer
   MoCoV3ViT
   SimMIMSwinTransformer

.. _target_generators:

Some self-supervise algorithms need an external **target generator** to
generate the optimization target. Here is a list of target generators.

.. autosummary::
   :toctree: generated
   :nosignatures:

   VQKD
   DALLEEncoder
   HOGGenerator
   CLIPGenerator

.. module:: mmpretrain.models.retrievers

Retrievers
------------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   BaseRetriever
   ImageToImageRetriever

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
   ViTSAM
   XCiT

.. module:: mmpretrain.models.necks

Necks
------------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   BEiTV2Neck
   CAENeck
   ClsBatchNormNeck
   DenseCLNeck
   GeneralizedMeanPooling
   GlobalAveragePooling
   HRFuseScales
   LinearNeck
   MAEPretrainDecoder
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

   ArcFaceClsHead
   BEiTV1Head
   BEiTV2Head
   CAEHead
   CSRAClsHead
   ClsHead
   ConformerHead
   ContrastiveHead
   DeiTClsHead
   EfficientFormerClsHead
   LatentCrossCorrelationHead
   LatentPredictHead
   LeViTClsHead
   LinearClsHead
   MAEPretrainHead
   MIMHead
   MixMIMPretrainHead
   MoCoV3Head
   MultiLabelClsHead
   MultiLabelLinearClsHead
   MultiTaskHead
   SimMIMHead
   StackedLinearClsHead
   SwAVHead
   VigClsHead
   VisionTransformerClsHead

.. module:: mmpretrain.models.losses

Losses
------------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   AsymmetricLoss
   CAELoss
   CosineSimilarityLoss
   CrossCorrelationLoss
   CrossEntropyLoss
   FocalLoss
   LabelSmoothLoss
   PixelReconstructionLoss
   SeesawLoss
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

   ConditionalPositionEncoding
   CosineEMA
   HybridEmbed
   InvertedResidual
   LayerScale
   MultiheadAttention
   PatchEmbed
   PatchMerging
   SELayer
   ShiftWindowMSA
   WindowMSA
   WindowMSAV2

.. _helpers:

Helper Functions
^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated
   :nosignatures:

   channel_shuffle
   is_tracing
   make_divisible
   resize_pos_embed
   resize_relative_position_bias_table
   to_ntuple
