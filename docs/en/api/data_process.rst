.. role:: hidden
    :class: hidden-section

Data Process
=================

In MMPreTrain, the data process and the dataset is decomposed. The
datasets only define how to get samples' basic information from the file
system. These basic information includes the ground-truth label and raw
images data / the paths of images.The data process includes data transforms,
data preprocessors and batch augmentations.

- :mod:`Data Transforms <mmpretrain.datasets.transforms>`: Transforms includes loading, preprocessing, formatting and etc.
- :mod:`Data Preprocessors <mmpretrain.models.utils.data_preprocessor>`: Processes includes collate, normalization, stacking, channel fliping and etc.

  - :mod:`Batch Augmentations <mmpretrain.models.utils.batch_augments>`: Batch augmentation involves multiple samples, such as Mixup and CutMix.

.. module:: mmpretrain.datasets.transforms

Data Transforms
--------------------

To prepare the inputs data, we need to do some transforms on these basic
information. These transforms includes loading, preprocessing and
formatting. And a series of data transforms makes up a data pipeline.
Therefore, you can find the a ``pipeline`` argument in the configs of dataset,
for example:

.. code:: python

    train_pipeline = [
        dict(type='LoadImageFromFile'),
        dict(type='RandomResizedCrop', scale=224),
        dict(type='RandomFlip', prob=0.5, direction='horizontal'),
        dict(type='PackInputs'),
    ]

    train_dataloader = dict(
        ....
        dataset=dict(
            pipeline=train_pipeline,
            ....),
        ....
    )

Every item of a pipeline list is one of the following data transforms class. And if you want to add a custom data transformation class, the tutorial :doc:`Custom Data Pipelines </advanced_guides/pipeline>` will help you.

.. contents::
   :depth: 1
   :local:
   :backlinks: top

Loading and Formatting
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: data_transform.rst

   LoadImageFromFile
   PackInputs
   PackMultiTaskInputs
   PILToNumpy
   NumpyToPIL
   Transpose
   Collect

Processing and Augmentation
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: data_transform.rst

   Albumentations
   CenterCrop
   ColorJitter
   EfficientNetCenterCrop
   EfficientNetRandomCrop
   Lighting
   Normalize
   RandomCrop
   RandomErasing
   RandomFlip
   RandomGrayscale
   RandomResize
   RandomResizedCrop
   Resize
   ResizeEdge
   BEiTMaskGenerator
   SimMIMMaskGenerator

Composed Augmentation
"""""""""""""""""""""
Composed augmentation is a kind of methods which compose a series of data
augmentation transforms, such as ``AutoAugment`` and ``RandAugment``.

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: data_transform.rst

   AutoAugment
   RandAugment

The above transforms is composed from a group of policies from the below random
transforms:

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: data_transform.rst

   AutoContrast
   Brightness
   ColorTransform
   Contrast
   Cutout
   Equalize
   GaussianBlur
   Invert
   Posterize
   Rotate
   Sharpness
   Shear
   Solarize
   SolarizeAdd
   Translate
   BaseAugTransform

MMCV transforms
^^^^^^^^^^^^^^^

We also provides many transforms in MMCV. You can use them directly in the config files. Here are some frequently used transforms, and the whole transforms list can be found in :external+mmcv:doc:`api/transforms`.

Transform Wrapper
^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: data_transform.rst

   MultiView

.. module:: mmpretrain.models.utils.data_preprocessor


TorchVision Transforms
^^^^^^^^^^^^^^^^^^^^^^

We also provide all the transforms in TorchVision. You can use them the like following examples:

**1. Use some TorchVision Augs Surrounded by NumpyToPIL and PILToNumpy (Recommendation)**

Add TorchVision Augs surrounded by ``dict(type='NumpyToPIL', to_rgb=True),`` and ``dict(type='PILToNumpy', to_bgr=True),``

.. code:: python

    train_pipeline = [
        dict(type='LoadImageFromFile'),
        dict(type='NumpyToPIL', to_rgb=True),     # from BGR in cv2 to RGB  in PIL
        dict(type='torchvision/RandomResizedCrop',size=176),
        dict(type='PILToNumpy', to_bgr=True),     # from RGB  in PIL to BGR in cv2
        dict(type='RandomFlip', prob=0.5, direction='horizontal'),
        dict(type='PackInputs'),
    ]

    data_preprocessor = dict(
        num_classes=1000,
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True,                          # from BGR in cv2 to RGB  in PIL
    )


**2. Use TorchVision Augs and ToTensor&Normalize**

Make sure the 'img' has been converted to PIL format from BGR-Numpy format before being processed by TorchVision Augs.

.. code:: python

    train_pipeline = [
        dict(type='LoadImageFromFile'),
        dict(type='NumpyToPIL', to_rgb=True),       # from BGR in cv2 to RGB  in PIL
        dict(
            type='torchvision/RandomResizedCrop',
            size=176,
            interpolation='bilinear'),            # accept str format interpolation mode
        dict(type='torchvision/RandomHorizontalFlip', p=0.5),
        dict(
            type='torchvision/TrivialAugmentWide',
            interpolation='bilinear'),
        dict(type='torchvision/PILToTensor'),
        dict(type='torchvision/ConvertImageDtype', dtype=torch.float),
        dict(
            type='torchvision/Normalize',
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
        dict(type='torchvision/RandomErasing', p=0.1),
        dict(type='PackInputs'),
    ]

    data_preprocessor = dict(num_classes=1000, mean=None, std=None, to_rgb=False)  # Normalize in dataset pipeline


**3. Use TorchVision Augs Except ToTensor&Normalize**

.. code:: python

    train_pipeline = [
        dict(type='LoadImageFromFile'),
        dict(type='NumpyToPIL', to_rgb=True),   # from BGR in cv2 to RGB  in PIL
        dict(type='torchvision/RandomResizedCrop', size=176, interpolation='bilinear'),
        dict(type='torchvision/RandomHorizontalFlip', p=0.5),
        dict(type='torchvision/TrivialAugmentWide', interpolation='bilinear'),
        dict(type='PackInputs'),
    ]

    # here the Normalize params is for the RGB format
    data_preprocessor = dict(
        num_classes=1000,
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=False,
    )


Data Preprocessors
------------------

The data preprocessor is also a component to process the data before feeding data to the neural network.
Comparing with the data transforms, the data preprocessor is a module of the classifier,
and it takes a batch of data to process, which means it can use GPU and batch to accelebrate the processing.

The default data preprocessor in MMPreTrain could do the pre-processing like following:

1. Move data to the target device.
2. Pad inputs to the maximum size of current batch.
3. Stack inputs to a batch.
4. Convert inputs from bgr to rgb if the shape of input is (3, H, W).
5. Normalize image with defined std and mean.
6. Do batch augmentations like Mixup and CutMix during training.

You can configure the data preprocessor by the ``data_preprocessor`` field or ``model.data_preprocessor`` field in the config file. Typical usages are as below:

.. code-block:: python

    data_preprocessor = dict(
        # RGB format normalization parameters
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True,    # convert image from BGR to RGB
    )

Or define in ``model.data_preprocessor`` as following:

.. code-block:: python

   model = dict(
       backbone = ...,
       neck = ...,
       head = ...,
       data_preprocessor = dict(
                            mean=[123.675, 116.28, 103.53],
                            std=[58.395, 57.12, 57.375],
                            to_rgb=True)
       train_cfg=...,
   )

Note that the ``model.data_preprocessor`` has higher priority than ``data_preprocessor``.

.. autosummary::
   :toctree: generated
   :nosignatures:

   ClsDataPreprocessor
   SelfSupDataPreprocessor
   TwoNormDataPreprocessor
   VideoDataPreprocessor

.. module:: mmpretrain.models.utils.batch_augments

Batch Augmentations
^^^^^^^^^^^^^^^^^^^^

The batch augmentation is a component of data preprocessors. It involves multiple samples and mix them in some way, such as Mixup and CutMix.

These augmentations are usually only used during training, therefore, we use the ``model.train_cfg`` field to configure them in config files.

.. code-block:: python

   model = dict(
       backbone=...,
       neck=...,
       head=...,
       train_cfg=dict(augments=[
           dict(type='Mixup', alpha=0.8),
           dict(type='CutMix', alpha=1.0),
       ]),
   )

You can also specify the probabilities of every batch augmentation by the ``probs`` field.

.. code-block:: python

   model = dict(
       backbone=...,
       neck=...,
       head=...,
       train_cfg=dict(augments=[
           dict(type='Mixup', alpha=0.8),
           dict(type='CutMix', alpha=1.0),
       ], probs=[0.3, 0.7])
   )

Here is a list of batch augmentations can be used in MMPreTrain.

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: callable.rst

   Mixup
   CutMix
   ResizeMix
