.. role:: hidden
    :class: hidden-section

Data Process
=================

In MMClassification, the data process and the dataset is decomposed. The
datasets only define how to get samples' basic information from the file
system. These basic information includes the ground-truth label and raw
images data / the paths of images.The data process includes data transforms,
data preprocessors and batch augmentations.

- :mod:`Data Transforms <mmcls.datasets.transforms>`: Transforms includes loading, preprocessing, formatting and etc.
- :mod:`Data Preprocessors <mmcls.models.utils.data_preprocessor>`: Processes includes collate, normalization, stacking, channel fliping and etc.

  - :mod:`Batch Augmentations <mmcls.models.utils.batch_augments>`: Batch augmentation involves multiple samples, such as Mixup and CutMix.

.. module:: mmcls.datasets.transforms

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
        dict(type='PackClsInputs'),
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

Processing and Augmentation
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: data_transform.rst

   Albumentations
   ColorJitter
   EfficientNetCenterCrop
   EfficientNetRandomCrop
   Lighting
   RandomCrop
   RandomErasing
   RandomResizedCrop
   ResizeEdge

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

To specify the augmentation combination (The ``policies`` argument), you can use string to specify
from some preset policies.

.. list-table::
   :widths: 20 20 60
   :header-rows: 1

   * - Preset policy
     - Use for
     - Description
   * - "imagenet"
     - :class:`AutoAugment`
     - Policy for ImageNet, come from `DeepVoltaire/AutoAugment`_
   * - "timm_increasing"
     - :class:`RandAugment`
     - The ``_RAND_INCREASING_TRANSFORMS`` policy from `timm`_

.. _DeepVoltaire/AutoAugment: https://github.com/DeepVoltaire/AutoAugment
.. _timm: https://github.com/rwightman/pytorch-image-models

And you can also configure a group of policies manually by selecting from the below table.

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
   Invert
   Posterize
   Rotate
   Sharpness
   Shear
   Solarize
   SolarizeAdd
   Translate
   BaseAugTransform

Formatting
^^^^^^^^^^

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: data_transform.rst

   Collect
   PackClsInputs
   ToNumpy
   ToPIL
   Transpose


MMCV transforms
^^^^^^^^^^^^^^^

We also provides many transforms in MMCV. You can use them directly in the config files. Here are some frequently used transforms, and the whole transforms list can be found in :external+mmcv:doc:`api/transforms`.

.. list-table::
   :widths: 50 50

   * - :external:class:`~mmcv.transforms.LoadImageFromFile`
     - Load an image from file.
   * - :external:class:`~mmcv.transforms.Resize`
     - Resize images & bbox & seg & keypoints.
   * - :external:class:`~mmcv.transforms.RandomResize`
     - Random resize images & bbox & keypoints.
   * - :external:class:`~mmcv.transforms.RandomFlip`
     - Flip the image & bbox & keypoints & segmentation map.
   * - :external:class:`~mmcv.transforms.RandomGrayscale`
     - Randomly convert image to grayscale with a probability.
   * - :external:class:`~mmcv.transforms.CenterCrop`
     - Crop the center of the image, segmentation masks, bounding boxes and key points. If the crop area exceeds the original image and ``auto_pad`` is True, the original image will be padded before cropping.
   * - :external:class:`~mmcv.transforms.Normalize`
     - Normalize the image.
   * - :external:class:`~mmcv.transforms.Compose`
     - Compose multiple transforms sequentially.

.. module:: mmcls.models.utils.data_preprocessor

Data Preprocessors
------------------

The data preprocessor is also a component to process the data before feeding data to the neural network.
Comparing with the data transforms, the data preprocessor is a module of the classifier,
and it takes a batch of data to process, which means it can use GPU and batch to accelebrate the processing.

The default data preprocessor in MMClassification could do the pre-processing like following:

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

.. module:: mmcls.models.utils.batch_augments

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

Here is a list of batch augmentations can be used in MMClassification.

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: callable.rst

   Mixup
   CutMix
   ResizeMix
