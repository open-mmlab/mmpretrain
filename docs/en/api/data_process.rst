.. role:: hidden
    :class: hidden-section

Data Process
*****************

In MMClassification, the data process and the dataset is decomposed. The
datasets only define how to get samples' basic information from the file
system. These basic information includes the ground-truth label and raw
images data / the paths of images.The data process includes data transformations,
data preprocessors and batch augmentations.

- :ref:`datatransformations`: Transformations includes loading, preprocessing and formatting.
- :ref:`datapreprocessors`: Processes includes collate, normalization, stacking and channel fliping.
- :ref:`batchaugmentations`: Batch augmentation involves multiple samples, such as Mixup and CutMix.

.. _datatransformations:

Data Transformations
^^^^^^^^^^^^^^^^^^^^

To prepare the inputs data, we need to do some transformations on these basic
information. These transformations includes loading, preprocessing and
formatting. And a series of data transformations makes up a data pipeline.
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

Every item of a pipeline list is one of the following data transformations class. And if you want to add a custom data transformation class, the tutorial :doc:`Custom Data Pipelines </advanced_guides/pipeline.md>` will help you.

.. contents:: data_process
   :depth: 3
   :local:
   :backlinks: top

.. currentmodule:: mmcls.datasets.transforms


Loading
=======

LoadImageFromFile
-----------------

TODO: add MMCV Link

Compose Transforms
==================

Compose is a helper transform to combine a series of data transformations.

Compose
---------------------

TODO: add MMCV Link

Preprocessing and Augmentation
==============================

CenterCrop
----------

TODO: add MMCV Link

Lighting
---------------------
.. autoclass:: Lighting

Pad
---------------------

TODO: add MMCV Link

Resize
---------------------

TODO: add MMCV Link

ResizeEdge
---------------------
.. autoclass:: ResizeEdge

RandomCrop
---------------------
.. autoclass:: RandomCrop

RandomErasing
---------------------
.. autoclass:: RandomErasing

RandomFlip
---------------------

TODO: add MMCV Link

RandomGrayscale
---------------------

TODO: add MMCV Link

RandomResizedCrop
---------------------
.. autoclass:: RandomResizedCrop

ColorJitter
---------------------
.. autoclass:: ColorJitter

Albumentations
---------------------
.. autoclass:: Albumentations


Composed Augmentation
---------------------
Composed augmentation is a kind of methods which compose a series of data
augmentation transformations, such as ``AutoAugment`` and ``RandAugment``.

.. autoclass:: AutoAugment

.. autoclass:: RandAugment

In composed augmentation, we need to specify several data transformations or
several groups of data transformations (The ``policies`` argument) as the
random sampling space. These data transformations are chosen from the below
table. In addition, we provide some preset policies in `this folder`_.

.. _this folder: https://github.com/open-mmlab/mmclassification/tree/dev-1.x/configs/_base_/datasets/transforms

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: classtemplate.rst

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

Formatting
==========

PackClsInputs
---------------------
.. autoclass:: PackClsInputs

Collect
---------------------
.. autoclass:: Collect

ToNumpy
---------------------
.. autoclass:: ToNumpy

ToPIL
---------------------
.. autoclass:: ToPIL

Transpose
---------------------
.. autoclass:: Transpose

.. _datapreprocessors:

Data Preprocessors
^^^^^^^^^^^^^^^^^^

Data Preprocessors in MMClassification could do the pre-processing like following:

    - Collate and move data to the target device.
    - Pad inputs to the maximum size of current batch.
    - Stack inputs to batch_inputs.
    - Convert inputs from bgr to rgb if the shape of input is (3, H, W).
    - Normalize image with defined std and mean.
    - Do batch augmentations like Mixup and Cutmix during training.

In MMClassification, the data preprocessor could be fefined in the a ``preprocess_cfg`` argument or as a part of :ref:`classifiers`. Typical usages are as below:

.. code-block:: python

    preprocess_cfg = dict(
        # RGB format normalization parameters
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True,    # convert image from BGR to RGB
    )

Or define in ``classifiers.data_preprocessor`` as following:

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

Note that the ``classifiers.data_preprocessor`` has higher priority than ``preprocess_cfg``.

.. currentmodule:: mmcls.models

Datapreprocessors
------------------

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: classtemplate.rst

    ClsDataPreprocessor

.. _batchaugmentations:

Batch Augmentations
^^^^^^^^^^^^^^^^^^^^

Batch augmentation is the augmentation which involve multiple samples, such as Mixup and CutMix.

In MMClassification, these batch augmentation is executed in ``classifiers.data_preprocessor.forword`` but used as a part of :ref:`classifiers`. A typical usage is as below:

.. code-block:: python

   model = dict(
       backbone = ...,
       neck = ...,
       head = ...,
       train_cfg=dict(augments=[
           dict(type='BatchMixup', alpha=0.8, prob=0.5, num_classes=num_classes),
           dict(type='BatchCutMix', alpha=1.0, prob=0.5, num_classes=num_classes),
       ]))
   )

.. currentmodule:: mmcls.models.utils.batch_augments

Datapreprocessors
------------------

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: classtemplate.rst

    Mixup
    CutMix
    ResizeMix
    RandomBatchAugment
