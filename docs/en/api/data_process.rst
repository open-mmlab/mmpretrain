.. role:: hidden
    :class: hidden-section

Data Transformations
***********************************

In MMClassification, the data preparation and the dataset is decomposed. The
datasets only define how to get samples' basic information from the file
system. These basic information includes the ground-truth label and raw images
data / the paths of images.

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

Every item of a pipeline list is one of the following data transformations class. And if you want to add a custom data transformation class, the tutorial :doc:`Custom Data Pipelines </tutorials/data_pipeline>` will help you.

.. contents:: mmcls.datasets.transforms
   :depth: 2
   :local:
   :backlinks: top

.. currentmodule:: mmcls.datasets.pipelines

Loading
=======

LoadImageFromFile
-----------------
LoadImageFromFile
:ref:`LoadImageFromFile`

Compose Transforms
==================

Compose is a helper transform to combine a series of data transformations.

Compose
---------------------
.. autoclass:: Compose

Preprocessing and Augmentation
==============================

CenterCrop
----------
CenterCrop
:ref:`CenterCrop`

Lighting
---------------------
.. autoclass:: Lighting

Normalize
---------------------
.. autoclass:: Normalize

Pad
---------------------
.. autoclass:: Pad

Resize
---------------------
Resize
:ref:`Resize`

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
RandomFlip
:ref:`RandomFlip`

RandomGrayscale
---------------------
.. autoclass:: RandomGrayscale

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

.. _this folder: https://github.com/open-mmlab/mmclassification/tree/master/configs/_base_/datasets/pipelines

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

ImageToTensor
---------------------
.. autoclass:: ImageToTensor

ToNumpy
---------------------
.. autoclass:: ToNumpy

ToPIL
---------------------
.. autoclass:: ToPIL

ToTensor
---------------------
.. autoclass:: ToTensor

Transpose
---------------------
.. autoclass:: Transpose

.. role:: hidden
    :class: hidden-section

Batch Augmentation
===================================

Batch augmentation is the augmentation which involve multiple samples, such as Mixup and CutMix.

In MMClassification, these batch augmentation is used as a part of :ref:`classifiers`. A typical usage is as below:

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

Mixup
-----
.. autoclass:: Mixup

CutMix
------
.. autoclass:: CutMix

ResizeMix
---------
.. autoclass:: ResizeMix

BatchBaseTransform
------------------
.. autoclass:: RandomBatchAugment
