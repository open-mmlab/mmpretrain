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

    img_norm_cfg = dict(
        mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
    train_pipeline = [
        dict(type='LoadImageFromFile'),
        dict(type='RandomResizedCrop', size=224),
        dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
        dict(type='Normalize', **img_norm_cfg),
        dict(type='ImageToTensor', keys=['img']),
        dict(type='ToTensor', keys=['gt_label']),
        dict(type='Collect', keys=['img', 'gt_label'])
    ]
    test_pipeline = [
        dict(type='LoadImageFromFile'),
        dict(type='Resize', size=256),
        dict(type='CenterCrop', crop_size=224),
        dict(type='Normalize', **img_norm_cfg),
        dict(type='ImageToTensor', keys=['img']),
        dict(type='Collect', keys=['img'])
    ]

    data = dict(
        train=dict(..., pipeline=train_pipeline),
        val=dict(..., pipeline=test_pipeline),
        test=dict(..., pipeline=test_pipeline),
    )

Every item of a pipeline list is one of the following data transformations class. And if you want to add a custom data transformation class, the tutorial :doc:`Custom Data Pipelines </tutorials/data_pipeline>` will help you.

.. contents:: mmcls.datasets.pipelines
   :depth: 2
   :local:
   :backlinks: top

.. currentmodule:: mmcls.datasets.pipelines

Loading
=======

LoadImageFromFile
---------------------
.. autoclass:: LoadImageFromFile

Preprocessing and Augmentation
==============================

CenterCrop
---------------------
.. autoclass:: CenterCrop

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
.. autoclass:: Resize

RandomCrop
---------------------
.. autoclass:: RandomCrop

RandomErasing
---------------------
.. autoclass:: RandomErasing

RandomFlip
---------------------
.. autoclass:: RandomFlip

RandomGrayscale
---------------------
.. autoclass:: RandomGrayscale

RandomResizedCrop
---------------------
.. autoclass:: RandomResizedCrop

ColorJitter
---------------------
.. autoclass:: ColorJitter


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
