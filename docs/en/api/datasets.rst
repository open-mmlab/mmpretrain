.. role:: hidden
    :class: hidden-section

mmcls.datasets
===================================

The ``datasets`` package contains several usual datasets for image classification tasks and some dataset wrappers.

.. contents:: mmcls.datasets
   :depth: 2
   :local:
   :backlinks: top

.. currentmodule:: mmcls.datasets

Custom Dataset
--------------

.. autoclass:: CustomDataset

ImageNet
--------

.. autoclass:: ImageNet

.. autoclass:: ImageNet21k

CIFAR
-----

.. autoclass:: CIFAR10

.. autoclass:: CIFAR100

MNIST
-----

.. autoclass:: MNIST

.. autoclass:: FashionMNIST

VOC
---

.. autoclass:: VOC

CUB
---

.. autoclass:: CUB

Base classes
------------

.. autoclass:: BaseDataset

.. autoclass:: MultiLabelDataset

Dataset Wrappers
----------------

ConcatDataset

RepeatDataset

ClassBalancedDataset

.. autoclass:: KFoldDataset
