.. role:: hidden
    :class: hidden-section

.. module:: mmcls.datasets

mmcls.datasets
===================================

The ``datasets`` package contains several usual datasets for image classification tasks and some dataset wrappers.

.. contents:: mmcls.datasets
   :depth: 2
   :local:
   :backlinks: top

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

Retrieval
---------

.. autoclass:: InShop

LongTailDataSet
---

.. autoclass:: LongTailCIFAR10

.. autoclass:: LongTailCIFAR100

.. autoclass:: LongTailImageNet

Base classes
------------

.. autoclass:: BaseDataset

.. autoclass:: MultiLabelDataset

Dataset Wrappers
----------------

.. autoclass:: KFoldDataset

The dataset wrappers in the MMEngine can be directly used in MMClassification.

.. list-table::

   * - :class:`~mmengine.dataset.ConcatDataset`
     - A wrapper of concatenated dataset.
   * - :class:`~mmengine.dataset.RepeatDataset`
     - A wrapper of repeated dataset.
   * - :class:`~mmengine.dataset.ClassBalancedDataset`
     - A wrapper of class balanced dataset.
