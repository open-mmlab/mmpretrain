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

Base classes
------------

.. autoclass:: BaseDataset

.. autoclass:: MultiLabelDataset

Dataset Wrappers
----------------

ConcatDataset

TODO: add MMEngine Link

RepeatDataset

TODO: add MMEngine Link

ClassBalancedDataset

TODO: add MMEngine Link

.. autoclass:: KFoldDataset
